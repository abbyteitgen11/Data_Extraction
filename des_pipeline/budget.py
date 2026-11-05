"""
Spend safeguards for paid model calls. Nothing here makes a call.

Two caps, both enforced BEFORE a call is sent, never after:

    monthly   DES_MONTHLY_LIMIT_USD (default 80) minus DES_MONTHLY_MARGIN_USD (default 10):
              the total of every usage.csv row this calendar month, across ALL run tags
              (runs/*/usage.csv and data/usage.csv).
    per run   DES_BUDGET_USD: the total of this run tag's own usage.csv. 0 = no cap.

A call first RESERVES its worst case -- its estimated input plus the full `max_tokens`
as output, at the model's price -- and is refused if spent + reserved + worst case would
pass either cap. When the call returns, the reservation is released and the real cost
(from the API's usage block) is added. Reservations make the check correct when eight
calls are in flight at once: none of them can assume the others cost nothing.

What this cannot see: spend on the same API key from anywhere else (another project,
the Console workbench, a colleague). Set a spend limit on the key's workspace in the
Claude Console as the final backstop; this module is the first one.

Stopping loses nothing: every answer received is in the response cache, so running the
same command again resumes where it stopped.
"""
import csv
import os
import threading
import time

from . import config

MONTHLY_LIMIT_USD = float(os.environ.get("DES_MONTHLY_LIMIT_USD", 80) or 80)
MONTHLY_MARGIN_USD = float(os.environ.get("DES_MONTHLY_MARGIN_USD", 10) or 0)
# Characters per token for the worst-case input estimate. Claude's tokenizer averages
# ~3.5-4 characters per token on English and chemistry text; 3 over-reserves on purpose.
CHARS_PER_TOKEN = 3.0


class BudgetExceeded(RuntimeError):
    """A cap would be passed. Every answer so far is cached; re-run to resume."""


_LOCK = threading.Lock()
_STATE = {"month": None, "month_spent": None, "run_spent": None, "reserved": 0.0}


def monthly_cap():
    return MONTHLY_LIMIT_USD - MONTHLY_MARGIN_USD


def _usage_files():
    files = list((config.ROOT / "runs").glob("*/usage.csv"))
    shared = config.SHARED_DATA / "usage.csv"
    if shared.exists():
        files.append(shared)
    return files


def _sum(path, month=None):
    total = 0.0
    try:
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                if month and not str(row.get("time", "")).startswith(month):
                    continue
                try:
                    total += float(row.get("cost_usd") or 0)
                except ValueError:
                    pass
    except OSError:
        pass
    return total


def month_spent(refresh=False):
    """US dollars spent on paid calls this calendar month, across every run tag."""
    month = time.strftime("%Y-%m")
    if refresh or _STATE["month"] != month or _STATE["month_spent"] is None:
        _STATE["month"] = month
        _STATE["month_spent"] = sum(_sum(p, month) for p in _usage_files())
    return _STATE["month_spent"]


def run_spent(refresh=False):
    """US dollars this run tag has spent (its own usage.csv, all months)."""
    if refresh or _STATE["run_spent"] is None:
        _STATE["run_spent"] = _sum(config.USAGE_CSV) if config.USAGE_CSV.exists() else 0.0
    return _STATE["run_spent"]


def price(model):
    """($ per input token, $ per output token) for a model; (0, 0) if unpriced."""
    per_m_in, per_m_out = config.ANTHROPIC_PRICES.get(model, (0.0, 0.0))
    return per_m_in / 1e6, per_m_out / 1e6


def worst_case(model, prompt_chars, max_tokens):
    p_in, p_out = price(model)
    return (prompt_chars / CHARS_PER_TOKEN) * p_in + max_tokens * p_out


def reserve(model, prompt_chars, max_tokens):
    """Reserve a call's worst case, or raise BudgetExceeded. -> the amount reserved."""
    if model not in config.ANTHROPIC_PRICES:
        raise BudgetExceeded(f"no price known for {model!r}; add it to "
                             f"config.ANTHROPIC_PRICES before spending on it")
    amount = worst_case(model, prompt_chars, max_tokens)
    with _LOCK:
        month = month_spent()
        if month + _STATE["reserved"] + amount > monthly_cap():
            raise BudgetExceeded(
                f"monthly cap: ${month:.2f} spent this month + ${_STATE['reserved']:.2f} "
                f"in flight + ${amount:.2f} for this call would pass ${monthly_cap():.2f} "
                f"(limit ${MONTHLY_LIMIT_USD:.0f} - margin ${MONTHLY_MARGIN_USD:.0f})")
        if config.BUDGET_USD:
            spent = run_spent()
            if spent + _STATE["reserved"] + amount > config.BUDGET_USD:
                raise BudgetExceeded(
                    f"run cap: ${spent:.2f} spent by run {config.RUN_TAG or '(untagged)'} "
                    f"+ ${_STATE['reserved']:.2f} in flight + ${amount:.2f} would pass "
                    f"DES_BUDGET_USD=${config.BUDGET_USD:.2f}")
        _STATE["reserved"] += amount
    return amount


def settle(reserved, actual_cost):
    """Release a reservation and record what the call really cost."""
    with _LOCK:
        _STATE["reserved"] = max(0.0, _STATE["reserved"] - reserved)
        _STATE["month_spent"] = month_spent() + (actual_cost or 0.0)
        _STATE["run_spent"] = run_spent() + (actual_cost or 0.0)


def status():
    """One line for logs: what has been spent and what is left."""
    month = month_spent(refresh=True)
    left = monthly_cap() - month
    run = run_spent(refresh=True)
    return (f"spend this month ${month:.2f} of the ${monthly_cap():.2f} cap "
            f"(${left:.2f} left); this run ${run:.2f}"
            f"{f' of ${config.BUDGET_USD:.2f}' if config.BUDGET_USD else ''}")
