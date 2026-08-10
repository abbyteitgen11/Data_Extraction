"""
des_pipeline — extract deep eutectic solvent data from a paper into a Neo4j graph.

Pipeline stages, in order:

    XML  --router-->  tables      --extract_table------>  wide + long CSV
                      figures     --extract_figures---->  figures.csv (human review)
                      references  --extract_references->  references.csv (Crossref DOIs)
                      prose       --extract_text_llm--->  sections_llm.csv (human review)

    component names   --enrich_components-->  components.csv (PubChem / NIST)

    CSVs              --build_graph-------->  Neo4j

Run it with ``python run_pipeline.py --steps ...`` from the repository root.
"""
