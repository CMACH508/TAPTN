"""Dataset / model display names for TAPTN ICL tables."""

MAIN_DATASETS = ["cora", "arxiv_2023", "texas", "wisconsin", "cornell"]
DS_PRINT = {
    "cora": "Cora",
    "arxiv_2023": "arXiv-2023",
    "texas": "Texas",
    "wisconsin": "Wisconsin",
    "cornell": "Cornell",
}

CURRENT_MODELS = ["llama", "gemma", "qwen", "glm"]
CURRENT_TAPTN_MODELS = ["gemma", "qwen", "glm"]

MODEL_PRINT = {
    "llama": "Llama-3.3-70B (anchor)",
    "gemma": "Gemma-4-31B-it",
    "qwen": "Qwen3.5-27B",
    "glm": "GLM-5.1",
}

MAIN_METHODS = ["0hop", "gicl1", "gicl2", "taptn1", "taptn2"]
METHOD_PRINT = {
    "0hop": "0-hop (zero-shot CoT)",
    "gicl1": "GraphICL+SAT 1-hop",
    "gicl2": "GraphICL+SAT 2-hop",
    "taptn1": "TAPTN 1-hop",
    "taptn2": "TAPTN 2-hop",
}

FACTORIAL_ROWS = ["gicl", "instr", "aggr", "taptn"]
FACTORIAL_PRINT = {
    "gicl": "GraphICL+SAT (no instr., no aggr.)",
    "instr": "+ instructions (no aggr.)",
    "aggr": "+ iterative aggregation (no instr.)",
    "taptn": "TAPTN (instr. + aggr.)",
}
