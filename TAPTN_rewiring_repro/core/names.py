"""Model / dataset name maps used by the rewiring tables."""

DATASETS = ["cornell", "texas", "washington", "wisconsin"]
DS_PRINT = {
    "cornell": "Cornell",
    "texas": "Texas",
    "washington": "Washington",
    "wisconsin": "Wisconsin",
}

# Short keys as they appear in Section-2 pickle filenames ([:15] truncation).
LMARENA = {
    "gpt_3.5_turbo_0125": 1224,
    "gpt_3.5_turbo_0": 1224,
    "phi_4": 1255,
    "gemma_2_9b_it": 1265,
    "llama_3_70b_ins": 1276,
    "llama_3.1_70b_i": 1293,
    "qwen2.5_vl_72b": 1302,
    "qwen_2.5_72b_in": 1302,
    "qwen_2.5_72b": 1302,
    "llama_3.3_70b_i": 1319,
    "gpt_oss_120b": 1353,
    "qwen3.5_27b": 1409,
    "gemma_4_31b_it": 1451,
    "glm_5.1": 1475,
}

DISPLAY = {
    "gpt_3.5_turbo_0125": "GPT-3.5-Turbo-0125",
    "gpt_3.5_turbo_0": "GPT-3.5-Turbo-0125",
    "phi_4": "Phi-4",
    "gemma_2_9b_it": "Gemma-2-9B",
    "llama_3_70b_ins": "Llama-3-70B-Instruct",
    "llama_3.1_70b_i": "Llama-3.1-70B-Instruct",
    "qwen2.5_vl_72b": "Qwen2.5-VL-72B-Instruct",
    "qwen_2.5_72b_in": "Qwen-2.5-72B",
    "llama_3.3_70b_i": "Llama-3.3-70B-Instruct",
    "gpt_oss_120b": "GPT-OSS-120B",
    "qwen3.5_27b": "Qwen3.5-27B",
    "gemma_4_31b_it": "Gemma-4-31B-it",
    "glm_5.1": "GLM-5.1",
}

PARAMS = {
    "gpt_3.5_turbo_0125": "~20B",
    "phi_4": "14B",
    "gemma_2_9b_it": "9B",
    "llama_3_70b_ins": "70B",
    "llama_3.1_70b_i": "70B",
    "qwen2.5_vl_72b": "72B",
    "llama_3.3_70b_i": "70B",
}

# Paper Table 2 / rewiring_stats / nolabel panel (keep VL, not the non-VL Qwen).
PANEL_7 = [
    "gpt_3.5_turbo_0125",
    "phi_4",
    "gemma_2_9b_it",
    "llama_3_70b_ins",
    "llama_3.1_70b_i",
    "qwen2.5_vl_72b",
    "llama_3.3_70b_i",
]

PANEL_4_NEW = [
    "gpt_oss_120b",
    "qwen3.5_27b",
    "gemma_4_31b_it",
    "glm_5.1",
]

# CLI short names -> OpenRouter-style model ids used by run_taptn.py
CLI_MODELS = {
    "phi-4": "microsoft/phi-4",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "llama-3-70b-instruct": "meta-llama/llama-3-70b-instruct",
    "llama-3.1-70b-instruct": "meta-llama/llama-3.1-70b-instruct",
    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
    "qwen2.5-vl-72b-instruct": "qwen/qwen2.5-vl-72b-instruct",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo-0125",
    "gemma-4-31b-it": "google/gemma-4-31b-it",
    "qwen3.5-27b": "qwen/qwen3.5-27b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "glm-5.1": "z-ai/glm-5.1",
}

CANON = {
    "gpt_3.5_turbo_0": "gpt_3.5_turbo_0125",
    "gpt_3.5_turbo_0125": "gpt_3.5_turbo_0125",
}


def canon_model(name: str) -> str:
    name = (name or "").strip("_")
    if name.startswith("gpt_3"):
        return "gpt_3.5_turbo_0125"
    return CANON.get(name, name)


def display(name: str) -> str:
    name = canon_model(name)
    return DISPLAY.get(name, name)
