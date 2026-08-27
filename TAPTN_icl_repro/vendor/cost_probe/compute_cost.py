"""Turn measured tokens + OpenRouter prices into the cost/runtime table (per 400 test nodes
and per 1000 test nodes). All inputs are the measured quantities in results.json."""
import json, os
OUT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(OUT, "results.json")))

# OpenRouter standard prices ($ per 1e6 tokens), retrieved 2026-06-10
P70_IN, P70_OUT = 0.10, 0.32      # meta-llama/llama-3.3-70b-instruct
P8_IN,  P8_OUT  = 0.02, 0.05      # meta-llama/llama-3.1-8b-instruct

# ---- measured per-call tokens ----
g_in  = R["graphicl_2hop_input_recon"]["input_total"]["mean"]     # 1724
g_out = R["graphicl_2hop_output_pkl"]["mean"]                     # 393
t1_in, t1_out = R["taptn_iter1_from_json"]["input_total"]["mean"], R["taptn_iter1_output_pkl"]["mean"]  # 3365 / 640
t2_in, t2_out = R["taptn_iter2_from_json"]["input_total"]["mean"], R["taptn_iter2_output_pkl"]["mean"]  # 3091 / 593
# extraction (8B): input ~= category list system (~250) + reasoning text; output ~20
ex_sys = 250
ex_out = 20

N_TEST = 400
N_ITER1_NODES = R["taptn_iter1_output_pkl"]["n_records"]   # 1430 (test + 1-hop neighbours)

def cost(n, ti, to, pin, pout):
    return n * (ti * pin + to * pout) / 1e6

def fmt(d):
    return {k: round(v, 4) for k, v in d.items()}

summary = {}

# ============ GraphICL 2-hop ============
g_reason = cost(N_TEST, g_in, g_out, P70_IN, P70_OUT)
g_extract = cost(N_TEST, ex_sys + g_out, ex_out, P8_IN, P8_OUT)
summary["graphicl_2hop"] = {
    "calls_70b": N_TEST, "calls_8b_extract": N_TEST,
    "input_tokens": round(N_TEST * (g_in + ex_sys + g_out)),
    "output_tokens": round(N_TEST * (g_out + ex_out)),
    "total_tokens": round(N_TEST * (g_in + g_out + ex_sys + g_out + ex_out)),
    "cost_usd": round(g_reason + g_extract, 4),
    "cost_per_1k_nodes": round((g_reason + g_extract) / N_TEST * 1000, 3),
    "accuracy": R["graphicl_2hop_output_pkl"]["accuracy"],
}

# ============ TAPTN uniform (all reasoning on 70B) ============
u_i1 = cost(N_ITER1_NODES, t1_in, t1_out, P70_IN, P70_OUT)
u_i2 = cost(N_TEST,        t2_in, t2_out, P70_IN, P70_OUT)
u_ex = cost(N_ITER1_NODES, ex_sys + t1_out, ex_out, P8_IN, P8_OUT) + cost(N_TEST, ex_sys + t2_out, ex_out, P8_IN, P8_OUT)
u_total_tok = (N_ITER1_NODES * (t1_in + t1_out) + N_TEST * (t2_in + t2_out)
               + N_ITER1_NODES * (ex_sys + t1_out + ex_out) + N_TEST * (ex_sys + t2_out + ex_out))
summary["taptn_uniform_70b"] = {
    "calls_70b": N_ITER1_NODES + N_TEST, "calls_8b_extract": N_ITER1_NODES + N_TEST,
    "reasoning_calls_per_test_node": round((N_ITER1_NODES + N_TEST) / N_TEST, 2),
    "total_tokens": round(u_total_tok),
    "tokens_per_test_node": round(u_total_tok / N_TEST),
    "cost_usd": round(u_i1 + u_i2 + u_ex, 4),
    "cost_per_1k_nodes": round((u_i1 + u_i2 + u_ex) / N_TEST * 1000, 3),
    "accuracy": R["taptn_iter2_output_pkl"]["accuracy"],
}

# ============ TAPTN economy (iter1 pre-pass on 8B, iter2 on 70B) ============
e_i1 = cost(N_ITER1_NODES, t1_in, t1_out, P8_IN, P8_OUT)
e_i2 = cost(N_TEST,        t2_in, t2_out, P70_IN, P70_OUT)
e_ex = u_ex
summary["taptn_economy_8b_70b"] = {
    "calls_8b_reason": N_ITER1_NODES, "calls_70b_reason": N_TEST,
    "cost_usd": round(e_i1 + e_i2 + e_ex, 4),
    "cost_per_1k_nodes": round((e_i1 + e_i2 + e_ex) / N_TEST * 1000, 3),
}

# ============ ratios ============
summary["ratios"] = {
    "taptn_uniform_vs_graphicl_cost": round(summary["taptn_uniform_70b"]["cost_usd"] / summary["graphicl_2hop"]["cost_usd"], 2),
    "taptn_uniform_vs_economy_cost": round(summary["taptn_uniform_70b"]["cost_usd"] / summary["taptn_economy_8b_70b"]["cost_usd"], 2),
    "instruction_input_overhead_tokens": round(R["taptn_iter1_from_json"]["system"]["mean"] - R["graphicl_2hop_input_recon"]["system"]["mean"]),
}

# ============ wall-clock (measured latency × calls / concurrency) ============
lat70 = R["timing"]["llama33_70b_main_latency_s"]["mean"]
lat8  = R["timing"]["llama31_8b_extract_latency_s"]["mean"]
C = 100  # max_workers used in production runs
def wall(n_calls, lat):
    import math
    return math.ceil(n_calls / C) * lat
# TAPTN: iter1 then iter2 (dependency); per node reasoning+extraction sequential
taptn_wall = (wall(N_ITER1_NODES, lat70 + lat8) + wall(N_TEST, lat70 + lat8))
gicl_wall  = wall(N_TEST, lat70 + lat8)
summary["wallclock_est_s"] = {
    "latency_70b_s": lat70, "latency_8b_s": lat8, "concurrency": C,
    "graphicl_2hop_total_s": round(gicl_wall),
    "taptn_total_s": round(taptn_wall),
    "graphicl_s_per_node": round(gicl_wall / N_TEST, 2),
    "taptn_s_per_node": round(taptn_wall / N_TEST, 2),
}

print(json.dumps(summary, indent=2))
json.dump(summary, open(os.path.join(OUT, "cost_summary.json"), "w"), indent=2)
print("\nsaved ->", os.path.join(OUT, "cost_summary.json"))
