"""固化：全邻居多数票门控（2-hop 选择性聚合的推荐落地形态）。

规则：仅当 2-hop 改判方向与该节点全部 1-hop 邻居标签的多数票一致时才采纳 2-hop，
否则保留 1-hop。零额外推理（复用已算 1-hop / 2-hop / 邻居标签），跨模型统一。

为每个模型产出 cora_2hop_gated_{model}.pkl：
  {final, gold, acc1, acc2, acc_gate, n_adopt, n_flip, decisions}
"""
import pickle, numpy as np, torch
from utils.utils import load_data
from collections import Counter

THRS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 2.01]
GOLD2OPT = {"Theory": "Computational Learning Theory"}

CFG = {
    "gemma-4-31b-it": dict(
        h1="cora_hop1_noanon_intv-era_gemma-4-31b-it_s2era_on_era.pkl",
        h2="cora_hop1_noanon_gemma-4-31b-it_2hop_fc_v1v0_s2.pkl",
        nb="cora_neighbors_eraS1_gemma-4-31b-it_s2.pkl", nbthr=0.5),
    "glm-5.1": dict(
        h1="cora_hop1_noanon_noguide_glm-5.1_s2era_on_base_wr.pkl",
        h2="cora_hop1_noanon_glm-5.1_2hop_fc_v0v1X_s2.pkl",
        nb="cora_neighbors_noguideS1_glm-5.1_s2.pkl", nbthr=0.1),
    "qwen3.5-27b": dict(
        h1="cora_hop1_noanon_noguide_qwen3.5-27b_s2era_on_base_wr.pkl",
        h2="cora_hop1_noanon_qwen3.5-27b_2hop_part_v1_s2.pkl",
        nb="cora_neighbors_noguideS1_qwen3.5-27b_s2.pkl", nbthr=0.1),
}


def tg(c):
    return "Theory" if c == "Computational Learning Theory" else c


def label_from_rec(r, thr):
    fired = bool(r["has_disc"] and r["pick"] is not None and (r["s1"] - r["s2"]) < thr)
    return tg(r["pick"]) if fired else tg(r["base"])


def finals_at(records, thr):
    return {int(n): label_from_rec(r, thr) for n, r in records.items()}


def best_thr(records, gold):
    best, bt = -1, None
    for thr in THRS:
        f = finals_at(records, thr)
        acc = sum(1 for n in f if f[n] == gold.get(n)) / len(f)
        if acc > best:
            best, bt = acc, thr
    return bt


def main():
    data, _, text = load_data("cora", use_text=True, seed=42)
    ei = data.edge_index.numpy()
    nbr = {}
    for s, d in zip(ei[0], ei[1]):
        nbr.setdefault(int(s), set()).add(int(d))
        nbr.setdefault(int(d), set()).add(int(s))
    test_ids = set(int(x) for x in np.where(data.test_mask.numpy())[0])

    print(f"{'model':16s} {'1-hop':>7} {'2-hop':>7} {'gate':>7} {'采纳/分歧':>10}")
    for model, cfg in CFG.items():
        d1 = pickle.load(open(cfg["h1"], "rb"))
        d2 = pickle.load(open(cfg["h2"], "rb"))
        r1, r2 = d1["records"], d2["records"]
        gold = {int(k): tg(v) for k, v in d1["gold"].items()}
        t1, t2 = best_thr(r1, gold), best_thr(r2, gold)
        f1, f2 = finals_at(r1, t1), finals_at(r2, t2)

        # 全部邻居标签：测试邻居用 f1，增补邻居用邻居 S2 文件
        nblab = {}
        dn = pickle.load(open(cfg["nb"], "rb"))["records"]
        for k, r in dn.items():
            nblab[int(k)] = label_from_rec(r, cfg["nbthr"])

        def nb_label(x):
            return f1.get(x) if x in test_ids else nblab.get(x)

        keys = sorted(set(f1) & set(f2))
        final, decisions = {}, {}
        n_flip = n_adopt = 0
        for n in keys:
            l1, l2 = f1[n], f2[n]
            adopt = False
            if l1 != l2:
                n_flip += 1
                votes = [nb_label(x) for x in nbr.get(n, set())]
                votes = [v for v in votes if v]
                fullmaj = Counter(votes).most_common(1)[0][0] if votes else l1
                adopt = (l2 == fullmaj)
            chosen = l2 if adopt else l1
            if adopt:
                n_adopt += 1
            final[n] = GOLD2OPT.get(chosen, chosen)
            decisions[n] = dict(l1=l1, l2=l2, adopt=adopt, gold=gold[n])
        N = len(keys)
        acc1 = sum(1 for n in keys if f1[n] == gold[n]) / N * 100
        acc2 = sum(1 for n in keys if f2[n] == gold[n]) / N * 100
        accg = sum(1 for n in keys if tg(final[n]) == gold[n]) / N * 100
        out = dict(final=final, gold={n: GOLD2OPT.get(gold[n], gold[n]) for n in gold},
                   acc1=acc1, acc2=acc2, acc_gate=accg, n_adopt=n_adopt, n_flip=n_flip,
                   t1=t1, t2=t2, decisions=decisions, model=model)
        op = f"cora_2hop_gated_{model}.pkl"
        pickle.dump(out, open(op, "wb"))
        print(f"{model:16s} {acc1:7.2f} {acc2:7.2f} {accg:7.2f} {f'{n_adopt}/{n_flip}':>10}  -> {op}")


if __name__ == "__main__":
    main()
