"""D1 pilot, step 4 (CPU): probes, controls and the three separate properties R / A / U (owner GO 2026-11-13).

Inputs: d1_extract.py output dir (regions.json, taps_<arm>.npz, taps_REF.npz, use_<arm>.json) + d1_sample.json +
d1_align.json. Every split is FAMILY-grouped (GroupKFold on the family id), so no family is in both probe-train and
probe-test. Every tap is reduced by PCA to the same d (fitted on the probe-train fold only) and probed by the same
L2 logistic regression (C picked by an inner grouped CV). Reported per tap, arm and representation (region mean, region
centre frame, whole-utterance mean = context confound).

R  recoverability
   digit       10-way accuracy of the digit word (digit-by-digit postcodes)
   capacity    two-seat vs three-seat accuracy
   name_ver    verification AUC on pairs (same first name vs different), pair features |a-b| and a*b; also the AUC on
               HARD negatives only (different name, same voice + same language, or edit distance <= 2)
   concept_ver verification AUC (same concept vs different), hard negatives = same category
   xling       IT -> FR retrieval of the same family's region (same task), top-1 and mean rank, centred cosine
A  alignment with the original VC EN path (EN only): grouped-CV ridge R^2 from the tap (PCA d) to REF (PCA d), and
   linear CKA between tap and REF region means
U  use by the Thinker: per slot (full_name / postcode / query) token accuracy, slot exact and mean gold margin
Controls: shuffled labels (train fold), random projection of T1 (1024 -> 4480 Gaussian, then the same PCA), chance.
Pilot pass checks (method, not result): alignment durations, T1 digit > shuffled and > utterance mean, PCA-128 vs
PCA-256 agreement, U computed for every selected clip.
Usage: d1_probe.py --dir D --sample s.json --align a.json --arms s103,s200 --out report.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

TAPS = ("T1", "T2", "T3", "T4")
RNG = np.random.default_rng(20261113)


def edit(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]


def fit_pca(X, d):
    d = min(d, X.shape[0] - 1, X.shape[1])
    return PCA(n_components=d, random_state=0).fit(X)


def logreg(Xtr, ytr, gtr):
    best, bc = -1, 1.0
    if len(set(gtr)) >= 3:
        for C in (0.01, 0.1, 1.0):
            sc = []
            for a, b in GroupKFold(3).split(Xtr, ytr, gtr):
                if len(set(ytr[a])) < 2:
                    continue
                m = LogisticRegression(C=C, max_iter=2000).fit(Xtr[a], ytr[a])
                sc.append((m.predict(Xtr[b]) == ytr[b]).mean())
            if sc and np.mean(sc) > best:
                best, bc = np.mean(sc), C
    return LogisticRegression(C=bc, max_iter=2000).fit(Xtr, ytr)


def cls_cv(X, y, g, d, shuffle=False, k=5):
    y, g = np.asarray(y), np.asarray(g)
    pred = np.empty(len(y), dtype=object)
    unseen = 0                                    # test items whose class is absent from that fold's probe-train
    for a, b in GroupKFold(min(k, len(set(g)))).split(X, y, g):
        unseen += int(sum(1 for v in y[b] if v not in set(y[a])))
        p = fit_pca(X[a], d)
        ytr = RNG.permutation(y[a]) if shuffle else y[a]
        if len(set(ytr)) < 2:
            pred[b] = ytr[0]
            continue
        m = logreg(p.transform(X[a]), ytr, g[a])
        pred[b] = m.predict(p.transform(X[b]))
    acc = (pred == y)
    fams = sorted(set(g))
    boot = []
    for _ in range(300):
        s = RNG.choice(fams, len(fams))
        idx = np.concatenate([np.where(g == f)[0] for f in s])
        boot.append(acc[idx].mean())
    return {"acc": round(float(acc.mean()), 3), "ci95": [round(float(np.percentile(boot, 2.5)), 3),
                                                          round(float(np.percentile(boot, 97.5)), 3)], "n": int(len(y)),
            "test_items_with_class_unseen_in_train": unseen, "n_classes": len(set(y)),
            "chance_majority": round(float(Counter(y).most_common(1)[0][1] / len(y)), 3)}


def pairs_for(idx, labels, meta, hard_fn, n_max=4000):
    pos, neg, hard = [], [], []
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            a, b = idx[i], idx[j]
            if meta[a]["id"] == meta[b]["id"]:
                continue
            if labels[a] == labels[b]:
                pos.append((a, b))
            else:
                (hard if hard_fn(meta[a], meta[b]) else neg).append((a, b))
    RNG.shuffle(pos), RNG.shuffle(neg), RNG.shuffle(hard)
    return pos[:n_max], neg[:n_max], hard[:n_max]


def pair_feats(Z, prs):
    a = np.stack([Z[i] for i, _ in prs])
    b = np.stack([Z[j] for _, j in prs])
    return np.concatenate([np.abs(a - b), a * b], 1)


def verification(X, meta, sel, labels, d, hard_fn):
    """Family-split verification: PCA + pair classifier fitted on train-fold families, AUC on test-fold pairs."""
    fam = np.array([meta[i]["id"] for i in sel])
    out = {"auc": [], "auc_hard": []}
    for a, b in GroupKFold(min(4, len(set(fam)))).split(sel, groups=fam):
        tr, te = [sel[i] for i in a], [sel[i] for i in b]
        p = fit_pca(X[tr], d)
        Z = {i: z for i, z in zip(tr + te, p.transform(X[tr + te]))}
        ptr, ntr, htr = pairs_for(tr, labels, meta, hard_fn)
        pte, nte, hte = pairs_for(te, labels, meta, hard_fn)
        if not ptr or not pte or not (ntr + htr) or not (nte + hte):
            continue
        Xp = pair_feats(Z, ptr + ntr + htr)
        yp = np.r_[np.ones(len(ptr)), np.zeros(len(ntr) + len(htr))]
        m = LogisticRegression(C=0.1, max_iter=3000).fit(Xp, yp)
        s_all = m.decision_function(pair_feats(Z, pte + nte + hte))
        out["auc"].append(roc_auc_score(np.r_[np.ones(len(pte)), np.zeros(len(nte) + len(hte))], s_all))
        if hte:
            s_h = m.decision_function(pair_feats(Z, pte + hte))
            out["auc_hard"].append(roc_auc_score(np.r_[np.ones(len(pte)), np.zeros(len(hte))], s_h))
    return {k: (round(float(np.mean(v)), 3) if v else None) for k, v in out.items()}


def xling(X, meta, task):
    it = [i for i, r in enumerate(meta) if r["task"] == task and r["lang"] == "it"]
    fr = {(meta[i]["id"], meta[i].get("pos")): i for i, r in enumerate(meta) if r["task"] == task and r["lang"] == "fr"}
    q = [(i, fr[(meta[i]["id"], meta[i].get("pos"))]) for i in it if (meta[i]["id"], meta[i].get("pos")) in fr]
    if len(q) < 5:
        return None
    A = X[[i for i, _ in q]].astype(np.float32)
    B = X[[j for _, j in q]].astype(np.float32)
    A, B = A - A.mean(0), B - B.mean(0)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-9
    S = A @ B.T
    ranks = [(S[k] > S[k, k]).sum() + 1 for k in range(len(q))]
    return {"n": len(q), "top1": round(float(np.mean(np.array(ranks) == 1)), 3),
            "mean_rank": round(float(np.mean(ranks)), 2), "chance_top1": round(1 / len(q), 3)}


def cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y) ** 2
    return float(hsic / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def alignment(Xtap, Xref, fam, d):
    r2 = []
    for a, b in GroupKFold(5).split(Xtap, groups=fam):
        px, pr = fit_pca(Xtap[a], d), fit_pca(Xref[a], d)
        m = Ridge(alpha=10.0).fit(px.transform(Xtap[a]), pr.transform(Xref[a]))
        Yt, Yp = pr.transform(Xref[b]), m.predict(px.transform(Xtap[b]))
        r2.append(1 - ((Yt - Yp) ** 2).sum() / ((Yt - Yt.mean(0)) ** 2).sum())
    return {"ridge_r2": round(float(np.mean(r2)), 3), "cka": round(cka(Xtap.astype(np.float64), Xref.astype(np.float64)), 3)}


def use_summary(rows):
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for slot in {t["slot"] for t in r["tokens"] if t["slot"]}:
            ts = [t for t in r["tokens"] if t["slot"] == slot]
            for key in (f"{slot}|all", f"{slot}|{r['lang']}"):
                by[key]["tok_acc"].append(np.mean([t["argmax"] == t["gold"] for t in ts]))
                by[key]["exact"].append(all(t["argmax"] == t["gold"] for t in ts))
                by[key]["margin"].append(np.mean([t["margin"] for t in ts]))
    return {k: {"n": len(v["exact"]), "slot_exact": f"{sum(v['exact'])}/{len(v['exact'])}",
                "tok_acc": round(float(np.mean(v["tok_acc"])), 3), "mean_margin": round(float(np.mean(v["margin"])), 2)}
            for k, v in sorted(by.items())}


def _group_dur(meta):
    g = defaultdict(list)
    for r in meta:
        g[(r["lang"], r["task"])].append(r["t1"] - r["t0"])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--align", required=True)
    ap.add_argument("--arms", default="s103,s200")
    ap.add_argument("--dims", default="128,256")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} exists")
    D = Path(a.dir)
    reg = json.load(open(D / "regions.json"))
    meta = reg["regions"]
    clips = json.load(open(a.sample))["clips"]
    al = json.load(open(a.align))
    rep = {"n_regions": len(meta), "tasks": dict(Counter(r["task"] for r in meta)), "arms": {},
           "ctc_blank_used": sorted({v.get("blank") for v in al["alignments"].values() if "blank" in v}),
           "ctc_blank_expected": 0,
           "alignment_failures": {k: v["error"] for k, v in al["alignments"].items() if "error" in v},
           "extracted": sorted(p.name for p in D.glob("taps_*.npz")) + sorted(p.name for p in D.glob("use_*.json"))}
    cov = defaultdict(Counter)
    for c in clips:
        v = al["alignments"].get(f"{c['id']}|{c['lang']}", {})
        for s_ in c["spans"]:
            ok = "words" in v and all(w is not None for w in v["words"][s_["w0"]:s_["w1"]])
            cov[f"{c['lang']}|{s_['task']}"]["aligned" if ok else "missing"] += 1
    rep["alignment_coverage"] = {k: dict(v) for k, v in sorted(cov.items())}
    sc = defaultdict(list)
    for c in clips:
        v = al["alignments"].get(f"{c['id']}|{c['lang']}", {})
        if "score" in v:
            sc[c["lang"]].append(v["score"])
    rep["alignment_score_mean_logp"] = {l: round(float(np.mean(x)), 3) for l, x in sc.items()}

    # pilot check 1: alignment sanity
    rep["region_duration_s"] = {f"{l}|{t}": [round(float(np.percentile(v, q)), 3) for q in (5, 50, 95)]
                                for (l, t), v in _group_dur(meta).items()}
    dur = np.array([r["t1"] - r["t0"] for r in meta if r["task"] == "digit"])
    inside = np.mean([r["t1"] <= clips[r["clip"]]["dur"] + 0.05 and r["t0"] >= 0 for r in meta])
    rep["check_alignment"] = {"digit_dur_median_s": round(float(np.median(dur)), 3),
                              "digit_dur_in_0.15_0.8": round(float(np.mean((dur >= 0.15) & (dur <= 0.8))), 3),
                              "spans_inside_clip": round(float(inside), 3),
                              "examples": [{"text": clips[r["clip"]]["text"][:90], "task": r["task"], "label": r["label"],
                                            "t": [r["t0"], r["t1"]], "dur": clips[r["clip"]]["dur"]}
                                           for r in meta[::max(1, len(meta) // 20)][:20]]}
    dims = [int(x) for x in a.dims.split(",")]
    for arm in a.arms.split(","):
        f = D / f"taps_{arm}.npz"
        if not f.exists():
            rep["arms"][arm] = "missing (stage not reached)"
            continue
        z = np.load(f)
        R = defaultdict(dict)
        for tap in TAPS + ("RP",):
            if tap == "RP":                                   # random-projection control of T1
                W = RNG.standard_normal((z["T1_mean"].shape[1], 4480)).astype(np.float32) / np.sqrt(4480)
                X = {"mean": z["T1_mean"].astype(np.float32) @ W}
            else:
                X = {"mean": z[f"{tap}_mean"].astype(np.float32), "centre": z[f"{tap}_centre"].astype(np.float32)}
                cl = list(z["utt_clips"])
                U = z[f"{tap}_utt"].astype(np.float32)
                X["utt"] = np.stack([U[cl.index(r["clip"])] for r in meta])
            for d in dims:
                for rep_name, Xr in X.items():
                    key = f"{tap}|{rep_name}|d{d}"
                    dg = [i for i, r in enumerate(meta) if r["task"] == "digit"]
                    R[key]["digit"] = cls_cv(Xr[dg], [meta[i]["label"] for i in dg], [meta[i]["id"] for i in dg], d)
                    if tap == "T1" and rep_name == "mean":
                        R[key]["digit_shuffled"] = cls_cv(Xr[dg], [meta[i]["label"] for i in dg],
                                                          [meta[i]["id"] for i in dg], d, shuffle=True)
                    cp = [i for i, r in enumerate(meta) if r["task"] == "capacity"]
                    R[key]["capacity"] = cls_cv(Xr[cp], [meta[i]["label"] for i in cp], [meta[i]["id"] for i in cp], d)
                    if rep_name != "mean":
                        continue
                    nm = [i for i, r in enumerate(meta) if r["task"] == "first_name"]
                    labels = {i: meta[i]["label"] for i in nm}
                    R[key]["name_ver"] = verification(
                        Xr, meta, nm, labels, d,
                        lambda p, q: (p["voice"] == q["voice"] and p["lang"] == q["lang"])
                        or edit(p["label"].lower(), q["label"].lower()) <= 2)
                    cc = [i for i, r in enumerate(meta) if r["task"] == "concept"]
                    labels = {i: meta[i]["label"] for i in cc}
                    R[key]["concept_ver"] = verification(Xr, meta, cc, labels, d,
                                                         lambda p, q: p["category"] == q["category"])
            if tap != "RP":
                R[f"{tap}|xling"] = {t: xling(X["mean"], meta, t) for t in ("first_name", "digit", "concept", "capacity")}
        A = {}
        fr = D / "taps_REF.npz"
        if fr.exists():
            zr = np.load(fr)
            ri = list(zr["region_index"])
            fam = np.array([meta[i]["id"] for i in ri])
            for tap in TAPS:
                Xt = z[f"{tap}_mean"].astype(np.float32)[ri]
                A[tap] = {f"d{d}": alignment(Xt, zr["REF_mean"].astype(np.float32), fam, d) for d in dims}
        uf = D / f"use_{arm}.json"
        Uu = use_summary(json.load(open(uf))) if uf.exists() else "missing"
        rep["arms"][arm] = {"R": R, "A": A, "U": Uu}
    # pilot checks 2-4
    chk = {}
    s = rep["arms"].get("s103")
    if isinstance(s, dict):
        for d in dims:
            t1 = s["R"][f"T1|mean|d{d}"]
            chk[f"d{d}_T1_digit_vs_shuffled"] = [t1["digit"]["ci95"], t1["digit_shuffled"]["ci95"]]
            chk[f"d{d}_T1_digit_vs_utt"] = [t1["digit"]["ci95"], s["R"][f"T1|utt|d{d}"]["digit"]["ci95"]]
            chk[f"d{d}_digit_by_tap"] = [s["R"][f"{t}|mean|d{d}"]["digit"]["acc"] for t in TAPS]
        uf = D / "use_s103.json"
        chk["use_clips"] = len(json.load(open(uf))) if uf.exists() else 0
    rep["pilot_checks"] = chk
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str))
    print(json.dumps({"check_alignment": {k: v for k, v in rep["check_alignment"].items() if k != "examples"},
                      "pilot_checks": chk}, indent=1))


if __name__ == "__main__":
    main()
