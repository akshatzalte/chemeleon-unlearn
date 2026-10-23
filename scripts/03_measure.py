"""Drift measurement (resumable, per task).

    03_measure.py [--only SLUG ...] [--device cuda] [--no-ownalpha] [--no-pc] [--combine]

Per task writes results/drift/<slug>/{drift_long,families,eta2,posctrl}.parquet then
done.json (last). PubChem rand anchor + correlation clusters live in results/drift/_global/.
--combine concatenates everything into results/{drift_long,families,eta2,posctrl,corr_clusters}.parquet.
Embeddings are cached as float16 under data/emb/<set>/<encoder_id>.npy.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from drift import config
from drift.config import BENCHMARKS, DATA, RESULTS, RUNS, SEEDS, slug
from drift.descriptors import Targets, descriptor_modules, prepare_targets
from drift.encoders import embed, load_mp, random_mp
from drift.measure import (effdim_table, fit_pca, pc_long_rows, pc_scores, alpha_rows, clean_rows, compare_to_pre, family_labels, measure_own_alpha, family_outputs, long_rows,
                           measure_probe_set, posctrl_rows, predict_probe, weight_change)
from drift.measure import LONG_COLS as LONG_COLS_
from drift.probe import choose_alpha, make_folds
from drift.stats import correlation_clusters, positive_control
from drift.tasks import load_task

OUT = RESULTS / "drift"
GLOBAL = OUT / "_global"
EMB = DATA / "emb"
N_FIT = config.N_PUBCHEM_FIT
FIT = np.arange(N_FIT)
EVAL = np.arange(N_FIT, config.N_PUBCHEM)
CKPTS = ("es", "last")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get_emb(set_name: str, enc_id: str, make_mp, smiles, device: str):
    """float16-cached embedding; always returns the float16-rounded values as float32."""
    path = EMB / set_name / f"{enc_id}.npy"
    if path.exists():
        Z = np.load(path)
    else:
        mp = make_mp()
        Z32 = embed(mp, smiles, device=device)
        del mp
        Z = Z32.astype(np.float16)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".tmp.npy"), Z)
        path.with_suffix(".tmp.npy").rename(path)
    Z = Z.astype(np.float32)
    bad = ~np.isfinite(Z).all(1)
    if bad.any():
        raise RuntimeError(f"{set_name}/{enc_id}: {bad.sum()} rows with non-finite embedding")
    return Z


class Pubchem:
    """PubChem probe set: targets, alpha, pre embeddings and pre predictions (lazy, per process)."""

    def __init__(self, device):
        self.device = device
        self.smiles = (DATA / "pubchem/pubchem10k.smiles").read_text().split()
        assert len(self.smiles) == config.N_PUBCHEM
        mods = descriptor_modules()
        D = pd.read_parquet(DATA / "descriptors/pubchem10k.parquet")[list(mods.index)]
        self.targets = prepare_targets(D, FIT)
        self.D = D
        self.Z_pre = get_emb("pubchem", "pre", lambda: load_mp(config.CHEMELEON_MP), self.smiles, device)
        GLOBAL.mkdir(parents=True, exist_ok=True)
        meta_p = GLOBAL / "meta.json"
        if meta_p.exists():
            self.alpha = json.loads(meta_p.read_text())["alpha_pubchem"]
        else:
            self.alpha = choose_alpha(self.Z_pre[FIT], self.targets.Y[FIT].astype(np.float64))
            meta_p.write_text(json.dumps({"alpha_pubchem": self.alpha,
                                          "n_retained": len(self.targets.names),
                                          "n_dropped": len(self.targets.dropped)}))
        self._pre_pred = None

    @property
    def Yhat_pre(self):
        if self._pre_pred is None:
            self._pre_pred = predict_probe(self.Z_pre, self.targets.Y.astype(np.float64), "split",
                                           alpha=self.alpha, fit_rows=FIT)
        return self._pre_pred

    def embed_enc(self, enc_id, make_mp):
        return get_emb("pubchem", enc_id, make_mp, self.smiles, self.device)


def ensure_global(pub: Pubchem, mods: pd.Series):
    if (GLOBAL / "done.json").exists():
        return
    log("global: PubChem rand anchor + clusters")
    cl = correlation_clusters(pub.targets.Y[FIT], pub.targets.names)
    pd.DataFrame({"descriptor": cl.index, "cluster": cl.values}).to_parquet(GLOBAL / "corr_clusters.parquet")
    Zr = {f"rand{i}": pub.embed_enc(f"rand{i}", lambda i=i: random_mp(i)) for i in range(3)}
    res = measure_probe_set(pub.Z_pre, Zr, pub.targets, {e: EVAL for e in Zr}, "split",
                            alpha=pub.alpha, fit_rows=FIT, Yhat_pre=pub.Yhat_pre)
    base = dict(task="", regime="", ckpt="", probe_set="pubchem")
    df = long_rows(res, pub.targets.names, base, "rand")
    extra = pd.DataFrame([dict(base, encoder="pre", measure="alpha", descriptor="", value=pub.alpha)])
    pd.concat([df, extra], ignore_index=True).to_parquet(GLOBAL / "drift_long.parquet")
    (GLOBAL / "done.json").write_text(json.dumps({"ok": True}))
    r2 = res["r2_pre"]
    log(f"global done: pre PubChem R2 median={np.nanmedian(r2):.3f} "
        f"rand median={np.nanmedian(np.concatenate([v for v in res['r2'].values()])):.3f} alpha={pub.alpha}")


def pc_names(k):
    return [f"PC{i + 1}" for i in range(k)]


def load_pca():
    z = np.load(GLOBAL / "pca.npz", allow_pickle=True)
    return {"loadings": z["loadings"], "evr": z["evr"], "mean": z["mean"], "mu": z["mu"], "sd": z["sd"],
            "names": [str(n) for n in z["names"]]}


def pubchem_pc(pub, pca):
    """(Ypc (n,k), Yhat_pre_pc) for PubChem; cached on the Pubchem object."""
    if getattr(pub, "_pc", None) is None:
        Ypc = pc_scores(pub.D[pca["names"]].to_numpy(dtype=np.float64, na_value=np.nan), pca["mu"], pca["sd"], pca)
        Yh = predict_probe(pub.Z_pre, Ypc, "split", alpha=pub.alpha, fit_rows=FIT)
        pub._pc = (Ypc, Yh)
    return pub._pc


def ensure_global_pc(pub: Pubchem, mods):
    """PCA basis, effective dimensionality, PubChem rand anchor + evr rows (once)."""
    if (GLOBAL / "pca_done.json").exists():
        return
    log("global: PCA + effdim + PubChem PC rand anchor")
    names = pub.targets.names
    Yf = pub.targets.Y[FIT].astype(np.float64)
    pca = fit_pca(Yf)
    X = pub.D[names].to_numpy(dtype=np.float64, na_value=np.nan)[FIT]
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    np.savez(GLOBAL / "pca.npz", loadings=pca["loadings"], evr=pca["evr"], mean=pca["mean"], mu=mu, sd=sd,
             names=np.array(names))
    ed = effdim_table(Yf, names, mods)
    ed.to_parquet(GLOBAL / "effdim.parquet")
    pca.update(mu=mu, sd=sd, names=list(names))
    k = pca["loadings"].shape[1]
    Ypc, Yh = pubchem_pc(pub, pca)
    T = Targets(Ypc.astype(np.float32), pc_names(k), [])
    Zr = {f"rand{i}": pub.embed_enc(f"rand{i}", lambda i=i: random_mp(i)) for i in range(3)}
    res = measure_probe_set(pub.Z_pre, Zr, T, {e: EVAL for e in Zr}, "split", alpha=pub.alpha,
                            fit_rows=FIT, Yhat_pre=Yh)
    base = dict(task="", regime="", ckpt="", probe_set="pubchem_pc")
    evr = pd.DataFrame({**{c: [v] * k for c, v in base.items()}, "encoder": "pca", "measure": "evr",
                        "descriptor": pc_names(k), "value": pca["evr"]})[LONG_COLS_]
    pd.concat([pc_long_rows(res, pc_names(k), base, "rand", pca["evr"]), evr], ignore_index=True
              ).to_parquet(GLOBAL / "drift_long_pc.parquet")
    (GLOBAL / "pca_done.json").write_text(json.dumps({"n_components": k, "n_descriptors": len(names),
                                                      "n99_overall": int(ed.iloc[0]["n99"])}))
    log(f"global PCA: {k} components for {len(names)} descriptors; effdim ALL: {ed.iloc[0].to_dict()}")


def pc_task(name, pub: Pubchem, pca):
    """Decorrelated (PC-target) views for one finished task; marker pc_done.json."""
    d = OUT / slug(name)
    if (d / "pc_done.json").exists() or not (d / "done.json").exists():
        return
    t0 = time.time()
    s = slug(name)
    runs = expected_runs(name)
    meta = json.loads((d / "done.json").read_text())
    task = load_task(name)
    n = len(task.train_smiles)
    smi = list(task.train_smiles)
    Draw = pd.read_parquet(DATA / f"descriptors/{s}.parquet")
    Ypc = pc_scores(Draw[pca["names"]].to_numpy(dtype=np.float64, na_value=np.nan), pca["mu"], pca["sd"], pca)
    k = Ypc.shape[1]
    pcn, evr = pc_names(k), pca["evr"]
    Ypub, Yhpub = pubchem_pc(pub, pca)
    Tpub = Targets(Ypub.astype(np.float32), pcn, [])
    folds = make_folds(n, config.N_FOLDS, 0)
    kw = dict(mode="crossfit", folds=folds, alpha=meta["alpha_task"])
    Zp = get_emb(s, "pre", lambda: load_mp(config.CHEMELEON_MP), smi, device_of(pub))
    Yp = predict_probe(Zp, Ypc, **kw)
    allrows = np.arange(n)
    out = []
    Zr = {f"rand{i}": get_emb(s, f"rand{i}", lambda i=i: random_mp(i), smi, device_of(pub)) for i in range(3)}
    res = compare_to_pre(Ypc, Yp, {e: predict_probe(Z, Ypc, **kw) for e, Z in Zr.items()}, Zp, Zr,
                         {e: allrows for e in Zr})
    out.append(pc_long_rows(res, pcn, dict(task=name, regime="", ckpt="", probe_set="task_all_pc"), "rand", evr))
    for regime in config.REGIMES:
        clean = {seed: clean_rows(runs[(regime, seed)], n) for seed in SEEDS}
        for ckpt in CKPTS:
            Zt, Zq = {}, {}
            for seed in SEEDS:
                dd = runs[(regime, seed)]["dir"]
                eid = f"{s}__{regime}__s{seed}__{ckpt}"
                loader = lambda dd=dd, ckpt=ckpt: load_mp(dd / f"{ckpt}_mp.pt")
                Zt[f"seed{seed}"] = get_emb(s, eid, loader, smi, device_of(pub))
                Zq[f"seed{seed}"] = pub.embed_enc(eid, loader)
            b = dict(task=name, regime=regime, ckpt=ckpt)
            res = measure_probe_set(pub.Z_pre, Zq, Tpub, {e: EVAL for e in Zq}, "split", alpha=pub.alpha,
                                    fit_rows=FIT, Yhat_pre=Yhpub)
            out.append(pc_long_rows(res, pcn, dict(b, probe_set="pubchem_pc"), "ft", evr))
            preds = {e: predict_probe(Z, Ypc, **kw) for e, Z in Zt.items()}
            for view, rows in (("task_all_pc", {e: allrows for e in Zt}),
                               ("task_clean_pc", {f"seed{sd}": clean[sd] for sd in SEEDS})):
                res = compare_to_pre(Ypc, Yp, preds, Zp, Zt, rows)
                out.append(pc_long_rows(res, pcn, dict(b, probe_set=view), "ft", evr))
            log(f"  {name} PC {regime}/{ckpt} done ({time.time() - t0:.0f}s)")
    pd.concat(out, ignore_index=True).to_parquet(d / "drift_long_pc.parquet")
    (d / "pc_done.json").write_text(json.dumps({"seconds": time.time() - t0, "n_components": k}))
    log(f"{name}: PC views done in {time.time() - t0:.0f}s")


def device_of(pub):
    return pub.device


def _cached_alphas(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ensure_global_ownalpha(pub: Pubchem):
    """rand0..2 with per-encoder alpha on PubChem (probe_set pubchem_ownalpha)."""
    if (GLOBAL / "ownalpha_done.json").exists():
        return
    log("global: own-alpha rand anchor")
    Zr = {f"rand{i}": pub.embed_enc(f"rand{i}", lambda i=i: random_mp(i)) for i in range(3)}
    res = measure_own_alpha(pub.Z_pre, Zr, pub.targets, EVAL, FIT, pub.alpha, pub.Yhat_pre)
    base = dict(task="", regime="", ckpt="", probe_set="pubchem_ownalpha")
    df = pd.concat([long_rows(res, pub.targets.names, base, "rand"), alpha_rows(base, res["alphas"], pub.alpha)],
                   ignore_index=True)
    df.to_parquet(GLOBAL / "drift_long_ownalpha.parquet")
    (GLOBAL / "ownalpha_done.json").write_text(json.dumps(res["alphas"]))


def ownalpha_task(name, pub: Pubchem):
    """Secondary: es-checkpoint ft encoders (both regimes) on PubChem with per-encoder alpha."""
    d = OUT / slug(name)
    if (d / "ownalpha_done.json").exists():
        return
    runs = expected_runs(name)
    if runs is None:
        return
    t0 = time.time()
    s = slug(name)
    cache_p = d / "ownalpha_alphas.json"
    cache = _cached_alphas(cache_p)
    out = []
    for regime in config.REGIMES:
        Zq = {}
        for seed in SEEDS:
            dd = runs[(regime, seed)]["dir"]
            Zq[f"seed{seed}"] = pub.embed_enc(f"{s}__{regime}__s{seed}__es",
                                              lambda dd=dd: load_mp(dd / "es_mp.pt"))
        res = measure_own_alpha(pub.Z_pre, Zq, pub.targets, EVAL, FIT, pub.alpha, pub.Yhat_pre,
                                alphas=cache.get(regime))
        cache[regime] = res["alphas"]
        cache_p.write_text(json.dumps(cache))
        base = dict(task=name, regime=regime, ckpt="es", probe_set="pubchem_ownalpha")
        out += [long_rows(res, pub.targets.names, base, "ft"), alpha_rows(base, res["alphas"], pub.alpha)]
    pd.concat(out, ignore_index=True).to_parquet(d / "drift_long_ownalpha.parquet")
    (d / "ownalpha_done.json").write_text(json.dumps(cache))
    log(f"{name}: own-alpha done in {time.time() - t0:.0f}s {cache}")


def expected_runs(name):
    """{(regime, seed): run dict} for ft runs; None if any is missing a done.json."""
    out = {}
    for regime in config.REGIMES:
        for seed in SEEDS:
            d = RUNS / slug(name) / regime / "ft" / f"seed{seed}"
            if not (d / "done.json").exists():
                return None
            run = json.loads((d / "run.json").read_text())
            run["dir"] = d
            out[(regime, seed)] = run
    return out


def measure_task(name, pub: Pubchem, mods, cluster_map, device):
    t0 = time.time()
    s = slug(name)
    runs = expected_runs(name)
    if runs is None:
        log(f"{name}: ft runs not all done, skipping")
        return False
    task = load_task(name)
    n = len(task.train_smiles)
    Draw = pd.read_parquet(DATA / f"descriptors/{s}.parquet")
    assert len(Draw) == n and list(Draw["smiles"]) == list(task.train_smiles)
    D = Draw[list(mods.index)]
    T = prepare_targets(D, np.arange(n))
    Y = T.Y.astype(np.float64)
    names = T.names
    folds = make_folds(n, config.N_FOLDS, 0)
    smi = list(task.train_smiles)

    Zp = get_emb(s, "pre", lambda: load_mp(config.CHEMELEON_MP), smi, device)
    alpha = choose_alpha(Zp, Y)
    kw = dict(mode="crossfit", folds=folds, alpha=alpha)
    Yp = predict_probe(Zp, Y, **kw)
    lab_task = family_labels(names, mods, cluster_map)
    lab_pub = family_labels(pub.targets.names, mods, cluster_map)
    top = positive_control(Draw, task.y_raw, names, k=config.TOP_K_POSCTRL)
    top_pub = [d for d in top if d in set(pub.targets.names)]

    L, F, E, P = [], [], [], []
    alpha_row = [dict(task=name, regime="", ckpt="", probe_set="task_all", encoder="pre",
                      measure="alpha", descriptor="", value=alpha)]

    def record(res, view_names, labels, tops, base, agg="ft"):
        L.append(long_rows(res, view_names, base, agg))
        if agg == "ft":
            f, e = family_outputs(res, view_names, labels, base)
            F.append(f), E.append(e)
            P.append(posctrl_rows(res, view_names, tops, base))

    # rand anchor on the task set (all rows); one per task
    Zr = {f"rand{i}": get_emb(s, f"rand{i}", lambda i=i: random_mp(i), smi, device) for i in range(3)}
    allrows = np.arange(n)
    res = compare_to_pre(Y, Yp, {e: predict_probe(Z, Y, **kw) for e, Z in Zr.items()}, Zp, Zr,
                         {e: allrows for e in Zr})
    L.append(long_rows(res, names, dict(task=name, regime="", ckpt="", probe_set="task_all"), "rand"))

    mp_pre = load_mp(config.CHEMELEON_MP)
    for regime in config.REGIMES:
        clean = {seed: clean_rows(runs[(regime, seed)], n) for seed in SEEDS}
        for ckpt in CKPTS:
            Zt, Zq = {}, {}
            for seed in SEEDS:
                d = runs[(regime, seed)]["dir"]
                eid = f"{s}__{regime}__s{seed}__{ckpt}"
                mp = load_mp(d / f"{ckpt}_mp.pt")
                wc = weight_change(mp, mp_pre)
                L.append(pd.DataFrame({"task": name, "regime": regime, "ckpt": ckpt, "probe_set": "",
                                       "encoder": f"seed{seed}", "measure": "wchange",
                                       "descriptor": list(wc), "value": list(wc.values())}))
                loader = lambda d=d: load_mp(d / f"{ckpt}_mp.pt")
                Zt[f"seed{seed}"] = get_emb(s, eid, loader, smi, device)
                Zq[f"seed{seed}"] = pub.embed_enc(eid, loader)
            # PubChem view
            pres = measure_probe_set(pub.Z_pre, Zq, pub.targets, {e: EVAL for e in Zq}, "split",
                                     alpha=pub.alpha, fit_rows=FIT, Yhat_pre=pub.Yhat_pre)
            record(pres, pub.targets.names, lab_pub, top_pub,
                   dict(task=name, regime=regime, ckpt=ckpt, probe_set="pubchem"))
            # task views share one set of cross-fitted predictions
            preds = {e: predict_probe(Z, Y, **kw) for e, Z in Zt.items()}
            for view, rows in (("task_all", {e: allrows for e in Zt}),
                               ("task_clean", {f"seed{sd}": clean[sd] for sd in SEEDS})):
                res = compare_to_pre(Y, Yp, preds, Zp, Zt, rows)
                record(res, names, lab_task, top, dict(task=name, regime=regime, ckpt=ckpt, probe_set=view))
            log(f"  {name} {regime}/{ckpt} done ({time.time() - t0:.0f}s)")

    d = OUT / s
    d.mkdir(parents=True, exist_ok=True)
    long = pd.concat(L + [pd.DataFrame(alpha_row)], ignore_index=True)
    long.to_parquet(d / "drift_long.parquet")
    pd.concat(F, ignore_index=True).to_parquet(d / "families.parquet")
    pd.concat(E, ignore_index=True).to_parquet(d / "eta2.parquet")
    pd.concat(P, ignore_index=True).to_parquet(d / "posctrl.parquet")
    (d / "done.json").write_text(json.dumps({
        "task": name, "n_train": n, "n_retained": len(names), "n_dropped": len(T.dropped),
        "alpha_task": alpha, "alpha_pubchem": pub.alpha, "low_confidence": bool(n < 1000),
        "n_clean": {f"{r}/s{sd}": int(len(clean_rows(runs[(r, sd)], n))) for r, sd in runs},
        "seconds": time.time() - t0}))
    log(f"{name}: done in {time.time() - t0:.0f}s (alpha_task={alpha}, n={n})")
    return True


def combine():
    for f in ("drift_long", "families", "eta2", "posctrl"):
        parts = sorted(OUT.glob(f"*/{f}.parquet"))
        parts = [p for p in parts if p.parent.name == "_global" or (p.parent / "done.json").exists()]
        if f == "drift_long":
            parts += sorted(OUT.glob("*/drift_long_ownalpha.parquet")) + sorted(OUT.glob("*/drift_long_pc.parquet"))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df.to_parquet(RESULTS / f"{f}.parquet")
        log(f"combined {f}: {len(df)} rows from {len(parts)} parts")
    pd.read_parquet(GLOBAL / "corr_clusters.parquet").to_parquet(RESULTS / "corr_clusters.parquet")
    for f in ("effdim.parquet",):
        if (GLOBAL / f).exists():
            pd.read_parquet(GLOBAL / f).to_parquet(RESULTS / f)
    if (GLOBAL / "pca.npz").exists():
        import shutil
        shutil.copy(GLOBAL / "pca.npz", RESULTS / "pca.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="task slugs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ownalpha", action=argparse.BooleanOptionalAction, default=True,
                    help="secondary per-encoder-alpha PubChem measurement (--no-ownalpha to skip)")
    ap.add_argument("--pc", action=argparse.BooleanOptionalAction, default=True,
                    help="decorrelated PC-target views + effective dimensionality (--no-pc to skip)")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()
    if a.combine:
        return combine()
    mods = descriptor_modules()
    pub = Pubchem(a.device)
    ensure_global(pub, mods)
    if a.ownalpha:
        ensure_global_ownalpha(pub)
    pca = None
    if a.pc:
        ensure_global_pc(pub, mods)
        pca = load_pca()
    cl = pd.read_parquet(GLOBAL / "corr_clusters.parquet")
    cluster_map = dict(zip(cl["descriptor"], cl["cluster"]))
    for name in BENCHMARKS:
        if a.only and slug(name) not in a.only:
            continue
        if (OUT / slug(name) / "done.json").exists():
            log(f"{name}: already done")
        else:
            measure_task(name, pub, mods, cluster_map, a.device)
        if a.ownalpha:
            ownalpha_task(name, pub)
        if a.pc:
            pc_task(name, pub, pca)


if __name__ == "__main__":
    main()
