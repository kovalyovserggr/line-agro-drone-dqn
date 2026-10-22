"""Основна таблиця Статті 3 (Table 4) і вибір агента за валідацією (Р47) — лише з сирих журналів.

Сценарії: помірна деградація × 12/15/17/19 принтерів; сильна і слабка деградація × 15.
Політики: B0; B1 з m*, налаштованим на seed 1000–1004 (помірна 0.5, сильна 0.35, слабка 0.5);
B2 θ = 1e-3; B3+ (лише помірна); B1* (покласові множники); DQN-TA — середнє 3 агентів
(середнє за агентами для кожного тестового seed) і агент, вибраний за середнім прибутком на
валідаційних seed 200000–200002 (Р47; тестові seed 100000–100004 у виборі не беруть участі).
Δ — парна різниця до B1* за тестовими seed, ± SE (n = 20, Р48: 100000–100019).

Друга частина (Р49) — робастність: «справжній» час до відмови зносових складових
логнормальний, політики заморожені; еталон — B1, перерахований на логнормальній правді.

Дані: test_finalists_L.csv, test_finalists_{strong,weak}_L.csv, test_b1star_{strong,weak}_L.csv,
dqn_ta_{np12,np15,np17,np19,strong,weak}_s{1,2,3}/{eval,val}.csv; Р48: test20_finalists{,_strong,_weak}_L.csv,
test20_dqn_L.csv; Р49: tune_logn_{moderate,strong}_L.csv, test_logn_{moderate,strong,dqn}_L.csv.

Запуск: python experiments/article3/make_table4.py --raw experiments/article3/raw --out docs/article3/tables
"""
import argparse, os
import numpy as np
import pandas as pd

SCEN = [("moderate", 12, "dqn_ta_np12"), ("moderate", 15, "dqn_ta_np15"), ("moderate", 17, "dqn_ta_np17"),
        ("moderate", 19, "dqn_ta_np19"), ("strong", 15, "dqn_ta_strong"), ("weak", 15, "dqn_ta_weak")]
B1_M = {"moderate": 0.5, "strong": 0.35, "weak": 0.5}
AGENTS = (1, 2, 3)
N_TEST = 20          # Р48: тестові seed 100000–100019


def load_baselines(raw):
    fs = ["test_finalists_L.csv", "test_finalists_strong_L.csv", "test_finalists_weak_L.csv",
          "test_b1star_strong_L.csv", "test_b1star_weak_L.csv",
          # Р48: дооцінка на seed 100005–100019
          "test20_finalists_L.csv", "test20_finalists_strong_L.csv", "test20_finalists_weak_L.csv"]
    d = []
    for f in fs:
        x = pd.read_csv(os.path.join(raw, f))
        if "scenario" not in x:
            x["scenario"] = "moderate"
        d.append(x)
    return pd.concat(d, ignore_index=True)


def by_seed(df):
    return df.groupby("seed").profit.mean()


def paired(a, ref):
    d = (a - ref).dropna() / 1e3
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def wilcoxon_p(a, ref):
    """Двобічний знаковий ранговий тест Вілкоксона для парних різниць (CRN)."""
    from scipy.stats import wilcoxon
    d = (a - ref).dropna()
    if len(d) == 0 or np.allclose(d, 0):
        return np.nan
    return float(wilcoxon(d, zero_method="wilcox").pvalue)


def holm(ps):
    """Поправка Холма–Бонферроні; NaN пропускаються."""
    ps = np.asarray(ps, dtype=float)
    out = np.full_like(ps, np.nan)
    idx = [i for i in np.argsort(ps) if not np.isnan(ps[i])]
    m, run = len(idx), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (m - k) * ps[i]))
        out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/article3/raw")
    ap.add_argument("--out", default="docs/article3/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = load_baselines(a.raw)
    d20 = pd.read_csv(os.path.join(a.raw, "test20_dqn_L.csv"))       # Р48
    rows, sel_rows = [], []
    for sc, n, pre in SCEN:
        b = base[(base.scenario == sc) & (base.n_printers == n)]
        pol = {"B0": b[b.policy == "B0"],
               "B1": b[(b.policy == "B1") & np.isclose(pd.to_numeric(b.param, errors="coerce"), B1_M[sc])],
               "B2": b[b.policy == "B2"], "B3+": b[b.policy == "B3+"], "B1*": b[b.policy == "B1*"]}
        ev = {i: pd.concat([pd.read_csv(os.path.join(a.raw, f"{pre}_s{i}", "eval.csv")),
                            d20[d20.agent_dir == f"{pre}_s{i}"]], ignore_index=True) for i in AGENTS}
        va = {i: pd.read_csv(os.path.join(a.raw, f"{pre}_s{i}", "val.csv")) for i in AGENTS}
        vmean = {i: va[i].profit.mean() for i in AGENTS}
        tmean = {i: ev[i].profit.mean() for i in AGENTS}
        best = max(AGENTS, key=lambda i: vmean[i])
        ref = by_seed(pol["B1*"])
        for i in AGENTS:
            dv, sev, _ = paired(by_seed(ev[i]), ref)
            sel_rows.append({"scenario": sc, "n_printers": n, "agent": i, "val_profit_k": vmean[i] / 1e3,
                             "test_profit_k": tmean[i] / 1e3, "test_d_vs_B1star": dv, "test_se": sev,
                             "selected": i == best})
        cols = ["profit", "throughput_per_week", "OEE", "pm", "failures"]
        pol["DQN-TA (mean of 3)"] = pd.concat(ev.values()).groupby("seed", as_index=False)[cols].mean()
        pol["DQN-TA (val.-selected)"] = ev[best]
        for name, df in pol.items():
            if df.empty:
                continue
            s = by_seed(df)
            d, se, k = paired(s, ref)
            assert k == N_TEST and s.index.nunique() == N_TEST, f"{sc} {n} {name}: {k} seed замість {N_TEST}"
            par = df.param.iloc[0] if "param" in df and name.startswith("B") else (f"agent {best}" if "selected" in name else "")
            rows.append({"scenario": sc, "n_printers": n, "policy": name, "param": par,
                         "profit_k": s.mean() / 1e3, "d_vs_B1star_k": d, "se_k": se, "n_seeds": k,
                         "p_wilcoxon": wilcoxon_p(s, ref) if name != "B1*" else np.nan,
                         "throughput": df.throughput_per_week.mean(), "OEE": df.OEE.mean(),
                         "pm": df.pm.mean(), "failures": df.failures.mean()})
    t = pd.DataFrame(rows)
    # Холм — окремо для кожної політики по 6 сценаріях (сімейство порівнянь «політика vs B1*»)
    t["p_holm"] = np.nan
    for pol_name, g in t.groupby("policy"):
        t.loc[g.index, "p_holm"] = holm(g.p_wilcoxon.values)
    sel = pd.DataFrame(sel_rows)
    t.round(6).to_csv(os.path.join(a.out, "table4.csv"), index=False)
    sel.round(6).to_csv(os.path.join(a.out, "table4_agent_selection.csv"), index=False)

    # Markdown-версія (англійською, як у статті)
    lab = {("moderate", 12): "Moderate, 12 pr.", ("moderate", 15): "Moderate, 15 pr.",
           ("moderate", 17): "Moderate, 17 pr.", ("moderate", 19): "Moderate, 19 pr.",
           ("strong", 15): "Strong, 15 pr.", ("weak", 15): "Weak, 15 pr."}
    order = ["B0", "B1", "B2", "B3+", "B1*", "DQN-TA (mean of 3)", "DQN-TA (val.-selected)"]
    md = ["| Scenario | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
    for sc, n, _ in SCEN:
        cells = []
        for p in order:
            r = t[(t.scenario == sc) & (t.n_printers == n) & (t.policy == p)]
            if r.empty:
                cells.append("—"); continue
            r = r.iloc[0]
            star = "" if p == "B1*" or np.isnan(r.p_holm) else ("**" if r.p_holm < 0.01 else ("*" if r.p_holm < 0.05 else ""))
            cells.append(f"{r.profit_k:.1f}" if p == "B1*" else f"{r.profit_k:.1f} ({r.d_vs_B1star_k:+.1f} ± {r.se_k:.1f}){star}")
        md.append(f"| {lab[(sc, n)]} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "table4.md"), "w", encoding="utf-8") as f:
        f.write("Profit, k USD/year, test seeds 100000–100019 (n = 20); in brackets — paired Δ vs B1* ± SE. "
                "* / ** — Wilcoxon signed-rank vs B1*, Holm-adjusted over the six scenarios per policy, p < 0.05 / p < 0.01.\n\n")
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print()
    print(sel.round(1).to_string(index=False))
    print()
    r = robustness(a.raw, a.out, t, sel)
    print(r.round(1).to_string(index=False))


ROBUST = [("moderate", 15, "dqn_ta_np15"), ("moderate", 17, "dqn_ta_np17"), ("strong", 15, "dqn_ta_strong")]


def robustness(raw, out, t4, sel):
    """Р49: заморожені політики на логнормальній «правді» (20 seed) проти вейбулівської (Table 4)."""
    fin = pd.concat([pd.read_csv(os.path.join(raw, f)) for f in ("test_logn_moderate_L.csv", "test_logn_strong_L.csv")])
    dq = pd.read_csv(os.path.join(raw, "test_logn_dqn_L.csv"))
    tune = pd.concat([pd.read_csv(os.path.join(raw, f)) for f in ("tune_logn_moderate_L.csv", "tune_logn_strong_L.csv")])
    rows = []
    for sc, n, pre in ROBUST:
        tn = tune[(tune.scenario == sc) & (tune.n_printers == n)].groupby("param").profit.mean()
        m_logn = float(tn.idxmax())
        f = fin[(fin.scenario == sc) & (fin.n_printers == n)]
        ref = by_seed(f[f.policy == "B1*"])
        b1 = f[(f.policy == "B1") & np.isclose(pd.to_numeric(f.param, errors="coerce"), B1_M[sc])]
        assert np.isclose(m_logn, B1_M[sc]), f"{sc} {n}: m*_logn = {m_logn} ≠ m*_Weibull — потрібен окремий тест B1(m*_logn)"
        best = int(sel[(sel.scenario == sc) & (sel.n_printers == n) & sel.selected].agent.iloc[0])
        ag = {i: dq[dq.agent_dir == f"{pre}_s{i}"] for i in AGENTS}
        pols = {"B0": f[f.policy == "B0"], "B1 (m* Weibull = m* lognormal)": b1, "B2": f[f.policy == "B2"],
                "B1*": f[f.policy == "B1*"],
                "DQN-TA (mean of 3)": pd.concat(ag.values()).groupby("seed", as_index=False)[["profit", "failures", "pm"]].mean(),
                "DQN-TA (val.-selected)": ag[best]}
        for name, df in pols.items():
            s_ = by_seed(df)
            d, se, k = paired(s_, ref)
            assert k == N_TEST, (sc, n, name, k)
            w = t4[(t4.scenario == sc) & (t4.n_printers == n) &
                   (t4.policy == {"B1 (m* Weibull = m* lognormal)": "B1"}.get(name, name))]
            rows.append({"scenario": sc, "n_printers": n, "policy": name, "m_star_lognormal": m_logn,
                         "profit_k": s_.mean() / 1e3, "d_vs_B1star_k": d, "se_k": se,
                         "p_wilcoxon": wilcoxon_p(s_, ref) if name != "B1*" else np.nan,
                         "weibull_d_vs_B1star_k": w.d_vs_B1star_k.iloc[0] if len(w) else np.nan,
                         "failures": df.failures.mean(), "pm": df.pm.mean()})
        for i in AGENTS:
            d, se, _ = paired(by_seed(ag[i]), ref)
            rows.append({"scenario": sc, "n_printers": n, "policy": f"DQN-TA agent {i}", "profit_k": by_seed(ag[i]).mean() / 1e3,
                         "d_vs_B1star_k": d, "se_k": se, "p_wilcoxon": wilcoxon_p(by_seed(ag[i]), ref), "failures": ag[i].failures.mean(), "pm": ag[i].pm.mean()})
    r = pd.DataFrame(rows)
    r.round(6).to_csv(os.path.join(out, "table_robust_lognormal.csv"), index=False)
    lab = {("moderate", 15): "Moderate, 15 pr.", ("moderate", 17): "Moderate, 17 pr.", ("strong", 15): "Strong, 15 pr."}
    order = ["B0", "B1 (m* Weibull = m* lognormal)", "B2", "B1*", "DQN-TA (mean of 3)", "DQN-TA (val.-selected)"]
    md = ["| Scenario | " + " | ".join(order) + " |", "|---" * (len(order) + 1) + "|"]
    for sc, n, _ in ROBUST:
        cells = []
        for p in order:
            x = r[(r.scenario == sc) & (r.n_printers == n) & (r.policy == p)].iloc[0]
            cells.append(f"{x.profit_k:.1f}" if p == "B1*" else f"{x.profit_k:.1f} ({x.d_vs_B1star_k:+.1f} ± {x.se_k:.1f})")
        md.append(f"| {lab[(sc, n)]} | " + " | ".join(cells) + " |")
    with open(os.path.join(out, "table_robust_lognormal.md"), "w", encoding="utf-8") as fh:
        fh.write("Robustness: true time-to-failure of wear modes is lognormal (same mean and CV as the Weibull model); "
                 "all policies frozen (tuned / trained on Weibull). Profit, k USD/year, test seeds 100000–100019; "
                 "in brackets — paired Δ vs B1* ± SE.\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
    return r


if __name__ == "__main__":
    main()
