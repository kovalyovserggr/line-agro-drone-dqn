"""Основна таблиця Статті 3 (Table 4) і вибір агента за валідацією (Р47) — лише з сирих журналів.

Сценарії: помірна деградація × 12/15/17/19 принтерів; сильна і слабка деградація × 15.
Політики: B0; B1 з m*, налаштованим на seed 1000–1004 (помірна 0.5, сильна 0.35, слабка 0.5);
B2 θ = 1e-3; B3+ (лише помірна); B1* (покласові множники); DQN-TA — середнє 3 агентів
(середнє за агентами для кожного тестового seed) і агент, вибраний за середнім прибутком на
валідаційних seed 200000–200002 (Р47; тестові seed 100000–100004 у виборі не беруть участі).
Δ — парна різниця до B1* за тестовими seed, ± SE (n = 5).

Дані: test_finalists_L.csv, test_finalists_{strong,weak}_L.csv, test_b1star_{strong,weak}_L.csv,
dqn_ta_{np12,np15,np17,np19,strong,weak}_s{1,2,3}/{eval,val}.csv.

Запуск: python experiments/article3/make_table4.py --raw experiments/article3/raw --out docs/article3/tables
"""
import argparse, os
import numpy as np
import pandas as pd

SCEN = [("moderate", 12, "dqn_ta_np12"), ("moderate", 15, "dqn_ta_np15"), ("moderate", 17, "dqn_ta_np17"),
        ("moderate", 19, "dqn_ta_np19"), ("strong", 15, "dqn_ta_strong"), ("weak", 15, "dqn_ta_weak")]
B1_M = {"moderate": 0.5, "strong": 0.35, "weak": 0.5}
AGENTS = (1, 2, 3)


def load_baselines(raw):
    fs = ["test_finalists_L.csv", "test_finalists_strong_L.csv", "test_finalists_weak_L.csv",
          "test_b1star_strong_L.csv", "test_b1star_weak_L.csv"]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/article3/raw")
    ap.add_argument("--out", default="docs/article3/tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base = load_baselines(a.raw)
    rows, sel_rows = [], []
    for sc, n, pre in SCEN:
        b = base[(base.scenario == sc) & (base.n_printers == n)]
        pol = {"B0": b[b.policy == "B0"],
               "B1": b[(b.policy == "B1") & np.isclose(pd.to_numeric(b.param, errors="coerce"), B1_M[sc])],
               "B2": b[b.policy == "B2"], "B3+": b[b.policy == "B3+"], "B1*": b[b.policy == "B1*"]}
        ev = {i: pd.read_csv(os.path.join(a.raw, f"{pre}_s{i}", "eval.csv")) for i in AGENTS}
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
            par = df.param.iloc[0] if "param" in df and name.startswith("B") else (f"agent {best}" if "selected" in name else "")
            rows.append({"scenario": sc, "n_printers": n, "policy": name, "param": par,
                         "profit_k": s.mean() / 1e3, "d_vs_B1star_k": d, "se_k": se, "n_seeds": k,
                         "throughput": df.throughput_per_week.mean(), "OEE": df.OEE.mean(),
                         "pm": df.pm.mean(), "failures": df.failures.mean()})
    t = pd.DataFrame(rows)
    sel = pd.DataFrame(sel_rows)
    t.round(3).to_csv(os.path.join(a.out, "table4.csv"), index=False)
    sel.round(3).to_csv(os.path.join(a.out, "table4_agent_selection.csv"), index=False)

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
            cells.append(f"{r.profit_k:.1f}" if p == "B1*" else f"{r.profit_k:.1f} ({r.d_vs_B1star_k:+.1f} ± {r.se_k:.1f})")
        md.append(f"| {lab[(sc, n)]} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "table4.md"), "w", encoding="utf-8") as f:
        f.write("Profit, k USD/year, test seeds 100000–100004; in brackets — paired Δ vs B1* ± SE.\n\n")
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print()
    print(sel.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
