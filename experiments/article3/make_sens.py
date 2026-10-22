"""Таблиця чутливості до параметрів без джерела (Р50) — лише з сирих журналів.

η_C2 ∈ {750; 1 500 (база); 3 000}: усі політики переналаштовані / перенавчені під кожне
значення (sens_pipeline.py). η_C4 ∈ {2 500; 5 000 (база); 10 000}: заморожені політики бази.
Δ — парна різниця до B1* того самого варіанта за тестовими seed 100000–100019, ± SE.
База (η_C2 = 1 500, η_C4 = 5 000) — з docs/article3/tables/table4.csv (помірна, 15 принтерів).

Запуск: python experiments/article3/make_sens.py --raw experiments/article3/raw --out docs/article3/tables
"""
import argparse, json, os
import numpy as np
import pandas as pd

N_TEST = 20
BASE_SELECTED = None   # береться з table4_agent_selection.csv


def by_seed(df):
    return df.groupby("seed").profit.mean()


def paired(a, ref):
    d = ((a - ref) / 1e3).dropna()
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)


def wilcoxon_p(a, ref):
    from scipy.stats import wilcoxon
    d = (a - ref).dropna()
    return np.nan if len(d) == 0 or np.allclose(d, 0) else float(wilcoxon(d).pvalue)


def block(label, eta_lab, fin, dq, selected, extra):
    ref = by_seed(fin[fin.policy == "B1*"])
    ag = {int(i): g for i, g in dq.groupby("param")}
    pols = [("B0", fin[fin.policy == "B0"])]
    for nm in ("B1", "B2"):
        if (fin.policy == nm).any():
            pols.append((nm, fin[fin.policy == nm]))
    pols += [("B1*", fin[fin.policy == "B1*"]),
             ("DQN-TA (mean of 3)", pd.concat(ag.values()).groupby("seed", as_index=False)[["profit", "pm", "failures"]].mean()),
             ("DQN-TA (val.-selected)", ag[selected])]
    rows = []
    for name, df in pols:
        s = by_seed(df)
        d, se, k = paired(s, ref)
        assert k == N_TEST and s.size == N_TEST, (label, eta_lab, name, k)
        par = ""
        if name in ("B1", "B2"):
            par = str(df.param.iloc[0])
        elif name == "B1*":
            par = str(df.param.iloc[0])
        elif "selected" in name:
            par = f"agent {selected}"
        rows.append({"study": label, "variant": eta_lab, "policy": name, "param": par, "profit_k": s.mean() / 1e3,
                     "d_vs_B1star_k": d, "se_k": se, "p_wilcoxon": wilcoxon_p(s, ref) if name != "B1*" else np.nan, "pm": df.pm.mean(), "failures": df.failures.mean(), **extra})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/article3/raw")
    ap.add_argument("--out", default="docs/article3/tables")
    a = ap.parse_args()
    t4 = pd.read_csv(os.path.join(a.out, "table4.csv"))
    sel4 = pd.read_csv(os.path.join(a.out, "table4_agent_selection.csv"))
    base_sel = int(sel4[(sel4.scenario == "moderate") & (sel4.n_printers == 15) & sel4.selected].agent.iloc[0])
    rows = []
    # база — з Table 4
    for p in ("B0", "B1", "B2", "B1*", "DQN-TA (mean of 3)", "DQN-TA (val.-selected)"):
        r = t4[(t4.scenario == "moderate") & (t4.n_printers == 15) & (t4.policy == p)].iloc[0]
        for study in ("eta_C2 (re-tuned)", "eta_C4 (frozen)"):
            if study.startswith("eta_C4") and p == "B2":
                continue
            rows.append({"study": study, "variant": "base", "policy": p, "param": r.param if isinstance(r.param, str) else "",
                         "profit_k": r.profit_k, "d_vs_B1star_k": r.d_vs_B1star_k, "se_k": r.se_k,
                         "pm": r.pm, "failures": r.failures})
    # η_C2 — переналаштування
    for tag, lab in (("c2eta750", "750 h"), ("c2eta3000", "3000 h")):
        sel = json.load(open(os.path.join(a.raw, f"sens_{tag}_selection.json")))
        fin = pd.read_csv(os.path.join(a.raw, f"sens_{tag}_test_L.csv"))
        dq = pd.read_csv(os.path.join(a.raw, f"sens_{tag}_test_dqn_L.csv"))
        rows += block("eta_C2 (re-tuned)", lab, fin, dq, sel["selected_agent"],
                      {"m_star": sel["m_star"], "theta_star": sel["theta_star"], "b1star": json.dumps(sel["b1star"], sort_keys=True)})
    # η_C4 — заморожені
    for tag, lab in (("c4eta2500", "2500 h"), ("c4eta10000", "10000 h")):
        fin = pd.read_csv(os.path.join(a.raw, f"sens_{tag}_test_L.csv"))
        dq = pd.read_csv(os.path.join(a.raw, f"sens_{tag}_test_dqn_L.csv"))
        rows += block("eta_C4 (frozen)", lab, fin, dq, base_sel, {})
    t = pd.DataFrame(rows)
    t.round(6).to_csv(os.path.join(a.out, "table_sens_eta.csv"), index=False)
    order = ["B0", "B1", "B2", "B1*", "DQN-TA (mean of 3)", "DQN-TA (val.-selected)"]
    md = ["Sensitivity to unsourced wear parameters (moderate degradation, 15 printers). Profit, k USD/year, "
          "test seeds 100000–100019; in brackets — paired Δ vs B1* of the same variant ± SE. "
          "η_C2: all policies re-tuned / re-trained per value; η_C4: base policies frozen.", ""]
    md.append("| Study | η | " + " | ".join(order) + " |")
    md.append("|---" * (len(order) + 2) + "|")
    for study in ("eta_C2 (re-tuned)", "eta_C4 (frozen)"):
        vs = ["750 h", "base", "3000 h"] if study.startswith("eta_C2") else ["2500 h", "base", "10000 h"]
        for v in vs:
            cells = []
            for p in order:
                r = t[(t.study == study) & (t.variant == v) & (t.policy == p)]
                if r.empty:
                    cells.append("—"); continue
                r = r.iloc[0]
                cells.append(f"{r.profit_k:.1f}" if p == "B1*" else f"{r.profit_k:.1f} ({r.d_vs_B1star_k:+.1f} ± {r.se_k:.1f})")
            vl = v if v != "base" else ("1500 h (base)" if study.startswith("eta_C2") else "5000 h (base)")
            md.append(f"| {study} | {vl} | " + " | ".join(cells) + " |")
    with open(os.path.join(a.out, "table_sens_eta.md"), "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(t[["study", "variant", "policy", "param", "pm", "failures"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
