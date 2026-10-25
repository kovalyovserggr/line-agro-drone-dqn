"""Аркуш чисел для розділу Results (WORKFLOW_CHECKLIST D: кожне число в тексті — зі скрипту).

Читає сирі журнали і таблиці (make_table4.py, make_sens.py) і пише
docs/article3/draft/results_numbers.md — список «ключ: значення» з джерелом.
Текст 03_results.md посилається на ці ключі у коментарях; check_draft_numbers.py (пізніше)
звіряє числа тексту з аркушем.

Запуск (після make_table4.py і make_sens.py):
  python experiments/article3/results_numbers.py --raw experiments/article3/raw --tables docs/article3/tables
"""
import argparse, json, os
import numpy as np
import pandas as pd

CAP_WORK_PRINT_H, CAP_SETUP_PRINT_H = 62.69, 5.80      # BOM, год/дрон (capacity_diag.analytic_bound)
CAP_WORK_EXTR_H, CAP_SETUP_EXTR_H = 19.18, 1.92
WEEK_H = 168.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/article3/raw")
    ap.add_argument("--tables", default="docs/article3/tables")
    ap.add_argument("--out", default="docs/article3/draft/results_numbers.md")
    a = ap.parse_args()
    R, T = a.raw, a.tables
    N = []                                   # (ключ, значення, джерело)

    def put(k, v, src, fmt="{:.1f}"):
        N.append((k, fmt.format(v) if isinstance(v, (float, np.floating)) else str(v), src))

    # --- лінія і цінність ТО (помірна, вісь принтерів) -------------------------------
    fin = pd.concat([pd.read_csv(os.path.join(R, f)) for f in ("test_finalists_L.csv", "test20_finalists_L.csv")])
    for n in (12, 15, 17, 19):
        f = fin[fin.n_printers == n]
        b0, bs = f[f.policy == "B0"], f[f.policy == "B1*"]
        put(f"cap_print_{n}", n * WEEK_H / (CAP_WORK_PRINT_H + CAP_SETUP_PRINT_H), "BOM", "{:.1f}")
        put(f"thr_B0_{n}", b0.throughput_per_week.mean(), "test(_20)_finalists_L", "{:.2f}")
        put(f"thr_B1star_{n}", bs.throughput_per_week.mean(), "test(_20)_finalists_L", "{:.2f}")
        put(f"gainPM_pct_{n}", 100 * (bs.profit.mean() / b0.profit.mean() - 1), "test(_20)_finalists_L")
        for p in ("B0", "B1", "B1*", "B2", "B3+"):
            g = f[f.policy == p]
            put(f"OEE_{p}_{n}", g.OEE.mean(), "test(_20)_finalists_L", "{:.3f}")
            put(f"C1OEE_{p}_{n}", g.C1_OEE.mean(), "test(_20)_finalists_L", "{:.3f}")
            put(f"C2OEE_{p}_{n}", g.C2_OEE.mean(), "test(_20)_finalists_L", "{:.3f}")
            put(f"pm_{p}_{n}", g.pm.mean(), "test(_20)_finalists_L", "{:.0f}")
            put(f"pmwait_{p}_{n}", g.pm_waiting.mean(), "test(_20)_finalists_L", "{:.1f}")
            put(f"fail_{p}_{n}", g.failures.mean(), "test(_20)_finalists_L", "{:.1f}")
    put("cap_extr", 5 * WEEK_H / (CAP_WORK_EXTR_H + CAP_SETUP_EXTR_H), "BOM", "{:.1f}")

    # --- DQN-TA: поведінка (вісь принтерів) -------------------------------------------
    t20 = pd.read_csv(os.path.join(R, "test20_dqn_L.csv"))
    sel = pd.read_csv(os.path.join(T, "table4_agent_selection.csv"))
    for n, pre in ((12, "np12"), (15, "np15"), (17, "np17"), (19, "np19")):
        best = int(sel[(sel.scenario == "moderate") & (sel.n_printers == n) & sel.selected].agent.iloc[0])
        d = pd.concat([pd.read_csv(os.path.join(R, f"dqn_ta_{pre}_s{best}", "eval.csv")),
                       t20[t20.agent_dir == f"dqn_ta_{pre}_s{best}"]])
        for k, c, fmt in (("pm", "pm", "{:.0f}"), ("pmwait", "pm_waiting", "{:.1f}"), ("fail", "failures", "{:.1f}"),
                          ("OEE", "OEE", "{:.3f}"), ("C1OEE", "C1_OEE", "{:.3f}"), ("C2OEE", "C2_OEE", "{:.3f}"),
                          ("thr", "throughput_per_week", "{:.2f}")):
            put(f"{k}_SEL_{n}", d[c].mean(), f"dqn_ta_{pre}_s{best} eval + test20", fmt)
        put(f"pmwait_share_SEL_{n}", 100 * d.pm_waiting.sum() / d.pm.sum(), "same", "{:.1f}")

    # --- Table 4 ----------------------------------------------------------------------
    t4 = pd.read_csv(os.path.join(T, "table4.csv"))
    tag = {"B0": "B0", "B1": "B1", "B2": "B2", "B3+": "B3p", "B1*": "B1star",
           "DQN-TA (mean of 3)": "MEAN", "DQN-TA (val.-selected)": "SEL"}
    for _, r in t4.iterrows():
        k = f"{r.scenario[:3]}{r.n_printers}_{tag[r.policy]}"
        put(f"T4_profit_{k}", r.profit_k, "table4.csv")
        if r.policy != "B1*":
            put(f"T4_d_{k}", r.d_vs_B1star_k, "table4.csv", "{:+.1f}")
            put(f"T4_se_{k}", r.se_k, "table4.csv")
            put(f"T4_pholm_{k}", r.p_holm, "table4.csv", "{:.3f}")
    for p in ("DQN-TA (val.-selected)", "DQN-TA (mean of 3)", "B1", "B3+"):
        x = t4[t4.policy == p]
        put(f"T4_pooled_d_{tag[p]}", x.d_vs_B1star_k.mean(), "table4.csv", "{:+.1f}")
        put(f"T4_pooled_se_{tag[p]}", x.d_vs_B1star_k.std(ddof=1) / np.sqrt(len(x)), "table4.csv")
        put(f"T4_nsig_better_{tag[p]}", int(((x.p_holm < 0.05) & (x.d_vs_B1star_k > 0)).sum()), "table4.csv", "{}")
        put(f"T4_nsig_worse_{tag[p]}", int(((x.p_holm < 0.05) & (x.d_vs_B1star_k < 0)).sum()), "table4.csv", "{}")
    put("T4_rel_gain_SEL_min_pct", 100 * (t4[t4.policy == "DQN-TA (val.-selected)"].d_vs_B1star_k /
                                          t4[t4.policy == "B1*"].set_index(["scenario", "n_printers"]).profit_k.reindex(
                                              t4[t4.policy == "DQN-TA (val.-selected)"].set_index(["scenario", "n_printers"]).index).values).min(),
        "table4.csv", "{:.2f}")
    put("T4_rel_gain_SEL_max_pct", 100 * (t4[t4.policy == "DQN-TA (val.-selected)"].d_vs_B1star_k /
                                          t4[t4.policy == "B1*"].set_index(["scenario", "n_printers"]).profit_k.reindex(
                                              t4[t4.policy == "DQN-TA (val.-selected)"].set_index(["scenario", "n_printers"]).index).values).max(),
        "table4.csv", "{:.2f}")
    # вибір агента
    for _, r in sel.iterrows():
        put(f"sel_{r.scenario[:3]}{r.n_printers}_a{r.agent}_val", r.val_profit_k, "table4_agent_selection.csv")
        put(f"sel_{r.scenario[:3]}{r.n_printers}_a{r.agent}_d", r.test_d_vs_B1star, "table4_agent_selection.csv", "{:+.1f}")
    n_match = 0
    for (sc, n), g in sel.groupby(["scenario", "n_printers"]):
        n_match += int(g.loc[g.test_profit_k.idxmax()].agent == g[g.selected].agent.iloc[0])
    put("sel_match_test_best", n_match, "table4_agent_selection.csv", "{}")

    # --- навчання ----------------------------------------------------------------------
    for pre in ("np12", "np15", "np17", "np19", "strong", "weak"):
        L = [pd.read_csv(os.path.join(R, f"dqn_ta_{pre}_s{i}", "train_log.csv")) for i in (1, 2, 3)]
        put(f"train_pm_first_{pre}", np.mean([l.pm.iloc[0] for l in L]), "train_log.csv", "{:.0f}")
        put(f"train_pm_last10_{pre}", np.mean([l.pm.iloc[-10:].mean() for l in L]), "train_log.csv", "{:.0f}")
        put(f"train_dec_per_ep_{pre}", np.mean([l.decisions.iloc[-10:].mean() for l in L]), "train_log.csv", "{:.0f}")
        put(f"train_wall_h_{pre}", np.mean([l.wall_s.sum() / 3600 for l in L]), "train_log.csv", "{:.2f}")
        last = [l.profit_k.iloc[-10:].mean() for l in L]
        put(f"train_last10_spread_{pre}", max(last) - min(last), "train_log.csv")

    # --- абляція (20 seed) і OEE класів за 17 принтерів -----------------------------------
    sm = pd.read_csv(os.path.join("docs/article3/figures", "fig_scenario_map_data.csv"))
    for _, r in sm.iterrows():
        if not np.isnan(r.get("dLOC", np.nan)):
            put(f"abl_dLOC_{int(r.n)}", r.dLOC, "fig_scenario_map_data.csv", "{:+.1f}")
            put(f"abl_seLOC_{int(r.n)}", r.seLOC, "fig_scenario_map_data.csv")
    abl = pd.concat([pd.read_csv(os.path.join(R, f"dqn_g_np17_s{i}", "eval.csv")) for i in (1, 2, 3)] +
                    [pd.read_csv(os.path.join(R, "test20_ablation_L.csv")).query("agent_dir.str.startswith('dqn_g_np17')", engine="python")])
    put("abl17_C1OEE_LOC", abl.C1_OEE.mean(), "dqn_g_np17 eval + test20_ablation", "{:.3f}")
    put("abl17_C2OEE_LOC", abl.C2_OEE.mean(), "dqn_g_np17 eval + test20_ablation", "{:.3f}")
    ta17 = pd.concat([pd.read_csv(os.path.join(R, f"dqn_ta_np17_s{i}", "eval.csv")) for i in (1, 2, 3)] +
                     [t20[t20.agent_dir.str.startswith("dqn_ta_np17")]])
    put("abl17_C1OEE_TA", ta17.C1_OEE.mean(), "dqn_ta_np17 eval + test20", "{:.3f}")
    put("abl17_C2OEE_TA", ta17.C2_OEE.mean(), "dqn_ta_np17 eval + test20", "{:.3f}")

    # --- вісь деградації ------------------------------------------------------------------
    dg = pd.read_csv(os.path.join("docs/article3/figures", "fig_degradation_data.csv"))
    for _, r in dg.iterrows():
        put(f"gainPM_pct_{r.scenario}", r.gain_B1star_pct, "fig_degradation_data.csv")

    # --- робастність і чутливість -------------------------------------------------------
    rb = pd.read_csv(os.path.join(T, "table_robust_lognormal.csv"))
    for _, r in rb.iterrows():
        k = f"{r.scenario[:3]}{r.n_printers}_" + r.policy.replace(" ", "").replace("(", "").replace(")", "").replace("*", "star").replace(".", "")[:24]
        put(f"RB_d_{k}", r.d_vs_B1star_k, "table_robust_lognormal.csv", "{:+.1f}")
        put(f"RB_se_{k}", r.se_k, "table_robust_lognormal.csv")
        put(f"RB_fail_{k}", r.failures, "table_robust_lognormal.csv")
        put(f"RB_p_{k}", r.p_wilcoxon, "table_robust_lognormal.csv", "{:.3f}")
        put(f"RB_pm_{k}", r.pm, "table_robust_lognormal.csv", "{:.0f}")
    st = pd.read_csv(os.path.join(T, "table_sens_eta.csv"))
    for _, r in st.iterrows():
        k = f"{r.study[:6]}_{r.variant.replace(' ', '')}_" + r.policy.replace(" ", "").replace("(", "").replace(")", "").replace("*", "star").replace(".", "")[:20]
        put(f"SE_d_{k}", r.d_vs_B1star_k, "table_sens_eta.csv", "{:+.1f}")
        put(f"SE_se_{k}", r.se_k, "table_sens_eta.csv")
        put(f"SE_profit_{k}", r.profit_k, "table_sens_eta.csv")
        if "p_wilcoxon" in r and not pd.isna(r.p_wilcoxon):
            put(f"SE_p_{k}", r.p_wilcoxon, "table_sens_eta.csv", "{:.3f}")
    for tag_ in ("c2eta750", "c2eta3000"):
        s = json.load(open(os.path.join(R, f"sens_{tag_}_selection.json")))
        put(f"SE_{tag_}_mstar", s["m_star"], "selection.json", "{}")
        put(f"SE_{tag_}_theta", s["theta_star"], "selection.json", "{}")
        put(f"SE_{tag_}_b1star", json.dumps(s["b1star"], sort_keys=True), "selection.json", "{}")
        put(f"SE_{tag_}_selected", s["selected_agent"], "selection.json", "{}")
        for ag, v in s["val_profit_k"].items():
            put(f"SE_{tag_}_val_a{ag}", v, "selection.json")
        d = pd.read_csv(os.path.join(R, f"sens_{tag_}_test_dqn_L.csv")); f = pd.read_csv(os.path.join(R, f"sens_{tag_}_test_L.csv"))
        ref = f[f.policy == "B1*"].set_index("seed").profit
        for ag, g in d.groupby("param"):
            put(f"SE_{tag_}_test_d_a{ag}", ((g.set_index("seed").profit - ref) / 1e3).mean(), "sens test", "{:+.1f}")

    # --- поведінка політик за журналом подій (Р51) -------------------------------------
    pe = os.path.join(R, "policy_events_L.csv")
    if os.path.exists(pe):
        ev = pd.read_csv(pe)
        runs = pd.read_csv(os.path.join(R, "policy_events_runs_L.csv"))
        put("EV_all_identical", bool(runs.identical.all()), "policy_events_runs_L.csv", "{}")
        put("EV_n_runs", len(runs), "policy_events_runs_L.csv", "{}")
        ns = runs.groupby(["key", "policy"]).seed.nunique()
        for (k, p, c), g in ev.groupby(["key", "policy", "cls"]):
            pol = "B1star" if p == "B1*" else "DQN"
            pm, fl = g[g.kind == "pm"], g[g.kind == "failure"]
            n = ns[(k, p)]
            put(f"EV_{k}_{pol}_{c}_pm_yr", len(pm) / n, "policy_events_L.csv")
            put(f"EV_{k}_{pol}_{c}_fail_yr", len(fl) / n, "policy_events_L.csv")
            if len(pm) >= 3:
                put(f"EV_{k}_{pol}_{c}_r_med", pm.r.median(), "policy_events_L.csv", "{:.3f}")
                put(f"EV_{k}_{pol}_{c}_r_cv", pm.r.std() / pm.r.mean(), "policy_events_L.csv", "{:.2f}")
                put(f"EV_{k}_{pol}_{c}_wait_pct", 100 * pm.pm_waiting.mean(), "policy_events_L.csv", "{:.0f}")
                put(f"EV_{k}_{pol}_{c}_young_pct", 100 * (pm.r < 0.5 * pm.r.median()).mean(), "policy_events_L.csv", "{:.1f}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\r\n") as fh:
        fh.write("# Аркуш чисел для Results (згенеровано results_numbers.py — не редагувати вручну)\n\n")
        fh.write("| Ключ | Значення | Джерело |\n|---|---|---|\n")
        for k, v, s in N:
            fh.write(f"| {k} | {v} | {s} |\n")
    print(f"{len(N)} чисел → {a.out}")
    return {k: v for k, v, _ in N}


if __name__ == "__main__":
    main()
