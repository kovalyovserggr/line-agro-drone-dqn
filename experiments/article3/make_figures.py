"""Рисунки Статті 3 — лише з сирих журналів (WORKFLOW_CHECKLIST: рисунки тільки з логів).

  fig_scenario_map  — (a) прибуток B0 / B1* / DQN-TA за кількістю принтерів;
                      (b) різниця до B1* (± SE за 5 тестовими seed): DQN-TA, DQN з
                      локальною винагородою (абляція), B1; B2 — поза шкалою, підписом.
  fig_degradation   — вісь деградації (слабка / помірна / сильна, 15 принтерів): (a) цінність
                      ТО понад B0, %; (b) Δ до B1*: DQN-TA (середнє 3 і вибраний за валідацією, Р47),
                      B1 (m*), B2.
  fig_robust        — Р49: Δ до B1* для заморожених політик при вейбулівській і логнормальній
                      «правді» (з таблиць make_table4.py; запускати після нього).
  fig_sens          — Р50: Δ до B1* за η_C2 (переналаштування) і η_C4 (заморожені політики);
                      з table_sens_eta.csv (make_sens.py; запускати після нього).
  fig_learning      — криві навчання DQN-TA (прибуток за 26-тижневий навчальний
                      епізод), 3 агенти, малі мультиплікатори за n_printers.
  fig_ablation17    — OEE екструзії (C1) і друку (C2) за 17 принтерів: B1*, DQN-локальна,
                      DQN-TA (облік за вузьким місцем).

Дані: test_finalists_L.csv (a3-test-finalists-L), dqn_ta_np*/ (a3-dqn-ta-map-L),
dqn_g_np12, dqn_g_t300, dqn_g_np17 (a3-dqn-grid-*). Текст підписів — англійською (мова статті).

Запуск: python experiments/article3/make_figures.py --raw experiments/article3/raw --out docs/article3/figures
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Категоріальні кольори (перевірено validate_palette.js, світлий режим; aqua < 3:1 до тла →
# кожна серія має також власний маркер і прямий підпис)
C_TA, C_LOC, C_B1, C_REF = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NS = [12, 15, 17, 19]
LOCAL_DIRS = {12: "dqn_g_np12", 15: "dqn_g_t300", 17: "dqn_g_np17"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})


T20 = {"test_finalists_L.csv": "test20_finalists_L.csv", "test_finalists_strong_L.csv": "test20_finalists_strong_L.csv",
       "test_finalists_weak_L.csv": "test20_finalists_weak_L.csv"}


def load_fin(raw, f):
    """Тестові фіналісти: seed 100000–100004 + дооцінка 100005–100019 (Р48), якщо є."""
    d = [pd.read_csv(os.path.join(raw, f))]
    if f in T20 and os.path.exists(os.path.join(raw, T20[f])):
        d.append(pd.read_csv(os.path.join(raw, T20[f])))
    return pd.concat(d, ignore_index=True)


def load_dqn(raw, prefix):
    """eval.csv агентів (seed 100000–100004) + рядки test20_dqn_L.csv (Р48), якщо є для цього каталогу.
    Абляція з локальною винагородою (dqn_g_*) — з test20_ablation_L.csv (Р50), якщо є; інакше 5 seed."""
    ex = [pd.read_csv(os.path.join(raw, f)) for f in ("test20_dqn_L.csv", "test20_ablation_L.csv")
          if os.path.exists(os.path.join(raw, f))]                     # Р48; абляція — Р50
    extra = pd.concat(ex, ignore_index=True) if ex else None
    out = []
    for i in (1, 2, 3):
        d = pd.read_csv(os.path.join(raw, f"{prefix}_s{i}", "eval.csv"))
        if extra is not None:
            d = pd.concat([d, extra[extra.agent_dir == f"{prefix}_s{i}"]], ignore_index=True)
        out.append(d.assign(agent=i))
    return pd.concat(out, ignore_index=True)


def selected_agent(raw, prefix):
    """Р47: агент із найбільшим середнім прибутком на валідаційних seed 200000–200002."""
    v = {i: pd.read_csv(os.path.join(raw, f"{prefix}_s{i}", "val.csv")).profit.mean() for i in (1, 2, 3)}
    return max(v, key=v.get)


def paired(m_by_seed, ref_by_seed):
    d = ((m_by_seed - ref_by_seed) / 1e3).dropna()     # лише спільні seed
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def fig_scenario_map(raw, out):
    fin = load_fin(raw, "test_finalists_L.csv")
    rows = []
    for n in NS:
        fn = fin[fin.n_printers == n]
        ref = fn[fn.policy == "B1*"].set_index("seed").profit
        ta = load_dqn(raw, f"dqn_ta_np{n}").groupby("seed").profit.mean()
        sel = load_dqn(raw, f"dqn_ta_np{n}").pipe(lambda d: d[d.agent == selected_agent(raw, f"dqn_ta_np{n}")])
        r = {}
        r["dSEL"], r["seSEL"] = paired(sel.set_index("seed").profit, ref)
        r |= {"n": n, "B0": fn[fn.policy == "B0"].profit.mean() / 1e6, "B1*": ref.mean() / 1e6,
             "TA": ta.mean() / 1e6}
        r["dTA"], r["seTA"] = paired(ta, ref)
        r["dB1"], r["seB1"] = paired(fn[fn.policy == "B1"].set_index("seed").profit, ref)
        r["dB2"], r["seB2"] = paired(fn[fn.policy == "B2"].set_index("seed").profit, ref)
        if n in LOCAL_DIRS:
            loc = load_dqn(raw, LOCAL_DIRS[n]).groupby("seed").profit.mean()
            r["dLOC"], r["seLOC"] = paired(loc, ref)      # лише 5 спільних seed (100000–100004)
            r["nLOC"] = int(loc.index.isin(ref.index).sum())
        rows.append(r)
    t = pd.DataFrame(rows)
    t.round(3).to_csv(os.path.join(out, "fig_scenario_map_data.csv"), index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) абсолютний прибуток: DQN-TA — лише маркери поверх лінії B1* (вони майже збігаються)
    a.plot(t.n, t["B0"], color=INK2, lw=2, marker="s", ms=6, mec="white", mew=1, label="B0 (no PM)")
    a.plot(t.n, t["B1*"], color=C_B1, lw=2, marker="D", ms=7, mec="white", mew=1, label="B1* (best classical)")
    a.plot(t.n, t["TA"], color=C_TA, lw=0, marker="o", ms=4.5, label="DQN-TA")
    a.set_xticks(NS); a.set_xlabel("Number of printers"); a.set_ylabel("Profit, M USD / year")
    a.set_title("(a) Line profit on test seeds", loc="left")
    a.axvspan(16, 19.5, color="#f1f0ec", zorder=0, lw=0)
    a.set_xlim(11.5, 19.5); a.set_ylim(4.7, 7.8)
    a.text(12.0, 4.78, "bottleneck: printing", va="bottom", fontsize=7.5, color=INK2)
    a.text(16.15, 4.78, "bottleneck: extrusion", va="bottom", fontsize=7.5, color=INK2)
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=1)

    # (b) різниця до B1*
    off = {"TA": -0.3, "SEL": -0.1, "LOC": 0.1, "B1": 0.3}
    b.axhline(0, color=INK2, lw=1)
    b.text(13.4, 1.5, "B1* = 0", ha="center", va="bottom", fontsize=7.5, color=INK2)
    for key, c, mk, lab in [("TA", C_TA, "o", "DQN-TA, mean of 3 agents"),
                            ("SEL", C_TA, "s", "DQN-TA, agent selected on validation seeds"),
                            ("LOC", C_LOC, "^", "DQN, local reward (ablation" + ("; 5 seeds)" if t["nLOC"].max() < 20 else ")")),
                            ("B1", C_REF, "D", "B1 (single period multiplier)")]:
        s = t.dropna(subset=[f"d{key}"])
        hollow = key == "SEL"
        b.errorbar(s.n + off[key], s[f"d{key}"], yerr=s[f"se{key}"], color=c, marker=mk, ms=6,
                   mec=c if hollow else "white", mfc="white" if hollow else c, mew=1.5 if hollow else 1,
                   lw=0, elinewidth=1.5, capsize=0, label=lab)
    b.set_xticks(NS); b.set_xlabel("Number of printers")
    b.set_ylabel("Δ profit vs B1*, k USD / year")
    b.set_title("(b) Difference to the best classical policy (± SE)", loc="left")
    lo = min(t[["dTA", "dB1"]].min().min(), t["dLOC"].min()) - 25
    b.set_ylim(lo, max(t["dTA"].max(), t["dSEL"].max(), t["dLOC"].max()) + 18)
    b.set_xlim(11.5, 19.5)
    off_scale = [f"{int(r.n)} printers: {r['dB2']:.0f}" for _, r in t.iterrows() if r["dB2"] < lo]
    if off_scale:
        b.text(11.65, lo + 3, "B2 off scale — " + "; ".join(off_scale), fontsize=7, color=INK2, va="bottom")
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=1)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_scenario_map.{ext}"))
    plt.close(fig)
    return t


DEGR = [("weak", "Weak", "test_finalists_weak_L.csv", "test_b1star_weak_L.csv", "dqn_ta_weak", 0.5),
        ("moderate", "Moderate", "test_finalists_L.csv", None, "dqn_ta_np15", 0.5),
        ("strong", "Strong", "test_finalists_strong_L.csv", "test_b1star_strong_L.csv", "dqn_ta_strong", 0.35)]


def fig_degradation(raw, out):
    """Вісь деградації (15 принтерів): (a) цінність ТО — прибуток B1* і DQN-TA понад B0, %;
    (b) Δ до B1* (± SE): DQN-TA середнє 3 агентів, вибраний за валідацією агент, B1 (m*), B2."""
    rows = []
    for sc, lab, ff, fb, pre, m_star in DEGR:
        fin = load_fin(raw, ff)
        fin = fin[fin.n_printers == 15]
        if fb:
            fin = pd.concat([fin, pd.read_csv(os.path.join(raw, fb))])
        ref = fin[fin.policy == "B1*"].set_index("seed").profit
        b0 = fin[fin.policy == "B0"].set_index("seed").profit
        d = load_dqn(raw, pre)
        ta = d.groupby("seed").profit.mean()
        sel = d[d.agent == selected_agent(raw, pre)].set_index("seed").profit
        b1 = fin[(fin.policy == "B1") & np.isclose(pd.to_numeric(fin.param, errors="coerce"), m_star)].set_index("seed").profit
        b2 = fin[fin.policy == "B2"].set_index("seed").profit
        r = {"scenario": sc, "label": lab, "B0": b0.mean() / 1e6, "B1*": ref.mean() / 1e6,
             "gain_B1star_pct": 100 * (ref.mean() / b0.mean() - 1), "gain_TA_pct": 100 * (ta.mean() / b0.mean() - 1),
             "gain_SEL_pct": 100 * (sel.mean() / b0.mean() - 1)}
        for k, v in [("TA", ta), ("SEL", sel), ("B1", b1), ("B2", b2)]:
            r[f"d{k}"], r[f"se{k}"] = paired(v, ref)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.round(3).to_csv(os.path.join(out, "fig_degradation_data.csv"), index=False)
    x = np.arange(len(t))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    a.bar(x - 0.2, t.gain_B1star_pct, 0.38, color=C_B1, label="B1* (best classical)")
    a.bar(x + 0.2, t.gain_SEL_pct, 0.38, color=C_TA, label="DQN-TA (selected agent)")
    for xi, v in zip(x, t.gain_SEL_pct):
        a.text(xi + 0.2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    for xi, v in zip(x, t.gain_B1star_pct):
        a.text(xi - 0.2, v + 0.4, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    a.set_xticks(x); a.set_xticklabels(t.label); a.set_xlabel("Degradation level (15 printers)")
    a.set_ylabel("Profit gain over B0 (no PM), %"); a.set_title("(a) Value of preventive maintenance", loc="left")
    a.set_ylim(0, t[["gain_B1star_pct", "gain_SEL_pct"]].max().max() * 1.15)
    a.grid(axis="x", visible=False)
    a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=1)

    off = {"TA": -0.24, "SEL": -0.08, "B1": 0.08, "B2": 0.24}
    b.axhline(0, color=INK2, lw=1)
    lo = min(t[["dTA", "dSEL", "dB1"]].min().min() - t["seTA"].max(), -60) - 10
    for key, c, mk, lab in [("TA", C_TA, "o", "DQN-TA, mean of 3 agents"),
                            ("SEL", C_TA, "s", "DQN-TA, selected agent"),
                            ("B1", C_REF, "D", "B1 (m* per level)"),
                            ("B2", INK2, "v", "B2 (risk threshold)")]:
        hollow = key == "SEL"
        ok = t[f"d{key}"] >= lo
        b.errorbar(x[ok] + off[key], t[f"d{key}"][ok], yerr=t[f"se{key}"][ok], color=c, marker=mk, ms=6,
                   mec=c if hollow else "white", mfc="white" if hollow else c, mew=1.5 if hollow else 1,
                   lw=0, elinewidth=1.5, capsize=0, label=lab)
    off_scale = [f"{r.label.lower()}: {r.dB2:.0f}" for _, r in t.iterrows() if r.dB2 < lo]
    if off_scale:
        b.text(len(t) - 0.55, lo + 3, "B2 off scale — " + "; ".join(off_scale), fontsize=7, color=INK2, va="bottom", ha="right")
    b.set_ylim(lo, t[["dTA", "dSEL"]].max().max() + t["seSEL"].max() + 12)
    b.set_xticks(x); b.set_xticklabels(t.label); b.set_xlabel("Degradation level (15 printers)")
    b.set_ylabel("Δ profit vs B1*, k USD / year"); b.set_title("(b) Difference to B1* (± SE)", loc="left")
    b.set_xlim(-0.5, len(t) - 0.5)
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_degradation.{ext}"))
    plt.close(fig)
    return t


def fig_robust(table_dir, out):
    """Р49: Δ до B1* при вейбулівській і логнормальній «правді» (політики заморожені), ± SE, 20 seed.
    Дані — docs/article3/tables/table4.csv і table_robust_lognormal.csv (make_table4.py)."""
    t4 = pd.read_csv(os.path.join(table_dir, "table4.csv"))
    rb = pd.read_csv(os.path.join(table_dir, "table_robust_lognormal.csv"))
    sc = [("moderate", 15, "Moderate,\n15 printers"), ("moderate", 17, "Moderate,\n17 printers"), ("strong", 15, "Strong,\n15 printers")]
    pols = [("DQN-TA (val.-selected)", "DQN-TA (val.-selected)", C_TA, "s", True),
            ("DQN-TA (mean of 3)", "DQN-TA (mean of 3)", C_TA, "o", False),
            ("B1", "B1 (m* Weibull = m* lognormal)", C_REF, "D", False)]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.axhline(0, color=INK2, lw=1)
    x = np.arange(len(sc))
    for j, (pw, pl, c, mk, hollow) in enumerate(pols):
        for k, (src, key, ls, alpha) in enumerate([(t4, pw, "Weibull", 0.45), (rb, pl, "lognormal", 1.0)]):
            ys, es = [], []
            for s_, n, _ in sc:
                r = src[(src.scenario == s_) & (src.n_printers == n) & (src.policy == key)].iloc[0]
                ys.append(r.d_vs_B1star_k); es.append(r.se_k)
            xo = x + (j - 1) * 0.25 + (k - 0.5) * 0.1
            ax.errorbar(xo, ys, yerr=es, color=c, alpha=alpha, marker=mk, ms=6, lw=0, elinewidth=1.5,
                        mec=c if hollow else "white", mfc="white" if hollow else c, mew=1.5 if hollow else 1,
                        label=f"{pw.replace(' (val.-selected)', ', selected agent').replace(' (mean of 3)', ', mean of 3')} — {ls} truth")
    ax.set_xticks(x); ax.set_xticklabels([l for *_, l in sc])
    ax.set_ylabel("Δ profit vs B1*, k USD / year")
    ax.set_title("Frozen policies under a misspecified failure model (± SE, 20 seeds)", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_robust.{ext}"))
    plt.close(fig)


def fig_sens(table_dir, out):
    """Р50: Δ до B1* за η_C2 (усе переналаштовано) і η_C4 (заморожені політики), ± SE, 20 seed.
    Дані — docs/article3/tables/table_sens_eta.csv (make_sens.py)."""
    t = pd.read_csv(os.path.join(table_dir, "table_sens_eta.csv"))
    xv = {"750 h": 750, "base": None, "3000 h": 3000, "2500 h": 2500, "10000 h": 10000}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, study, base, title in [(axs[0], "eta_C2 (re-tuned)", 1500, "(a) Printer wear scale η_C2 (all policies re-tuned)"),
                                   (axs[1], "eta_C4 (frozen)", 5000, "(b) Tooling wear scale η_C4 (policies frozen)")]:
        d = t[t.study == study].copy()
        d["x"] = d.variant.map(lambda v: base if v == "base" else xv[v])
        ax.axhline(0, color=INK2, lw=1)
        for k, (pol, c, mk, hollow, lab) in enumerate([
                ("DQN-TA (val.-selected)", C_TA, "s", True, "DQN-TA, selected agent"),
                ("DQN-TA (mean of 3)", C_TA, "o", False, "DQN-TA, mean of 3 agents"),
                ("B1", C_REF, "D", False, "B1 (single multiplier)")]):
            g = d[d.policy == pol].sort_values("x")
            ax.errorbar(g.x * (1 + (k - 1) * 0.06), g.d_vs_B1star_k, yerr=g.se_k, color=c, marker=mk, ms=6, lw=1 if not hollow else 0,
                        alpha=0.9, elinewidth=1.5, mec=c if hollow else "white", mfc="white" if hollow else c,
                        mew=1.5 if hollow else 1, label=lab)
        ax.set_xscale("log")
        xs = sorted(d.x.unique())
        ax.set_xticks(xs); ax.set_xticklabels([f"{int(x):,}".replace(",", " ") + (" (base)" if x == base else "") for x in xs], fontsize=7.5)
        ax.minorticks_off()
        ax.set_xlabel("η, h of operation")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(axis="x", visible=False)
        b2 = d[(d.policy == "B2") & (d.variant != "base")]
        if study.startswith("eta_C2"):
            txt = "B2: " + "; ".join(f"{int(r.x)} h: {r.d_vs_B1star_k:.0f}" for _, r in d[d.policy == "B2"].sort_values("x").iterrows())
            ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=7, color=INK2)
    axs[0].set_ylabel("Δ profit vs B1*, k USD / year")
    axs[0].set_ylim(-25, 35)
    axs[0].legend(frameon=False, loc="upper right", fontsize=7.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_sens.{ext}"))
    plt.close(fig)


def fig_learning(raw, out):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True)
    for ax, n in zip(axs.flat, NS):
        logs = [pd.read_csv(os.path.join(raw, f"dqn_ta_np{n}_s{i}", "train_log.csv")) for i in (1, 2, 3)]
        prof = np.vstack([l.profit_k.values / 1e3 for l in logs])
        ep = logs[0].episode.values
        ax.fill_between(ep, prof.min(0), prof.max(0), color=C_TA, alpha=0.18, lw=0)
        ax.plot(ep, prof.mean(0), color=C_TA, lw=2, label="mean of 3 agents")
        e_end = int(ep[np.argmax(logs[0].eps.values <= 0.0501)])
        ax.axvline(e_end, color=INK2, lw=0.8, ls=":")
        ax.text(e_end + 0.5, ax.get_ylim()[0], "ε = 0.05", fontsize=7, color=INK2, va="bottom")
        ax.set_title(f"{n} printers", loc="left")
        if n in (12, 17):
            ax.set_ylabel("Profit per episode,\nM USD / 26 weeks")
    for ax in axs[1]:
        ax.set_xlabel("Training episode")
    axs[0, 0].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_learning.{ext}"))
    plt.close(fig)


def fig_ablation17(raw, out):
    fin = load_fin(raw, "test_finalists_L.csv")
    fn = fin[fin.n_printers == 17]
    sets = [("B1*", fn[fn.policy == "B1*"], C_B1, "D"),
            ("DQN, local reward", load_dqn(raw, "dqn_g_np17"), C_LOC, "^"),
            ("DQN-TA", load_dqn(raw, "dqn_ta_np17"), C_TA, "o")]
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for j, (cls, lab_cls) in enumerate([("C1", "Extrusion (bottleneck)"), ("C2", "Printing")]):
        for k, (lab, d, c, mk) in enumerate(sets):
            v = d[f"{cls}_OEE"]
            y = j + (k - 1) * 0.22
            ax.errorbar(v.mean(), y, xerr=v.std(ddof=1) / np.sqrt(len(v)), color=c, marker=mk, ms=7,
                        mec="white", mew=1, lw=0, elinewidth=1.5, label=lab if j == 0 else None)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Extrusion\n(bottleneck)", "Printing"])
    ax.set_ylim(-0.6, 1.6); ax.invert_yaxis()
    ax.set_xlabel("Class OEE (17 printers, test seeds, ± SE)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig_ablation17.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="experiments/article3/raw")
    ap.add_argument("--out", default="docs/article3/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = fig_scenario_map(a.raw, a.out)
    print(t.round(1).to_string(index=False))
    print(fig_degradation(a.raw, a.out).round(1).to_string(index=False))
    tdir = os.path.join(os.path.dirname(os.path.normpath(a.out)), "tables")
    if os.path.exists(os.path.join(tdir, "table_robust_lognormal.csv")):
        fig_robust(tdir, a.out)
    if os.path.exists(os.path.join(tdir, "table_sens_eta.csv")):
        fig_sens(tdir, a.out)
    fig_learning(a.raw, a.out)
    fig_ablation17(a.raw, a.out)
    print("рисунки збережено в", a.out)


if __name__ == "__main__":
    main()
