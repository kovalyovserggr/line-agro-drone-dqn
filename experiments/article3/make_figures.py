"""Рисунки Статті 3 — лише з сирих журналів (WORKFLOW_CHECKLIST: рисунки тільки з логів).

  fig_scenario_map  — (a) прибуток B0 / B1* / DQN-TA за кількістю принтерів;
                      (b) різниця до B1* (± SE за 5 тестовими seed): DQN-TA, DQN з
                      локальною винагородою (абляція), B1; B2 — поза шкалою, підписом.
  fig_degradation   — вісь деградації (слабка / помірна / сильна, 15 принтерів): (a) цінність
                      ТО понад B0, %; (b) Δ до B1*: DQN-TA (середнє 3 і вибраний за валідацією, Р47),
                      B1 (m*), B2.
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


def load_dqn(raw, prefix):
    return pd.concat([pd.read_csv(os.path.join(raw, f"{prefix}_s{i}", "eval.csv")).assign(agent=i)
                      for i in (1, 2, 3)])


def selected_agent(raw, prefix):
    """Р47: агент із найбільшим середнім прибутком на валідаційних seed 200000–200002."""
    v = {i: pd.read_csv(os.path.join(raw, f"{prefix}_s{i}", "val.csv")).profit.mean() for i in (1, 2, 3)}
    return max(v, key=v.get)


def paired(m_by_seed, ref_by_seed):
    d = (m_by_seed - ref_by_seed) / 1e3
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def fig_scenario_map(raw, out):
    fin = pd.read_csv(os.path.join(raw, "test_finalists_L.csv"))
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
            r["dLOC"], r["seLOC"] = paired(loc, ref)
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
                            ("LOC", C_LOC, "^", "DQN, local reward (ablation)"),
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
        fin = pd.read_csv(os.path.join(raw, ff))
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
    fin = pd.read_csv(os.path.join(raw, "test_finalists_L.csv"))
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
    fig_learning(a.raw, a.out)
    fig_ablation17(a.raw, a.out)
    print("рисунки збережено в", a.out)


if __name__ == "__main__":
    main()
