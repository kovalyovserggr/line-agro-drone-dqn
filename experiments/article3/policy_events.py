"""Аналіз поведінки політик за журналом подій (Р51; рисунки 7–8 статті).

Що робить:
  1. --run: прогони з журналом подій (JSONL.gz) для B1* і агента DQN-TA, вибраного за
     валідацією (Р47), на тих самих тестових seed (CRN) у трьох сценаріях:
       moderate, 12 принтерів — вузьке місце друк, значимий виграш DQN-TA;
       moderate, 19 принтерів — вузьке місце екструзія, значимий виграш;
       strong,   15 принтерів — найбільший виграш (сильна деградація).
     Протокол — як у тесті: розігрів 26 тижнів, вікно 1 рік. Журнал неінвазивний: прибуток
     кожного прогону звіряється з уже записаним тестовим результатом (має збігтися побайтово).
  2. --parse: з журналів — таблиці подій (у git, компактні):
       raw/policy_events_L.csv  — кожне ТО і кожна відмова у вікні вимірювання: сценарій,
                                  політика, seed, вузол, клас, час, вид (pm/failure), r = max v/η
                                  зносових складових, напрацювання від ТО, ТО під час очікування;
       raw/policy_gantt_L.csv   — інтервали станів вибраних вузлів (5 екструдерів + 5 принтерів)
                                  у 21-денному вікні з початку вимірювання — для діаграми Ганта.
     Самі журнали (десятки МБ) у git не йдуть: raw/eventlogs/ (у .gitignore; архів — Zenodo).

Запуск (L):
  python experiments/article3/policy_events.py --run --parse --workers 5
Пілот (C): --seeds 100000 --scenarios mod12
"""
import argparse, contextlib, csv, gzip, io, json, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
sys.path.insert(0, HERE)
import pandas as pd

WARMUP_H, WINDOW_H = 26 * 7 * 24, 8760
SCEN = {"mod12": ("moderate", 12, "dqn_ta_np12"), "mod19": ("moderate", 19, "dqn_ta_np19"),
        "str15": ("strong", 15, "dqn_ta_strong")}
GANTT_DAYS = 21


def b1star_mults(raw, scenario, n):
    """Множники B1* — з тестового CSV, яким оцінювали B1* (те саме джерело, що й Table 4)."""
    files = {"moderate": "test_finalists_L.csv", "strong": "test_b1star_strong_L.csv"}
    d = pd.read_csv(os.path.join(raw, files[scenario]))
    d = d[(d.policy == "B1*") & (d.n_printers == n)]
    return json.loads(d.param.iloc[0])


def reference_profit(raw, scenario, n, policy, agent_dir, seed):
    if policy == "B1*":
        files = {"moderate": "test_finalists_L.csv", "strong": "test_b1star_strong_L.csv"}
        d = pd.read_csv(os.path.join(raw, files[scenario]))
        d = d[(d.policy == "B1*") & (d.n_printers == n) & (d.seed == seed)]
    else:
        d = pd.read_csv(os.path.join(raw, agent_dir, "eval.csv"))
        d = d[d.seed == seed]
    return float(d.profit.iloc[0]) if len(d) else None


def job(a):
    key, policy, seed, raw, logdir = a
    import torch
    torch.set_num_threads(1)
    from LineV3 import AgroDroneLineV3, PeriodicPM, _snapshot, line_metrics, DT
    from line_config_v3 import scenario_classes
    from analytic_v3 import class_T_star_seconds
    from EventLog import EventLogger
    scenario, n, pre = SCEN[key]
    cl, ov = scenario_classes(scenario)
    T = class_T_star_seconds(cl)
    agent_dir = None
    if policy == "B1*":
        mult = b1star_mults(raw, scenario, n)
        pol = PeriodicPM({c: m * T[c] for c, m in mult.items()})
        pdesc = json.dumps(mult, sort_keys=True)
    else:
        from DQNAgentV3 import DoubleDQNAgentV3
        from DQNLineV3 import DQNPMPolicy, OBS_DIM
        sel = pd.read_csv(os.path.join(HERE, "..", "..", "docs", "article3", "tables", "table4_agent_selection.csv"))
        best = int(sel[(sel.scenario == scenario) & (sel.n_printers == n) & sel.selected].agent.iloc[0])
        agent_dir = f"{pre}_s{best}"
        cfg = json.load(open(os.path.join(raw, agent_dir, "config.json")))
        agent = DoubleDQNAgentV3(OBS_DIM, 2, seed=cfg["agent_seed"], eps_steps=10**9, total_steps=10**9,
                                 buffer=1000, warmup=cfg.get("warmup", 1000))
        agent.online.load_state_dict(torch.load(os.path.join(raw, agent_dir, "model.pt")))
        pol = DQNPMPolicy(agent, train=False, grid=cfg.get("grid", 0.0))
        pdesc = agent_dir
    path = os.path.join(logdir, f"{key}_{'B1star' if policy == 'B1*' else 'DQN'}_{seed}.jsonl.gz")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        line = AgroDroneLineV3(seed, pol, classes=cl, node_overrides=ov, crew=1, n_printers=n)
        meta = {"model": "v0.3", "scenario": scenario, "n_printers": n, "seed": seed, "policy": policy,
                "policy_param": pdesc, "warmup_h": WARMUP_H, "window_h": WINDOW_H,
                "classes": {str(u.id): u.cls for u in line.units},
                "eta_h": {str(u.id): [m.eta / 3600 for m in u.rel.modes] for u in line.units},
                "T_star_h": {c: v / 3600 for c, v in T.items()}}
        log = EventLogger(path, line, meta=meta, stock_every=1e12)
        n_ticks = int(round((WARMUP_H + WINDOW_H) * 3600 / DT)); warm = int(round(WARMUP_H * 3600 / DT))
        t, s0 = 0.0, None
        for k in range(1, n_ticks + 1):
            t += DT
            line.tick(t)
            if k == warm:
                s0 = _snapshot(line)
        log.close("horizon", t)
    m = line_metrics(s0, _snapshot(line), line)
    ref = reference_profit(raw, scenario, n, policy, agent_dir, seed)
    return {"key": key, "policy": policy, "policy_param": pdesc, "seed": seed, "profit": m["profit"],
            "ref_profit": ref, "identical": ref is not None and m["profit"] == ref, "pm": m["pm"],
            "failures": m["failures"], "log": os.path.basename(path), "wall_s": round(time.time() - t0, 1)}


def parse(logdir, runs, out_events, out_gantt):
    ev_rows, g_rows = [], []
    for r in runs:
        path = os.path.join(logdir, r["log"])
        with gzip.open(path, "rt", encoding="utf-8") as f:
            meta = json.loads(f.readline())
            cls = {int(float(k)): v for k, v in meta["classes"].items()}
            w0 = meta["warmup_h"] * 3600.0
            g_end = w0 + GANTT_DAYS * 86400.0
            c1 = sorted(i for i, c in cls.items() if c == "C1")
            c2 = sorted(i for i, c in cls.items() if c == "C2")[:5]
            gnodes = set(c1 + c2)
            cur, since, last_pw = {}, {}, {}
            for line in f:
                e = json.loads(line)
                if e["ev"] != "state":
                    continue
                nid, t = int(float(e["node"])), e["t"]
                if nid in gnodes and nid in cur and t > w0 and since[nid] < g_end:
                    g_rows.append({"key": r["key"], "policy": r["policy"], "seed": r["seed"], "node": nid, "cls": cls[nid],
                                   "state": cur[nid], "t0_h": (max(since[nid], w0) - w0) / 3600,
                                   "t1_h": (min(t, g_end) - w0) / 3600})
                cur[nid], since[nid] = e["to"], t
                waiting = False
                if "pm_wait" in e:                      # ТО під час очікування: приріст лічильника (Р51)
                    waiting = e["to"] == "MAINTENANCE" and e["pm_wait"] > last_pw.get(nid, 0)
                    last_pw[nid] = e["pm_wait"]
                if e["to"] in ("MAINTENANCE", "REPAIR") and t >= w0 and "r" in e:
                    ev_rows.append({"pm_waiting": waiting, "key": r["key"], "scenario": meta["scenario"], "n_printers": meta["n_printers"],
                                    "policy": r["policy"], "seed": r["seed"], "node": nid, "cls": cls[nid],
                                    "t_h": (t - w0) / 3600, "kind": "pm" if e["to"] == "MAINTENANCE" else "failure",
                                    "from_state": e["from"], "r": e["r"], "op_pm_h": e["op_pm_h"],
                                    "mode": e.get("mode", "")})
    ev = pd.DataFrame(ev_rows).sort_values(["key", "policy", "seed", "node", "t_h"])
    ev.to_csv(out_events, index=False)
    pd.DataFrame(g_rows).to_csv(out_gantt, index=False)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--scenarios", nargs="+", default=list(SCEN))
    ap.add_argument("--seeds", nargs="+", type=int, default=[100000, 100001, 100002])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    a = ap.parse_args()
    logdir = os.path.join(a.raw, "eventlogs")
    os.makedirs(logdir, exist_ok=True)
    runs_csv = os.path.join(a.raw, "policy_events_runs_L.csv")
    if a.run:
        jobs = [(k, p, s, a.raw, logdir) for k in a.scenarios for p in ("B1*", "DQN") for s in a.seeds]
        rows = []
        with Pool(a.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
                rows.append(r)
                print(f"[{i}/{len(jobs)}] {r['key']} {r['policy']} seed={r['seed']}: прибуток {r['profit']/1e3:.1f} тис. "
                      f"(тест: {r['ref_profit']/1e3 if r['ref_profit'] else float('nan'):.1f}; збіг: {r['identical']}), "
                      f"ТО {r['pm']}, відмов {r['failures']} ({r['wall_s']} с)", flush=True)
        pd.DataFrame(rows).sort_values(["key", "policy", "seed"]).to_csv(runs_csv, index=False)
        bad = [r for r in rows if not r["identical"]]
        if bad:
            print(f"УВАГА: {len(bad)} прогонів не збіглися з тестовими результатами — журнал не має змінювати траєкторію!")
    if a.parse:
        runs = pd.read_csv(runs_csv).to_dict("records")
        ev = parse(logdir, runs, os.path.join(a.raw, "policy_events_L.csv"), os.path.join(a.raw, "policy_gantt_L.csv"))
        s = ev.groupby(["key", "policy", "cls", "kind"]).agg(n=("r", "size"), r_mean=("r", "mean"), r_cv=("r", lambda x: x.std() / x.mean()),
                                                           waiting=("pm_waiting", "mean")).round(3)
        s["n_per_year"] = (s.n / ev.groupby(["key", "policy"]).seed.nunique().reindex(s.index.droplevel([2, 3])).values).round(1)
        print(s.to_string())


if __name__ == "__main__":
    main()
