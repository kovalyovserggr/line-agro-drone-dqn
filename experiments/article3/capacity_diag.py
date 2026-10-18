"""Діагностика верхньої межі резерву диспетчеризації (Р37, п. 5).

Дві частини:
  (1) Аналітична межа потужності за специфікацією (BOM) — для кожного пулу вузлів:
      потрібний час на один дрон (робота + переналагодження) проти доступного часу пулу.
      Межа «без втрат» і межа «з виміряними втратами» (частка часу на ТО/ремонт,
      P і Q пулу з того самого прогону).
  (2) Розклад часу вузлів у симуляції за класами: робота, переналагодження,
      ТО, ремонт, голодування (WAITING_FOR_RESOURCE — чекає вхідний ресурс),
      блокування (IDLE — немає дозволеної операції через кеп буфера).
      Варіанти: правило диспетчеризації × buffer_drones × type0_cap (кеп філаменту).
      Кепи — структурні параметри; тут вони лише локалізують, де втрачається потік.

Запуск (L): PYTHONPATH=src python experiments/article3/capacity_diag.py --dispatch D17 quota random --type0-cap 4000 auto --workers 5 --out experiments/article3/raw/capacity_diag_L.csv  (a3-capacity-diag-L)
            PYTHONPATH=src python experiments/article3/capacity_diag.py --dt 25 50 100 200 --n-printers 12 15 17 19 --workers 5 --out experiments/article3/raw/dt_printers_L.csv  (a3-dt-printers-L)
"""
import argparse, contextlib, csv, io, itertools, os, sys, time
from collections import Counter
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from LineV3 import AgroDroneLineV3, _snapshot, line_metrics
from line_config_v3 import scenario_classes
from analytic_v3 import class_T_star_seconds
from baselines_tune import make_policy, WARMUP_H, WINDOW_H

STATES = ["WORKING", "IDLE", "WAITING_FOR_RESOURCE", "MAINTENANCE", "REPAIR"]


def bom_need(line):
    """Кількість операцій кожного типу на один дрон (повне розгортання BOM) і їх тривалості."""
    ops = {}
    for n in line.units:
        for op in n.operations:
            ops[op.name] = op
    need = Counter()
    def expand(name, q):
        need[name] += q
        for r, k in ops[name].input_resources.items():
            if r in ops and r != "Type0":
                expand(r, q * k)
    expand("Type22", 1)
    filament = sum(need[t] * ops[t].input_resources.get("Type0", 0) for t in need)
    need["Type0"] = filament / 1000.0          # операція екструзії дає 1000 г
    return need, ops


def job(a):
    disp, kind, param, seed, buf, cap0, scale, n_pr, dt, tc = a
    cl, ov = scenario_classes("moderate")
    pol = make_policy(kind, param, class_T_star_seconds(cl))
    t0 = time.time()
    line = AgroDroneLineV3(seed, pol, classes=cl, node_overrides=ov, crew=1, dispatch=disp,
                           buffer_drones=buf, type0_cap=cap0, n_printers=n_pr, time_carry=tc)
    n_ticks = int(round((WARMUP_H + WINDOW_H) * scale * 3600 / dt))
    warm = int(round(WARMUP_H * scale * 3600 / dt))
    t, s0 = 0.0, None
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(1, n_ticks + 1):
            t += dt
            line.tick(t)
            if k == warm:
                s0 = _snapshot(line)
    s1 = _snapshot(line)
    m = line_metrics(s0, s1, line)
    T = s1["t"] - s0["t"]
    row = {"dispatch": disp, "policy": kind, "param": param, "seed": seed, "buffer_drones": buf,
           "type0_cap": cap0 if cap0 == "auto" else int(cap0), "n_printers": n_pr, "dt": dt, "time_carry": tc, "throughput_per_week": m["throughput_per_week"], "profit": m["profit"],
           "OEE": m["OEE"]}
    # розклад часу за класами (частки календарного часу вікна)
    by_cls = {}
    for n in line.units:
        d1, d0 = s1["nodes"][n.id], s0["nodes"][n.id]
        c = by_cls.setdefault(n.cls, Counter())
        c["nodes"] += 1
        for st in STATES:
            c[st] += d1["dur"][st] - d0["dur"][st]
        c["setup"] += d1["setup"] - d0["setup"]
        c["ideal"] += d1["ideal"] - d0["ideal"]
        c["comp"] += d1["n_comp"] - d0["n_comp"]
        c["good"] += d1["n_good"] - d0["n_good"]
    for cls, c in sorted(by_cls.items()):
        tot = c["nodes"] * T
        row[f"{cls}_work"] = (c["WORKING"] - c["setup"]) / tot
        row[f"{cls}_setup"] = c["setup"] / tot
        row[f"{cls}_pm"] = c["MAINTENANCE"] / tot
        row[f"{cls}_cm"] = c["REPAIR"] / tot
        row[f"{cls}_starved"] = c["WAITING_FOR_RESOURCE"] / tot
        row[f"{cls}_blocked"] = c["IDLE"] / tot
        row[f"{cls}_P"] = c["ideal"] / (c["WORKING"] - c["setup"]) if c["WORKING"] > c["setup"] else float("nan")
        row[f"{cls}_Q"] = c["good"] / c["comp"] if c["comp"] else float("nan")
    row["wall_s"] = round(time.time() - t0, 1)
    return row


def analytic_bound(rows):
    """Межа потужності за пулами: без втрат і з виміряними втратами (середнє по рядках)."""
    cl, ov = scenario_classes("moderate")
    with contextlib.redirect_stdout(io.StringIO()):
        line = AgroDroneLineV3(0, make_policy("B0", 0.0, class_T_star_seconds(cl)), classes=cl,
                               node_overrides=ov, crew=1)
    need, ops = bom_need(line)
    pools = {}
    for n in line.units:
        for op in n.operations:
            pools.setdefault(op.name, set()).add(n.id)
    # групуємо типи операцій за однаковими наборами вузлів
    groups = {}
    for name, nodes in pools.items():
        groups.setdefault(frozenset(nodes), []).append(name)
    week = 7 * 24 * 3600
    print("\nАналітична межа потужності (BOM, 1 дрон):")
    out = []
    for nodes, names in groups.items():
        cls = next(n.cls for n in line.units if n.id in nodes)
        work = sum(need[t] * ops[t].duration for t in names)
        setup = sum(need[t] * ops[t].duration_letup for t in names)
        cap0 = len(nodes) * week / (work + setup)
        if rows:
            avail = sum(1 - r[f"{cls}_pm"] - r[f"{cls}_cm"] for r in rows) / len(rows)
            P = sum(r[f"{cls}_P"] for r in rows) / len(rows)
            Q = sum(r[f"{cls}_Q"] for r in rows) / len(rows)
            cap1 = len(nodes) * week * avail / (work / P / Q + setup / Q)
        else:
            avail = P = Q = cap1 = float("nan")
        out.append((cap1, cls, names, len(nodes), work / 3600, setup / 3600, cap0, avail, P, Q))
    for cap1, cls, names, k, w, s, cap0, av, P, Q in sorted(out, key=lambda x: x[0]):
        nm = names[0] if len(names) == 1 else f"{names[0]}…{names[-1]} ({len(names)} типів)"
        print(f"  {cls} {nm:28s} вузлів {k:2d}: робота {w:6.2f} год + налагодження {s:5.2f} год/дрон → "
              f"межа без втрат {cap0:6.1f}, з втратами {cap1:6.1f} дрон/тиж (доступн. {av:.3f}, P {P:.3f}, Q {Q:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispatch", nargs="+", default=["D17"])
    ap.add_argument("--buffer", nargs="+", type=float, default=[2.0])
    ap.add_argument("--type0-cap", nargs="+", default=["auto"], help="г або auto (Р38)")
    ap.add_argument("--policies", nargs="+", default=["B0:0", "B3:0.2"], help="політика:параметр")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002])
    ap.add_argument("--n-printers", nargs="+", type=int, default=[15])
    ap.add_argument("--dt", nargs="+", type=float, default=[100.0], help="крок такту, с (перевірка чутливості, Р38)")
    ap.add_argument("--time-carry", nargs="+", type=int, default=[1], help="1 — такт без втрат на переходах (Р39), 0 — як до Р39")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--horizon-scale", type=float, default=1.0, help="лише для димового тесту (<1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    caps = [c if c == "auto" else float(c) for c in a.type0_cap]
    pols = [(p.split(":")[0], float(p.split(":")[1])) for p in a.policies]
    jobs = [(d, k, prm, s, b, c, a.horizon_scale, npr, dt, bool(tc))
            for (k, prm), tc, npr, dt, d, b, c, s in itertools.product(pols, a.time_carry, a.n_printers, a.dt,
                                                                        a.dispatch, a.buffer, caps, a.seeds)]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows, writer = [], None
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs), 1):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(r)); writer.writeheader()
            writer.writerow(r); f.flush(); rows.append(r)
            print(f"[{i}/{len(jobs)}] tc={int(r['time_carry'])} n_pr={r['n_printers']} dt={r['dt']:.0f} {r['dispatch']} buf={r['buffer_drones']} cap0={r['type0_cap']} {r['policy']} "
                  f"seed={r['seed']}: {r['throughput_per_week']:.2f} дрон/тиж; принтери C2: робота "
                  f"{r['C2_work']:.3f}, голод {r['C2_starved']:.3f}, блок {r['C2_blocked']:.3f}; "
                  f"екструзія C1: робота {r['C1_work']:.3f}, блок {r['C1_blocked']:.3f} ({r['wall_s']} с)", flush=True)
    # підсумок за конфігураціями
    print("\nСередні за seed (дрон/тиж; частки часу принтерів C2 і екструдерів C1):")
    keys = sorted({(r["policy"], r["n_printers"], r["dt"], r["dispatch"], r["buffer_drones"], str(r["type0_cap"]), int(r["time_carry"])) for r in rows})
    for k in keys:
        g = [r for r in rows if (r["policy"], r["n_printers"], r["dt"], r["dispatch"], r["buffer_drones"], str(r["type0_cap"]), int(r["time_carry"])) == k]
        mean = lambda f: sum(r[f] for r in g) / len(g)
        print(f"  {k[0]} tc={k[6]} n_pr={k[1]:2d} dt={k[2]:4.0f} {k[3]:5s} buf={k[4]:.0f} cap0={k[5]:>6s}: {mean('throughput_per_week'):6.2f} дрон/тиж, "
              f"прибуток {mean('profit')/1e3:7.1f} тис. | "
              f"C2 роб {mean('C2_work'):.3f} налаг {mean('C2_setup'):.3f} ТО {mean('C2_pm'):.3f} рем {mean('C2_cm'):.3f} "
              f"голод {mean('C2_starved'):.3f} блок {mean('C2_blocked'):.3f} | "
              f"C1 роб {mean('C1_work'):.3f} голод {mean('C1_starved'):.3f} блок {mean('C1_blocked'):.3f}")
    base = [r for r in rows if r["policy"] == "B3" and r["buffer_drones"] == 2.0 and r["type0_cap"] == "auto"
            and r["n_printers"] == 15 and r["dt"] == 100.0 and r["time_carry"]]
    analytic_bound(base or rows)


if __name__ == "__main__":
    main()
