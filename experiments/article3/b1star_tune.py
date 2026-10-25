"""B1* — найсильніший класичний бейслайн: періодичне ТО з окремим множником m_c
для кожного класу обладнання (Р25 + Р32), налаштування покоординатним спуском
за прибутком на навчальних seed.

Процедура: старт — найкращий спільний множник (m = 0.5 для всіх класів,
a3-tune-econ-L). Далі по черзі для кожного класу (C2, C1, C3, C4, C6) перебирається
сітка m_c ∈ {0.25, 0.35, 0.5, 0.75, 1, 1.5, ∞} (∞ — без ТО для цього класу) при
фіксованих множниках інших класів; обирається значення з найбільшим середнім
прибутком по seed. Один прохід. Усі оцінені конфігурації пишуться в CSV
(однакові конфігурації не переоцінюються).

Запуск: PYTHONPATH=src python experiments/article3/b1star_tune.py --workers 5 --out experiments/article3/raw/b1star_econ_L.csv
(v0.13, Р42: --n-printers 12 15 17 19 --order C2 C1 --grid 0.35 0.5 0.75 1.0)
"""
import argparse, csv, json, math, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from LineV3 import run_v3, PeriodicPM
from line_config_v3 import scenario_classes
from analytic_v3 import class_T_star_seconds

WARMUP_H, WINDOW_H = 26 * 7 * 24, 8760
INF = "inf"


def job(args):
    scenario, mults, seed, crew, horizon_scale, n_pr, clock = args
    cl, ov = scenario_classes(scenario)
    Tstar = class_T_star_seconds(cl)
    T = {c: float(mults[c]) * Tstar[c] for c in Tstar if mults.get(c, INF) != INF}
    t0 = time.time()
    m = run_v3(seed, PeriodicPM(T, clock=clock), horizon_h=(WARMUP_H + WINDOW_H) * horizon_scale,
               warmup_h=WARMUP_H * horizon_scale, classes=cl, node_overrides=ov, crew=crew,
               n_printers=n_pr)
    m.update({"scenario": scenario, "policy": "B1star" if clock == "op" else "B1astar", "mults": json.dumps(mults, sort_keys=True),
              "seed": seed, "n_printers": n_pr, "wall_s": round(time.time() - t0, 1)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="moderate")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--grid", nargs="+", default=["0.25", "0.35", "0.5", "0.75", "1.0", "1.5", INF])
    ap.add_argument("--order", nargs="+", default=["C2", "C1", "C3", "C4", "C6"])
    ap.add_argument("--start", type=float, default=0.5)
    ap.add_argument("--crew", type=int, default=1)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--horizon-scale", type=float, default=1.0, help="лише для димового тесту (<1)")
    ap.add_argument("--n-printers", nargs="+", type=int, default=[15])
    ap.add_argument("--clock", default="op", choices=["op", "age"],
                    help="op — напрацювання з ТО (B1*); age — віртуальний вік зносової складової (B1a*, Р52)")
    ap.add_argument("--resume", action="store_true",
                    help="дочитати вже пораховані конфігурації з --out (повні за seed) і дописувати нові рядки")
    ap.add_argument("--summary", default=None, help="JSON з підсумковими множниками B1* (Р50: для конвеєра)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    summary = {}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prev = {}                                        # (n_printers, ключ конфігурації) -> [прибутки]
    writer = None
    if a.resume and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        with open(a.out, newline="") as fr:
            rows = list(csv.DictReader(fr))
        for r in rows:
            if r["scenario"] == a.scenario and int(r["seed"]) in a.seeds and \
                    r.get("policy", "B1star") == ("B1star" if a.clock == "op" else "B1astar"):
                prev.setdefault((int(r["n_printers"]), r["mults"]), {})[int(r["seed"])] = float(r["profit"])
        complete = {k for k, v in prev.items() if len(v) == len(a.seeds)}
        with open(a.out, "w", newline="") as fw:   # лишаємо лише повні конфігурації
            keep = [r for r in rows if (int(r["n_printers"]), r["mults"]) in complete]
            if rows:
                w0 = csv.DictWriter(fw, fieldnames=list(rows[0].keys())); w0.writeheader(); w0.writerows(keep)
        prev = {k: list(v.values()) for k, v in prev.items() if k in complete}
        print(f"дозапуск: {len(prev)} повних конфігурацій з {a.out}", flush=True)
        f = open(a.out, "a", newline="")
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
    else:
        f = open(a.out, "w", newline="")
    pool = Pool(a.workers)
    for n_pr in a.n_printers:
        cur = {c: a.start for c in a.order}
        cache = {k[1]: sum(v) / len(v) for k, v in prev.items() if k[0] == n_pr}   # ключ конфігурації -> середній прибуток
        print(f"=== n_printers = {n_pr}", flush=True)
        for cls in a.order:
            cands = []
            for g in a.grid:
                mults = dict(cur)
                mults[cls] = g if g == INF else float(g)
                cands.append(mults)
            todo = [m for m in cands if json.dumps(m, sort_keys=True) not in cache]
            jobs = [(a.scenario, m, s, a.crew, a.horizon_scale, n_pr, a.clock) for m in todo for s in a.seeds]
            acc = {}
            for r in pool.imap_unordered(job, jobs):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(r)); writer.writeheader()
                writer.writerow(r); f.flush()
                acc.setdefault(r["mults"], []).append(r["profit"])
            for k, v in acc.items():
                cache[k] = sum(v) / len(v)
            scores = [(cache[json.dumps(m, sort_keys=True)], m) for m in cands]
            best_val, best = max(scores, key=lambda x: x[0])
            print(f"{cls}: " + ", ".join(f"{m[cls]}→{v/1e3:.1f}" for v, m in scores)
                  + f"  | обрано {cls}={best[cls]} (прибуток {best_val/1e3:.1f} тис./рік)", flush=True)
            cur = best
        print(f"B1* підсумок (n_printers = {n_pr}):", json.dumps(cur, sort_keys=True), flush=True)
        summary[n_pr] = {"mults": cur, "profit_k": max(cache.values()) / 1e3}
    pool.close()
    f.close()
    if a.summary:
        with open(a.summary, "w") as fs:
            json.dump(summary, fs, indent=1)


if __name__ == "__main__":
    main()
