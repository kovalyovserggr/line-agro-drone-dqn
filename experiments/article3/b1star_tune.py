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
    scenario, mults, seed, crew, horizon_scale, n_pr = args
    cl, ov = scenario_classes(scenario)
    Tstar = class_T_star_seconds(cl)
    T = {c: float(mults[c]) * Tstar[c] for c in Tstar if mults.get(c, INF) != INF}
    t0 = time.time()
    m = run_v3(seed, PeriodicPM(T), horizon_h=(WARMUP_H + WINDOW_H) * horizon_scale,
               warmup_h=WARMUP_H * horizon_scale, classes=cl, node_overrides=ov, crew=crew,
               n_printers=n_pr)
    m.update({"scenario": scenario, "policy": "B1star", "mults": json.dumps(mults, sort_keys=True),
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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    f = open(a.out, "w", newline="")
    writer = None
    pool = Pool(a.workers)
    for n_pr in a.n_printers:
        cur = {c: a.start for c in a.order}
        cache = {}                    # ключ конфігурації -> середній прибуток
        print(f"=== n_printers = {n_pr}", flush=True)
        for cls in a.order:
            cands = []
            for g in a.grid:
                mults = dict(cur)
                mults[cls] = g if g == INF else float(g)
                cands.append(mults)
            todo = [m for m in cands if json.dumps(m, sort_keys=True) not in cache]
            jobs = [(a.scenario, m, s, a.crew, a.horizon_scale, n_pr) for m in todo for s in a.seeds]
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
    pool.close()
    f.close()


if __name__ == "__main__":
    main()
