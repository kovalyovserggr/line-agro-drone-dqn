"""Тестове оцінювання фіналістів-бейслайнів (Р42) на відкладених seed 100000–100004.

Фіналісти (параметри — з налаштування на навчальних seed 1000–1004):
  B0      — без ТО;
  B1      — періодичне ТО, m = 0.5 (a3-tune-econ-v13-L);
  B2      — поріг ризику θ = 1e-3;
  B3+     — опортуністичне ТО x = 0.3 поверх періодичного m = 0.5 (a3-b3plus-v13-L);
  B1*     — покласові множники C1/C2 (a3-b1star-v13-L), окремо для кожної кількості принтерів.
Ці seed не використовувались ні для налаштування бейслайнів, ні для навчання DQN.

Запуск (L): PYTHONPATH=src python experiments/article3/eval_finalists.py --workers 3 --out experiments/article3/raw/test_finalists_L.csv
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from LineV3 import run_v3, PeriodicPM
from line_config_v3 import scenario_classes
from analytic_v3 import class_T_star_seconds
from baselines_tune import make_policy, WARMUP_H, WINDOW_H

# B1* за a3-b1star-v13-L (класи без запису — без ТО; C3/C4/C6 ТО не спрацьовує, Р35)
B1STAR = {12: {"C1": 1.0, "C2": 0.5}, 15: {"C1": 1.0, "C2": 0.5},
          17: {"C1": 0.35, "C2": 0.5}, 19: {"C1": 0.35, "C2": 0.75}}
FINALISTS = [("B0", 0.0, None), ("B1", 0.5, None), ("B2", 1e-3, None), ("B3+", 0.3, 0.5), ("B1*", None, None)]


def job(a):
    name, param, b3m, n_pr, seed = a
    cl, ov = scenario_classes("moderate")
    T = class_T_star_seconds(cl)
    if name == "B1*":
        pol = PeriodicPM({c: m * T[c] for c, m in B1STAR[n_pr].items()})
        param = json.dumps(B1STAR[n_pr], sort_keys=True)
    elif name == "B3+":
        pol = make_policy("B3", param, T, b3m)
    else:
        pol = make_policy(name, param, T)
    t0 = time.time()
    m = run_v3(seed, pol, horizon_h=WARMUP_H + WINDOW_H, warmup_h=WARMUP_H, classes=cl, node_overrides=ov,
               crew=1, n_printers=n_pr)
    m.update({"policy": name, "param": param, "b3_m": b3m if b3m else "", "n_printers": n_pr, "seed": seed,
              "wall_s": round(time.time() - t0, 1)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-printers", nargs="+", type=int, default=[12, 15, 17, 19])
    ap.add_argument("--seeds", nargs="+", type=int, default=[100000, 100001, 100002, 100003, 100004])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = [(n, p, b, npr, s) for npr in a.n_printers for (n, p, b) in FINALISTS for s in a.seeds]
    writer, done = None, 0
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        for m in pool.imap_unordered(job, jobs):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(m)); writer.writeheader()
            writer.writerow(m); f.flush(); done += 1
            print(f"[{done}/{len(jobs)}] n_pr={m['n_printers']} {m['policy']} seed={m['seed']}: "
                  f"{m['throughput_per_week']:.2f} дрон/тиж, прибуток {m['profit']/1e3:.1f} тис. ({m['wall_s']} с)",
                  flush=True)


if __name__ == "__main__":
    main()
