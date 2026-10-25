"""Налаштування бейслайнів B1–B3 на навчальних seed (Фаза 3, Р25, Р26, Р28).

Кожна конфігурація оцінюється на однакових seed (спільні випадкові числа, Р6):
розігрів 26 тижнів, вікно 1 рік (Р26). Результат — CSV з усіма метриками
по (політика, параметр, seed). Вибір найкращого параметра — за середньою
пропускною здатністю на навчальних seed; перевірка — на тестових (окремий запуск).

Запуск: PYTHONPATH=src python experiments/article3/baselines_tune.py --scenario moderate --seeds 1000 1001 1002 1003 1004 --out experiments/article3/raw/tune_moderate.csv --workers 2
"""
import argparse, csv, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from LineV3 import run_v3, NoPM, PeriodicPM, RiskPM, CONTINUE, PM, E1, E2, E3
from line_config_v3 import scenario_classes
from analytic_v3 import class_T_star_seconds

WARMUP_H, WINDOW_H = 26 * 7 * 24, 8760


class B3Opportunistic:
    """B3: під час простою (E2/E3) — ТО, якщо зношеність > x; на межі операцій (E1) — як B1 з m=1."""
    def __init__(self, x, T_by_class):
        self.x, self.b1 = x, PeriodicPM(T_by_class)
        self.name = "B3_opportunistic"
    def decide(self, node, epoch, d_next):
        if epoch in (E2, E3):
            return PM if node.wear_ratio() > self.x else CONTINUE
        return self.b1.decide(node, epoch, d_next)


def make_policy(kind, param, Tstar, b3_m=1.0):
    if kind == "B0":
        return NoPM()
    if kind == "B1":
        return PeriodicPM({c: param * T for c, T in Tstar.items()})
    if kind == "B1a":                      # Р52: за віртуальним віком зносової складової
        return PeriodicPM({c: param * T for c, T in Tstar.items()}, clock="age")
    if kind == "B2":
        return RiskPM({c: param for c in Tstar})
    if kind == "B3":
        return B3Opportunistic(param, {c: b3_m * T for c, T in Tstar.items()})
    raise ValueError(kind)


def job(args):
    scenario, kind, param, seed, crew, dispatch, cap0, n_pr, tc, b3_m, ttf = args
    cl, ov = scenario_classes(scenario, ttf=ttf)
    Tstar = class_T_star_seconds(cl)
    t0 = time.time()
    m = run_v3(seed, make_policy(kind, param, Tstar, b3_m), horizon_h=WARMUP_H + WINDOW_H, warmup_h=WARMUP_H,
               classes=cl, node_overrides=ov, crew=crew, dispatch=dispatch, type0_cap=cap0, n_printers=n_pr, time_carry=tc)
    m.update({"scenario": scenario, "ttf": ttf, "policy": kind, "param": param, "seed": seed, "dispatch": dispatch, "type0_cap": cap0, "n_printers": n_pr, "time_carry": tc, "b3_m": b3_m if kind == "B3" else "",
              "wall_s": round(time.time() - t0, 1)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="moderate")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--b1", nargs="+", type=float, default=[0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--b2", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3])
    ap.add_argument("--b3", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--crew", nargs="+", default=["inf"], help="кількість техніків (Р30); inf — необмежено")
    ap.add_argument("--policies", nargs="+", default=["B0", "B1", "B2", "B3"])
    ap.add_argument("--dispatch", nargs="+", default=["D17"], help="правила диспетчеризації: D17 list quota random")
    ap.add_argument("--type0-cap", default="auto", help="кеп філаменту: auto — на buffer_drones дронів (Р38); або число, г (4000 — серії до a3-v0.10)")
    ap.add_argument("--time-carry", type=int, default=1, help="1 — такт без втрат на переходах (Р39), 0 — як до Р39")
    ap.add_argument("--b3-m", type=float, default=1.0, help="B3: множник періодичного ТО на E1 (1.0 — аналітичний T*)")
    ap.add_argument("--n-printers", nargs="+", type=int, default=[15], help="кількість принтерів (вісь сценаріїв, Р38)")
    ap.add_argument("--ttf", default="weibull", choices=["weibull", "lognormal"],
                    help="«справжній» розподіл часу до відмови зносових складових (Р49)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    crews = [None if c == "inf" else int(c) for c in a.crew]
    grid = {"B0": [0.0], "B1": a.b1, "B1a": a.b1, "B2": a.b2, "B3": a.b3}
    cap0 = a.type0_cap if a.type0_cap == "auto" else float(a.type0_cap)
    jobs = [(a.scenario, k, p, s, c, dsp, cap0, npr, bool(a.time_carry), a.b3_m, a.ttf) for npr in a.n_printers for dsp in a.dispatch for c in crews
            for k in a.policies for p in grid[k] for s in a.seeds]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    writer, f, done = None, open(a.out, "w", newline=""), 0
    with Pool(a.workers) as pool:
        for m in pool.imap_unordered(job, jobs):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(m)); writer.writeheader()
            writer.writerow(m); f.flush(); done += 1
            print(f"[{done}/{len(jobs)}] n_pr={m['n_printers']} {m['dispatch']} crew={m['crew']} {m['policy']} {m['param']} seed={m['seed']}: "
                  f"{m['throughput_per_week']:.2f} дронів/тиж, OEE {m['OEE']:.4f} ({m['wall_s']} с)", flush=True)
    f.close()

if __name__ == "__main__":
    main()
