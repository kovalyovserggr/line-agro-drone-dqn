"""Вибір горизонту і розігріву для оцінювання на лінії (Р19, Фаза 3).

Для політики B0 (без ТО) проганяє лінію v0.3 на довгому горизонті і пише
тижневі кількості готових дронів. Аналіз: (а) розігрів — за методом Велча
(ковзне середнє тижневого випуску по репліках); (б) горизонт — напівширина
95% довірчого інтервалу середнього тижневого випуску (метод групових середніх)
як функція довжини вікна.

Запуск: PYTHONPATH=src python experiments/article3/horizon_study.py --seeds 1 2 3 4 --years 3 --out experiments/article3/raw/horizon_B0.csv
"""
import argparse, contextlib, csv, io, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from LineV3 import AgroDroneLineV3, NoPM, DT

WEEK = 7 * 24 * 3600.0

def weekly_series(seed, years):
    line = AgroDroneLineV3(seed, NoPM())
    n_ticks = int(years * 365 * 24 * 3600 / DT)
    t = 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_ticks):
            t += DT
            line.tick(t)
    n_weeks = int(t // WEEK)
    counts = [0] * n_weeks
    for td in line.drone_times:
        w = int(td // WEEK)
        if w < n_weeks:
            counts[w] += 1
    return counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--years", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["seed", "week", "drones"])
        for s in a.seeds:
            t0 = time.time()
            c = weekly_series(s, a.years)
            for i, x in enumerate(c): w.writerow([s, i, x])
            f.flush()
            print(f"seed={s}: {len(c)} тижнів, середнє {sum(c)/len(c):.2f} дронів/тиж ({time.time()-t0:.0f} с)", flush=True)

if __name__ == "__main__":
    main()
