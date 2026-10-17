"""Параметричне дослідження T_pm парку 3D-друку (стаття 1, розділ 4.2).

Сітка T_pm x N реплікацій; для кожної репліки: seed, статус завершення,
makespan (elapsed_time на момент SimulationComplete), готовність парку друку
(частка часу поза MAINTENANCE/REPAIR/FAILURE серед вузлів 2-16).

Без TimeManager/sleep — тік-цикл driving AgroDroneLine напряму (див.
docs/DECISIONS.md D1, D4). stdout вузлів придушується (дуже багатослівний),
за винятком короткого прогрес-логу цього скрипта.

Використання:
    PYTHONPATH=src python experiments/run_parametric_study.py \
        --t-pm 6000 24000 --reps 3 --max-ticks 50000 --out experiments/raw/pilot.csv
"""
import argparse
import contextlib
import csv
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from AgroDroneLine import AgroDroneLine, SimulationComplete  # noqa: E402

DT = 100.0  # крок модельного часу на тік (100с/тік, як у TimeManager acceleration=10^4, 100Hz)


def availability_print_park(line: AgroDroneLine) -> float:
    """Частка часу поза MAINTENANCE/REPAIR/FAILURE серед вузлів 2-16 (парк друку)."""
    from EnumsLine import NodeState
    down_states = (NodeState.MAINTENANCE, NodeState.REPAIR, NodeState.FAILURE)
    total, down = 0.0, 0.0
    for n in line.units:
        if 2 <= n.id <= 16:
            node_total = sum(n.state_durations.values())
            node_down = sum(v for k, v in n.state_durations.items() if k in down_states)
            total += node_total
            down += node_down
    return 1.0 - (down / total if total > 0 else 0.0)


def run_one(seed: int, t_pm: float, lim_drones_made: int, max_ticks: int,
            n_filament_suppliers: int = 1):
    line = AgroDroneLine(seed=seed, lim_drones_made=lim_drones_made,
                          T_pm_print_park=t_pm, verbose=False,
                          n_filament_suppliers=n_filament_suppliers)
    elapsed = 0.0
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            for _ in range(max_ticks):
                elapsed += DT
                line.tick(elapsed)
    except SimulationComplete as sc:
        return {
            "seed": seed, "T_pm": t_pm, "status": "completed",
            "makespan_s": sc.elapsed_time, "drones_made": sc.number_drones_made,
            "availability_print_park": round(availability_print_park(line), 4),
        }
    return {
        "seed": seed, "T_pm": t_pm, "status": "timeout_at_max_ticks",
        "makespan_s": elapsed, "drones_made": line.number_drones_made,
        "availability_print_park": round(availability_print_park(line), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-pm", nargs="+", type=float, required=True,
                     help="сітка T_pm, с (напр. 1000 2000 4000 6000 8000 12000 16000 24000)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--lim-drones", type=int, default=10)
    ap.add_argument("--max-ticks", type=int, default=50000,
                     help="запобіжний ліміт тіків на репліку (50000 тіків = 5*10^6 с модельного часу)")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--n-filament-suppliers", type=int, default=1,
                     help="кількість паралельних постачальників Type0 (D9/D10); "
                          "1 = поведінка до D9 (структурна стеля вузла 1)")
    ap.add_argument("--out", type=str, default="experiments/raw/results.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows = []
    t0 = time.time()
    run_idx = 0
    total_runs = len(args.t_pm) * args.reps
    for t_pm in args.t_pm:
        for rep in range(args.reps):
            seed = args.seed_base + rep
            run_idx += 1
            t_run0 = time.time()
            row = run_one(seed, t_pm, args.lim_drones, args.max_ticks, args.n_filament_suppliers)
            row["n_filament_suppliers"] = args.n_filament_suppliers
            row["rep"] = rep
            dt_run = time.time() - t_run0
            rows.append(row)
            print(f"[{run_idx}/{total_runs}] T_pm={t_pm:>7.0f} seed={seed} "
                  f"-> {row['status']:<20} makespan={row['makespan_s']:.0f}s "
                  f"avail={row['availability_print_park']:.3f} ({dt_run:.1f}s реального часу)",
                  flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["T_pm", "rep", "seed", "status",
                                                "makespan_s", "drones_made",
                                                "availability_print_park",
                                                "n_filament_suppliers"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nГотово: {len(rows)} реплік за {time.time()-t0:.1f}с реального часу -> {args.out}")


if __name__ == "__main__":
    main()
