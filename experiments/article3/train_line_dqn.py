"""Навчання і оцінювання DQN-політики ТО на всій лінії (Стаття 3, Р41).

Навчання: епізоди по --ep-hours модельного часу на навчальних seed середовища
(0…9999, вибираються RNG агента — не перетинаються з тестовими 100000+ і з
seed-ами налаштування бейслайнів 1000–1004 не обов'язково збігаються).
Кожен епізод — нова лінія (вік вузлів 0). ε і lr згасають лінійно; графік
задається після першого епізоду за фактичною кількістю рішень.

Оцінювання: жадібна політика, протокол бейслайнів (розігрів 26 тижнів, вікно
1 рік, Р26) на --eval-seeds; CSV з тими самими метриками, що й baselines_tune.

Запуск (L, 3 агенти паралельно; виходи raw/dqn_np15_s1, _s2, _s3):
  PYTHONPATH=src python experiments/article3/train_line_dqn.py --n-printers 15 --agent-seeds 1 2 3 \
      --episodes 30 --out experiments/article3/raw/dqn_np15
"""
import argparse, contextlib, csv, io, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
import torch
from LineV3 import AgroDroneLineV3, run_v3, _snapshot, line_metrics, DT
from line_config_v3 import scenario_classes
from DQNAgentV3 import DoubleDQNAgentV3
from DQNLineV3 import DQNPMPolicy, OBS_DIM

WARMUP_H, WINDOW_H = 26 * 7 * 24, 8760


def run_episode(seed, policy, hours, cl, ov, n_printers, dt):
    with contextlib.redirect_stdout(io.StringIO()):
        line = AgroDroneLineV3(seed, policy, classes=cl, node_overrides=ov, crew=1, n_printers=n_printers)
        s0 = _snapshot(line)
        t = 0.0
        for _ in range(int(round(hours * 3600 / dt))):
            t += dt
            line.tick(t)
    return line_metrics(s0, _snapshot(line), line), line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-seeds", nargs="+", type=int, default=None,
                    help="кілька seed агента паралельно (окремі процеси); вихід — <out>_s<seed>")
    ap.add_argument("--scenario", default="moderate")
    ap.add_argument("--n-printers", type=int, default=15)
    ap.add_argument("--agent-seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--ep-hours", type=float, default=4368.0, help="тривалість навчального епізоду (26 тижнів)")
    ap.add_argument("--tau-h", type=float, default=200.0, help="стала дисконтування SMDP, год")
    ap.add_argument("--dt", type=float, default=DT)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-end", type=float, default=3e-5)
    ap.add_argument("--eps-frac", type=float, default=0.4, help="частка навчання, за яку ε згасає до 0.05")
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--eval-seeds", nargs="+", type=int, default=[100000, 100001, 100002, 100003, 100004])
    ap.add_argument("--eval-only", default=None, help="шлях до model.pt — лише оцінювання")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.agent_seeds:
        import copy
        from multiprocessing import Pool
        jobs = []
        for sd in a.agent_seeds:
            b = copy.copy(a); b.agent_seed, b.agent_seeds, b.out = sd, None, f"{a.out}_s{sd}"
            jobs.append(b)
        with Pool(len(jobs)) as p:
            p.map(train_and_eval, jobs)
        return
    train_and_eval(a)


def train_and_eval(a):
    torch.set_num_threads(1)
    os.makedirs(a.out, exist_ok=True)
    cl, ov = scenario_classes(a.scenario)
    agent = DoubleDQNAgentV3(OBS_DIM, 2, seed=a.agent_seed, lr=a.lr, lr_end=a.lr_end,
                             eps_steps=10**9, total_steps=10**9, buffer=200_000, warmup=5_000)
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(vars(a), f, indent=1)

    if a.eval_only is None:
        env_rng = np.random.default_rng([a.agent_seed, 4242])
        log = open(os.path.join(a.out, "train_log.csv"), "w", newline="")
        w = None
        pol = DQNPMPolicy(agent, tau_h=a.tau_h, train=True, train_every=a.train_every)
        # початкова оцінка ρ_c — калібрувальний епізод без ТО (B0), щоб уже перші
        # переходи в буфері мали правильний зсув
        from LineV3 import NoPM
        _, cal = run_episode(int(env_rng.integers(0, 10000)), NoPM(), a.ep_hours / 2, cl, ov, a.n_printers, a.dt)
        pol.update_rho(cal, a.ep_hours / 2 * 3600, w=1.0)
        print("ρ_c (USD/год на вузол):", {k: round(v * 3600, 2) for k, v in sorted(pol.rho.items())}, flush=True)
        for ep in range(a.episodes):
            seed = int(env_rng.integers(0, 10000))
            pol.pending, pol.losses, pol.n_pm = {}, [], 0
            # ε — лінійно за епізодами: 1 → 0.05 за перші eps_frac·episodes епізодів, далі 0.05
            agent.eps_fixed = max(0.05, 1.0 - 0.95 * ep / max(1.0, a.eps_frac * a.episodes))
            d0, t0 = pol.n_dec, time.time()
            m, line = run_episode(seed, pol, a.ep_hours, cl, ov, a.n_printers, a.dt)
            # ρ_c: за епізодом із B3-подібною поведінкою було б краще, але досить і
            # згладженої оцінки за власними епізодами (зсув не змінює оптимальну політику)
            pol.update_rho(line, a.ep_hours * 3600)
            n_ep = pol.n_dec - d0
            # lr — лінійно до lr_end за оцінкою загальної кількості кроків (оновлюється щоепізоду)
            agent.total_steps = max(agent.steps, int(pol.n_dec / (ep + 1) * a.episodes))
            row = {"episode": ep, "env_seed": seed, "decisions": n_ep, "pm": pol.n_pm,
                   "pm_per_1k_dec": 1000 * pol.n_pm / max(1, n_ep), "eps": round(agent.eps, 4),
                   "loss": float(np.mean(pol.losses)) if pol.losses else float("nan"),
                   "profit_k": m["profit"] / 1e3, "throughput": m["throughput_per_week"],
                   "OEE": m["OEE"], "failures": m["failures"],
                   "rho_C2_per_h": pol.rho.get("C2", 0.0) * 3600, "wall_s": round(time.time() - t0, 1)}
            if w is None:
                w = csv.DictWriter(log, fieldnames=list(row)); w.writeheader()
            w.writerow(row); log.flush()
            print(f"ep {ep:3d} seed {seed:4d}: рішень {n_ep}, ТО {pol.n_pm}, ε {agent.eps:.3f}, "
                  f"loss {row['loss']:.4f}, прибуток {row['profit_k']:.1f} тис., {row['throughput']:.2f} дрон/тиж "
                  f"({row['wall_s']} с)", flush=True)
            torch.save(agent.online.state_dict(), os.path.join(a.out, "model.pt"))
        log.close()
    else:
        agent.online.load_state_dict(torch.load(a.eval_only))

    # оцінювання: жадібна політика, протокол бейслайнів
    pol = DQNPMPolicy(agent, train=False)
    rows = []
    for s in a.eval_seeds:
        t0 = time.time()
        pol.n_pm = 0
        m = run_v3(s, pol, horizon_h=WARMUP_H + WINDOW_H, warmup_h=WARMUP_H, classes=cl, node_overrides=ov,
                   crew=1, n_printers=a.n_printers, dt=a.dt)
        m.update({"scenario": a.scenario, "policy": "DQN", "param": a.agent_seed, "seed": s,
                  "n_printers": a.n_printers, "wall_s": round(time.time() - t0, 1)})
        rows.append(m)
        print(f"оцінка seed {s}: {m['throughput_per_week']:.2f} дрон/тиж, прибуток {m['profit']/1e3:.1f} тис., "
              f"ТО {m['pm']}, відмов {m['failures']}", flush=True)
    with open(os.path.join(a.out, "eval.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"середній прибуток {np.mean([r['profit'] for r in rows])/1e3:.1f} тис. USD/рік")


if __name__ == "__main__":
    main()
