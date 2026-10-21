"""Оцінювання збережених агентів DQN-TA на довільному наборі seed — пулом процесів (Р48, Р49).

Кожна задача = (каталог агента, seed): модель <dir>/model.pt, сценарій, кількість
принтерів і сітка рішень — з <dir>/config.json (параметри навчання). Політика жадібна,
протокол бейслайнів (розігрів 26 тижнів, вікно 1 рік). На відміну від
train_line_dqn.py --eval-only, задачі паралеляться за seed, а не лише за агентами, і
пишуться в один CSV (з колонками agent_dir, param = seed агента, ttf).

--ttf lognormal — перевірка робастності (Р49): «справжній» розподіл часу до відмови
зносових складових логнормальний (ті самі середнє і CV); агент не перенавчається.

Запуск (L):
  python experiments/article3/eval_dqn.py --dirs experiments/article3/raw/dqn_ta_np15_s1 ... \
      --seeds-range 100005 100020 --workers 5 --out experiments/article3/raw/test20_dqn_L.csv
"""
import argparse, contextlib, csv, io, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

WARMUP_H, WINDOW_H = 26 * 7 * 24, 8760


def job(a):
    d, seed, ttf, scen = a
    import torch
    torch.set_num_threads(1)
    from LineV3 import run_v3, DT
    from line_config_v3 import scenario_classes
    from DQNAgentV3 import DoubleDQNAgentV3
    from DQNLineV3 import DQNPMPolicy, OBS_DIM
    cfg = json.load(open(os.path.join(d, "config.json")))
    scenario = scen or cfg["scenario"]           # Р50: заморожений агент в іншому середовищі
    cl, ov = scenario_classes(scenario, ttf=ttf)
    agent = DoubleDQNAgentV3(OBS_DIM, 2, seed=cfg["agent_seed"], lr=cfg.get("lr", 3e-4), lr_end=cfg.get("lr_end", 3e-5),
                             eps_steps=10**9, total_steps=10**9, buffer=1000, warmup=cfg.get("warmup", 1000))
    agent.online.load_state_dict(torch.load(os.path.join(d, "model.pt")))
    pol = DQNPMPolicy(agent, train=False, grid=cfg.get("grid", 0.0))
    t0 = time.time()
    m = run_v3(seed, pol, horizon_h=WARMUP_H + WINDOW_H, warmup_h=WARMUP_H, classes=cl, node_overrides=ov,
               crew=1, n_printers=cfg["n_printers"], dt=cfg.get("dt", DT))
    m.update({"scenario": scenario, "trained_on": cfg["scenario"], "policy": "DQN", "param": cfg["agent_seed"], "seed": seed,
              "n_printers": cfg["n_printers"], "ttf": ttf, "agent_dir": os.path.basename(os.path.normpath(d)),
              "wall_s": round(time.time() - t0, 1)})
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="каталоги агентів (з model.pt і config.json)")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--seeds-range", nargs=2, type=int, default=None, metavar=("FROM", "TO"),
                    help="seed від FROM до TO−1 (замість --seeds)")
    ap.add_argument("--ttf", default="weibull", choices=["weibull", "lognormal"])
    ap.add_argument("--scenario", default=None,
                    help="середовище оцінювання замість сценарію навчання (Р50: заморожений агент, напр. moderate+C4eta2500)")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = a.seeds if a.seeds else list(range(*a.seeds_range))
    for d in a.dirs:
        assert os.path.exists(os.path.join(d, "model.pt")), f"немає моделі в {d}"
    jobs = [(d, s, a.ttf, a.scenario) for d in a.dirs for s in seeds]
    writer, done = None, 0
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        for m in pool.imap_unordered(job, jobs):
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(m)); writer.writeheader()
            writer.writerow(m); f.flush(); done += 1
            print(f"[{done}/{len(jobs)}] {m['agent_dir']} ttf={m['ttf']} seed={m['seed']}: "
                  f"{m['throughput_per_week']:.2f} дрон/тиж, прибуток {m['profit']/1e3:.1f} тис. ({m['wall_s']} с)", flush=True)


if __name__ == "__main__":
    main()
