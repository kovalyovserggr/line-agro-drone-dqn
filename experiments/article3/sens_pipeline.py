"""Конвеєр чутливості до параметра без джерела (Р50): повне переналаштування всіх політик
для одного варіанта середовища, з правилами вибору, зафіксованими заздалегідь.

Для сценарію S (напр. "moderate+C2eta750") і n принтерів:
  1. Навчання 3 агентів DQN-TA (той самий протокол, що й a3-dqn-ta-*: 40 епізодів,
     τ = 300 год, TA s = 20, сітка 0.025) — у фоні, паралельно з кроками 2–3.
  2. Налаштування B0/B1/B2 на seed 1000–1004 → m* = argmax середнього прибутку B1,
     θ* = argmax B2.
  3. B1* — покоординатний спуск (C2, потім C1) від m*, сітка {0.25, 0.35, 0.5, 0.75, 1.0}.
  4. Валідація агентів на seed 200000–200002 → вибраний агент = argmax (Р47).
  5. Тест на seed 100000–100019: B0, B1 m*, B2 θ*, B1*, 3 агенти.
Вибір фіксується в <prefix>_selection.json ДО тесту. Кожен крок викликає наявні скрипти
(baselines_tune, b1star_tune, train_line_dqn, eval_dqn, eval_finalists), тож логіка
моделі й оцінювання — та сама, що в основних серіях.

Запуск (L):
  python experiments/article3/sens_pipeline.py --scenario moderate+C2eta750 --tag c2eta750 --workers 3
Димовий тест (C): --smoke (1 seed, укорочене навчання; результати НЕ для статті).
Дозапуск: повторний виклик з тими самими аргументами пропускає завершені кроки (навчання з
усіма епізодами, повні CSV) і дораховує відсутні конфігурації налаштування / спуску B1*.
Сітка B2: θ ∈ {1e-4, 3e-4, 1e-3, 3e-3, 1e-2} (ширша, ніж в основних серіях: оптимальний θ
залежить від η).
"""
import argparse, json, os, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
# Windows: stdout дочірнього процесу, перенаправлений у файл, має кодування ANSI (cp1251) —
# символи «→», «ρ» тощо валили b1star_tune і train_line_dqn (a3-sens-eta-L, перша спроба).
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(cmd, log):
    print("▶", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    print(f"   ✓ {time.time() - t0:.0f} с", flush=True)


def csv_rows(path):
    try:
        return len(pd.read_csv(path))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0


def trained(dirs, episodes):
    """Навчання завершене: у кожному каталозі є model.pt і train_log.csv з усіма епізодами."""
    return all(os.path.exists(os.path.join(d, "model.pt")) and csv_rows(os.path.join(d, "train_log.csv")) == episodes
               for d in dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--tag", required=True, help="коротка мітка для імен файлів, напр. c2eta750")
    ap.add_argument("--n-printers", type=int, default=15)
    ap.add_argument("--workers", type=int, default=3, help="процесів для бейслайнів (3 ядра — навчання DQN)")
    ap.add_argument("--workers-after", type=int, default=5, help="процесів для валідації і тесту (після навчання)")
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    S, n, W, raw = a.scenario, str(a.n_printers), str(a.workers), a.raw
    WA = str(a.workers_after)
    pre = os.path.join(raw, f"sens_{a.tag}")
    os.makedirs(raw, exist_ok=True)
    log = open(pre + "_pipeline.log", "a", encoding="utf-8")
    print(f"=== {S} [{a.tag}] {time.strftime('%Y-%m-%d %H:%M:%S')}", file=log, flush=True)
    tune_seeds = ["1000"] if a.smoke else ["1000", "1001", "1002", "1003", "1004"]
    val_seeds = ["200000"] if a.smoke else ["200000", "200001", "200002"]
    test_rng = ["100000", "100001"] if a.smoke else ["100000", "100020"]
    b1_grid = ["0.35", "0.5"] if a.smoke else ["0.25", "0.35", "0.5", "0.75", "1.0", "1.5"]
    # θ масштабується з η (ризик на операцію ∝ (d/η)^β): сітка ширша, ніж в основних серіях
    b2_grid = ["1e-3", "3e-3"] if a.smoke else ["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]
    episodes = 1 if a.smoke else 40
    star_grid = ["0.5", "1.0"] if a.smoke else ["0.25", "0.35", "0.5", "0.75", "1.0"]
    dqn_out = os.path.join(raw, f"dqn_ta_sens_{a.tag}")
    dirs = [f"{dqn_out}_s{i}" for i in (1, 2, 3)]

    # 1. навчання DQN-TA у фоні
    tr = [PY, os.path.join(HERE, "train_line_dqn.py"), "--scenario", S, "--n-printers", n, "--agent-seeds", "1", "2", "3",
          "--tau-h", "300", "--ta", "20", "--grid", "0.025", "--out", dqn_out,
          "--eval-seeds", "100000"]                 # eval.csv кроку навчання не використовується (тест — крок 5)
    tr += ["--episodes", "1", "--ep-hours", "300"] if a.smoke else ["--episodes", "40"]
    p_train = None
    if trained(dirs, episodes):
        print("   навчання вже завершене — пропускаю", flush=True)
    else:
        print("▶ (фон)", " ".join(tr), flush=True)
        train_log = open(pre + "_train.log", "w", encoding="utf-8")
        p_train = subprocess.Popen(tr, stdout=train_log, stderr=subprocess.STDOUT, env=ENV)

    # 2. B0 / B1 / B2 — лише відсутні конфігурації (дозапуск після переривання)
    tune = pre + "_tune_L.csv"
    need = {"B0": [0.0], "B1": [float(x) for x in b1_grid], "B2": [float(x) for x in b2_grid]}
    have = pd.read_csv(tune) if csv_rows(tune) else pd.DataFrame(columns=["policy", "param", "seed"])
    done = {(p, round(float(q), 12)) for (p, q), g in have.groupby(["policy", "param"]) if g.seed.nunique() == len(tune_seeds)}
    miss = {k: [x for x in v if (k, round(x, 12)) not in done] for k, v in need.items()}
    miss = {k: v for k, v in miss.items() if v}
    if miss:
        part = pre + "_tune_part.csv"
        cmd = [PY, os.path.join(HERE, "baselines_tune.py"), "--scenario", S, "--crew", "1", "--n-printers", n,
               "--policies", *miss, "--seeds", *tune_seeds, "--workers", W, "--out", part]
        if "B1" in miss: cmd += ["--b1", *map(str, miss["B1"])]
        if "B2" in miss: cmd += ["--b2", *map(str, miss["B2"])]
        run(cmd, log)
        keep = have[have.apply(lambda r: (r.policy, round(float(r.param), 12)) in done, axis=1)] if len(have) else have
        pd.concat([keep, pd.read_csv(part)], ignore_index=True).to_csv(tune, index=False)
        os.remove(part)
    else:
        print("   налаштування B0/B1/B2 вже є — пропускаю", flush=True)
    t = pd.read_csv(tune)
    g = t.groupby(["policy", "param"]).profit.mean()
    m_star = float(g["B1"].idxmax()); th_star = float(g["B2"].idxmax())
    print(f"   m* = {m_star}, θ* = {th_star}", flush=True)

    # 3. B1*
    star_sum = pre + "_b1star.json"
    run([PY, os.path.join(HERE, "b1star_tune.py"), "--scenario", S, "--n-printers", n, "--order", "C2", "C1",
         "--grid", *star_grid, "--start", str(m_star), "--seeds", *tune_seeds, "--workers", W,
         "--out", pre + "_b1star_L.csv", "--summary", star_sum, "--resume"], log)
    mults = json.load(open(star_sum))[n]["mults"]
    b1star = {c: float(v) for c, v in mults.items() if v != "inf"}
    print(f"   B1* = {b1star}", flush=True)

    # 4. дочекатися навчання і валідувати
    if p_train is not None:
        rc = p_train.wait()
        if rc != 0:
            raise SystemExit(f"навчання DQN завершилось з кодом {rc}; див. {pre}_train.log")
    if not trained(dirs, episodes):
        raise SystemExit(f"навчання неповне в {dirs}")
    val = pre + "_val_L.csv"
    if csv_rows(val) != 3 * len(val_seeds):
        run([PY, os.path.join(HERE, "eval_dqn.py"), "--dirs", *dirs, "--seeds", *val_seeds, "--workers", WA,
             "--out", val], log)
    v = pd.read_csv(val).groupby("param").profit.mean()
    best = int(v.idxmax())
    sel = {"scenario": S, "n_printers": int(n), "m_star": m_star, "theta_star": th_star, "b1star": b1star,
           "val_profit_k": {int(k): x / 1e3 for k, x in v.items()}, "selected_agent": best, "smoke": a.smoke,
           "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(pre + "_selection.json"):          # вибір уже зафіксовано до тесту — не переписуємо
        old = json.load(open(pre + "_selection.json"))
        assert {k: old[k] for k in ("m_star", "theta_star", "b1star", "selected_agent")} == \
               {k: sel[k] for k in ("m_star", "theta_star", "b1star", "selected_agent")}, "вибір змінився після дозапуску"
        sel = old
    else:
        json.dump(sel, open(pre + "_selection.json", "w"), indent=1)
    print("   вибір зафіксовано:", json.dumps(sel), flush=True)

    # 5. тест
    n_test = int(test_rng[1]) - int(test_rng[0])
    if csv_rows(pre + "_test_L.csv") != 4 * n_test:
        run([PY, os.path.join(HERE, "eval_finalists.py"), "--scenario", S, "--n-printers", n,
             "--finalists", "B0", f"B1:{m_star}", f"B2:{th_star}", "B1*", "--b1star", json.dumps(b1star),
             "--seeds-range", *test_rng, "--workers", WA, "--out", pre + "_test_L.csv"], log)
    if csv_rows(pre + "_test_dqn_L.csv") != 3 * n_test:
        run([PY, os.path.join(HERE, "eval_dqn.py"), "--dirs", *dirs, "--seeds-range", *test_rng,
             "--workers", WA, "--out", pre + "_test_dqn_L.csv"], log)
    print(f"Р50 [{a.tag}] завершено", flush=True)


if __name__ == "__main__":
    main()
