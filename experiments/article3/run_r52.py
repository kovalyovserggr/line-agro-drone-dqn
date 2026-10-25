"""Р52: класична заміна за віком як додатковий бейслайн (B1a, B1a*) — для всіх 6 сценаріїв Table 4.

B1/B1* рахують напрацювання з останнього ТО; ремонт відмовлої зносової складової (q_cm = 0,
«як нова») цей лічильник не скидає, тож частина планових ТО припадає на щойно відновлену
складову (журнал подій, Р51: 8–21% ТО B1* — на складовій з віком < ½ медіанного). B1a/B1a*
використовують віртуальний вік зносової складової (скидається і ТО, і ремонтом) — це класична
політика заміни за віком (Barlow–Proschan) і найчесніший класичний конкурент «агента, що бачить вік».

Для кожного сценарію (правила вибору зафіксовані заздалегідь):
  1. B1a: m ∈ {0.25, 0.35, 0.5, 0.75, 1.0, 1.5} на seed 1000–1004 → m* = argmax середнього прибутку;
  2. B1a*: покоординатний спуск C2 → C1 від m*, сітка {0.25, 0.35, 0.5, 0.75, 1.0};
  3. тест на seed 100000–100019: B1a (m*) і B1a*.
Дозапуск: завершені кроки пропускаються (повні CSV), спуск B1a* — з --resume.

Запуск (L): python experiments/article3/run_r52.py --workers 5
"""
import argparse, json, os, subprocess, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
ENV = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
SCEN = [("moderate", 12, "mod12"), ("moderate", 15, "mod15"), ("moderate", 17, "mod17"),
        ("moderate", 19, "mod19"), ("strong", 15, "str15"), ("weak", 15, "wea15")]
TUNE_SEEDS = ["1000", "1001", "1002", "1003", "1004"]
B1_GRID = ["0.25", "0.35", "0.5", "0.75", "1.0", "1.5"]
STAR_GRID = ["0.25", "0.35", "0.5", "0.75", "1.0"]


def rows(path):
    try:
        return len(pd.read_csv(path))
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return 0


def run(cmd, log):
    print("▶", " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=ENV)
    print(f"   ✓ {time.time() - t0:.0f} с", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    ap.add_argument("--scenarios", nargs="+", default=[t for *_, t in SCEN])
    a = ap.parse_args()
    W = str(a.workers)
    for sc, n, tag in SCEN:
        if tag not in a.scenarios:
            continue
        pre = os.path.join(a.raw, f"b1a_{tag}")
        log = open(pre + "_pipeline.log", "a", encoding="utf-8")
        print(f"=== {tag} ({sc}, {n} принтерів)", flush=True)
        tune = pre + "_tune_L.csv"
        if rows(tune) != len(B1_GRID) * len(TUNE_SEEDS):
            run([PY, os.path.join(HERE, "baselines_tune.py"), "--scenario", sc, "--crew", "1", "--n-printers", str(n),
                 "--policies", "B1a", "--b1", *B1_GRID, "--seeds", *TUNE_SEEDS, "--workers", W, "--out", tune], log)
        g = pd.read_csv(tune).groupby("param").profit.mean()
        m_star = float(g.idxmax())
        star_sum = pre + "_b1star.json"
        run([PY, os.path.join(HERE, "b1star_tune.py"), "--scenario", sc, "--n-printers", str(n), "--clock", "age",
             "--order", "C2", "C1", "--grid", *STAR_GRID, "--start", str(m_star), "--seeds", *TUNE_SEEDS,
             "--workers", W, "--out", pre + "_b1star_L.csv", "--summary", star_sum, "--resume"], log)
        mults = json.load(open(star_sum))[str(n)]["mults"]
        b1a_star = {c: float(v) for c, v in mults.items() if v != "inf"}
        sel = {"scenario": sc, "n_printers": n, "m_star": m_star, "b1a_star": b1a_star,
               "tune_profit_k": {str(k): v / 1e3 for k, v in g.items()}, "fixed_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if not os.path.exists(pre + "_selection.json"):
            json.dump(sel, open(pre + "_selection.json", "w"), indent=1)
        print(f"   m* = {m_star}, B1a* = {b1a_star}", flush=True)
        test = pre + "_test_L.csv"
        if rows(test) != 2 * 20:
            run([PY, os.path.join(HERE, "eval_finalists.py"), "--scenario", sc, "--n-printers", str(n),
                 "--finalists", f"B1a:{m_star}", "B1a*", "--b1star", json.dumps(b1a_star),
                 "--seeds-range", "100000", "100020", "--workers", W, "--out", test], log)
    print("Р52 завершено", flush=True)


if __name__ == "__main__":
    main()
