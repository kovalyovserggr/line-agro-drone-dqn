"""Р53: B1a* (найсильніший класичний еталон, Р52) у дослідженнях робастності й чутливості —
щоб усі порівняння статті мали той самий еталон.

  A. Логнормальна «правда» (як Р49), політики заморожені: B1a (m*) і B1a* з параметрами,
     налаштованими на Weibull (b1a_<tag>_selection.json); помірна 15 і 17, сильна 15; 20 seed.
  B. η_C2 ∈ {750; 3 000}: повне переналаштування B1a / B1a* (як run_r52.pipeline).
  C. η_C4 ∈ {2 500; 10 000}: заморожені B1a / B1a* бази (помірна, 15 принтерів); 20 seed.
Дозапуск: завершені кроки пропускаються.

Запуск (L): python experiments/article3/run_r53.py --workers 5
"""
import argparse, json, os, subprocess, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_r52 import pipeline, run, rows, PY

LOGN = [("moderate", 15, "mod15"), ("moderate", 17, "mod17"), ("strong", 15, "str15")]


def frozen(sc, n, src_tag, out, W, raw, ttf="weibull"):
    if rows(out) == 2 * 20:
        print(f"   {os.path.basename(out)} уже є — пропускаю", flush=True)
        return
    sel = json.load(open(os.path.join(raw, f"b1a_{src_tag}_selection.json")))
    log = open(out.replace("_test_L.csv", "_pipeline.log"), "a", encoding="utf-8")
    run([PY, os.path.join(HERE, "eval_finalists.py"), "--scenario", sc, "--n-printers", str(n), "--ttf", ttf,
         "--finalists", f"B1a:{sel['m_star']}", "B1a*", "--b1star", json.dumps(sel["b1a_star"]),
         "--seeds-range", "100000", "100020", "--workers", W, "--out", out], log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--raw", default=os.path.join(HERE, "raw"))
    a = ap.parse_args()
    W, raw = str(a.workers), a.raw
    print("=== A. логнормальна правда, заморожені B1a / B1a*", flush=True)
    for sc, n, tag in LOGN:
        frozen(sc, n, tag, os.path.join(raw, f"b1a_logn_{tag}_test_L.csv"), W, raw, ttf="lognormal")
    print("=== B. η_C2: переналаштування B1a / B1a*", flush=True)
    for e in (750, 3000):
        pipeline(f"moderate+C2eta{e}", 15, f"c2eta{e}", W, raw)
    print("=== C. η_C4: заморожені B1a / B1a* бази", flush=True)
    for e in (2500, 10000):
        frozen(f"moderate+C4eta{e}", 15, "mod15", os.path.join(raw, f"b1a_c4eta{e}_test_L.csv"), W, raw)
    print("Р53 (серія) завершено", flush=True)


if __name__ == "__main__":
    main()
