"""Верифікація моделі лінії v0.3 (Стаття 3, Фаза 1).

(L1) Детермінованість: той самий master seed → ідентичні метрики.
(L2) Журнал подій не змінює траєкторію v0.3.
(L3) Інваріант: у черзі ResourceManager немає запитів від вузлів, що не в
     стані WAITING_FOR_RESOURCE (перевірка відкликання запиту при ТО під час очікування).
(L4) MTBF класу на лінії = теоретичному для послідовної системи складових з
     відновленням відмовлої складової «як нова»: 1 / Σ_c 1/(η_c·Γ(1+1/β_c)).
(L5) β = 1 для всіх складових І без деградації якості/швидкості: періодичне ТО
     не підвищує ні доступність, ні пропускну здатність (парне порівняння, CRN).
(L6) β = 1, але з деградацією якості/швидкості від віку: ТО знижує доступність
     (надійнісний канал не працює), але підвищує якість і продуктивність —
     окремий, ефективнісний канал дії ТО (не залежить від β).

Запуск: PYTHONPATH=src python tests/test_line_v3.py
"""
import contextlib, copy, io, math, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from LineV3 import AgroDroneLineV3, run_v3, NoPM, PeriodicPM, OpportunisticPM, DT
from EnumsLine import NodeState
from line_config_v3 import CLASSES

ok_all = True
def same(x, y):
    """Порівняння словників метрик з урахуванням NaN (NaN != NaN, напр. crew_util при crew=None)."""
    if x.keys() != y.keys():
        return False
    for k in x:
        u, v = x[k], y[k]
        if isinstance(u, float) and isinstance(v, float) and math.isnan(u) and math.isnan(v):
            continue
        if u != v:
            return False
    return True

def report(cond, msg):
    global ok_all
    print(("  OK   " if cond else "  FAIL ") + msg); ok_all &= bool(cond)

H = 3600.0
PERIODIC = {"C1": 800 * H, "C2": 300 * H, "C3": 5000 * H, "C4": 1500 * H, "C6": 4000 * H}

print("(L1) детермінованість")
a = run_v3(7, NoPM(), horizon_h=1500, warmup_h=100)
b = run_v3(7, NoPM(), horizon_h=1500, warmup_h=100)
report(same(a, b), f"seed=7 двічі: дрони {a['drones']} = {b['drones']}, усі метрики однакові")

print("(L2) журнал подій не змінює траєкторію")
path = os.path.join(tempfile.mkdtemp(), "v3.jsonl.gz")
c = run_v3(7, NoPM(), horizon_h=1500, warmup_h=100, logger_path=path)
report(same(a, c), f"з журналом: дрони {c['drones']}, метрики однакові; журнал {os.path.getsize(path)/1024:.0f} КБ")

print("(L3) інваріант черги запитів при опортуністичному ТО")
pol = OpportunisticPM({k: 0.05 for k in CLASSES}, {k: 0.5 for k in CLASSES})
with contextlib.redirect_stdout(io.StringIO()):
    line = AgroDroneLineV3(11, pol)
    bad, t = 0, 0.0
    for k in range(int(1500 * H / DT)):
        t += DT
        line.tick(t)
        if k % 10 == 0:
            waiting = {n.id for n in line.units if n.status == NodeState.WAITING_FOR_RESOURCE}
            bad += sum(1 for r in line.res_manager.requests if r.requester_node_id not in waiting)
n_pm_w = sum(n.n_pm_waiting for n in line.units)
report(bad == 0 and n_pm_w > 0, f"ТО під час простою: {n_pm_w}; «осиротілих» запитів у черзі: {bad}")

print("(L4) MTBF класу на лінії проти теорії (B0, 1 рік, 4 seed)")
def theory_mtbf_h(cls):
    rate = sum(1.0 / (eta * math.gamma(1 + 1 / b)) for (_, b, eta, _, _) in CLASSES[cls]["modes"])
    return 1.0 / rate
runs = [run_v3(s, NoPM(), horizon_h=8760, warmup_h=0.1) for s in (101, 102, 103, 104)]
for cls in ("C1", "C2"):
    # MTBF_h класу = напрацювання / відмови; оцінка й довірчий інтервал за Пуассоном
    est = [r[f"{cls}_MTBF_h"] for r in runs]
    th = theory_mtbf_h(cls)
    m = float(np.mean(est))
    rel_err = abs(m - th) / th
    report(rel_err < 0.15, f"{cls}: теорія {th:.0f} год, симуляція {m:.0f} год (по seed: {[round(e) for e in est]}), відхилення {rel_err*100:.1f}%")

def make_exp_classes(degradation: bool):
    cl = copy.deepcopy(CLASSES)
    for c in cl.values():
        c["modes"] = [(n, 1.0, eta, qp, qc) for (n, b, eta, qp, qc) in c["modes"]]
        if not degradation:
            c["speed_kappa"] = 0.0
            c["quality"] = {"p0": c["quality"]["p0"], "pmax": c["quality"]["p0"]}
    return cl

def paired(classes, seeds=(201, 202, 203, 204, 205, 206)):
    d = {k: [] for k in ("A", "P", "Q", "throughput_per_week")}
    for s in seeds:
        r0 = run_v3(s, NoPM(), horizon_h=4000, warmup_h=200, classes=classes)
        r1 = run_v3(s, PeriodicPM(PERIODIC), horizon_h=4000, warmup_h=200, classes=classes)
        for k in d: d[k].append(r1[k] - r0[k])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v)))) for k, v in d.items()}

import LineV3
from line_config_v3 import NODE_OVERRIDES
_saved = copy.deepcopy(NODE_OVERRIDES)
NODE_OVERRIDES.clear()   # вузол 22 має власну деградацію якості — для L5 вимикаємо

print("(L5) β = 1, без деградації: ТО не допомагає")
d = paired(make_exp_classes(False))
report(d["A"][0] < 0, f"ΔA = {d['A'][0]:+.5f} ± {d['A'][1]:.5f} (очікується < 0)")
report(d["throughput_per_week"][0] <= 2 * d["throughput_per_week"][1],
       f"Δпропускна здатність = {d['throughput_per_week'][0]:+.3f} ± {d['throughput_per_week'][1]:.3f} дронів/тиждень (очікується ≤ 0 у межах шуму)")

NODE_OVERRIDES.update(_saved)
print("(L6) β = 1, з деградацією: ТО діє через ефективність, не через надійність")
d = paired(make_exp_classes(True))
report(d["A"][0] < 0 and d["Q"][0] > 0 and d["P"][0] > 0,
       f"ΔA = {d['A'][0]:+.5f}, ΔP = {d['P'][0]:+.4f} ± {d['P'][1]:.4f}, ΔQ = {d['Q'][0]:+.4f} ± {d['Q'][1]:.4f}, "
       f"Δпропускна здатність = {d['throughput_per_week'][0]:+.3f} ± {d['throughput_per_week'][1]:.3f}")

print("ВСІ ПЕРЕВІРКИ ПРОЙДЕНО" if ok_all else "Є ПОМИЛКИ")
sys.exit(0 if ok_all else 1)
