"""Верифікація моделі надійності v0.3 (Стаття 3, Фаза 1, перевірки a–c).

(a) MTBF: для складової Weibull(β, η) з відновленням «як нова» середнє
    напрацювання між відмовами = η·Γ(1+1/β).
(b) β = 1: доступність не зростає від профілактики (оптимум — без ТО).
(c) Один ізольований вузол, політика «ТО при досягненні віку T» (ТО і ремонт
    «як новий»): симульована доступність A(T) збігається з аналітичною
    A(T) = ∫₀ᵀR / (∫₀ᵀR + d_pm·R(T) + d_cm·F(T)), а оптимум T* — з аналітичним.
(d) Дві складові: знос (ТО діє) + випадкова (ТО не діє, q_pm = 1): інтенсивність
    випадкових відмов на одиницю напрацювання = 1/η_r незалежно від T.
Тривалості ТО/ремонту — випадкові (усічений нормальний / логнормальний), аналітика
використовує їхні середні (теорема відновлення з винагородою).

Запуск: PYTHONPATH=src python tests/test_reliability_v3.py
"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from ReliabilityV3 import FailureMode, NodeReliability, sample_pm_duration, sample_repair_duration

ETA, D_PM, D_CM, CV_PM, CV_CM = 1000.0, 20.0, 200.0, 0.1, 0.4
EPS = 1e-9
ok_all = True

def report(cond, msg):
    global ok_all
    print(("  OK   " if cond else "  FAIL ") + msg); ok_all &= bool(cond)

def simulate_age_policy(modes, T, n_cycles, seed, pm_driver=0):
    """Подієва (точна) симуляція одного вузла: ТО, коли вік складової pm_driver
    досягає T; ремонт при відмові. Повертає (A, up, failures_by_mode)."""
    rng = np.random.default_rng(seed)
    rel = NodeReliability(modes, rng)
    up = down = 0.0
    for _ in range(n_cycles):
        to_fail = min(vf - v for vf, v in zip(rel.v_fail, rel.v))
        to_pm = (T - rel.v[pm_driver]) if T is not None else math.inf
        if to_fail <= to_pm:
            k = rel.advance(to_fail + EPS)
            up += to_fail
            down += sample_repair_duration(D_CM, CV_CM, rng)
            rel.corrective_repair(k)
        else:
            rel.advance(to_pm)
            up += to_pm
            down += sample_pm_duration(D_PM, CV_PM, rng)
            rel.preventive_maintenance()
    return up / (up + down), up, rel.n_failures

def analytic_A(beta, T):
    R = lambda t: math.exp(-(t / ETA) ** beta)
    integ = quad(R, 0, T, limit=200)[0]
    return integ / (integ + D_PM * R(T) + D_CM * (1 - R(T)))

def analytic_A_inf(beta):   # без ТО
    m = ETA * math.gamma(1 + 1 / beta)
    return m / (m + D_CM)

print("(a) MTBF = η·Γ(1+1/β)")
for beta in (1.0, 1.5, 2.5, 3.5):
    rng = np.random.default_rng(int(beta * 100))
    m = FailureMode("w", beta, ETA)
    lives = np.array([m.sample_failure_age(0.0, rng) for _ in range(40000)])
    _, up, nf = simulate_age_policy([FailureMode("w", beta, ETA)], None, 20000, seed=7)
    exp_m = m.mean_life()
    se = lives.std() / math.sqrt(len(lives))
    report(abs(lives.mean() - exp_m) < 3 * se and abs(up / nf[0] - exp_m) / exp_m < 0.02,
           f"β={beta}: теорія {exp_m:.1f}, вибірка {lives.mean():.1f}±{se:.1f}, симуляція відновлення {up/nf[0]:.1f}")

print("(b, c) Оптимум заміни за віком: симуляція проти аналітики")
for beta in (1.0, 1.5, 2.5, 3.5):
    grid = ETA * np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0])
    sims, anas = [], []
    for j, T in enumerate(grid):
        A_sim, _, _ = simulate_age_policy([FailureMode("w", beta, ETA)], T, 40000, seed=1000 + j)
        sims.append(A_sim); anas.append(analytic_A(beta, T))
    sims, anas = np.array(sims), np.array(anas)
    max_err = np.max(np.abs(sims - anas))
    res = minimize_scalar(lambda T: -analytic_A(beta, T), bounds=(1.0, 5 * ETA), method="bounded")
    T_star, A_star, A_inf = res.x, -res.fun, analytic_A_inf(beta)
    report(max_err < 0.004, f"β={beta}: max|A_sim − A_аналіт| по сітці = {max_err:.4f}")
    if beta == 1.0:
        report(np.all(np.diff(anas) > 0) and A_inf >= anas.max() - 1e-12 and sims.argmax() >= len(grid) - 2,
               f"β=1: A(T) монотонно зростає, найкраще — без ТО (A∞={A_inf:.4f}); симуляція — max на T={grid[sims.argmax()]:.0f}")
    else:
        i_sim = int(sims.argmax())
        near = grid[max(i_sim - 1, 0)] <= T_star <= grid[min(i_sim + 1, len(grid) - 1)]
        report(near and A_star > A_inf,
               f"β={beta}: аналітичний T*={T_star:.0f} (A*={A_star:.4f} > A∞={A_inf:.4f}); "
               f"симуляція — max на сітці T={grid[i_sim]:.0f}")

print("(d) Знос + випадкова складова: ТО не впливає на випадкові відмови")
ETA_R = 4000.0
for T in (300.0, 800.0, None):
    modes = [FailureMode("wear", 2.5, ETA, q_pm=0.0), FailureMode("rand", 1.0, ETA_R, q_pm=1.0)]
    _, up, nf = simulate_age_policy(modes, T, 60000, seed=55, pm_driver=0)
    rate = nf[1] / up
    se = math.sqrt(nf[1]) / up
    report(abs(rate - 1 / ETA_R) < 3 * se,
           f"T={T}: випадкових відмов на од. напрацювання {rate*1e4:.3f}e-4 (теорія {1e4/ETA_R:.3f}e-4 ± {3*se*1e4:.3f}e-4)")

print("(e) Р49: логнормальна «правда» з тими самими середнім і CV")
from scipy import stats as _st
for beta in (1.2, 1.5, 1.8, 2.0):
    m = FailureMode("w", beta, ETA, ttf="lognormal")
    rng = np.random.default_rng(11)
    x = np.array([m.sample_failure_age(0.0, rng) for _ in range(40000)])
    g1, g2 = math.gamma(1 + 1 / beta), math.gamma(1 + 2 / beta)
    mean_w, cv_w = ETA * g1, math.sqrt(g2 / g1 ** 2 - 1)
    se = x.std() / math.sqrt(len(x))
    ks = _st.kstest(x, _st.lognorm(s=m.ln_sigma, scale=math.exp(m.ln_mu)).cdf).pvalue
    report(abs(x.mean() - mean_w) < 3.5 * se and abs(x.std() / x.mean() - cv_w) < 0.03 and ks > 0.01,
           f"β={beta}: середнє {x.mean():.1f} (Weibull {mean_w:.1f} ± {3.5*se:.1f}), CV {x.std()/x.mean():.3f} (Weibull {cv_w:.3f}), KS p={ks:.2f}")
    # умовний семпл від віку v: розподіл залишкового часу = усічений логнормальний
    v0 = 0.8 * mean_w
    xc = np.array([m.sample_failure_age(v0, rng) for _ in range(20000)])
    d = _st.lognorm(s=m.ln_sigma, scale=math.exp(m.ln_mu))
    ks_c = _st.kstest(xc, lambda t: (d.cdf(t) - d.cdf(v0)) / d.sf(v0)).pvalue
    report(xc.min() >= v0 and ks_c > 0.01, f"β={beta}: умовний семпл від v={v0:.0f}: min {xc.min():.1f} ≥ v, KS p={ks_c:.2f}")
# CRN: однакова кількість випадкових чисел; Weibull за замовчуванням — побайтово як раніше
r1, r2 = np.random.default_rng(5), np.random.default_rng(5)
FailureMode("w", 1.5, ETA, ttf="lognormal").sample_failure_age(100.0, r1)
FailureMode("w", 1.5, ETA).sample_failure_age(100.0, r2)
report(r1.random() == r2.random(), "CRN: логнормальний семпл споживає стільки ж випадкових чисел, скільки вейбулівський")
mw = FailureMode("w", 1.5, ETA); rr = np.random.default_rng(9)
e = np.random.default_rng(9).exponential(1.0)
report(mw.sample_failure_age(50.0, rr) == ETA * ((50.0 / ETA) ** 1.5 + e) ** (1 / 1.5), "Weibull за замовчуванням — формула не змінилась")

print("ВСІ ПЕРЕВІРКИ ПРОЙДЕНО" if ok_all else "Є ПОМИЛКИ")
sys.exit(0 if ok_all else 1)
