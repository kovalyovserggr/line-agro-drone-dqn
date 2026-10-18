"""Перевірка DQN на одному вузлі (Стаття 3, Фаза 4, перевірка «аналітичний оптимум»).

Постановка (Р18: там, де аналітика існує, метод має з нею збігатися):
  * один вузол, одна зносова складова Weibull(β, η = 1), ремонт і ТО «як новий»;
  * вузол виконує однакові операції тривалістю d одну за одною; рішення
    «працювати / ТО» — лише на межі операцій (E1), як у моделі лінії;
  * відмова під час операції: операція втрачена, ремонт d_cm;
  * мета — максимум корисного часу (сума тривалостей завершених операцій).

Еталони:
  (1) точний оптимум дисконтованої задачі (ітерація цінностей на сітці віку
      j·d) — з тим самим дисконтом, що й DQN;
  (2) оптимальний поріг k*·d для середнього критерію (теорема відновлення з
      винагородою, точна формула для порогових політик);
  (3) неперервний аналітичний оптимум заміни за віком T* (Barlow–Proschan) — довідково.
Перевіряємо: поріг політики DQN = поріг (1) з точністю ±1 крок сітки, і
середня продуктивність політики DQN ≥ 99% від оптимальної (2). При β = 1
оптимум — не робити ТО.

Тривалості ТО і ремонту тут детерміновані (щоб еталон (1) був точним);
випадкові тривалості перевірено окремо в tests/test_reliability_v3.py.

Запуск: PYTHONPATH=src python experiments/article3/single_node_check.py --out experiments/article3/raw/single_node_check.csv
"""
import argparse, csv, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from ReliabilityV3 import FailureMode, NodeReliability
from DQNAgentV3 import DoubleDQNAgentV3

D, D_PM, D_CM = 0.05, 0.02, 0.20      # у одиницях η: операція, ТО, ремонт (d_pm/d_cm = 0.1)
TAU = 10.0                            # стала часу дисконтування: γ(Δt) = exp(−Δt/τ), τ = 10·η
J_MAX = 120                           # сітка віку для еталона: j·d, j ≤ J_MAX (6·η)

# ----------------------------------------------------------------------------- середовище
class SingleNodeEnv:
    def __init__(self, beta, seed):
        self.mode = FailureMode("wear", beta, 1.0)
        self.rel = NodeReliability([self.mode], np.random.default_rng(seed))

    def obs(self):
        v = self.rel.v[0]
        return np.array([v, self.rel.p_fail(D)], dtype=np.float32)   # вік/η і p_f наступної операції

    def step(self, a):
        """Повертає (reward, Δt)."""
        if a == 1:                                   # ТО
            self.rel.preventive_maintenance()
            return 0.0, D_PM
        to_fail = self.rel.v_fail[0] - self.rel.v[0]
        if to_fail >= D:                             # операцію завершено
            self.rel.advance(D)
            return D, D
        self.rel.advance(to_fail + 1e-12)            # відмова під час операції
        self.rel.corrective_repair(0)
        return 0.0, to_fail + D_CM

# ----------------------------------------------------------------------------- еталони
def R(beta, t): return math.exp(-t ** beta)

def dp_discounted(beta):
    """Ітерація цінностей: стан j (вік j·d на межі операцій), дії CONTINUE/PM."""
    g = lambda dt: math.exp(-dt / TAU)
    ps, fail_disc = [], []
    for j in range(J_MAX + 1):
        r0, r1 = R(beta, j * D), R(beta, (j + 1) * D)
        q = r1 / r0 if r0 > 0 else 0.0
        ps.append(q)
        # E[γ^(τ_f + d_cm) · 1{відмова в межах операції}] / P(виживання до j·d)
        dens = lambda x: beta * x ** (beta - 1) * R(beta, x) if x > 0 else (1.0 if beta == 1 else 0.0)
        val = quad(lambda x: dens(x) * g(x - j * D + D_CM), j * D, (j + 1) * D)[0] / r0 if r0 > 0 else g(D_CM)
        fail_disc.append(val)
    V = np.zeros(J_MAX + 1)
    for _ in range(20000):
        Vn = np.empty_like(V)
        for j in range(J_MAX + 1):
            jn = min(j + 1, J_MAX)
            cont = ps[j] * (D + g(D) * V[jn]) + fail_disc[j] * V[0]
            pm = g(D_PM) * V[0]
            Vn[j] = max(cont, pm)
        if np.max(np.abs(Vn - V)) < 1e-10:
            V = Vn
            break
        V = Vn
    act = []
    for j in range(J_MAX + 1):
        jn = min(j + 1, J_MAX)
        cont = ps[j] * (D + g(D) * V[jn]) + fail_disc[j] * V[0]
        pm = g(D_PM) * V[0]
        act.append(1 if pm > cont + 1e-12 else 0)
    k = next((j for j in range(1, J_MAX + 1) if act[j] == 1), None)   # поріг: перший вік із ТО
    return k

def avg_rate(beta, k):
    """Середня частка корисного часу для політики «ТО після k операцій» (k=None — без ТО)."""
    if k is None:
        m = math.gamma(1 + 1 / beta)
        # завершені операції за життя: Σ_j R(j d) · d
        comp = D * sum(R(beta, j * D) for j in range(1, 100000) if R(beta, j * D) > 1e-15)
        return comp / (m + D_CM)
    comp = D * sum(R(beta, j * D) for j in range(1, k + 1))
    worked = quad(lambda t: R(beta, t), 0, k * D)[0]
    return comp / (worked + D_PM * R(beta, k * D) + D_CM * (1 - R(beta, k * D)))

def best_k(beta):
    rates = {k: avg_rate(beta, k) for k in range(1, J_MAX + 1)}
    rates[None] = avg_rate(beta, None)
    k = max(rates, key=rates.get)
    return k, rates[k]

def continuous_T(beta):
    if beta == 1.0:
        return None
    f = lambda T: -(quad(lambda t: R(beta, t), 0, T)[0] /
                    (quad(lambda t: R(beta, t), 0, T)[0] + D_PM * R(beta, T) + D_CM * (1 - R(beta, T))))
    return minimize_scalar(f, bounds=(0.01, 5), method="bounded").x

# ----------------------------------------------------------------------------- навчання
def train(beta, seed, steps, lr_end=None, batch=64):
    env = SingleNodeEnv(beta, seed=10_000 + seed)
    ag = DoubleDQNAgentV3(2, 2, seed=seed, eps_steps=int(0.4 * steps), warmup=2000, lr=5e-4,
                          lr_end=lr_end, total_steps=steps, batch=batch)
    s = env.obs()
    for _ in range(steps):
        a = ag.act(s)
        r, dt = env.step(a)
        s2 = env.obs()
        ag.push(s, a, r, s2, math.exp(-dt / TAU))
        ag.train_step()
        s = s2
    return ag

def learned_threshold(ag, beta):
    """Поріг жадібної політики на сітці віку; перевірка, що політика порогова."""
    m = FailureMode("wear", beta, 1.0)
    acts = []
    for j in range(J_MAX + 1):
        v = j * D
        pf = 1 - math.exp(-(m.cum_hazard(v + D) - m.cum_hazard(v)))
        acts.append(ag.act(np.array([v, pf], np.float32), greedy=True))
    k = next((j for j in range(1, J_MAX + 1) if acts[j] == 1), None)
    monotone = k is None or all(acts[j] == 1 for j in range(k, min(k + 40, J_MAX + 1)))
    learned_threshold.last_acts = "".join(map(str, acts))   # повний вектор дій для діагностики
    return k, monotone

# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", nargs="+", type=float, default=[1.0, 1.5, 2.5, 3.5])
    ap.add_argument("--agent-seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--lr-end", type=float, default=None, help="лінійне згасання lr до цього значення")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    rows, ok_all = [], True
    for beta in args.betas:
        k_dp = dp_discounted(beta)
        k_avg, rate_opt = best_k(beta)
        T_c = continuous_T(beta)
        print(f"β={beta}: еталон (1) поріг DP k={k_dp} (вік {k_dp*D if k_dp else '∞'}), "
              f"(2) середній критерій k*={k_avg} (частка корисного часу {rate_opt:.4f}), "
              f"(3) неперервний T*={T_c if T_c is None else round(T_c, 3)}", flush=True)
        for sd in args.agent_seeds:
            t0 = time.time()
            ag = train(beta, sd, args.steps, args.lr_end, args.batch)
            k_q, mono = learned_threshold(ag, beta)
            rate_q = avg_rate(beta, k_q)
            if k_dp is None:
                thr_ok = k_q is None or k_q * D > 3.0       # «майже ніколи»: поріг > 3η
            else:
                thr_ok = k_q is not None and abs(k_q - k_dp) <= 1
            rel = rate_q / rate_opt
            ok = thr_ok and mono and rel >= 0.99
            ok_all &= ok
            print(f"   агент seed={sd}: поріг k={k_q} (вік {k_q*D if k_q else '∞'}), монотонна={mono}, "
                  f"частка корисного часу {rate_q:.4f} = {rel*100:.2f}% від оптимуму "
                  f"[{'OK' if ok else 'FAIL'}] ({time.time()-t0:.0f} с)", flush=True)
            rows.append({"beta": beta, "agent_seed": sd, "steps": args.steps, "lr_end": args.lr_end, "batch": args.batch, "k_dp": k_dp, "k_avg_opt": k_avg,
                         "T_continuous": T_c, "k_dqn": k_q, "monotone": mono, "rate_dqn": rate_q,
                         "rate_opt": rate_opt, "rel": rel, "ok": ok,
                         "acts": learned_threshold.last_acts})
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    print("УСІ ПЕРЕВІРКИ ПРОЙДЕНО" if ok_all else "Є НЕВДАЛІ ПЕРЕВІРКИ")
    return 0 if ok_all else 1

if __name__ == "__main__":
    sys.exit(main())
