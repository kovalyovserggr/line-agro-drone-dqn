"""Модель надійності вузла v0.3 (Стаття 3, рішення Р3, Р6, Р8, Р12, Р20).

Вузол = набір складових (механізмів відмов). Кожна складова має власний
віртуальний вік v, розподіл часу до відмови Weibull(β, η) (β = 1 — експоненційний)
і параметри відновлення Kijima типу II: після ТО v ← q_pm·v, після ремонту
відмовлої складової v ← q_cm·v (q = 0 — «як нова», q = 1 — без ефекту).
Вузол відмовляє, щойно відмовить будь-яка складова (послідовна система).

Вік зростає лише під час напрацювання (стан WORKING) — його подає викликач
через advance(dt). Випадковість — лише з власних numpy.random.Generator,
переданих ззовні (окремий потік на кожну пару «вузол, процес», Р6).

Модуль не залежить від решти симулятора: його перевіряє
tests/test_reliability_v3.py (MTBF = η·Γ(1+1/β); оптимум заміни за віком
збігається з аналітичним; при β = 1 профілактика не дає виграшу).

Перевірка робастності (Р49): «справжній» розподіл часу до відмови складової може
бути логнормальним (ttf = "lognormal") з тими самими середнім і CV, що й Weibull(β, η).
Тоді випадкові пороги відмов семплюються з логнормального розподілу, а
cum_hazard / hazard / p_fail (ознаки агента, ризик B2, T* для B1) лишаються
вейбулівськими — це модель, якою користується обслуговування (хибна специфікація).
Кількість випадкових чисел на семпл однакова (одне Exp(1)), тож потоки CRN не зсуваються.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import log_ndtr, ndtri


@dataclass
class FailureMode:
    """Одна складова вузла (механізм відмов)."""
    name: str
    beta: float          # параметр форми Вейбулла (1 — випадкові відмови)
    eta: float           # параметр масштабу, с напрацювання
    q_pm: float = 0.0    # Kijima II після ТО: 0 — як нова, 1 — ТО не впливає
    q_cm: float = 0.0    # Kijima II після ремонту цієї складової
    ttf: str = "weibull"  # «справжній» розподіл порогів відмов: weibull | lognormal (Р49)

    def __post_init__(self):
        if self.ttf == "lognormal":
            # логнормальний з тими самими середнім і CV, що й Weibull(β, η)
            g1, g2 = math.gamma(1 + 1 / self.beta), math.gamma(1 + 2 / self.beta)
            cv2 = g2 / g1 ** 2 - 1.0
            self.ln_sigma = math.sqrt(math.log(1.0 + cv2))
            self.ln_mu = math.log(self.eta * g1) - 0.5 * self.ln_sigma ** 2
        elif self.ttf != "weibull":
            raise ValueError(f"невідомий розподіл {self.ttf}")

    def cum_hazard(self, v: float) -> float:
        return (v / self.eta) ** self.beta

    def hazard(self, v: float) -> float:
        if v <= 0.0:
            return (1.0 / self.eta) if self.beta == 1.0 else (math.inf if self.beta < 1.0 else 0.0)
        return (self.beta / self.eta) * (v / self.eta) ** (self.beta - 1.0)

    def mean_life(self) -> float:
        """Середній час до відмови нової складової: η·Γ(1 + 1/β)."""
        return self.eta * math.gamma(1.0 + 1.0 / self.beta)

    def sample_failure_age(self, v: float, rng: np.random.Generator) -> float:
        """Вік, на якому складова відмовить, за умови, що вона дожила до віку v:
        H(v_f) = H(v) − ln U  ⇒  v_f = η·[(v/η)^β − ln U]^(1/β)."""
        e = rng.exponential(1.0)  # −ln U
        if self.ttf == "lognormal":
            # S(v_f) = S(v)·U  ⇒  ln S(v_f) = ln S(v) − e;  S(x) = Φ(−z), z = (ln x − μ)/σ
            log_s = (float(log_ndtr(-(math.log(v) - self.ln_mu) / self.ln_sigma)) if v > 0 else 0.0) - e
            z = -float(ndtri(math.exp(log_s)))
            return math.exp(self.ln_mu + self.ln_sigma * z)
        return self.eta * (self.cum_hazard(v) + e) ** (1.0 / self.beta)


@dataclass
class NodeReliability:
    """Стан надійності вузла: віртуальні віки і пороги відмов складових."""
    modes: list[FailureMode]
    rng: np.random.Generator
    v: list[float] = field(default_factory=list)
    v_fail: list[float] = field(default_factory=list)
    n_failures: list[int] = field(default_factory=list)

    def __post_init__(self):
        self.v = [0.0] * len(self.modes)
        self.v_fail = [m.sample_failure_age(0.0, self.rng) for m in self.modes]
        self.n_failures = [0] * len(self.modes)

    # --- динаміка -----------------------------------------------------------
    def advance(self, dt: float) -> int | None:
        """Додає dt напрацювання всім складовим. Повертає індекс складової, що
        відмовила (першою за віком у межах цього кроку), або None."""
        failed, first_over = None, math.inf
        for k in range(len(self.modes)):
            self.v[k] += dt
            if self.v[k] >= self.v_fail[k]:
                over = self.v_fail[k] - (self.v[k] - dt)   # коли саме в межах dt
                if over < first_over:
                    failed, first_over = k, over
        if failed is not None:
            self.n_failures[failed] += 1
        return failed

    def _restore(self, k: int, q: float):
        if q >= 1.0:
            return  # вік не змінюється → поріг лишається чинним (без пам'яті не потрібно перевибирати)
        self.v[k] = q * self.v[k]
        self.v_fail[k] = self.modes[k].sample_failure_age(self.v[k], self.rng)

    def preventive_maintenance(self):
        for k, m in enumerate(self.modes):
            self._restore(k, m.q_pm)

    def corrective_repair(self, k: int):
        """Ремонт відмовлої складової k. Інші складові не змінюються."""
        self._restore(k, self.modes[k].q_cm)
        if self.v[k] >= self.v_fail[k]:   # q_cm = 1 (мінімальний ремонт): новий поріг від поточного віку
            self.v_fail[k] = self.modes[k].sample_failure_age(self.v[k], self.rng)

    # --- детерміновані ознаки для агента (з параметрів розподілів) -----------
    def p_fail(self, d: float) -> float:
        """Ймовірність відмови вузла протягом наступних d с напрацювання."""
        s = sum(m.cum_hazard(vk + d) - m.cum_hazard(vk) for m, vk in zip(self.modes, self.v))
        return 1.0 - math.exp(-s)

    def node_hazard(self) -> float:
        return sum(m.hazard(vk) for m, vk in zip(self.modes, self.v))


# --- тривалості втручань (Р8, Р12) ---------------------------------------------
def sample_pm_duration(mean: float, cv: float, rng: np.random.Generator, floor_frac: float = 0.2) -> float:
    """ТО: нормальний розподіл, усічений знизу на floor_frac·mean (Р8)."""
    sd = cv * mean
    while True:
        x = rng.normal(mean, sd)
        if x >= floor_frac * mean:
            return x


def sample_repair_duration(mean: float, cv: float, rng: np.random.Generator) -> float:
    """Ремонт: логнормальний розподіл із заданими середнім і CV (Р12)."""
    s2 = math.log(1.0 + cv * cv)
    mu = math.log(mean) - 0.5 * s2
    return float(rng.lognormal(mu, math.sqrt(s2)))


def make_streams(master_seed: int, n_nodes: int, processes=("failure", "quality", "pm", "repair")):
    """Окремі генератори для кожної пари (вузол, процес) з одного master seed (Р6).
    Повертає dict[(node_index, process)] -> Generator. Порядок і кількість потоків
    фіксовані, тож додавання політики чи журналу не зсуває випадкові «долі»."""
    ss = np.random.SeedSequence(master_seed)
    children = ss.spawn(n_nodes * len(processes))
    out, i = {}, 0
    for n in range(n_nodes):
        for p in processes:
            out[(n, p)] = np.random.default_rng(children[i])
            i += 1
    return out
