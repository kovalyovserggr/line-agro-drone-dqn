"""DQN-політика ТО для всієї лінії v0.3 (Стаття 3, Р41).

Один агент зі спільною мережею для всіх вузлів (parameter sharing): кожен
вузол — окремий «потік» SMDP-переходів, рішення ухвалюються в моменти E1/E2/E3
(STATE_ACTION_SPEC §3), лише коли ТО дозволене маскою (`pm_allowed`).

Перехід вузла i: (s_k, a_k) → наступне рішення того ж вузла (s_{k+1}),
винагорода — приріст локального економічного рахунку вузла `r_acc` між
рішеннями (Р41, варіант A: додана цінність придатних операцій − втрачена цінність
входів при браку/відмові − матеріал − вартість ТО − вартість ремонту), у тис. USD.
Сума r_acc усіх вузлів = прибуток лінії з точністю до зміни незавершеного
виробництва (перевірено: 0.13% на 2 000 год).
Дисконт переходу γ_k = exp(−Δt/τ) (SMDP).

Ознаки (детерміновані, без випадкових семплів) — STATE_ACTION_SPEC §4 з
доповненнями про вузьке місце (частки принтерів і екструдерів у голодуванні /
блокуванні) і ціну ТО.
"""
from __future__ import annotations

import math
import numpy as np

from EnumsLine import NodeState, OperationState
from LineV3 import CONTINUE, PM, E1, E2, E3
from line_config_v3 import maint_cost, ECONOMICS

CLASSES = ["C1", "C2", "C3", "C4", "C5", "C6"]
H = 3600.0
OBS_DIM = len(CLASSES) + 20


def _ref_duration(node) -> float:
    """Типова тривалість операції вузла (для ознак у моменти без наступної операції)."""
    if not hasattr(node, "_ref_d"):
        node._ref_d = float(np.mean([op.duration for op in node.operations]))
    return node._ref_d


def observe(node, epoch: str, d_next: float, now: float) -> np.ndarray:
    line = node.line_ref
    cfg = node.cfg
    x = np.zeros(OBS_DIM, dtype=np.float32)
    x[CLASSES.index(node.cls)] = 1.0
    i = len(CLASSES)
    d = d_next if d_next > 0 else _ref_duration(node)
    r = node.wear_ratio()
    x[i + 0] = min(r, 3.0)                                       # нормований вік
    x[i + 1] = node.rel.p_fail(d)                                # ризик відмови на наступній операції
    x[i + 2] = node.rel.p_fail(24 * H)                           # ризик відмови за добу роботи
    x[i + 3] = 1.0 - math.exp(-r)                                # рівень деградації якості (0…1)
    x[i + 4] = (node.speed_factor() - 1.0) * 5.0                 # уповільнення
    # момент рішення: E1 / очікування ресурсу / блокування буфером
    blocked = epoch != E1 and node.current_op is None
    x[i + 5] = 1.0 if epoch == E1 else 0.0
    x[i + 6] = 1.0 if (epoch != E1 and not blocked) else 0.0
    x[i + 7] = 1.0 if blocked else 0.0
    waited = (now - node.wait_since) if (epoch != E1 and node.wait_since is not None) else 0.0
    x[i + 8] = min(waited / (24 * H), 1.0)
    # покриття входів наступної операції запасом
    op = node.current_op
    if op is None and epoch == E1:
        cands = [o for o in node.operations if o.state == OperationState.IDLE and node.planner.can_accept(o.name)]
        op = cands[0] if cands else None
    cov, fill = 0.0, 1.0
    if op is not None:
        rm = node.res_manager
        cov = min((min(1.0, (rm.resources[k].number() if rm.resources[k] is not None else 0) / q)
                   for k, q in op.input_resources.items() if q > 0), default=1.0)
        fill = min(node._quota_ratio(op), 2.0)
    x[i + 9] = cov
    x[i + 10] = fill if fill != float("inf") else 2.0            # заповненість вихідного буфера
    x[i + 11] = math.log10(cfg["pm_mean"] / d)                    # тривалість ТО / операції
    x[i + 12] = math.log10(cfg["cm_mean"] / d)                    # тривалість ремонту / операції
    # глобальні ознаки
    st = line._status_counts()
    x[i + 13] = st["down"]
    x[i + 14] = st["down_cls"].get(node.cls, 0.0)
    x[i + 15] = st["C2_idle"]                                     # частка принтерів без роботи
    x[i + 16] = st["C1_idle"]                                     # частка екструдерів без роботи
    x[i + 17] = 1.0 if line.crew_free() else 0.0
    x[i + 18] = maint_cost(node.id) / ECONOMICS["drone_value"] * 10.0
    x[i + 19] = maint_cost(node.id) * ECONOMICS["repair_cost_factor"] / ECONOMICS["drone_value"] * 10.0
    return x


class DQNPMPolicy:
    """Політика ТО для `AgroDroneLineV3(policy=...)`. У режимі навчання записує
    SMDP-переходи кожного вузла в буфер агента і викликає train_step."""

    def __init__(self, agent, tau_h: float = 200.0, train: bool = True, train_every: int = 4,
                 reward_scale: float = 1e-3):
        self.agent, self.tau, self.train = agent, tau_h * H, train
        self.train_every, self.k = train_every, reward_scale
        self.pending = {}            # node.id -> (obs, action, time, r_acc)
        # Р41: базова швидкість винагороди класу ρ_c (USD/с на вузол). Від винагороди
        # переходу віднімається ρ_c·τ·(1 − e^{−Δt/τ}) — дисконтований інтеграл сталої
        # швидкості за час переходу. Сумарний зсув = −ρ_c·τ для будь-якої траєкторії,
        # тож оптимальна політика не змінюється; зникає хибна перевага довгих переходів
        # (після ТО наступне рішення вузла настає лише через ~1% η напрацювання).
        self.rho = {}
        self.n_dec = 0
        self.n_pm = 0
        self.losses = []
        self.name = "DQN"

    def decide(self, node, epoch, d_next):
        now = node._now
        obs = observe(node, epoch, d_next, now)
        if self.train:
            p = self.pending.get(node.id)
            if p is not None:
                s, a, t0, r0 = p
                g = math.exp(-(now - t0) / self.tau)
                base = self.rho.get(node.cls, 0.0) * self.tau * (1.0 - g)
                r = (node.r_acc - r0 - base) * self.k
                self.agent.push(s, a, r, obs, g)
        a = self.agent.act(obs, greedy=not self.train)
        if self.train:
            # ТО скидає r_acc на ціну ТО в _do_pm — вона входить у винагороду цього рішення
            self.pending[node.id] = (obs, a, now, node.r_acc)
            self.n_dec += 1
            if self.n_dec % self.train_every == 0:
                loss = self.agent.train_step()
                if loss is not None:
                    self.losses.append(loss)
        self.n_pm += a == PM
        return a

    def update_rho(self, line, elapsed_s: float, w: float = 0.5):
        """Оновлює ρ_c за підсумком епізоду: середня винагорода на вузол класу за секунду
        (експоненційне згладжування з вагою w)."""
        acc, cnt = {}, {}
        for n in line.units:
            acc[n.cls] = acc.get(n.cls, 0.0) + n.r_acc
            cnt[n.cls] = cnt.get(n.cls, 0) + 1
        for c in acc:
            r = acc[c] / cnt[c] / elapsed_s
            self.rho[c] = r if c not in self.rho else (1 - w) * self.rho[c] + w * r
