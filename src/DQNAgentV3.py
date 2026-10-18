"""Double Dueling DQN + PER для Статті 3 (SMDP-версія).

Архітектура та алгоритм — як у пакеті дисертації (dqn_integration: DuelingDQN,
PrioritizedReplayBuffer, DoubleDQNAgent), з трьома змінами, потрібними для
коректності й відтворюваності:
  1. SMDP: дисконт кожного переходу γ_k = exp(−Δt/τ) зберігається в буфері
     (час між рішеннями змінний — STATE_ACTION_SPEC §5.3).
  2. Уся випадковість агента — з власного numpy.random.Generator і
     torch.Generator, заданих seed-ом агента (Р6; у пакеті використовувався
     глобальний np.random).
  3. ε змінюється лінійно за кроками (eps_start → eps_end за eps_steps), а не
     множенням на кожному кроці — щоб графік дослідження не залежав від
     кількості кроків в епізоді.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden=(128, 128)):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU()]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.value = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, action_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        f = self.backbone(x)
        a = self.adv(f)
        return self.value(f) + a - a.mean(dim=1, keepdim=True)


class PER:
    """Prioritized Experience Replay (пропорційний, SumTree), з власним RNG."""

    def __init__(self, capacity: int, rng: np.random.Generator, alpha=0.6, beta0=0.4, beta_steps=50_000):
        self.cap, self.rng, self.alpha = capacity, rng, alpha
        self.beta0, self.beta_steps, self.t = beta0, beta_steps, 0
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.ptr = self.size = 0
        self.max_p = 1.0

    def _update(self, i, p):
        ch = p - self.tree[i]
        self.tree[i] = p
        while i:
            i = (i - 1) // 2
            self.tree[i] += ch

    def push(self, tr):
        i = self.ptr + self.cap - 1
        self.data[self.ptr] = tr
        self._update(i, self.max_p ** self.alpha)
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def _get(self, s):
        i = 0
        while i < self.cap - 1:
            l = 2 * i + 1
            if s <= self.tree[l]:
                i = l
            else:
                s -= self.tree[l]
                i = l + 1
        return i

    def sample(self, n):
        self.t += 1
        beta = min(1.0, self.beta0 + (1 - self.beta0) * self.t / self.beta_steps)
        seg = self.tree[0] / n
        idx = [self._get(self.rng.uniform(seg * k, seg * (k + 1))) for k in range(n)]
        pri = self.tree[idx]
        w = (self.size * pri / self.tree[0]) ** (-beta)
        w /= w.max()
        batch = [self.data[i - self.cap + 1] for i in idx]
        return batch, np.array(idx), w.astype(np.float32)

    def update(self, idx, td):
        for i, e in zip(idx, td):
            p = (abs(e) + 1e-6) ** self.alpha
            self._update(i, p)
            self.max_p = max(self.max_p, abs(e) + 1e-6)

    def __len__(self):
        return self.size


class DoubleDQNAgentV3:
    def __init__(self, obs_dim, action_dim, seed: int, lr=3e-4, tau_soft=0.005, batch=64,
                 buffer=100_000, eps_start=1.0, eps_end=0.05, eps_steps=20_000, hidden=(128, 128),
                 warmup=1_000, lr_end=None, total_steps=None):
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        self.action_dim, self.batch, self.tau_soft, self.warmup = action_dim, batch, tau_soft, warmup
        self.eps_start, self.eps_end, self.eps_steps = eps_start, eps_end, eps_steps
        self.online = DuelingDQN(obs_dim, action_dim, hidden)
        self.target = DuelingDQN(obs_dim, action_dim, hidden)
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buf = PER(buffer, np.random.default_rng(seed + 1))
        self.steps = 0
        # лінійне згасання швидкості навчання lr → lr_end за total_steps (якщо задано)
        self.lr0, self.lr_end, self.total_steps = lr, lr_end, total_steps

    @property
    def eps(self):
        if getattr(self, "eps_fixed", None) is not None:      # графік ε задає зовнішній цикл (за епізодами)
            return self.eps_fixed
        f = min(1.0, self.steps / self.eps_steps)
        return self.eps_start + f * (self.eps_end - self.eps_start)

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            return self.online(torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0))[0].numpy()

    def act(self, obs, mask=None, greedy=False) -> int:
        """mask — допустимі дії (bool-масив) або None."""
        allowed = np.arange(self.action_dim) if mask is None else np.flatnonzero(mask)
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.choice(allowed))
        q = self.q_values(obs)
        return int(allowed[np.argmax(q[allowed])])

    def push(self, s, a, r, s2, gamma_k, done=False):
        """gamma_k = exp(−Δt/τ) — дисконт саме цього переходу (SMDP)."""
        self.buf.push((np.asarray(s, np.float32), a, r, np.asarray(s2, np.float32), gamma_k, float(done)))
        self.steps += 1

    def train_step(self):
        if len(self.buf) < max(self.batch, self.warmup):
            return None
        batch, idx, w = self.buf.sample(self.batch)
        s = torch.as_tensor(np.stack([b[0] for b in batch]))
        a = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
        r = torch.as_tensor([b[2] for b in batch], dtype=torch.float32)
        s2 = torch.as_tensor(np.stack([b[3] for b in batch]))
        g = torch.as_tensor([b[4] for b in batch], dtype=torch.float32)
        d = torch.as_tensor([b[5] for b in batch], dtype=torch.float32)
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            a2 = self.online(s2).argmax(1, keepdim=True)
            y = r + g * (1 - d) * self.target(s2).gather(1, a2).squeeze(1)
        td = (q - y).detach().numpy()
        self.buf.update(idx, td)
        loss = (torch.as_tensor(w) * F.huber_loss(q, y, reduction="none")).mean()
        if self.lr_end is not None and self.total_steps:
            f = min(1.0, self.steps / self.total_steps)
            for gr in self.opt.param_groups:
                gr["lr"] = self.lr0 + f * (self.lr_end - self.lr0)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        with torch.no_grad():
            for t, o in zip(self.target.parameters(), self.online.parameters()):
                t.mul_(1 - self.tau_soft).add_(self.tau_soft * o)
        return float(loss.item())
