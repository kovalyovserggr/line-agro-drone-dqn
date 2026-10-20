"""Аналітичний оптимум заміни за віком для зносової складової класу (Р25, бейслайн B1).

Класичний підхід (Barlow–Proschan): вузол розглядається ізольовано, ТО і ремонт
відновлюють «як новий», максимізується доступність
    A(T) = ∫₀ᵀR / (∫₀ᵀR + d_pm·R(T) + d_cm·F(T)).
Враховується лише надійнісний канал — так, як це робиться в інженерній практиці.
"""
import math
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

H = 3600.0


def age_replacement_T_star(beta, eta, d_pm, d_cm):
    """Оптимальний вік ТО (у тих самих одиницях, що й η) або None, якщо ТО не потрібне."""
    if beta <= 1.0:
        return None
    R = lambda t: math.exp(-(t / eta) ** beta)
    A = lambda T: quad(R, 0, T, limit=200)[0] / (quad(R, 0, T, limit=200)[0] + d_pm * R(T) + d_cm * (1 - R(T)))
    res = minimize_scalar(lambda T: -A(T), bounds=(1e-3 * eta, 5 * eta), method="bounded")
    m = eta * math.gamma(1 + 1 / beta)
    return res.x if -res.fun > m / (m + d_cm) else None


def class_T_star_seconds(classes):
    """{клас: T* у секундах напрацювання} для зносової складової (q_pm < 1)."""
    out = {}
    for c, cfg in classes.items():
        wear = [m for m in cfg["modes"] if m[3] < 1.0]
        if not wear:
            continue
        _, b, eta_h = wear[0][:3]      # T* — за вейбулівською моделлю (Р49: і для логнормальної «правди»)
        T = age_replacement_T_star(b, eta_h, cfg["pm_h"][0], cfg["cm_h"][0])
        if T is not None:
            out[c] = T * H
    return out
