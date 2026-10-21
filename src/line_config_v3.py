"""Конфігурація класів обладнання для моделі v0.3 (Стаття 3).

⚠ ТИМЧАСОВІ ЗНАЧЕННЯ. Числа нижче — робочі припущення для налагодження й
верифікації моделі v0.3 (Фаза 1). Порядок величин β і η спирається на довідкові
діапазони компонентів (docs/article3/PARAMETER_SOURCES.md, база Barringer /
Bloch & Geitner), але конкретні значення для класів ще НЕ обґрунтовані
джерелами. Перед офіційними серіями запусків кожне значення має отримати
джерело або позначку «припущення» з діапазоном для аналізу чутливості (Фаза 2).

Одиниці: усі часи в годинах (переводяться в секунди в make_class_runtime()).
Класи — STATE_ACTION_SPEC §10, рішення Р9, Р13, Р14, Р20.
"""
H = 3600.0

# Складова: (назва, β, η [год напрацювання], q_pm, q_cm)
#   q_pm = 1 — ТО на складову не впливає (електроніка, випадкові відмови)
CLASSES = {
    "C1": {  # екструзія / рециклінг полімеру (вузол 1 і додаткові постачальники)
        "modes": [("wear", 1.8, 4000.0, 0.0, 0.0), ("electronics", 1.0, 30000.0, 1.0, 0.0)],
        "pm_h": (1.0, 0.1), "cm_h": (6.0, 0.4),
        "quality": {"p0": 0.005, "pmax": 0.05},
        "speed_kappa": 0.10,
    },
    "C2": {  # FDM-принтер (вузли 2–16)
        "modes": [("wear", 1.5, 1500.0, 0.0, 0.0), ("electronics", 1.0, 20000.0, 1.0, 0.0)],
        "pm_h": (0.5, 0.1), "cm_h": (4.0, 0.4),
        "quality": {"p0": 0.01, "pmax": 0.10},   # верхня межа браку FDM ~19% маси (Song & Telenko 2016)
        "speed_kappa": 0.20,
    },
    "C3": {  # промисловий робот складання (вузли 17–21)
        "modes": [("wear", 2.0, 20000.0, 0.0, 0.0), ("electronics", 1.0, 50000.0, 1.0, 0.0)],
        "pm_h": (2.0, 0.1), "cm_h": (8.0, 0.4),
        "quality": {"p0": 0.002, "pmax": 0.02},
        "speed_kappa": 0.10,
    },
    "C4": {  # ручна робота спеціаліста (вузли 22–24): відмови оснащення (Р9, Р14)
        "modes": [("tooling", 1.2, 5000.0, 0.0, 0.0)],
        "pm_h": (0.25, 0.1), "cm_h": (1.0, 0.4),
        "quality": {"p0": 0.005, "pmax": 0.005},  # 23, 24 — без деградації (Р14)
        "speed_kappa": 0.0,
    },
    "C5": {  # встановлення ПЗ, електроніка/ІТ-стенд (вузол 25)
        "modes": [("electronics", 1.0, 20000.0, 1.0, 0.0)],
        "pm_h": (0.5, 0.1), "cm_h": (2.0, 0.4),
        "quality": {"p0": 0.002, "pmax": 0.002},
        "speed_kappa": 0.0,
    },
    "C6": {  # тестовий стенд (вузол 26)
        "modes": [("stand", 1.2, 15000.0, 0.0, 0.0)],
        "pm_h": (0.5, 0.1), "cm_h": (3.0, 0.4),
        "quality": {"p0": 0.001, "pmax": 0.001},
        "speed_kappa": 0.0,
    },
}

# Вузол 22 (електропроводка): якість деградує зі зносом інструменту (Р14)
NODE_OVERRIDES = {
    22: {"quality": {"p0": 0.005, "pmax": 0.03}},
}


def node_class(node_id: float) -> str:
    i = int(node_id)
    if i == 1 or 100 <= i < 200:      # екструзія: вузол 1 і постачальники 101+
        return "C1"
    if 2 <= i <= 16 or 200 <= i < 300:  # принтери: 2–16 і додаткові 201+ (Р38)
        return "C2"
    if 17 <= i <= 21:
        return "C3"
    if 22 <= i <= 24:
        return "C4"
    if i == 25:
        return "C5"
    if i == 26:
        return "C6"
    raise ValueError(f"невідомий вузол {node_id}")


def class_config(node_id: float, classes: dict | None = None, overrides: dict | None = None) -> dict:
    """Конфіг вузла в секундах: класовий + перевизначення для окремих вузлів."""
    classes = classes or CLASSES
    overrides = NODE_OVERRIDES if overrides is None else overrides
    c = dict(classes[node_class(node_id)])
    c.update(overrides.get(int(node_id), {}))
    return {
        "cls": node_class(node_id),
        "modes": [(m[0], m[1], m[2] * H, *m[3:]) for m in c["modes"]],
        "pm_mean": c["pm_h"][0] * H, "pm_cv": c["pm_h"][1],
        "cm_mean": c["cm_h"][0] * H, "cm_cv": c["cm_h"][1],
        "p0": c["quality"]["p0"], "pmax": c["quality"]["pmax"],
        "speed_kappa": c["speed_kappa"],
        "v_cap": 1.0,   # деградація швидкості насичується на v/η = 1
    }


# --- Сценарії деградації якості та швидкості (Р28) ------------------------------
# Множник s застосовується до приросту браку (p_max − p0) і до κ; p_max не вище 0.5.
# «Помірний» = базові значення вище. «Сильний» для C2: p_max = 0.01 + 2·0.09 = 0.19 —
# верхня межа частки невдалих друків (Song & Telenko 2016, 19% маси матеріалу).
DEGRADATION_SCENARIOS = {"weak": 0.5, "moderate": 1.0, "strong": 2.0}


def scenario_classes(name: str = "moderate", classes: dict | None = None,
                     overrides: dict | None = None, ttf: str = "weibull"):
    """Повертає (classes, node_overrides) для сценарію деградації.
    ttf = "lognormal" — зносові складові (β > 1, q_pm < 1) отримують логнормальний
    «справжній» розподіл порогів відмов з тими самими середнім і CV (Р49).
    name може містити модифікатори η: "moderate+C2eta750" (Р50)."""
    import copy, re
    # Р50: модифікатори параметрів у назві сценарію: "moderate+C2eta750+C4eta2500" —
    # η зносової складової класу (год). Назва з модифікаторами потрапляє в усі CSV і config.json.
    base, *mods = name.split("+")
    s = DEGRADATION_SCENARIOS[base]
    cl = copy.deepcopy(classes or CLASSES)
    for mod in mods:
        mm = re.fullmatch(r"(C\d)eta(\d+(?:\.\d+)?)", mod)
        if not mm or mm.group(1) not in cl:
            raise ValueError(f"невідомий модифікатор сценарію: {mod}")
        c, eta = mm.group(1), float(mm.group(2))
        wear = [k for k, m in enumerate(cl[c]["modes"]) if m[3] < 1.0]
        assert len(wear) == 1, f"{c}: очікується одна зносова складова"
        m = list(cl[c]["modes"][wear[0]]); m[2] = eta
        cl[c]["modes"][wear[0]] = tuple(m)
    for c in cl.values():
        q = c["quality"]
        c["quality"] = {"p0": q["p0"], "pmax": min(0.5, q["p0"] + s * (q["pmax"] - q["p0"]))}
        c["speed_kappa"] = s * c["speed_kappa"]
        if ttf != "weibull":
            c["modes"] = [tuple(m[:5]) + ((ttf,) if (m[1] > 1.0 and m[3] < 1.0) else ()) for m in c["modes"]]
    ov = copy.deepcopy(overrides if overrides is not None else NODE_OVERRIDES)
    for v in ov.values():
        if "quality" in v:
            q = v["quality"]
            v["quality"] = {"p0": q["p0"], "pmax": min(0.5, q["p0"] + s * (q["pmax"] - q["p0"]))}
    return cl, ov


# --- Економічні параметри (Р32) ----------------------------------------------------
# Вартість одного ТО вузла — поле maintenance_cost з параметрів ефективності моделі
# v0.2 (AgroDroneLine.py, efficiency_paramsN), умовні одиниці = USD (автор, 2026-10-18).
MAINT_COST_BY_NODE = {1: 700.0, **{i: 500.0 for i in range(2, 17)}, 17: 600.0, 18: 600.0,
                      19: 550.0, 20: 550.0, 21: 600.0, 22: 400.0, 23: 450.0, 24: 500.0,
                      25: 450.0, 26: 700.0}
ECONOMICS = {
    "drone_value": 4000.0,          # USD за придатний дрон (автор: 3000–5000, база 4000)
    "repair_cost_factor": 5.0,      # ремонт = k × вартість ТО того ж вузла (ПРИПУЩЕННЯ, сценарна вісь)
    "filament_cost_per_g": 0.02,    # USD/г (≈20 USD/кг; ПРИПУЩЕННЯ, звірити з цінами)
}


def maint_cost(node_id: float) -> float:
    i = int(node_id)
    if i in MAINT_COST_BY_NODE:
        return MAINT_COST_BY_NODE[i]
    if 100 <= i < 200:
        return MAINT_COST_BY_NODE[1]        # додаткові екструдери
    if 200 <= i < 300:
        return MAINT_COST_BY_NODE[2]        # додаткові принтери (Р38)
    return 500.0
