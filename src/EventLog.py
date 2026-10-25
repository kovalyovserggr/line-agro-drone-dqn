"""Журнал подій симуляції у форматі JSONL (Стаття 3, рішення Р1).

Неінвазивний спостерігач: після кожного такту лінії порівнює стан вузлів і
лічильники з попереднім тактом і пише події. НЕ викликає жодних випадкових
генераторів і нічого не змінює в моделі, тому увімкнений журнал не впливає на
траєкторію симуляції (перевіряється tests/test_event_log.py).

Підключення:
    line = AgroDroneLine(seed=..., ...)
    log = EventLogger("run.jsonl.gz", line, meta={...})   # .gz -> стиснення
    try:
        for ...: line.tick(t)          # tick уже обгорнутий логером
    except SimulationComplete: ...
    finally:
        log.close()

Типи подій (поле "ev"):
    meta     — параметри прогону (seed, тег, політика, ...)
    state    — зміна стану вузла: node, from, to; для вузлів v0.3 при вході в MAINTENANCE /
               REPAIR додатково (Р51): r — нормований вік зносових складових (max v/η),
               ages — v/η кожної складової, op_pm_h — напрацювання від останнього ТО, год,
               pm_wait — лічильник ТО під час очікування (приріст = це ТО почато в E2/E3),
               mode — індекс відмовлої складової (лише REPAIR). Вік скидається в кінці ТО
               чи ремонту, тож на такті входу він ще дорівнює віку в момент втручання.
    failure  — відмова (перехід у REPAIR): node
    produce  — виготовлено одиниць ресурсу: node, type, qty
    defect   — приріст списаного в брак: node, res, qty  (облік D15/D18)
    drone    — готовий дрон (Type22): node, total
    stock    — періодичний знімок складу: qty {res: кількість}
    end      — кінець прогону: reason, t
Час "t" — модельний час (с) такту, на якому подію зафіксовано.
"""
import gzip
import json

from EnumsLine import NodeState


class EventLogger:
    def __init__(self, path, line, meta: dict | None = None, stock_every: float = 10000.0):
        self.line = line
        self.stock_every = stock_every
        self._next_stock = 0.0
        self._f = gzip.open(path, "wt", encoding="utf-8") if str(path).endswith(".gz") \
            else open(path, "w", encoding="utf-8")
        self._prev = {}
        for n in line.units:
            self._prev[n.id] = (n.status, dict(n.produced), dict(n.defects))
        self._write({"ev": "meta", "t": 0.0, **(meta or {}),
                     "nodes": [n.id for n in line.units]})
        for n in line.units:
            self._write({"ev": "state", "t": 0.0, "node": n.id, "from": None, "to": n.status.name})
        # обгортаємо tick лінії: спершу оригінальний такт, потім спостереження
        self._orig_tick = line.tick

        def wrapped_tick(elapsed_time):
            try:
                return self._orig_tick(elapsed_time)
            finally:
                # finally — щоб зафіксувати й останній такт, на якому
                # drone_made_callback піднімає SimulationComplete
                self.observe(elapsed_time)

        line.tick = wrapped_tick
        self._closed = False

    def _write(self, rec: dict):
        self._f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")

    def observe(self, t: float):
        for n in self.line.units:
            st0, prod0, def0 = self._prev[n.id]
            if n.status != st0:
                rec = {"ev": "state", "t": t, "node": n.id, "from": st0.name, "to": n.status.name}
                if n.status in (NodeState.MAINTENANCE, NodeState.REPAIR) and hasattr(n, "rel"):
                    # лише читання атрибутів — генератори не викликаються (неінвазивно, Р51)
                    rec["r"] = round(n.wear_ratio(), 6)
                    rec["ages"] = [round(v / m.eta, 6) for m, v in zip(n.rel.modes, n.rel.v)]
                    rec["op_pm_h"] = round(n.op_time_since_pm / 3600.0, 4)
                    rec["pm_wait"] = n.n_pm_waiting
                    if n.status == NodeState.REPAIR:
                        rec["mode"] = n.failed_mode
                self._write(rec)
                if n.status == NodeState.REPAIR:
                    self._write({"ev": "failure", "t": t, "node": n.id})
            for res, q in n.produced.items():
                dq = q - prod0.get(res, 0)
                if dq:
                    self._write({"ev": "produce", "t": t, "node": n.id, "type": res, "qty": dq})
                    if res == "Type22":
                        self._write({"ev": "drone", "t": t, "node": n.id, "total": self.line.number_drones_made})
            for res, q in n.defects.items():
                dq = q - def0.get(res, 0)
                if dq:
                    self._write({"ev": "defect", "t": t, "node": n.id, "res": res, "qty": dq})
            self._prev[n.id] = (n.status, dict(n.produced), dict(n.defects))
        if t >= self._next_stock:
            stock = {k: float(p.number()) for k, p in self.line.res_manager.resources.items()}
            self._write({"ev": "stock", "t": t, "qty": stock})
            self._next_stock = t + self.stock_every

    def close(self, reason: str = "", t: float | None = None):
        if self._closed:
            return
        self._write({"ev": "end", "t": t if t is not None else self.line.current_time, "reason": reason,
                     "last_time": {str(n.id): n.last_time for n in self.line.units}})
        self._f.close()
        self.line.tick = self._orig_tick
        self._closed = True


def metrics_from_log(path) -> dict:
    """Відновлює з журналу ключові лічильники: тривалості станів кожного вузла,
    виготовлене, брак, кількість відмов, кількість дронів. Використовується для
    перевірки «метрики з журналу = онлайн-метрики» і для рисунків статті."""
    opener = gzip.open if str(path).endswith(".gz") else open
    cur, since, dur = {}, {}, {}
    produced, defects, failures = {}, {}, {}
    drones, last_time = 0, {}
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            ev = e["ev"]
            if ev == "state":
                nid = e["node"]
                if e["from"] is not None:
                    d = dur.setdefault(nid, {})
                    d[cur[nid]] = d.get(cur[nid], 0.0) + (e["t"] - since[nid])
                cur[nid], since[nid] = e["to"], e["t"]
            elif ev == "failure":
                failures[e["node"]] = failures.get(e["node"], 0) + 1
            elif ev == "produce":
                p = produced.setdefault(e["node"], {})
                p[e["type"]] = p.get(e["type"], 0) + e["qty"]
            elif ev == "defect":
                p = defects.setdefault(e["node"], {})
                p[e["res"]] = p.get(e["res"], 0) + e["qty"]
            elif ev == "drone":
                drones = e["total"]
            elif ev == "end":
                last_time = {float(k): v for k, v in e["last_time"].items()}
    # закриваємо поточні стани на момент останнього такту кожного вузла
    for nid, st in cur.items():
        lt = last_time.get(float(nid), since[nid])
        d = dur.setdefault(nid, {})
        d[st] = d.get(st, 0.0) + (lt - since[nid])
    return {"state_durations": dur, "produced": produced, "defects": defects,
            "failures": failures, "drones": drones}
