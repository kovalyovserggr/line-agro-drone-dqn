"""Перевірка журналу подій (Стаття 3, Фаза 0).

1. Журнал не змінює траєкторію: той самий seed з журналом і без -> ідентичні
   makespan, кількість дронів, тривалості станів і лічильники кожного вузла.
2. Метрики, відновлені з журналу, = онлайн-метрики симулятора (тривалості
   станів, виготовлене, брак, кількість відмов, кількість дронів).

Запуск: PYTHONPATH=src python tests/test_event_log.py [seed ...]
"""
import contextlib, io, os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from AgroDroneLine import AgroDroneLine, SimulationComplete
from EnumsLine import NodeState
from EventLog import EventLogger, metrics_from_log
import ProductionNode as PN

DT, MAX_TICKS, T_PM, N_SUP, LIM = 100.0, 50000, 6000.0, 5, 10

_failures = {}
_orig_interrupt = PN.ProductionNode._interrupt_to
def _counting_interrupt(self, new_status):          # лише рахує, поведінку не змінює
    if new_status == NodeState.REPAIR:
        _failures[self.id] = _failures.get(self.id, 0) + 1
    return _orig_interrupt(self, new_status)
PN.ProductionNode._interrupt_to = _counting_interrupt

def run(seed, log_path=None):
    _failures.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        line = AgroDroneLine(seed=seed, lim_drones_made=LIM, T_pm_print_park=T_PM,
                             verbose=False, n_filament_suppliers=N_SUP)
        log = EventLogger(log_path, line, meta={"seed": seed}) if log_path else None
        t, reason = 0.0, "max_ticks"
        try:
            for _ in range(MAX_TICKS):
                t += DT
                line.tick(t)
        except SimulationComplete:
            reason = "completed"
        finally:
            if log: log.close(reason, t)
    snap = {n.id: ({k.name: v for k, v in n.state_durations.items() if v},
                   {k: v for k, v in n.produced.items() if v},
                   {k: v for k, v in n.defects.items() if v}) for n in line.units}
    return {"t": t, "reason": reason, "drones": line.number_drones_made,
            "snap": snap, "failures": dict(_failures)}

def check(seed):
    a = run(seed)
    path = os.path.join(tempfile.mkdtemp(), f"log_{seed}.jsonl.gz")
    b = run(seed, path)
    ok = True
    def req(cond, msg):
        nonlocal ok
        print(("  OK   " if cond else "  FAIL ") + msg); ok &= bool(cond)
    print(f"seed={seed}: {a['reason']} t={a['t']:.0f} drones={a['drones']}")
    req(a["t"] == b["t"] and a["drones"] == b["drones"], "журнал не змінює makespan і кількість дронів")
    req(a["snap"] == b["snap"], "журнал не змінює тривалості станів, виготовлене і брак жодного вузла")
    req(a["failures"] == b["failures"], "журнал не змінює кількість відмов")
    m = metrics_from_log(path)
    dur_ok = all(abs(m["state_durations"].get(nid, {}).get(st, 0.0) - v) < 1e-6
                 for nid, (sd, _, _) in b["snap"].items() for st, v in sd.items()) and \
             all(abs(v - b["snap"][nid][0].get(st, 0.0)) < 1e-6
                 for nid, d in m["state_durations"].items() for st, v in d.items())
    req(dur_ok, "тривалості станів з журналу = онлайн state_durations (усі вузли, усі стани)")
    req({k: v for k, v in m["produced"].items()} == {nid: p for nid, (_, p, _) in b["snap"].items() if p},
        "виготовлене з журналу = онлайн produced")
    req({k: v for k, v in m["defects"].items()} == {nid: d for nid, (_, _, d) in b["snap"].items() if d},
        "брак з журналу = онлайн defects")
    req(m["failures"] == b["failures"], f"відмови з журналу = онлайн ({sum(b['failures'].values())} шт.)")
    req(m["drones"] == b["drones"], "кількість дронів з журналу = онлайн")
    print(f"  розмір журналу: {os.path.getsize(path)/1024:.0f} КБ (gzip)")
    return ok

if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [1000, 1003]
    results = [check(s) for s in seeds]
    print("ВСІ ПЕРЕВІРКИ ПРОЙДЕНО" if all(results) else "Є ПОМИЛКИ")
    sys.exit(0 if all(results) else 1)
