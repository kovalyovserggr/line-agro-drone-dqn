"""Модель лінії v0.3 (Стаття 3): вузол = набір складових з розподілами
надійності; вік вузла причинно впливає на брак і швидкість; рішення про ТО
приймає політика (бейслайн або DQN-агент) у моменти E1/E2/E3.

Код v0.2 (ProductionNode, AgroDroneLine) НЕ змінюється — Стаття 1 лишається
відтворюваною. v0.3 будується поверх нього:
  * AgroDroneLineV3 створює лінію базовим конструктором (без глобального seed),
    потім «підвищує» кожен вузол до ProductionNodeV3 (upgrade_to_v3) і
    замінює планувальник на ProductionPlannerV3.
  * Уся випадковість v0.3 — з власних генераторів (Р6, ReliabilityV3.make_streams).
  * Декоративне семплювання efficiency.get()/reliability.get() у циклі прибрано (Р15/Р21).

Рішення: Р3, Р5, Р6, Р8, Р10, Р12, Р13, Р14, Р19, Р20, Р21, Р22 (A3_DECISIONS.md).
"""
from __future__ import annotations

import contextlib
import io
import math

import numpy as np

from AgroDroneLine import AgroDroneLine
from EnumsLine import NodeState, OperationState
from ProductionNode import ProductionNode
from ProductionPlanner import ProductionPlanner
from Resource import RequestRes
from ReliabilityV3 import (FailureMode, NodeReliability, make_streams,
                           sample_pm_duration, sample_repair_duration)
from line_config_v3 import class_config, maint_cost, ECONOMICS

CONTINUE, PM = 0, 1
E1, E2, E3 = "E1", "E2", "E3"
BULK = {"Type0", "TypePL"}          # сипучі ресурси (Р10)
DELTA_W = 3600.0                    # повторне рішення під час очікування (Р11/Р13)


# =============================================================================
# Планувальник v0.3: буферні ліміти (kanban) замість BOM-квоти на N дронів (Р22)
# =============================================================================
class ProductionPlannerV3(ProductionPlanner):
    """На довгому горизонті (Р19) квота «рівно на N дронів» не має сенсу.
    Виробництво проміжного ресурсу дозволено, лише поки (запас + у роботі)
    менше за буферний ліміт cap[res]. Готовий дрон (Type22) — без ліміту."""

    def __init__(self, base: ProductionPlanner, caps: dict, get_stock, get_in_progress):
        super().__init__()
        self.__dict__.update({k: v for k, v in base.__dict__.items()})
        self.caps = caps
        self.get_stock = get_stock
        self.get_in_progress = get_in_progress

    def can_accept(self, name: str) -> bool:
        cap = self.caps.get(name)
        return cap is None or self.get_stock(name) + self.get_in_progress(name) < cap

    def process_request(self, request: RequestRes, res_manager) -> bool:
        if not self.can_accept(request.requester_name):
            return False
        if self.get_state(request.requester_node_id) != NodeState.IDLE:
            return False
        res_manager.add_requests(request)
        return True


# =============================================================================
# Вузол v0.3
# =============================================================================
class ProductionNodeV3(ProductionNode):
    """Підвищений вузол. Створюється лише через upgrade_to_v3()."""

    # ---- ініціалізація -------------------------------------------------------
    def upgrade_to_v3(self, cfg: dict, streams: dict, idx: int, policy):
        self.cfg = cfg
        self.cls = cfg["cls"]
        self.policy = policy
        self.rng_q = streams[(idx, "quality")]
        self.rng_pm = streams[(idx, "pm")]
        self.rng_cm = streams[(idx, "repair")]
        modes = [FailureMode(n, b, eta, qp, qc) for (n, b, eta, qp, qc) in cfg["modes"]]
        self.rel = NodeReliability(modes, streams[(idx, "failure")])
        self.cur_dur = 0.0           # тривалість поточного ТО/ремонту
        self.failed_mode = None
        self.wait_since = None
        self.next_wait_decision = None
        self.op_time_since_pm = 0.0
        # лічильники для метрик
        self.n_pm = 0
        self.n_pm_waiting = 0
        self.n_fail = 0
        self.n_completed = 0
        self.n_good = 0
        self.ideal_done = 0.0
        self.waste = {}
        self.decisions = 0
        self.t_setup = 0.0           # час переналагодження (LETUP) — втрата доступності в OEE
        self.awaiting_crew = False   # ремонт чекає вільного техніка (Р30)
        self.t_crew_wait = 0.0       # час відмовленого вузла в очікуванні техніка
        self.line = None             # посилання на лінію (бригада), задає AgroDroneLineV3

    # ---- деградація (Р3, Р14) ------------------------------------------------
    def wear_ratio(self) -> float:
        """Нормований вік зносових складових (ТО на які діє): max v/η."""
        r = 0.0
        for m, v in zip(self.rel.modes, self.rel.v):
            if m.q_pm < 1.0:
                r = max(r, v / m.eta)
        return r

    def p_def(self) -> float:
        c = self.cfg
        return c["p0"] + (c["pmax"] - c["p0"]) * (1.0 - math.exp(-self.wear_ratio()))

    def speed_factor(self) -> float:
        c = self.cfg
        return 1.0 + c["speed_kappa"] * min(self.wear_ratio(), c["v_cap"])

    def pm_allowed(self) -> bool:
        """Маска: ТО заборонено одразу після ТО (усі зносові складові майже нові)
        і коли немає вільного техніка (Р30) — планове ТО не чекає в черзі."""
        if self.line is not None and not self.line.crew_free():
            return False
        return self.wear_ratio() >= 0.01

    # ---- відмови (замість update_status v0.2) --------------------------------
    def update_status(self):
        if self.status != NodeState.WORKING:
            return
        # вік зростає на крок напрацювання; delta вже пораховано в tick()
        failed = self.rel.advance(self._delta)
        self.op_time_since_pm += self._delta
        if failed is not None:
            self.failed_mode = failed
            self.n_fail += 1
            self._interrupt_v3(NodeState.REPAIR)

    def _interrupt_v3(self, new_status: NodeState):
        """Переривання операції відмовою (Р10, Р13): сипучий вхід — пропорційно
        прогресу (решта повертається на склад), дискретні деталі — наскрізне
        списання за BOM (D18)."""
        op = self.current_op
        if op is not None and op.state == OperationState.EXECUTING:
            f = min(1.0, op.progress / op.duration) if op.duration > 0 else 1.0
            for res, qty in op.input_resources.items():
                if res in BULK:
                    lost = f * qty
                    self.waste[res] = self.waste.get(res, 0.0) + lost
                    back = float(qty) - lost
                    if back > 0:
                        self.res_manager.resources[res].add(float(back))
            footprint = self.planner.get_bom_footprint(op.name)
            for res, qty in footprint.items():
                if res not in BULK:
                    self.defects[res] = self.defects.get(res, 0) + qty
            op.resourses = []
        if op is not None:
            op.progress = 0.0
            op.state = OperationState.IDLE
            self.current_op = None
        self._start_downtime(new_status)

    def _start_downtime(self, new_status: NodeState):
        self.status = new_status
        self.progress = 0.0
        if new_status == NodeState.REPAIR:
            self.cur_dur = sample_repair_duration(self.cfg["cm_mean"], self.cfg["cm_cv"], self.rng_cm)
        else:
            self.cur_dur = sample_pm_duration(self.cfg["pm_mean"], self.cfg["pm_cv"], self.rng_pm)
        # бригада (Р30): ТО стартує лише з вільним техніком (див. pm_allowed);
        # ремонт без вільного техніка стає в чергу (FIFO), вузол стоїть у REPAIR
        self.awaiting_crew = False
        if self.line is not None and not self.line.crew_acquire():
            self.awaiting_crew = True
            self.line.crew_queue.append(self)
        self.wait_since = None
        self.next_wait_decision = None

    def _withdraw_request(self):
        rm = self.res_manager
        rm.requests = [r for r in rm.requests if r.requester_node_id != self.id]
        if self.current_op is not None:
            self.current_op.state = OperationState.IDLE
            self.current_op.progress = 0.0
        self.current_op = None

    def _do_pm(self, waiting: bool):
        self.n_pm += 1
        if waiting:
            self.n_pm_waiting += 1
            self._withdraw_request()
        self._start_downtime(NodeState.MAINTENANCE)

    def _ask(self, epoch: str, d_next: float) -> int:
        if not self.pm_allowed():
            return CONTINUE
        self.decisions += 1
        return self.policy.decide(self, epoch, d_next)

    # ---- головний цикл -------------------------------------------------------
    def tick(self, current_time: float, is_point: int):
        lr = getattr(self, "line_ref", None)
        if lr is not None and getattr(lr, "time_carry", False):
            return self._tick_carry(current_time)
        return self._tick_legacy(current_time, is_point)

    def _tick_carry(self, now: float):
        """Р39: такт без втрати часу на переходах. Бюджет такту витрачається
        послідовно на стани вузла; залишок після завершення операції, LETUP, ТО чи
        ремонту переходить у наступний стан; запит на ресурс задовольняється в тому
        ж такті, якщо ресурс є на складі. Відмова — у точний момент усередині такту
        (вік зростає лише під час виконання операції, не під час LETUP)."""
        budget = now - self.last_time
        self.last_time = now
        self._now = now
        self._delta = budget
        sd = self.state_durations
        for _ in range(64):                                # захист від зациклення
            st = self.status
            if st == NodeState.WORKING:
                op = self.current_op
                if op is None:
                    self.status = NodeState.IDLE
                    continue
                if op.state == OperationState.EXECUTING:
                    if budget <= 0.0:
                        break
                    sf = self.speed_factor()
                    need = max(0.0, (op.duration - op.progress) * sf)
                    use = min(budget, need)
                    to_fail = min(vf - v for vf, v in zip(self.rel.v_fail, self.rel.v))
                    if to_fail < use:                       # відмова всередині такту
                        use = max(0.0, to_fail)
                        sd[NodeState.WORKING] += use
                        self.op_time_since_pm += use
                        op.progress += use / sf
                        budget -= use
                        failed = self.rel.advance(use + 1e-9)
                        if failed is None:                  # чисельний край — дотягуємо
                            failed = int(np.argmin([vf - v for vf, v in zip(self.rel.v_fail, self.rel.v)]))
                            self.rel.v[failed] = self.rel.v_fail[failed]
                            self.rel.n_failures[failed] += 1
                        self.failed_mode = failed
                        self.n_fail += 1
                        self._interrupt_v3(NodeState.REPAIR)
                        continue
                    self.rel.advance(use)
                    sd[NodeState.WORKING] += use
                    self.op_time_since_pm += use
                    op.progress += use / sf
                    budget -= use
                    if use >= need:
                        op.progress = op.duration
                        self._complete(op)                  # → LETUP, progress = 0
                        continue
                    break
                elif op.state == OperationState.LETUP:
                    if budget <= 0.0:
                        break
                    need = max(0.0, op.duration_letup - op.progress)
                    use = min(budget, need)
                    sd[NodeState.WORKING] += use
                    self.t_setup += use
                    op.progress += use
                    budget -= use
                    if use >= need:
                        op.progress = 0.0
                        op.state = OperationState.IDLE
                        self.current_op = None
                        self.status = NodeState.IDLE
                        continue
                    break
                else:
                    self.status = NodeState.IDLE
                    continue

            elif st == NodeState.MAINTENANCE:
                if budget <= 0.0:
                    break
                need = max(0.0, self.cur_dur - self.progress)
                use = min(budget, need)
                sd[st] += use
                self.progress += use
                budget -= use
                if use >= need:
                    self.rel.preventive_maintenance()
                    self.op_time_since_pm = 0.0
                    self.status = NodeState.IDLE
                    self.progress = 0.0
                    if self.line is not None:
                        self.line.crew_release()
                    continue
                break

            elif st == NodeState.REPAIR:
                if budget <= 0.0:
                    break
                if self.awaiting_crew:
                    sd[st] += budget
                    self.t_crew_wait += budget
                    budget = 0.0
                    break
                need = max(0.0, self.cur_dur - self.progress)
                use = min(budget, need)
                sd[st] += use
                self.progress += use
                budget -= use
                if use >= need:
                    self.rel.corrective_repair(self.failed_mode)
                    self.failed_mode = None
                    self.status = NodeState.IDLE
                    self.progress = 0.0
                    if self.line is not None:
                        self.line.crew_release()
                    continue
                break

            elif st == NodeState.WAITING_FOR_RESOURCE:
                op = self.current_op
                if op is not None and op.state != OperationState.EXECUTING:
                    self.line_ref.allocate_pass()           # ресурс міг з'явитися в цьому такті
                if op is not None and op.state == OperationState.EXECUTING:
                    op.progress = 0.0
                    self.status = NodeState.WORKING
                    self.wait_since = None
                    self.next_wait_decision = None
                    continue
                d_next = op.duration if op is not None else 0.0
                if self._idle_wait_epoch(now, d_next):
                    self._do_pm(waiting=True)
                    continue
                sd[st] += budget
                budget = 0.0
                break

            elif st == NodeState.IDLE:
                candidates = [op for op in self.operations if op.state == OperationState.IDLE
                              and self.planner.can_accept(op.name)]
                if not candidates:
                    if self._idle_wait_epoch(now, 0.0):
                        self._do_pm(waiting=True)
                        continue
                    sd[st] += budget
                    budget = 0.0
                    break
                self.wait_since = None
                self.next_wait_decision = None
                self._sort_candidates(candidates)
                if self._ask(E1, candidates[0].duration) == PM:
                    self._do_pm(waiting=False)
                    continue
                placed = False
                for op in candidates:
                    if self.planner.process_request(RequestRes(self.id, op.name, op.input_resources, op.callback),
                                                    self.res_manager):
                        self.current_op = op
                        self.status = NodeState.WAITING_FOR_RESOURCE
                        self.reserved[op.name] = self.reserved.get(op.name, 0) + (1000 if op.name == "Type0" else 1)
                        placed = True
                        break
                if placed:
                    continue
                sd[st] += budget
                budget = 0.0
                break
            else:
                sd[st] += budget
                budget = 0.0
                break
        if budget > 0.0:                                   # не мало б траплятися; облік часу не губимо
            sd[self.status] += budget

    def _tick_legacy(self, current_time: float, is_point: int):
        delta = current_time - self.last_time
        self.last_time = current_time
        self._delta = delta
        self._now = current_time
        self.state_durations[self.status] += delta
        self.update_status()

        st = self.status
        if st == NodeState.MAINTENANCE:
            self.progress += delta
            if self.progress >= self.cur_dur:
                self.rel.preventive_maintenance()
                self.op_time_since_pm = 0.0
                self.status = NodeState.IDLE
                self.progress = 0.0
                if self.line is not None:
                    self.line.crew_release()

        elif st == NodeState.REPAIR:
            if self.awaiting_crew:
                self.t_crew_wait += delta
                return
            self.progress += delta
            if self.progress >= self.cur_dur:
                self.rel.corrective_repair(self.failed_mode)
                self.failed_mode = None
                self.status = NodeState.IDLE
                self.progress = 0.0
                if self.line is not None:
                    self.line.crew_release()

        elif st == NodeState.WORKING:
            op = self.current_op
            if op is None:
                self.status = NodeState.IDLE
                return
            if op.state == OperationState.EXECUTING:
                op.progress += delta / self.speed_factor()
                if op.progress >= op.duration:
                    self._complete(op)
            elif op.state == OperationState.LETUP:
                op.progress += delta
                self.t_setup += delta
                if op.progress >= op.duration_letup:
                    op.progress = 0.0
                    op.state = OperationState.IDLE
                    self.current_op = None
                    self.status = NodeState.IDLE

        elif st == NodeState.WAITING_FOR_RESOURCE:
            op = self.current_op
            if op is not None and op.state == OperationState.EXECUTING:
                op.progress = 0.0
                self.status = NodeState.WORKING
                self.wait_since = None
                self.next_wait_decision = None
            else:
                # E2 — перший такт очікування; E3 — кожні DELTA_W с
                d_next = op.duration if op is not None else 0.0
                if self._idle_wait_epoch(current_time, d_next):
                    self._do_pm(waiting=True)

        elif st == NodeState.IDLE:
            candidates = [op for op in self.operations if op.state == OperationState.IDLE
                          and self.planner.can_accept(op.name)]
            if not candidates:
                # вузол заблоковано буферами (простій без роботи) — як і очікування
                # ресурсу, це момент для опортуністичного ТО: E2, далі E3 кожні DELTA_W
                if self._idle_wait_epoch(current_time, 0.0):
                    self._do_pm(waiting=True)
                return
            self.wait_since = None
            self.next_wait_decision = None
            self._sort_candidates(candidates)
            # E1: рішення до подачі запиту; d_next — тривалість першого кандидата (D17)
            if self._ask(E1, candidates[0].duration) == PM:
                self._do_pm(waiting=False)
                return
            for op in candidates:
                if self.planner.process_request(RequestRes(self.id, op.name, op.input_resources, op.callback),
                                                self.res_manager):
                    self.current_op = op
                    self.status = NodeState.WAITING_FOR_RESOURCE
                    self.reserved[op.name] = self.reserved.get(op.name, 0) + (1000 if op.name == "Type0" else 1)
                    break

    def _idle_wait_epoch(self, now: float, d_next: float) -> bool:
        """Спільна логіка E2/E3 для простою (очікування ресурсу або блокування
        буфером). Повертає True, якщо політика обрала ТО."""
        if self.wait_since is None:
            self.wait_since = now
            self.next_wait_decision = now + DELTA_W
            return self._ask(E2, d_next) == PM
        if now >= self.next_wait_decision:
            self.next_wait_decision = now + DELTA_W
            return self._ask(E3, d_next) == PM
        return False

    def _sort_candidates(self, candidates):
        """Правило диспетчеризації (який тип операції запускати), задається лінією:
        D17 — за реальним попитом (за замовчуванням); list — порядок списку операцій;
        quota — найменша заповненість буфера; random — випадково (окремий RNG лінії)."""
        mode = getattr(self.line_ref, "dispatch", "D17") if getattr(self, "line_ref", None) else "D17"
        if mode == "D17":
            candidates.sort(key=lambda op: (-self._pending_demand(op), self._quota_ratio(op)))
        elif mode == "quota":
            candidates.sort(key=lambda op: self._quota_ratio(op))
        elif mode == "random":
            self.line_ref.dispatch_rng.shuffle(candidates)
        # "list" — без сортування (порядок self.operations)

    # ---- D17: вибір операції за реальним попитом (без змін логіки) ------------
    def _pending_demand(self, op):
        total = sum(r.input_resources.get(op.name, 0) for r in self.res_manager.requests)
        pool = self.res_manager.resources.get(op.name)
        stock = pool.number() if pool is not None else 0
        return max(0.0, total - stock)

    def _quota_ratio(self, op):
        cap = self.planner.caps.get(op.name)
        if not cap:
            return float("inf")
        return (self.planner.get_stock(op.name) + self.planner.get_in_progress(op.name)) / cap

    # ---- завершення операції: якість (Р3, Р10) -------------------------------
    def _complete(self, op):
        self.n_completed += 1
        self.ideal_done += op.duration
        defective = self.rng_q.random() < self.p_def()
        if defective:
            for res, qty in op.input_resources.items():
                if res in BULK:
                    self.waste[res] = self.waste.get(res, 0.0) + qty
            for res, qty in self.planner.get_bom_footprint(op.name).items():
                if res not in BULK:
                    self.defects[res] = self.defects.get(res, 0) + qty
            op.resourses = []
        else:
            self.n_good += 1
            op.create_resource(self.id)
            self.produced[op.name] = self.produced.get(op.name, 0) + (1000 if op.name == "Type0" else 1)
            if self.made_dron:
                self.made_dron(self.id)
        op.progress = 0.0
        op.state = OperationState.LETUP


# =============================================================================
# Лінія v0.3
# =============================================================================
class AgroDroneLineV3(AgroDroneLine):
    def __init__(self, master_seed: int, policy, n_filament_suppliers: int = 5,
                 buffer_drones: float = 2.0, type0_cap: float | str = "auto", classes: dict | None = None,
                 node_overrides: dict | None = None, crew: int | None = None, dispatch: str = "D17",
                 n_printers: int = 15, time_carry: bool = False):
        with contextlib.redirect_stdout(io.StringIO()):
            super().__init__(seed=None, lim_drones_made=10**9, T_pm_print_park=None,
                             verbose=False, n_filament_suppliers=n_filament_suppliers,
                             n_printers=max(15, n_printers))
        self.master_seed = master_seed
        self.drone_times: list[float] = []
        # правило диспетчеризації (тест резерву, Р36); окремий RNG — не зсуває інші потоки
        self.dispatch = dispatch
        # Р39: такт без втрати часу на переходах (False — поведінка до Р39)
        self.time_carry = time_carry
        self.dispatch_rng = np.random.default_rng([master_seed, 777])
        # ремонтна бригада (Р30): crew=None — необмежена (поведінка до Р30)
        self.crew = crew
        self.crew_busy = 0
        self.crew_queue: list = []
        self.crew_busy_time = 0.0
        # потоки RNG — за індексом вузла в ПОВНОМУ списку (до вилучення принтерів),
        # щоб у вузлів, що лишаються, потоки не залежали від n_printers (CRN, Р6)
        streams = make_streams(master_seed, len(self.units))
        orig_idx = {id(n): i for i, n in enumerate(self.units)}
        self.n_printers = n_printers
        if n_printers < 15:                       # Р38: вилучаємо принтери з найбільшими id
            drop = {int(i) for i in range(2 + n_printers, 17)}
            self.units = [n for n in self.units if int(n.id) not in drop]
        # буферні ліміти (Р22): запас на buffer_drones дронів для дискретних деталей
        per_drone = self.getResourcesForOneDrone("Type22")
        caps = {k: max(1, math.ceil(buffer_drones * v)) for k, v in per_drone.items()
                if k not in BULK and k != "Type22"}
        # Type0: "auto" (за замовчуванням з Р38) — на buffer_drones дронів за BOM, як і для
        # дискретних деталей; число — кеп у грамах (4000 — Р22, серії до a3-v0.10)
        caps["Type0"] = buffer_drones * per_drone["Type0"] if type0_cap == "auto" else float(type0_cap)
        self.planner = ProductionPlannerV3(self.planner, caps, self._stock, self._in_progress)
        self.planner.get_bom_footprint = self.get_bom_footprint
        for node in self.units:
            node.__class__ = ProductionNodeV3
            node.planner = self.planner
            node.upgrade_to_v3(class_config(node.id, classes, node_overrides), streams, orig_idx[id(node)], policy)
            node.line = self if crew is not None else None
            node.line_ref = self
            node.made_dron = None
        last = next(n for n in self.units if int(n.id) == 26)   # вузол 26 — випуск дрона
        last.made_dron = self._drone_done

    # ---- бригада (Р30) ----
    def crew_free(self) -> bool:
        return self.crew is None or self.crew_busy < self.crew

    def crew_acquire(self) -> bool:
        if not self.crew_free():
            return False
        self.crew_busy += 1
        return True

    def crew_release(self):
        self.crew_busy -= 1
        if self.crew_queue:                      # вільний технік одразу береться за ремонт із черги
            nxt = self.crew_queue.pop(0)
            self.crew_busy += 1
            nxt.awaiting_crew = False

    def allocate_pass(self):
        """Р39: прохід розподілу ресурсів посеред такту — та сама логіка і той самий
        пріоритет черги (за часом очікування), що й ResourceManager.tick, але без
        збільшення лічильників очікування."""
        rm = self.res_manager
        done = []
        for item in sorted(rm.requests, key=lambda x: x.wait_ticks, reverse=True):
            ok = all((rm.resources[r].number() if rm.resources[r] is not None else 0) >= q
                     for r, q in item.input_resources.items())
            if ok:
                used = {r: rm.resources[r].get(q) for r, q in item.input_resources.items()}
                done.append(item)
                item.callback(used)
        for r in done:
            rm.requests.remove(r)

    def _stock(self, name):
        pool = self.res_manager.resources.get(name)
        return pool.number() if pool is not None else 0

    def _in_progress(self, name):
        k = 1000 if name == "Type0" else 1
        return k * sum(1 for n in self.units
                       if n.current_op is not None and n.current_op.name == name
                       and n.current_op.state != OperationState.LETUP)

    def _drone_done(self, unit_id):
        self.number_drones_made += 1
        self.drone_times.append(self.current_time)

    def tick(self, elapsed_time):
        step = elapsed_time - self.current_time      # крок такту (не залежить від константи DT)
        self.current_time = elapsed_time
        if self.crew is not None:
            self.crew_busy_time += self.crew_busy * step
        self.res_manager.tick()
        for item in self.units:
            item.tick(elapsed_time, 0)


# =============================================================================
# Бейслайни (STATE_ACTION_SPEC §7.1)
# =============================================================================
class NoPM:                       # B0
    name = "B0_noPM"
    def decide(self, node, epoch, d_next): return CONTINUE

class PeriodicPM:                 # B1: напрацювання з останнього ТО ≥ T (на межі операцій, E1)
    def __init__(self, T_by_class: dict): self.T = T_by_class; self.name = "B1_periodic"
    def decide(self, node, epoch, d_next):
        T = self.T.get(node.cls)
        return PM if (epoch == E1 and T is not None and node.op_time_since_pm >= T) else CONTINUE

class RiskPM:                     # B2: p_f(наступна операція) > θ (на межі операцій, E1)
    def __init__(self, theta_by_class: dict): self.th = theta_by_class; self.name = "B2_risk"
    def decide(self, node, epoch, d_next):
        th = self.th.get(node.cls)
        return PM if (epoch == E1 and th is not None and node.rel.p_fail(d_next) > th) else CONTINUE

class OpportunisticPM:            # B3: під час очікування — якщо зношеність > x; інакше B2
    def __init__(self, x_by_class: dict, theta_by_class: dict):
        self.x = x_by_class; self.risk = RiskPM(theta_by_class); self.name = "B3_opportunistic"
    def decide(self, node, epoch, d_next):
        if epoch in (E2, E3):
            x = self.x.get(node.cls)
            return PM if (x is not None and node.wear_ratio() > x) else CONTINUE
        return self.risk.decide(node, epoch, d_next)

class RandomPM:                   # B4: випадкове ТО з заданою ймовірністю на рішення
    def __init__(self, p: float, seed: int): self.p = p; self.rng = np.random.default_rng(seed); self.name = "B4_random"
    def decide(self, node, epoch, d_next): return PM if self.rng.random() < self.p else CONTINUE


# =============================================================================
# Прогін і метрики
# =============================================================================
DT = 100.0

def _snapshot(line):
    s = {}
    for n in line.units:
        s[n.id] = {"dur": {k.name: v for k, v in n.state_durations.items()},
                   "n_pm": n.n_pm, "n_pm_w": n.n_pm_waiting, "n_fail": n.n_fail,
                   "n_comp": n.n_completed, "n_good": n.n_good, "ideal": n.ideal_done,
                   "waste0": n.waste.get("Type0", 0.0), "setup": n.t_setup,
                   "crew_wait": n.t_crew_wait}
    return {"t": line.current_time, "drones": line.number_drones_made, "nodes": s,
            "crew_busy_time": line.crew_busy_time, "crew": line.crew}

def run_v3(master_seed: int, policy, horizon_h: float, warmup_h: float, logger_path=None,
           classes=None, node_overrides=None, crew=None, dt: float = DT, **line_kw) -> dict:
    """Один прогін v0.3: горизонт і розігрів — у годинах модельного часу (Р19).
    Метрики рахуються на вікні [warmup, horizon]."""
    line = AgroDroneLineV3(master_seed, policy, classes=classes, node_overrides=node_overrides, crew=crew, **line_kw)
    log = None
    if logger_path:
        from EventLog import EventLogger
        log = EventLogger(logger_path, line, meta={"model": "v0.3", "seed": master_seed,
                                                    "policy": getattr(policy, "name", "?")})
    n_ticks = int(round(horizon_h * 3600 / dt))
    warm_tick = int(round(warmup_h * 3600 / dt))
    t, s0 = 0.0, None
    with contextlib.redirect_stdout(io.StringIO()):
        for k in range(1, n_ticks + 1):
            t += dt
            line.tick(t)
            if k == warm_tick:
                s0 = _snapshot(line)
    if log:
        log.close("horizon", t)
    s1 = _snapshot(line)
    return line_metrics(s0, s1, line)

def line_metrics(s0, s1, line) -> dict:
    T = s1["t"] - s0["t"]
    drones = s1["drones"] - s0["drones"]
    tot = {"run": 0.0, "setup": 0.0, "pm": 0.0, "cm": 0.0, "ideal": 0.0, "comp": 0, "good": 0,
           "fail": 0, "n_pm": 0, "n_pm_w": 0, "waste0": 0.0, "pm_cost": 0.0, "cm_cost": 0.0}
    per_cls = {}
    for nid, a1 in s1["nodes"].items():
        a0 = s0["nodes"][nid]
        d = {k: a1["dur"][k] - a0["dur"][k] for k in a1["dur"]}
        setup = a1["setup"] - a0["setup"]
        x = {"run": d["WORKING"] - setup, "setup": setup, "pm": d["MAINTENANCE"], "cm": d["REPAIR"],
             "ideal": a1["ideal"] - a0["ideal"], "comp": a1["n_comp"] - a0["n_comp"],
             "good": a1["n_good"] - a0["n_good"], "fail": a1["n_fail"] - a0["n_fail"],
             "n_pm": a1["n_pm"] - a0["n_pm"], "n_pm_w": a1["n_pm_w"] - a0["n_pm_w"],
             "waste0": a1["waste0"] - a0["waste0"]}
        x["pm_cost"] = x["n_pm"] * maint_cost(nid)
        x["cm_cost"] = x["fail"] * maint_cost(nid) * ECONOMICS["repair_cost_factor"]
        for k in tot: tot[k] += x[k]
        c = per_cls.setdefault(next(n.cls for n in line.units if n.id == nid), {k: 0.0 for k in tot})
        for k in tot: c[k] += x[k]
    def oee(x):
        # OEE (STATE_ACTION_SPEC §7.2): ТО, ремонт і переналагодження — втрати доступності;
        # простій через відсутність ресурсу чи заблокований буфер (зовнішній) не рахується
        load = x["run"] + x["setup"] + x["pm"] + x["cm"]
        A = x["run"] / load if load > 0 else float("nan")
        P = x["ideal"] / x["run"] if x["run"] > 0 else float("nan")
        Q = x["good"] / x["comp"] if x["comp"] > 0 else float("nan")
        return A, P, Q, A * P * Q
    A, P, Q, O = oee(tot)
    out = {"throughput_per_week": drones / (T / (7 * 24 * 3600)), "drones": drones, "window_h": T / 3600,
           "A": A, "P": P, "Q": Q, "OEE": O,
           "MTBF_h": (tot["run"] / tot["fail"] / 3600) if tot["fail"] else float("inf"),
           "failures": tot["fail"], "pm": tot["n_pm"], "pm_waiting": tot["n_pm_w"],
           "waste_Type0": tot["waste0"],
           # економіка (Р32), за рік вікна; базові ставки — line_config_v3.ECONOMICS
           "pm_cost": tot["pm_cost"], "cm_cost": tot["cm_cost"],
           "material_cost": tot["waste0"] * ECONOMICS["filament_cost_per_g"],
           "revenue": drones * ECONOMICS["drone_value"],
           "profit": drones * ECONOMICS["drone_value"] - tot["pm_cost"] - tot["cm_cost"]
                     - tot["waste0"] * ECONOMICS["filament_cost_per_g"],
           "crew": s1["crew"] if s1["crew"] is not None else "inf",
           "crew_util": ((s1["crew_busy_time"] - s0["crew_busy_time"]) / (s1["crew"] * T)) if s1["crew"] else float("nan"),
           "crew_wait_h": sum(a1["crew_wait"] - s0["nodes"][nid]["crew_wait"] for nid, a1 in s1["nodes"].items()) / 3600}
    for cls, x in sorted(per_cls.items()):
        a, p, q, o = oee(x)
        out[f"{cls}_A"], out[f"{cls}_OEE"] = a, o
        out[f"{cls}_MTBF_h"] = (x["run"] / x["fail"] / 3600) if x["fail"] else float("inf")
    return out
