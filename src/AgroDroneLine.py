from ProductionOperation import ProductionOperation
from ResourceManager import ResourceManager 
from Resource import  ResourcePool
from ProductionNode import ProductionNode, NodeState
from Efficiency import Efficiency, Reliability, NormalDistribution, UniformDistribution, TimeDependentExponential, TimeDependentLogNormal, IncreasingParameter, DecreasingParameter, ConstantParameter
from ProductionPlanner import ProductionPlanner
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # без GUI, тільки рендер у файл
import matplotlib.pyplot as plt


class SimulationComplete(Exception):
    """Контрольоване завершення прогону: піднімається замість sys.exit(), щоб
    experiments/run_parametric_study.py міг ловити завершення однієї репліки
    і продовжувати сітку в тому самому процесі (sys.exit() зупиняв би весь
    Python-процес після першого ж прогону)."""
    def __init__(self, elapsed_time: float, number_drones_made: int):
        self.elapsed_time = elapsed_time
        self.number_drones_made = number_drones_made
        super().__init__(f"Виготовлено {number_drones_made} дронів за {elapsed_time:.1f} с")


 #Клас виробничої лінії агродронів
class AgroDroneLine:
    def __init__(self, seed: int | None = None, lim_drones_made: int = 3,
                 T_pm_print_park: float | None = None, verbose: bool = True,
                 n_filament_suppliers: int = 1, n_printers: int = 15):
        # 🎲 Сідування RNG для відтворюваності репліки: усі stats.*.rvs() у
        # Efficiency.py семплюють через глобальний numpy-стан (не отримують
        # явний random_state), тому np.random.seed() перед побудовою лінії
        # робить прогін повністю відтворюваним для даного seed.
        if seed is not None:
            np.random.seed(seed)
        self.seed = seed
        self.verbose = verbose         # False у батч-експерименті — прибирає підсумкові print()
        self.current_time = 0.0        # оновлюється в tick(); потрібен для SimulationComplete
        self.units = []                # Список виробничих вузлів
        self.res_manager = ResourceManager() # Менеджер ресурсів лінії
        self.planner = ProductionPlanner() # Ініціалізація планувальника
        self.T_pm_print_park = T_pm_print_park  # T_pm для парку 3D-друку (вузли 2-16), None = природний розподіл
        # 🔧 D9/D10: вузол 1 (постачальник філаменту) структурно не встигає за
        # апетитом парку 3D-друку (0.556 од/с проти ~2.55 од/с при повному
        # навантаженні парку) — незалежно від T_pm. n_filament_suppliers > 1
        # додає паралельні копії вузла 1 (кожна — власний ProductionNode з
        # тими самими node_params1/reliability_params1), що моделює кілька
        # паралельних екструдерів філаменту на реальній лінії. За замовчуванням
        # =1 — поведінка, ідентична до фіксу D9 (не ламає старі прогони).
        self.n_filament_suppliers = n_filament_suppliers
        # Стаття 3, Р38: n_printers > 15 додає копії принтера (node_params2) з id 201+
        # у КІНЕЦЬ списку вузлів (індекси вузлів 1–26 і 101+ не зсуваються);
        # за замовчуванням 15 — поведінка без змін. Зменшення — у LineV3.
        self.n_printers = n_printers
        self.initialize_units(self.res_manager, T_pm_print_park=self.T_pm_print_park,
                               n_filament_suppliers=self.n_filament_suppliers,
                               n_printers=self.n_printers)  # Ініціалізація вузлів при створенні лінії
        self.initialize_resources()    # Ініціалізація складських запасів
        self.number_drones_made = 0    # Лічильник виготовлених дронів
        self.lim_drones_made = lim_drones_made  # Ліміт на виготовлення дронів
        self.bind_planner_delegates()  # Прив'язка делегатів планера
        self.time_point = 1000        # Інтервал часу для запису в історію
        self.memory_points: list = []  #

   
    def plot_history(self, node_id: int, op_name: str | None, keys_y: list[str]):
        node = next((n for n in self.units if int(n.id) == int(node_id)), None)
        if node is None:
            print(f"⚠️ Вузол {node_id} не знайдено")
            return

        # якщо задана операція → беремо її пам'ять
        if op_name:
            op = next((o for o in node.operations if str(o.name) == str(op_name)), None)
            if op is None:
                print(f"⚠️ Операція {op_name} у вузлі {node_id} не знайдена")
                return
            source_memory = op.memory
            title = f"Графік парамтрів ефективності операції - {op_name}, що функціонує на вузлі {node_id},"
        else:
            source_memory = node.memory
            title = f"Графік параметрів надійності вузла {node_id}"

        xs = getattr(self, "memory_points", list(range(len(source_memory))))
        plt.figure(figsize=(10, 6))

        for key_y in keys_y:
            ys = [snap.get(key_y, None) for snap in source_memory]
            if all(v is None for v in ys):
                print(f"⚠️ Ключ {key_y} не знайдено у пам'яті")
                continue
            plt.plot(xs, ys, marker="o", label=f"{key_y}")

        plt.xlabel("Час, с")
        plt.ylabel("value")
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"history_node{node_id}_{op_name or 'node'}.png")
        plt.close()
    
    def bind_planner_delegates(self):
        self.planner.resource_limits = self.getResourcesForOneDrone("Type22")
        self.planner.limit_drones = self.lim_drones_made

        print("📦 Ініціалізація планувальника")
        self.planner.get_state = self.get_state
        self.planner.get_reserved = self.get_reserved
        self.planner.get_produced = self.get_produced  # D11: throttle-гейт має звірятись з РЕАЛЬНО виробленим
        self.planner.get_defects = self.get_defects  # D15: компенсація браку від REPAIR-переривань мід-флайт
        self.planner.get_bom_footprint = self.get_bom_footprint  # D18: наскрізний BOM-слід для нарахування браку
        for node in self.units:
            node.planner = self.planner

    # повертає поточний статус вузла за його ID
    def get_state(self, id_node: int) -> NodeState:
        unit = next((u for u in self.units if u.id == id_node), None)
        return unit.get_status() if unit else NodeState.FAILURE
    # повертає загальну кількість ресурсів заданого типу на які існує запит усіма вузлами
    # (замовлено, ще не обовʼязково виготовлено — див. D11, для throttle-гейту НЕ використовується)
    def get_reserved(self, resource_name: str) -> int:
        return sum(unit.get_reserved(resource_name) for unit in self.units)
    # D11: скільки одиниць ресурсу РЕАЛЬНО виготовлено (усіма вузлами сумарно) —
    # саме це, а не get_reserved(), має обмежувати throttle-гейт у ProductionPlanner.
    def get_produced(self, resource_name: str) -> int:
        return sum(unit.get_produced(resource_name) for unit in self.units)
    # D15: скільки одиниць цього ресурсу списано в брак по всій лінії (сумарно
    # від REPAIR-переривань мід-флайт) — саме на цю величину throttle-гейт
    # розширює квоту, щоб дозволити компенсуючу заміну. Див. ProductionNode._interrupt_to().
    def get_defects(self, resource_name: str) -> int:
        return sum(unit.get_defects(resource_name) for unit in self.units)

    #Повертає словник {ресурс: кількість}, необхідний для виготовлення ОДНІЄЇ
    # одиниці заданого ресурсу — рекурсивно, наскрізь по всьому BOM-дереву
    # (включно з самим resource_name на множнику 1). Раніше ця логіка (unique_operations
    # + resolve()) жила лише всередині getResourcesForOneDrone("Type22") для
    # розрахунку квот. D18 (2026-09-11): винесено в окремий публічний метод, бо
    # тепер вона ще й потрібна ProductionNode._interrupt_to() — щоб при перериванні
    # операції мід-флайт нараховувати брак НАСКРІЗНО по всьому дереву сировини, що
    # пішла на неї, а не лише на її безпосередні input_resources (див. D18 у
    # DECISIONS.md: компенсація лише на один рівень вгору не встигала покрити
    # реальний сукупний брак на глибоких вузлах 19-22 — лінія впиралась у
    # вичерпання сировини Type1-15 вже ПІСЛЯ D15/D16/D17, застрягаючи, напр., на 8/10
    # дронів за 7·10⁶с без подальшого прогресу).
    def get_bom_footprint(self, resource_name: str) -> dict[str, int]:
        unique_operations: dict[str, dict[str, int]] = {}
        for unit in self.units:
            for op in unit.operations:
                if op.name not in unique_operations:
                    unique_operations[op.name] = op.input_resources

        def resolve(res_name: str, multiplier: int = 1, acc: dict[str, int] = None):
            if acc is None:
                acc = {}

            acc[res_name] = acc.get(res_name, 0) + multiplier

            if res_name not in unique_operations:
                return acc  # базовий ресурс

            for sub_res, qty in unique_operations[res_name].items():
                resolve(sub_res, multiplier * qty, acc)

            return acc

        return resolve(resource_name)

    #Повертає словник {ресурс: кількість}, необхідний для виготовлення одного екземпляра кінцевого продукту.
    def getResourcesForOneDrone(self, resource_name: str) -> dict[str, int]:
        return self.get_bom_footprint(resource_name)
    
    def info(self) -> str:
        info_str = "Статус виробничої лінії:\n"
        for unit in self.units:
            info_str += f"Вузол {unit.id}:\n"
            for name, num in unit.reserved.items():
                if num > 0: 
                    info_str += f"   {name}:  замовлено - {num},  вироблено - {unit.produced[name]}\n"
        return info_str
    
    def drone_made_callback(self, unit_id: int):
        self.number_drones_made += 1
        if self.number_drones_made >= self.lim_drones_made:
            if self.verbose:
                print(self.res_manager.info())
                print(self.info())
                print(f"🚜 Виготовлено агродрон в кількості {self.number_drones_made} штук!")
                self.plot_history(node_id=2, op_name=None, keys_y=["failure_frequency", "emergency_response"])
            # Контрольоване завершення репліки (замість sys.exit(), який зупинив би
            # увесь процес — неприйнятно для батч-прогону сітки T_pm × N реплікацій)
            raise SimulationComplete(self.current_time, self.number_drones_made)

    # Первинна ініціалізація вузлів та операцій виробничої лінії
    def initialize_units(self, resource_manager, T_pm_print_park: float | None = None,
                          n_filament_suppliers: int = 1, n_printers: int = 15):

        
        # === Розподіли ефективності ===
        efficiency_params1= {"productivity": 0.03, "utilization_rate": 0.90, "integration_capability": "Модульна конструкція","scaling_potential": "Можлива при збільшенні кількості вузлів","schedule_accuracy": (0.98, 0.02),"defect_rate": (0.015, 0.005),
                               "stability": (0.95, 0.99),"reconfiguration_speed": 10.0,"power_usage": 0.70, "maintenance_cost": 700.0,"profitability": 0.15, "certification_compliance": 0.99, "waste_generation": 0.05,"environmental_efficiency": 0.85}
        efficiency_params2= {"productivity": 0.00017, "utilization_rate": 0.80, "integration_capability": "Можливість встановлення додаткових екструдерів","scaling_potential": "Лінія містить 15 принтерів, можливо збільшувати кількість","schedule_accuracy": (0.95, 0.03),"defect_rate": (0.02, 0.007),
                              "stability": (0.92, 0.97), "reconfiguration_speed": 5.0, "power_usage": 0.75, "maintenance_cost": 500.0, "profitability": 0.18, "certification_compliance": 0.97, "waste_generation": 0.03,"environmental_efficiency": 0.90}
        efficiency_params17={"productivity": 1 / (3 * 60), "utilization_rate": 0.90, "integration_capability": "Додавання модулів для нових типів трансмісій", "scaling_potential": "Лінія містить 2 роботи, можливо додати ще", "schedule_accuracy": (0.97, 0.02), "defect_rate": (0.012, 0.004), 
                              "stability": (0.96, 0.98), "reconfiguration_speed": 2.0, "power_usage": 0.85, "maintenance_cost": 600.0, "profitability": 0.20, "certification_compliance": 0.98, "waste_generation": 0.02, "environmental_efficiency": 0.95}
        efficiency_params19={"productivity":1/(4*60),"utilization_rate":0.84,"integration_capability":"Адаптація для нових версій агрегату","scaling_potential":"Два вузли, можливе збільшення до 4","schedule_accuracy":(0.97,0.015),"defect_rate":(0.013,0.005),
                             "stability":(0.94,0.98),"reconfiguration_speed":3.0,"power_usage":0.80,"maintenance_cost":550.0,"profitability":0.18,"certification_compliance":0.97,"waste_generation":0.025,"environmental_efficiency":0.90}
        efficiency_params21={"productivity":1/(5*60),"utilization_rate":0.80,"integration_capability":"Адаптація для нових типів деталей","scaling_potential":"Один вузол, можливо інтегрувати допоміжні механізми","schedule_accuracy":(0.95,0.02),"defect_rate":(0.015,0.006),
                             "stability":(0.93,0.97),"reconfiguration_speed":4.0,"power_usage":0.78,"maintenance_cost":600.0,"profitability":0.17,"certification_compliance":0.96,"waste_generation":0.028,"environmental_efficiency":0.93}
        efficiency_params22={"productivity":1/(53*60),"utilization_rate":0.85,"integration_capability":"Адаптація до нових схем","scaling_potential":"Залежить від кількості операторів","schedule_accuracy":(0.95,0.02),"defect_rate":(0.01,0.004),
                             "stability":(0.95,0.99),"reconfiguration_speed":5.0,"power_usage":0.90,"maintenance_cost":400.0,"profitability":0.15,"certification_compliance":0.98,"waste_generation":0.02,"environmental_efficiency":0.85}
        efficiency_params23={"productivity":1/(49*60),"utilization_rate":0.80,"integration_capability":"Так","scaling_potential":"Залежить від кількості операторів","schedule_accuracy":(0.94,0.025),"defect_rate":(0.015,0.005),
                             "stability":(0.95,0.99),"reconfiguration_speed":6.0,"power_usage":0.85,"maintenance_cost":450.0,"profitability":0.12,"certification_compliance":0.97,"waste_generation":0.03,"environmental_efficiency":1.0}
        efficiency_params24={"productivity":1/(37*60),"utilization_rate":0.80,"integration_capability":"Так","scaling_potential":"Залежить від персоналу","schedule_accuracy":(0.94,0.025),"defect_rate":(0.015,0.005),
                             "stability":(0.95,0.99),"reconfiguration_speed":6.0,"power_usage":0.85,"maintenance_cost":500.0,"profitability":0.12,"certification_compliance":0.97,"waste_generation":0.03,"environmental_efficiency":1.0}
        efficiency_params25={"productivity":1/(60*60),"utilization_rate":0.75,"integration_capability":"Так (модульна конструкція)","scaling_potential":"Можливе розширення тестових зон","schedule_accuracy":(0.96,0.02),"defect_rate":(0.012,0.004),
                             "stability":(0.95,0.99),"reconfiguration_speed":10.0,"power_usage":0.80,"maintenance_cost":450.0,"profitability":0.14,"certification_compliance":0.99,"waste_generation":0.02,"environmental_efficiency":1.0}
        efficiency_params26={"productivity":1/(18*60),"utilization_rate":0.70, "integration_capability":"Так (можливе розширення тестових зон)", "scaling_potential":"Можлива при збільшенні кількості вузлів", "schedule_accuracy":(0.95,0.02),"defect_rate":(0.01,0.003),
                            "stability":(0.95,0.99),"reconfiguration_speed":10.0,"power_usage":0.78, "maintenance_cost":700.0,"profitability":0.13,"certification_compliance":0.99, "waste_generation":0.02,"environmental_efficiency":1.0}
        
        # D7 (2026-09-11): maintenance_interval μ підняно з 1000 до 10000 (×10,
        # σ пропорційно до тієї ж CV=0.2) — з μ=1000 вузол ішов на ТО раніше,
        # ніж встигав завершити одну операцію (duration_work=1800с), паралізуючи
        # єдиний постачальник філаменту. μ=10000 дає ~5.6 циклів роботи між ТО —
        # орієнтовний, а не виміряний показник (джерело реальних параметрів
        # надійності не закрито, див. README.md "Дані"); задокументовано в
        # docs/DECISIONS.md D7 і validation/validation_reliability_params.md.
        reliability_params1={"repairability":"Середня (стандартні комплектуючі)","preservation":1.0,"longevity":9.0,"failure_frequency":(0.01,0.0001),"maintenance_interval":(10000,2000,0.01),"reliability":(0.00005,0.000001),"emergency_response":0.0083}
        reliability_params2={"repairability":"Висока (стандартні комплектуючі, заміна екструдера)","preservation":1.0,"longevity":6.5,"failure_frequency":(0.02,0.0001),"maintenance_interval":(800,150,0.01),"reliability":(0.0001,0.000002),"emergency_response":0.0056}
        reliability_params17={"repairability":"Висока (стандартні роботизовані механізми)","preservation":1.0,"longevity":8.5,"failure_frequency":(0.015,0.0001),"maintenance_interval":(1200,250,0.01),"reliability":(0.00008,0.0000015),"emergency_response":0.0028}
        reliability_params19={"repairability":"Висока (стандартні механізми)","preservation":1.0,"longevity":8.0,"failure_frequency":(0.012,0.0001),"maintenance_interval":(1100,200,0.01),"reliability":(0.00008,0.0000015),"emergency_response":0.0033}
        reliability_params21={"repairability":"Висока (адаптивна заміна компонентів)","preservation":1.0,"longevity":7.5,"failure_frequency":(0.014,0.0001),"maintenance_interval":(1000,210,0.01),"reliability":(0.00012,0.0000025),"emergency_response":0.0042}
        # D7 (2026-09-11): вузли 22-25 — та сама проблема, що й вузол 1 (μ ТО
        # коротший за duration_work власної операції: 3180/2940/2220/3600с
        # відповідно, проти μ=1200). Піднято ×10 (μ=12000, σ пропорційно,
        # CV≈0.167 збережено) — орієнтовний, задокументований допуск, не
        # виміряне значення (див. D7/validation_reliability_params.md).
        # reliability_params2 (парк 3D-друку, вузли 2-16) НАВМИСНО не чіпаємо:
        # T_pm_print_park завжди явно задається сіткою експерименту й повністю
        # ЗАМІНЮЄ природний maintenance_interval (див. ProductionNode._maintenance_due),
        # тому цей параметр park-вузлів не впливає на дослідження 4.2.
        reliability_params22={"repairability":"Висока (стандартні електронні компоненти)","preservation":1.0,"longevity":8.5,"failure_frequency":(0.01,0.0001),"maintenance_interval":(12000,2000,0.01),"reliability":(0.00007,0.0000012),"emergency_response":0.0028}
        reliability_params23={"repairability":"Висока","preservation":1.0,"longevity":10.0,"failure_frequency":(0.000001,0.00000001),"maintenance_interval":(12000,2000,0.01),"reliability":(0.00006,0.000001),"emergency_response":0.0028}
        reliability_params24={"repairability":"Висока","preservation":1.0,"longevity":10.0,"failure_frequency":(0.000001,0.00000001),"maintenance_interval":(12000,2000,0.01),"reliability":(0.00009,0.0000018),"emergency_response":0.0028}
        reliability_params25={"repairability":"Середня (стандартні комплектуючі)","preservation":1.0,"longevity":8.5,"failure_frequency":(0.000001,0.00000001),"maintenance_interval":(12000,2000,0.01),"reliability":(0.00005,0.000001),"emergency_response":0.0028}
        reliability_params26={"repairability":"Середня (стандартні комплектуючі)","preservation":1.0,"longevity":8.5,"failure_frequency":(0.000001,0.00000001),"maintenance_interval":(1200,200,0.01),"reliability":(0.00004,0.0000008),"emergency_response":0.0028}
        
        op_params0={"op_name": "Type0", "description": " ", "resources": {"TypePL": 1},"duration_work": 1800, "duration_maintenance": 180}
        op_params1={"op_name": "Type1","description": " ","resources": {"Type0": 1538},"duration_work": 9047,"duration_maintenance":180}
        op_params2={"op_name":"Type2","description":" ","resources":{"Type0":803},"duration_work":4723,"duration_maintenance":180}
        op_params3={"op_name":"Type3","description":" ","resources":{"Type0":231},"duration_work":1359,"duration_maintenance":180}
        op_params4={"op_name":"Type4","description":" ","resources":{"Type0":210},"duration_work":1235,"duration_maintenance":180}
        op_params5={"op_name":"Type5","description":" ","resources":{"Type0":10},"duration_work":59,"duration_maintenance":180}
        op_params6={"op_name":"Type6","description":" ","resources":{"Type0":15},"duration_work":88,"duration_maintenance":180}
        op_params7={"op_name":"Type7","description":" ","resources":{"Type0":78},"duration_work":459,"duration_maintenance":180}
        op_params8={"op_name":"Type8","description":" ","resources":{"Type0":120},"duration_work":705,"duration_maintenance":180}
        op_params9={"op_name":"Type9","description":" ","resources":{"Type0":307},"duration_work":1805,"duration_maintenance":180}
        op_params10={"op_name":"Type10","description":" ","resources":{"Type0":174},"duration_work":1023,"duration_maintenance":180}
        op_params11={"op_name":"Type11","description":" ","resources":{"Type0":97},"duration_work":571,"duration_maintenance":180}
        op_params12={"op_name":"Type12","description":" ","resources":{"Type0":192},"duration_work":1129,"duration_maintenance":180}
        op_params13={"op_name":"Type13","description":" ","resources":{"Type0":1305},"duration_work":7676,"duration_maintenance":180}
        op_params14={"op_name":"Type14","description":" ","resources":{"Type0":762},"duration_work":4482,"duration_maintenance":180}
        op_params15={"op_name":"Type15","description":" ","resources":{"Type0":415},"duration_work":2441,"duration_maintenance":180}
        op_params16={"op_name":"Type16","description":" ","resources":{"Type1":1,"Type2":2,"Type3":4,"Type4":2,"Type13":2,"Type15":3},"duration_work":180,"duration_maintenance":120}
        op_params17={"op_name":"Type17","description":" ","resources":{"Type11":2,"Type12":6,"Type3":4,"Type4":1,"Type13":2},"duration_work":240,"duration_maintenance":120}
        op_params18={"op_name":"Type18","description":" ","resources":{"Type5":20,"Type6":7,"Type7":8,"Type8":12,"Type9":2,"Type10":4,"Type14":6},"duration_work":3180,"duration_maintenance":120}
        op_params19={"op_name":"Type19","description":" ","resources":{"Type16":3,"Type17":1},"duration_work":2940,"duration_maintenance":120}
        op_params20={"op_name":"Type20","description":" ","resources":{"Type18":1,"Type19":1},"duration_work":2220,"duration_maintenance":120}
        op_params21={"op_name":"Type21","description":" ","resources":{"Type20":1},"duration_work":3600,"duration_maintenance":120}
        op_params22={"op_name":"Type22","description":" ","resources":{"Type21":1},"duration_work":1080,"duration_maintenance":120}
        
        node_params1 = {"id": 1.0, "description": "Базовий виробничий вузол з модульною архітектурою та високою стабільністю","reliability": reliability_params1, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params0, efficiency_params1)]}
        node_params2 = {"id": 2.0, "description": "3D-друкарський вузол з гнучкою масштабованістю та екологічною ефективністю","reliability": reliability_params2, "duration_maintenance": 1200.0, "duration_repair": 10000.0,
                        "operations": [(op_params1,efficiency_params2 ), (op_params2, efficiency_params2 ),(op_params3,efficiency_params2 ),(op_params4,efficiency_params2 ),(op_params5,efficiency_params2 ),
                                       (op_params6,efficiency_params2 ), (op_params7,efficiency_params2 ), (op_params8, efficiency_params2 ), (op_params9,efficiency_params2 ),(op_params10,efficiency_params2 ),
                                       (op_params11,efficiency_params2 ),(op_params12,efficiency_params2 ),(op_params13, efficiency_params2 ), (op_params14,efficiency_params2 ),(op_params15,efficiency_params2 )] }
        node_params17 = { "id": 17.0, "description": "Вузол складання трансмісій", "reliability": reliability_params17, "duration_maintenance": 1200.0,"duration_repair": 10000.0, "operations": [(op_params16 , efficiency_params17)]}       
        node_params19 = { "id": 19.0, "description": "робот для складання ріжучого агрегату", "reliability": reliability_params19, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params17, efficiency_params19)] }
        node_params21 = { "id": 21.0, "description": "Універсальний робот складання агрегатів", "reliability": reliability_params21, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params16, efficiency_params21), (op_params17, efficiency_params21)] }
        node_params22 = { "id": 22.0, "description": "Вузол виготовлення електропроводки", "reliability": reliability_params22, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params18, efficiency_params22)] }
        node_params23 = { "id": 23.0, "description": "Вузол попереднього складання основних вузлів дрона", "reliability": reliability_params23, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params19, efficiency_params23)] }
        node_params24 = { "id": 24.0, "description": "Вузол остаточного складання дрона", "reliability": reliability_params24, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params20, efficiency_params24)] }
        node_params25 = { "id": 25.0, "description": "Вузол встановлення та налаштування програмного забезпечення", "reliability": reliability_params25, "duration_maintenance": 1200.0, "duration_repair": 10000.0, "operations": [(op_params21, efficiency_params25)] }
        node_params26 = { "id": 26.0, "description": "Вузол тестування та навчання дрона", "reliability": reliability_params26, "duration_maintenance": 1200.0, "duration_repair": 10000.0,"operations": [(op_params22, efficiency_params26)] }
        
        def make_reliability(params):
            return Reliability( repairability=params["repairability"], preservation=params["preservation"], longevity=params["longevity"],
                failure_frequency=TimeDependentExponential(rate=params["failure_frequency"][0],rate_of_change=params["failure_frequency"][1]),
                maintenance_interval=TimeDependentLogNormal(mu=params["maintenance_interval"][0], sigma=params["maintenance_interval"][1], rate_of_change=params["maintenance_interval"][2]),
                reliability=TimeDependentExponential(rate=params["reliability"][0], rate_of_change=params["reliability"][1]), emergency_response=ConstantParameter(params["emergency_response"]))

        def make_efficiency(params):
            return Efficiency(
                productivity=params["productivity"],
                utilization_rate=params["utilization_rate"],
                integration_capability=params["integration_capability"],
                scaling_potential=params["scaling_potential"],
                schedule_accuracy=NormalDistribution(params["schedule_accuracy"][0], params["schedule_accuracy"][1]),
                defect_rate=NormalDistribution(params["defect_rate"][0], params["defect_rate"][1]),
                stability=UniformDistribution(params["stability"][0], params["stability"][1]),
                reconfiguration_speed=DecreasingParameter(params["reconfiguration_speed"]),
                power_usage=IncreasingParameter(params["power_usage"]),
                maintenance_cost=IncreasingParameter(params["maintenance_cost"]),
                profitability=DecreasingParameter(params["profitability"]),
                certification_compliance=ConstantParameter(params["certification_compliance"]),
                waste_generation=IncreasingParameter(params["waste_generation"]),
                environmental_efficiency=DecreasingParameter(params["environmental_efficiency"])
            )

        def make_operation(op_params, efficiency_params, res_manager):
            return ProductionOperation(
                op_params["op_name"],
                op_params["description"],
                op_params["resources"],
                op_params["duration_work"],
                op_params["duration_maintenance"],
                make_efficiency(efficiency_params),
                res_manager
            )

        def make_node(node_params, res_manager, planner, T_pm=None):
            operations_list = [
                make_operation(op_params, efficiency_params, res_manager)
                for (op_params, efficiency_params) in node_params["operations"]
            ]
            return ProductionNode(
                id=node_params["id"],
                description=node_params["description"],
                reliability=make_reliability(node_params["reliability"]),
                duration_maintenance=node_params["duration_maintenance"],
                duration_repair=node_params["duration_repair"],
                operations=operations_list,
                res_manager=res_manager,
                planner=planner,
                T_pm=T_pm
            )

        node_1 = make_node(node_params1, resource_manager, self.planner)
        # 🔧 D9/D10: додаткові паралельні постачальники філаменту (Type0), якщо
        # запитано (n_filament_suppliers > 1) — копії node_params1/reliability_params1,
        # унікальні id 101, 102, ... (поза діапазоном 1-26, щоб не перетнутись
        # з жодною існуючою нумерацією вузлів чи фільтром "парк 2-16").
        extra_filament_suppliers = []
        for i in range(2, n_filament_suppliers + 1):
            extra_node = make_node(node_params1, resource_manager, self.planner)
            extra_node.id = 100 + i
            extra_filament_suppliers.append(extra_node)
        # 🔬 Вузли 2-16 — парк 3D-друку, предмет параметричного дослідження 4.2:
        # T_pm_print_park керує їхнім плановим ТО (None = природний LogNormal-розподіл)
        node_2 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_2.id = 2
        node_3 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_3.id = 3
        node_4 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_4.id = 4
        node_5 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_5.id = 5
        node_6 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_6.id = 6
        node_7 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_7.id = 7
        node_8 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_8.id = 8
        node_9 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_9.id = 9
        node_10 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_10.id = 10
        node_11 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_11.id = 11
        node_12 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_12.id = 12
        node_13 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_13.id = 13
        node_14 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_14.id = 14
        node_15 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_15.id = 15
        node_16 = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
        node_16.id = 16
        node_17 = make_node(node_params17, resource_manager, self.planner)
        node_17.id = 17
        node_18 = make_node(node_params17, resource_manager, self.planner)
        node_18.id = 18
        node_19 = make_node(node_params19, resource_manager, self.planner)
        node_19.id = 19
        node_20 = make_node(node_params19, resource_manager, self.planner)
        node_20.id = 20
        node_21 = make_node(node_params21, resource_manager, self.planner)
        node_21.id = 21
        node_22 = make_node(node_params22, resource_manager, self.planner)
        node_22.id = 22
        node_23 = make_node(node_params23, resource_manager, self.planner)
        node_23.id = 23
        node_24 = make_node(node_params24, resource_manager, self.planner)
        node_24.id = 24
        node_25 = make_node(node_params25, resource_manager, self.planner)
        node_25.id = 25
        node_26 = make_node(node_params26, resource_manager, self.planner)
        node_26.id = 26

        node_26.made_dron = self.drone_made_callback
        print(f"✅ додавання вузлів почато.")
        #Автоматично додає всі вузли виробничої лінії.
        self.units = [
            # 🔹 Додавання всіх вузлів з використанням оновлених параметрів
            node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_9, node_10, node_11,
            node_12, node_13, node_14, node_15, node_16, node_17, node_18, node_19, node_20, node_21,
            node_22, node_23, node_24, node_25, node_26
        ] + extra_filament_suppliers  # D9/D10: додаткові постачальники філаменту (id 101+), якщо запитано
        # Р38: додаткові принтери (id 201+), якщо n_printers > 15
        for i in range(1, n_printers - 15 + 1):
            extra_printer = make_node(node_params2, resource_manager, self.planner, T_pm=T_pm_print_park)
            extra_printer.id = 200 + i
            self.units.append(extra_printer)
        print(f"✅ додавання вузлів завершено.")

    # Первинна ініціалізація ресурсів на складі
    def initialize_resources(self):
        self.res_manager.resources = {
            "TypePL": ResourcePool(1000000.0),  # Пластикові відходи 
            "Type0": ResourcePool(0.0),  # феламент для 3D-друку (пластик)
            "Type1": ResourcePool([]),  # Деталі корпусу
            "Type2": ResourcePool([]),  # Деталі кріплення
            "Type3": ResourcePool([]),  # Механічні компоненти
            "Type4": ResourcePool([]),  # Елементи трансмісії
            "Type5": ResourcePool([]),  # Електронні компоненти
            "Type6": ResourcePool([]),  # Датчики
            "Type7": ResourcePool([]),  # Актори
            "Type8": ResourcePool([]),  # Кабелі та проводка
            "Type9": ResourcePool([]),  # Батареї
            "Type10": ResourcePool([]),  # Пропелери
            "Type11": ResourcePool([]),  # Ріжучі механізми
            "Type12": ResourcePool([]),  # Робочі агрегати
            "Type13": ResourcePool([]),  # Плати управління
            "Type14": ResourcePool([]),  # Корпусні елементи
            "Type15": ResourcePool([]),  # Кришки та захисні елементи
            "Type16": ResourcePool([]),  # Трансмісії
            "Type17": ResourcePool([]),  # Робочі частини
            "Type18": ResourcePool([]),  # Електросистеми
            "Type19": ResourcePool([]),  # Напівфабрикати дронів
            "Type20": ResourcePool([]),  # Зібрані дрони без ПЗ
            "Type21": ResourcePool([]),  # Дрони з ПЗ, що пройшли інсталяцію
            "Type22": ResourcePool([])   # Готові до використання дрони після тестування
        }

        for item in self.units:
            item.res_manager = self.units[0].res_manager
            for op in item.operations:
                op.res_manager = self.units[0].res_manager

    def tick(self, elapsed_time):
        self.current_time = elapsed_time   # потрібен drone_made_callback для SimulationComplete

        self.res_manager.tick()
        if elapsed_time % self.time_point == 0:
            self.memory_points.append(elapsed_time)
            for item in self.units: # запис у історію лише раз на N тіків
                item.tick(elapsed_time, 1)
        else:
            for item in self.units: # без запису історії
                item.tick(elapsed_time, 0)

        if self.verbose and elapsed_time % 50000 == 0:
            print(f"⏱️ Час: {elapsed_time:.2f} сек")
        # ПРИМІТКА: попередній жорсткий sys.exit() на elapsed_time==300000 прибрано —
        # він тихо обрізав би будь-який прогін, чий makespan перевищує цей поріг
        # (реалістично при поганих T_pm/реальних відмовах), спотворюючи результати
        # параметричного дослідження. Горизонт і критерій завершення репліки тепер
        # задає experiments/run_parametric_study.py (ліміт тіків + SimulationComplete).

