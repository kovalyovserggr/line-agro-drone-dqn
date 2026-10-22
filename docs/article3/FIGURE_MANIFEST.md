# FIGURE_MANIFEST.md — Стаття 3

Порядок рядків = порядок появи в тексті статті. Рисунки будуються лише
скриптами з сирих даних.

| № | Рисунок / таблиця | Скрипт | Дані (run_id) | Тег | Статус |
|---|---|---|---|---|---|
| Fig. 1 | Схема підходу: модель, SMDP, агент, бейслайни | — (схема) | — | — | заплановано |
| Fig. 2 | Структура лінії з класами вузлів | — (схема) | — | — | заплановано |
| Fig. 3 | h(t) за класами, ефект ТО за Kijima | TBD | — | — | заплановано |
| Fig. 4 | Криві навчання DQN-TA, 3 агенти, 4 сценарії (fig_learning) | experiments/article3/make_figures.py | a3-dqn-ta-map-L | a3-v0.19-tamap | чернетка |
| Fig. 5 | Карта сценаріїв: прибуток і різниця до B1* за кількістю принтерів, з агентом, вибраним за валідацією (fig_scenario_map) | experiments/article3/make_figures.py | a3-test-finalists-L, a3-test20-L, a3-dqn-ta-map-L, a3-dqn-grid-* + a3-sens-eta-L (абляція, 20 seed), a3-val-agents-L | a3-v0.29-sens | чернетка |
| Fig. 5b | Вісь деградації (15 принтерів): цінність ТО і різниця до B1* (fig_degradation) | experiments/article3/make_figures.py | a3-test-finalists-sw-L, a3-test-b1star-sw-L, a3-test20-L, a3-dqn-ta-sw-L, a3-val-agents-L | a3-v0.25-test20 | чернетка |
| Fig. 5c | Робастність: Δ до B1* заморожених політик при вейбулівській і логнормальній «правді» (fig_robust) | experiments/article3/make_figures.py (після make_table4.py) | a3-test20-L, a3-logn-L | a3-v0.25-test20 | чернетка |
| Fig. 5d | Чутливість до η_C2 (переналаштування) і η_C4 (заморожені політики): Δ до B1* (fig_sens) | experiments/article3/make_figures.py (після make_sens.py) | a3-sens-eta-L, a3-test20-L | a3-v0.29-sens | чернетка |
| Fig. 5a | Абляція винагороди: OEE екструзії і друку за 17 принтерів (fig_ablation17) | experiments/article3/make_figures.py | a3-test-finalists-L, a3-dqn-grid-np12-np17-L, a3-dqn-ta-map-L | a3-v0.19-tamap | чернетка |
| Fig. 6 | Теплова карта чутливості (β × d_pm/d_cm) | TBD | TBD | TBD | заплановано |
| Fig. 7 | Вік у момент ТО проти аналітичного оптимуму | TBD | TBD | TBD | заплановано |
| Fig. 8 | Фрагмент Ганта: DQN проти найкращого бейслайну | TBD | TBD | TBD | заплановано |
| Table 1 | Класи обладнання і параметри з джерелами | — | — | — | заплановано |
| Table 2 | Специфікація SMDP | — | — | — | заплановано |
| Table 3 | Гіперпараметри | — | — | — | заплановано |
| Table 4 | Основні результати (20 тестових seed): 6 сценаріїв × B0/B1/B2/B3+/B1*/DQN-TA (середнє 3; вибраний за валідацією), парні Δ до B1* ± SE (table4.csv, table4.md, table4_agent_selection.csv) | experiments/article3/make_table4.py | a3-test-finalists-L, a3-test-finalists-sw-L, a3-test-b1star-sw-L, a3-dqn-ta-map-L, a3-dqn-ta-sw-L, a3-val-agents-L, a3-test20-L | a3-v0.25-test20 | чернетка |
| Table 5 | Робастність до хибної моделі відмов (логнормальна «правда», заморожені політики; table_robust_lognormal.{csv,md}) | experiments/article3/make_table4.py | a3-logn-L, a3-test20-L | a3-v0.25-test20 | чернетка |
| Table 6 | Чутливість до параметрів без джерела: η_C2 ∈ {750; 1 500; 3 000}, η_C4 ∈ {2 500; 5 000; 10 000} (table_sens_eta.{csv,md}) | experiments/article3/make_sens.py | a3-sens-eta-L, a3-test20-L | a3-v0.29-sens | чернетка |

Рисунки: `docs/article3/figures/` (PNG 300 dpi + PDF). Палітра — категоріальна, перевірена валідатором
(CVD ΔE ≥ 9.2, normal-vision ≥ 27.6; aqua < 3:1 до тла → кожна серія має також власний маркер).
