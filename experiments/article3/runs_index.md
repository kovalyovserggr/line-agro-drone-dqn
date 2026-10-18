# runs_index.md — Стаття 3

Один рядок на кожну серію запусків. Наратив і трактування — `RESULTS_LOG.md`.

| run_id | Дата | Тег | Команда (повна) | Seed-и середовища | Seed-и агента | Параметри | Файли | Статус |
|---|---|---|---|---|---|---|---|---|
| a3-phase0-profile | 2026-10-17 | eafe93d | `python tests/test_event_log.py 1000 1003` (+ профілювання cProfile однієї репліки) | 1000, 1001, 1003 | — | T_pm=6000, n_filament_suppliers=5, lim=10, max_ticks=50000 | у хмарному середовищі асистента; сирих файлів не збережено (діагностика) | завершено |
| a3-phase1-relmodel | 2026-10-17 | (до коміту) | `python tests/test_reliability_v3.py` | фіксовані в тесті (7, 55, 100–350, 1000–1011) | — | η=1000, d_pm=20, d_cm=200, CV 0.1/0.4 | тест детермінований, сирих файлів немає | завершено, усі перевірки OK |
| a3-phase1-linev3 | 2026-10-17 | (до коміту) | `python tests/test_line_v3.py` | 7, 11, 101–104, 201–206 | — | тимчасові параметри класів (line_config_v3.py); horizon 1500/8760/4000 год | тест детермінований, сирих файлів немає | завершено, L1–L6 OK |
| a3-sn-pilot60k | 2026-10-17 | a3-v0.3-line + незакомічені DQNAgentV3.py, single_node_check.py | `python experiments/article3/single_node_check.py --betas 1.0 1.5` і `--betas 2.5 3.5` (паралельно) | середовище 10001–10003 | 1, 2, 3 | 60 000 кроків, lr 5e-4, batch 64, τ=10η, d=0.05η, d_pm=0.02η, d_cm=0.2η | raw/single_node_pilot_60k.csv | пілот, 6/12 за критеріями |
| a3-sn-diag200k | 2026-10-17 | те саме | `... --betas 1.5 --agent-seeds 3 --steps 200000 --lr-end 2e-5 --batch 128` і `--betas 2.5 --agent-seeds 2 ...` | 10003, 10002 | 3, 2 | 200 000 кроків, lr 5e-4→2e-5, batch 128 | лише лог (діагностика) | 2/2 OK |
| a3-sn-official-L | 2026-10-17/18 | a3-v0.4-singlenode | `python experiments/article3/single_node_check.py --betas 1.0 3.5 --steps 200000 --lr-end 2e-5 --batch 128 --out experiments/article3/raw/sn_official_local_1035.csv` і те саме з `--betas 1.5 2.5 ... sn_official_local_1525.csv` (паралельно) | 10001–10003 | 1, 2, 3 | 200 000 кроків, lr 5e-4→2e-5, batch 128, τ=10η, d=0.05η, d_pm=0.02η, d_cm=0.2η | raw/sn_official_local_1035.csv, raw/sn_official_local_1525.csv | L (i5-9600K, Win10, Py 3.14, torch 2.10); 10/12 за критеріями |
| a3-sn-official-C | 2026-10-17 | a3-v0.4-singlenode | ті самі команди | 10001 | 1 | те саме | лише лог (серію обірвано після 1-го агента в кожному процесі) | C (Xeon 2 vCPU, Linux, Py 3.13, torch 2.14); контроль відтворюваності: 2/2 збіг з L |
| a3-horizon-B0 | 2026-10-18 | a3-r1-singlenode + experiments/article3/horizon_study.py (до коміту) | `python experiments/article3/horizon_study.py --seeds 1 2 3 4 --years 3 --out experiments/article3/raw/horizon_B0.csv` | 1–4 | — | B0, тимчасові параметри класів | raw/horizon_B0.csv | C; завершено |
