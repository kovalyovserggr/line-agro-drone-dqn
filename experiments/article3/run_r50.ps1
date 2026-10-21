# Р50: чутливість до параметрів без джерела (η зносу C2 і C4) + дооцінка абляції на 20 seed.
# Запуск з кореня репо (PowerShell 7):  pwsh -ExecutionPolicy Bypass -File .\experiments\article3\run_r50.ps1
# Орієнтовно ~7 год (6 ядер). Кроки незалежні: перерваний крок можна перезапустити окремо.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$env:PYTHONPATH = "src"
$raw = "experiments/article3/raw"

# 1. η_C2 ∈ {750; 3000} (база 1500): повне переналаштування бейслайнів і перенавчання DQN-TA, помірна, 15 принтерів
python experiments/article3/sens_pipeline.py --scenario moderate+C2eta750 --tag c2eta750 --workers 3 --workers-after 5
python experiments/article3/sens_pipeline.py --scenario moderate+C2eta3000 --tag c2eta3000 --workers 3 --workers-after 5

# 2. η_C4 ∈ {2500; 10000} (база 5000): заморожені політики бази (ТО C4 на горизонті майже не спрацьовує, Р35)
foreach ($e in 2500, 10000) {
    python experiments/article3/eval_finalists.py --scenario "moderate+C4eta$e" --n-printers 15 --finalists B0 B1:0.5 "B1*" --seeds-range 100000 100020 --workers 5 --out $raw/sens_c4eta${e}_test_L.csv
    $dirs = foreach ($s in 1, 2, 3) { "$raw/dqn_ta_np15_s$s" }
    python experiments/article3/eval_dqn.py --scenario "moderate+C4eta$e" --dirs $dirs --seeds-range 100000 100020 --workers 5 --out $raw/sens_c4eta${e}_test_dqn_L.csv
}

# 3. Абляція з локальною винагородою (Р41/Р43) — дооцінка на seed 100005–100019
$abl = foreach ($p in "dqn_g_np12", "dqn_g_t300", "dqn_g_np17") { foreach ($s in 1, 2, 3) { "$raw/${p}_s$s" } }
python experiments/article3/eval_dqn.py --dirs $abl --seeds-range 100005 100020 --workers 5 --out $raw/test20_ablation_L.csv
Write-Host "Р50 завершено"
