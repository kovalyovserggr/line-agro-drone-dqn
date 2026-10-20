# Р48: розширення тестових seed Table 4 до 20 (100000–100019): додаються 100005–100019.
# Політики незмінні: B1* за B1STAR / --b1star, агенти DQN-TA — збережені model.pt.
# Запуск з кореня репо (PowerShell 7):  .\experiments\article3\run_r48.ps1
# Орієнтовно ~5 год на 5 процесах. Кожен крок пише CSV порядково — перерваний крок можна перезапустити окремо.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true   # зупинка, якщо python завершився з помилкою
$env:PYTHONPATH = "src"
$raw = "experiments/article3/raw"

python experiments/article3/eval_finalists.py --n-printers 12 15 17 19 --seeds-range 100005 100020 --workers 5 --out $raw/test20_finalists_L.csv
python experiments/article3/eval_finalists.py --scenario strong --n-printers 15 --finalists B0 B1:0.35 B1:0.5 B2:0.001 "B1*" --b1star '{"C1": 1.0, "C2": 0.25}' --seeds-range 100005 100020 --workers 5 --out $raw/test20_finalists_strong_L.csv
python experiments/article3/eval_finalists.py --scenario weak --n-printers 15 --finalists B0 B1:0.5 B1:0.75 B2:0.001 "B1*" --b1star '{"C1": 1.0, "C2": 0.5}' --seeds-range 100005 100020 --workers 5 --out $raw/test20_finalists_weak_L.csv

$dirs = foreach ($p in "np12", "np15", "np17", "np19", "strong", "weak") { foreach ($s in 1, 2, 3) { "$raw/dqn_ta_${p}_s$s" } }
python experiments/article3/eval_dqn.py --dirs $dirs --seeds-range 100005 100020 --workers 5 --out $raw/test20_dqn_L.csv
Write-Host "Р48 завершено"
