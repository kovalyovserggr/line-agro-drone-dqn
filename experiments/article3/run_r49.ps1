# Р49: робастність до хибної специфікації розподілу часу до відмови.
# «Правда» — логнормальний розподіл зносових складових (ті самі середнє і CV, що й Weibull);
# політики заморожені (налаштовані / навчені на Weibull). Еталон — B1, перерахований на логнормальній «правді».
# Сценарії: помірна 15 (вузьке місце — друк), помірна 17 (екструзія), сильна 15. Тест — 20 seed 100000–100019.
# Запуск з кореня репо (PowerShell 7):  .\experiments\article3\run_r49.ps1     Орієнтовно ~4 год на 5 процесах.
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true   # зупинка, якщо python завершився з помилкою
$env:PYTHONPATH = "src"
$raw = "experiments/article3/raw"

# 1. «Поінформований» B1: налаштування m на логнормальній правді (seed 1000–1004)
python experiments/article3/baselines_tune.py --scenario moderate --crew 1 --n-printers 15 17 --policies B1 --b1 0.25 0.35 0.5 0.75 1.0 1.5 --ttf lognormal --workers 5 --out $raw/tune_logn_moderate_L.csv
python experiments/article3/baselines_tune.py --scenario strong --crew 1 --n-printers 15 --policies B1 --b1 0.25 0.35 0.5 0.75 1.0 1.5 --ttf lognormal --workers 5 --out $raw/tune_logn_strong_L.csv

# 2. Заморожені правила на логнормальній правді (B1 — m* з Weibull-налаштування)
python experiments/article3/eval_finalists.py --ttf lognormal --n-printers 15 17 --finalists B0 B1:0.5 B2:0.001 "B1*" --seeds-range 100000 100020 --workers 5 --out $raw/test_logn_moderate_L.csv
python experiments/article3/eval_finalists.py --ttf lognormal --scenario strong --n-printers 15 --finalists B0 B1:0.35 B2:0.001 "B1*" --b1star '{"C1": 1.0, "C2": 0.25}' --seeds-range 100000 100020 --workers 5 --out $raw/test_logn_strong_L.csv

# 3. Заморожені агенти DQN-TA (без перенавчання)
$dirs = foreach ($p in "np15", "np17", "strong") { foreach ($s in 1, 2, 3) { "$raw/dqn_ta_${p}_s$s" } }
python experiments/article3/eval_dqn.py --ttf lognormal --dirs $dirs --seeds-range 100000 100020 --workers 5 --out $raw/test_logn_dqn_L.csv
Write-Host "Р49, кроки 1–3 завершено. Крок 4 (тест поінформованого B1 з m*_logn) — після вибору m* скриптом."
