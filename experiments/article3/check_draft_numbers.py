"""Звірка чисел тексту Results з аркушем results_numbers.md (WORKFLOW_CHECKLIST D).

Виписує всі десяткові числа з docs/article3/draft/03_results.md (без HTML-коментарів) і показує ті,
яких немає серед значень аркуша. Нормальні винятки — номери підрозділів, параметри сітки
(0.25, 0.35 …), пороги p (0.05) і p-значення, округлені до двох знаків. Решту — перевірити вручну.

Запуск: python experiments/article3/check_draft_numbers.py [--draft docs/article3/draft/03_results.md]
"""
import argparse, re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="docs/article3/draft/03_results.md")
    ap.add_argument("--sheet", default="docs/article3/draft/results_numbers.md")
    a = ap.parse_args()
    norm = lambda x: x.replace("−", "-").replace(" ", "").lstrip("+").lstrip("-")
    vals = set()
    for line in open(a.sheet, encoding="utf-8"):
        p = [x.strip() for x in line.split("|")]
        if len(p) > 3:
            vals.add(norm(p[2]))
    txt = re.sub(r"<!--.*?-->", "", open(a.draft, encoding="utf-8").read(), flags=re.S)
    nums = re.findall(r"(?<![\w.])[−+-]?\d[\d ]*\.\d+", txt)
    miss = sorted({n for n in nums if norm(n) not in vals})
    print(f"{len(nums)} чисел у тексті; не знайдено в аркуші ({len(miss)}): {miss}")


if __name__ == "__main__":
    main()
