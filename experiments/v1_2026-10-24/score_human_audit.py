"""Score the team's hand labels against each judge (round 2, part 2).

    python score_human_audit.py human_audit_sheet.csv human_audit_key.csv

The sheet's `human_label` column must hold CORRECT or WRONG for every item the team labelled (blank rows are skipped).
Prints, per judge: items compared, percent agreement and Cohen's kappa. Standard library only.
"""
import csv
import sys


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def main(sheet, key):
    human = {r["id"]: r["human_label"].strip().upper() for r in csv.DictReader(open(sheet, newline=""))}
    human = {k: v == "CORRECT" for k, v in human.items() if v in ("CORRECT", "WRONG")}
    rows = list(csv.DictReader(open(key, newline="")))
    print(f"{len(human)} items labelled by hand")
    for col in [c for c in rows[0] if c.endswith("_label")]:
        pairs = [(human[r["id"]], r[col] == "CORRECT") for r in rows if r["id"] in human and r[col] in ("CORRECT", "WRONG")]
        if not pairs:
            print(f"{col}: no labels"); continue
        h, j = zip(*pairs)
        k = kappa(list(h), list(j))
        print(f"{col}: n={len(pairs)}  agreement {100 * sum(x == y for x, y in pairs) / len(pairs):.1f}%  "
              f"kappa {'n/a' if k is None else f'{k:.3f}'}  human says correct {100 * sum(h) / len(h):.1f}%, judge {100 * sum(j) / len(j):.1f}%")


if __name__ == "__main__":
    main(*sys.argv[1:3])
