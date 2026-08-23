#!/usr/bin/env python3
"""
check_claim_ledger.py — validate a claim ledger against this project's schema
(see docs/governance/review-protocol.md).

Checks (deterministic):
  * required columns present: claim_id, claim, label, locator
  * every label is from the canonical set
  * claim_id is unique and non-empty
  * locator present unless label is Unchecked or Conjecture
  * if a grade is given, it is one of the allowed grades
  * (warn) load-bearing claims (grade High/Moderate, or label Corroborated/Provisional)
    that lack a severity note

Parses the first GitHub-flavoured Markdown table whose header contains the
required columns. Exit non-zero on any error, so it can gate a run.

Usage:
    python check_claim_ledger.py reports/<run>/claim-ledger.md
    python check_claim_ledger.py --init        # print a blank template
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

# Canonical labels — each names the kind of check performed; none means "true".
LABELS = {
    "Resolved",
    "Concordant",
    "Reported",
    "Corroborated",
    "Provisional",
    "Conjecture",
    "Contested",
    "Unchecked",
}
GRADES = {"High", "Moderate", "Low", "Very-Low", ""}
NO_LOCATOR_OK = {"Unchecked", "Conjecture"}
REQUIRED = ["claim_id", "claim", "label", "locator"]

TEMPLATE = """| claim_id | claim | label | grade | locator | severity | notes |
|---|---|---|---|---|---|---|
| C-001 | One falsifiable sentence. | Reported | Moderate | https://arxiv.org/abs/XXXX.XXXXX | What would falsify this; was it sought? | downgrade reasons / caveats |
"""


def parse_md_table(text: str):
    lines = text.splitlines()
    # find header row that contains claim_id
    for i, ln in enumerate(lines):
        if not ln.strip().startswith("|"):
            continue
        cells = [c.strip().lower() for c in ln.strip().strip("|").split("|")]
        if "claim_id" in cells:
            header = cells
            break
    else:
        return None, []
    # The table ends at the first line that is not a row. Collecting every
    # pipe-prefixed line in the document instead would splice any later
    # unrelated table onto the ledger, and its cells would then be graded as
    # claims against a header they were never written under.
    body = []
    for ln in lines[i + 1 :]:
        if not ln.strip().startswith("|"):
            break
        body.append(ln)
    out = []
    for ln in body:
        if set(ln.strip()) <= set("|-: "):  # separator row
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        out.append(dict(zip(header, cells)))
    return header, out


def check(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    header, rows = parse_md_table(text)
    errors, warnings = [], []
    if header is None:
        return (
            ["no claim-ledger table found (need a header row containing 'claim_id')"],
            [],
            0,
        )
    for col in REQUIRED:
        if col not in header:
            errors.append(f"missing required column: {col}")
    if errors:
        return errors, warnings, 0

    seen = set()
    for n, row in enumerate(rows, 1):
        cid = row.get("claim_id", "")
        label = row.get("label", "")
        grade = row.get("grade", "")
        locator = row.get("locator", "")
        severity = row.get("severity", "")
        tag = cid or f"row {n}"
        if not cid:
            errors.append(f"{tag}: empty claim_id")
        elif cid in seen:
            errors.append(f"{tag}: duplicate claim_id")
        seen.add(cid)
        if label not in LABELS:
            errors.append(f"{tag}: label '{label}' not in {sorted(LABELS)}")
        if grade not in GRADES:
            errors.append(f"{tag}: grade '{grade}' not in {sorted(GRADES - {''})}")
        if label not in NO_LOCATOR_OK and not locator:
            errors.append(f"{tag}: label '{label}' requires a locator")
        load_bearing = grade in {"High", "Moderate"} or label in {
            "Corroborated",
            "Provisional",
        }
        if load_bearing and not severity:
            warnings.append(f"{tag}: load-bearing claim missing a severity note")
    return errors, warnings, len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Validate a claim ledger.")
    ap.add_argument("file", nargs="?", type=Path)
    ap.add_argument(
        "--init", action="store_true", help="print a blank ledger template and exit"
    )
    args = ap.parse_args(argv)
    if args.init:
        print(TEMPLATE)
        return 0
    if not args.file:
        ap.error("provide a ledger file, or use --init")
    errors, warnings, n = check(args.file)
    print(f"=== {args.file} ===  ({n} claim row(s))")
    for e in errors:
        print(f"  ERROR  {e}")
    for w in warnings:
        print(f"  warn   {w}")
    if not errors:
        print("  PASS (schema valid)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
