"""Regression tests for the claim-ledger Markdown table parser."""

from scripts.check_claim_ledger import parse_md_table


def test_parser_stops_at_the_end_of_the_first_ledger_table():
    text = """# Claims
| claim_id | claim | label | grade | locator | severity | notes |
|---|---|---|---|---|---|---|
| C-001 | Main claim | Reported | Low | source | | |

## Unrelated table
| metric | value |
|---|---|
| latency | 10ms |
"""

    header, rows = parse_md_table(text)

    assert header == [
        "claim_id",
        "claim",
        "label",
        "grade",
        "locator",
        "severity",
        "notes",
    ]
    assert rows == [
        {
            "claim_id": "C-001",
            "claim": "Main claim",
            "label": "Reported",
            "grade": "Low",
            "locator": "source",
            "severity": "",
            "notes": "",
        }
    ]
