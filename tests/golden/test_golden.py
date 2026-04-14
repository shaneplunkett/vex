"""Golden dataset integration tests — validate retrieval quality.

Runs golden queries against a populated test DB and checks that
expected results appear. Requires the full pipeline to have run
on test data (import → chunk → coref → embed).

Run with: pytest tests/golden/test_golden.py -v --run-golden
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

GOLDEN_FILE = Path(__file__).parent.parent / "fixtures" / "golden_queries.json"


def _load_golden_queries() -> list[dict]:
    return json.loads(GOLDEN_FILE.read_text())


@pytest.fixture(scope="module")
def golden_queries() -> list[dict]:
    return _load_golden_queries()


@pytest.mark.golden
async def test_golden_recall_queries(golden_queries: list[dict]) -> None:
    """Run all recall-type golden queries and check results."""
    from app.tools.query import recall

    recall_queries = [q for q in golden_queries if q.get("tool") == "recall"]
    passed = 0
    failed = 0
    results_log: list[str] = []

    for q in recall_queries:
        depth = q.get("depth", 2)
        chunk_type = q.get("chunk_type")
        results = await recall(q["query"], depth=depth, limit=10, chunk_type=chunk_type)

        # Check minimum results
        min_results = q.get("min_results", 1)
        max_results = q.get("max_results")

        ok = True
        issues: list[str] = []

        if len(results) < min_results:
            ok = False
            issues.append(f"expected >= {min_results} results, got {len(results)}")

        if max_results is not None and len(results) > max_results:
            ok = False
            issues.append(f"expected <= {max_results} results, got {len(results)}")

        # Check expected snippets in top results content
        if q.get("expected_snippets") and results:
            all_content = " ".join(r.get("content", "") for r in results[:5])
            for snippet in q["expected_snippets"]:
                if snippet.lower() not in all_content.lower():
                    # Also check entities
                    all_entities = " ".join(e.get("name", "") for r in results[:5] for e in r.get("entities", []))
                    if snippet.lower() not in all_entities.lower():
                        issues.append(f"expected snippet '{snippet}' not in top 5 results")
                        ok = False

        # Check unexpected snippets don't appear in top results
        if q.get("unexpected_snippets") and results:
            all_content = " ".join(r.get("content", "") for r in results[:5])
            for snippet in q["unexpected_snippets"]:
                if snippet.lower() in all_content.lower():
                    issues.append(f"unexpected snippet '{snippet}' found in top 5 results")
                    ok = False

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results_log.append(f"  [{status}] {q['id']}: {q['query'][:40]}" + (f" — {', '.join(issues)}" if issues else ""))

    report = f"\n{'=' * 60}\nGolden Recall Results: {passed} passed, {failed} failed\n{'=' * 60}\n"
    report += "\n".join(results_log)
    print(report)

    assert failed == 0, f"{failed} golden recall queries failed:\n{report}"


@pytest.mark.golden
async def test_golden_search_queries(golden_queries: list[dict]) -> None:
    """Run all search-type golden queries and check results."""
    from app.tools.query import search

    search_queries = [q for q in golden_queries if q.get("tool") == "search"]
    passed = 0
    failed = 0
    results_log: list[str] = []

    for q in search_queries:
        results = await search(q["query"], limit=10)

        min_results = q.get("min_results", 1)
        max_results = q.get("max_results")

        ok = True
        issues: list[str] = []

        if len(results) < min_results:
            ok = False
            issues.append(f"expected >= {min_results} results, got {len(results)}")

        if max_results is not None and len(results) > max_results:
            ok = False
            issues.append(f"expected <= {max_results} results, got {len(results)}")

        if q.get("expected_snippets") and results:
            all_content = " ".join(r.get("content", "") for r in results[:5])
            for snippet in q["expected_snippets"]:
                if snippet.lower() not in all_content.lower():
                    issues.append(f"expected snippet '{snippet}' not in top 5 results")
                    ok = False

        if q.get("unexpected_snippets") and results:
            all_content = " ".join(r.get("content", "") for r in results[:5])
            for snippet in q["unexpected_snippets"]:
                if snippet.lower() in all_content.lower():
                    issues.append(f"unexpected snippet '{snippet}' found in top 5 results")
                    ok = False

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        results_log.append(f"  [{status}] {q['id']}: {q['query'][:40]}" + (f" — {', '.join(issues)}" if issues else ""))

    report = f"\n{'=' * 60}\nGolden Search Results: {passed} passed, {failed} failed\n{'=' * 60}\n"
    report += "\n".join(results_log)
    print(report)

    assert failed == 0, f"{failed} golden search queries failed:\n{report}"
