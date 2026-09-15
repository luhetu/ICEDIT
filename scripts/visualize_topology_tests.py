"""Run the edit-topology tests and write a standalone visual HTML report."""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sys
import time
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


GROUP_LABELS = {
    "test_contract_schema": "Contract schema",
    "test_demo_templates": "Demo templates",
    "test_dataset_runner": "Dataset planning",
    "test_dev_validation": "Development checks",
    "test_executor": "Contract executor",
    "test_model_parser": "VLM parser",
    "test_parser_trial": "Rule parser",
    "test_parser_v02": "V0.2 task coverage",
    "test_spatial_mask": "Spatial masks",
}


class RecordingResult(unittest.TestResult):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, object]] = []
        self._started: dict[str, float] = {}

    def startTest(self, test) -> None:
        super().startTest(test)
        self._started[test.id()] = time.perf_counter()

    def _record(self, test, status: str, detail: str = "") -> None:
        test_id = test.id()
        elapsed = time.perf_counter() - self._started.get(test_id, time.perf_counter())
        module = test.__class__.__module__.split(".")[-1]
        self.records.append(
            {
                "id": test_id,
                "group": GROUP_LABELS.get(module, module),
                "name": test._testMethodName,
                "description": test.shortDescription() or test._testMethodName.replace("_", " "),
                "status": status,
                "duration_ms": round(elapsed * 1000, 2),
                "detail": detail,
            }
        )

    def addSuccess(self, test) -> None:
        super().addSuccess(test)
        self._record(test, "PASS")

    def addFailure(self, test, err) -> None:
        super().addFailure(test, err)
        self._record(test, "FAIL", self._exc_info_to_string(err, test))

    def addError(self, test, err) -> None:
        super().addError(test, err)
        self._record(test, "ERROR", self._exc_info_to_string(err, test))

    def addSkip(self, test, reason) -> None:
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)


def _status_counts(records: list[dict[str, object]]) -> dict[str, int]:
    return {
        status: sum(record["status"] == status for record in records)
        for status in ("PASS", "FAIL", "ERROR", "SKIP")
    }


def _render_rows(records: list[dict[str, object]]) -> str:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["group"]), []).append(record)

    sections = []
    for group, rows in groups.items():
        passed = sum(row["status"] == "PASS" for row in rows)
        rendered_rows = []
        for row in rows:
            detail = str(row["detail"])
            detail_html = ""
            if detail:
                detail_html = (
                    "<details><summary>Failure detail</summary><pre>"
                    + html.escape(detail)
                    + "</pre></details>"
                )
            rendered_rows.append(
                "<tr>"
                f"<td><span class='status {str(row['status']).lower()}'>{row['status']}</span></td>"
                f"<td><strong>{html.escape(str(row['description']))}</strong>"
                f"<code>{html.escape(str(row['name']))}</code>{detail_html}</td>"
                f"<td class='duration'>{float(row['duration_ms']):.2f} ms</td>"
                "</tr>"
            )
        sections.append(
            f"<section><div class='section-title'><h2>{html.escape(group)}</h2>"
            f"<span>{passed}/{len(rows)} passed</span></div>"
            "<table><thead><tr><th>Result</th><th>Test</th><th>Time</th></tr></thead>"
            f"<tbody>{''.join(rendered_rows)}</tbody></table></section>"
        )
    return "".join(sections)


def render_report(records: list[dict[str, object]], elapsed: float) -> str:
    counts = _status_counts(records)
    total = len(records)
    passed = counts["PASS"]
    percent = round(100 * passed / total) if total else 0
    generated = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    overall = "All checks passed" if passed == total else "Attention needed"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Edit Topology test report</title>
<style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#637083; --line:#d9dee6;
  --surface:#ffffff; --page:#f4f6f8; --green:#137a4b; --green-bg:#e7f5ed;
  --red:#b42318; --red-bg:#feeceb; --amber:#8a5a00; --amber-bg:#fff4d6;
  --blue:#175cd3; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--page); color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,sans-serif; letter-spacing:0; }}
main {{ width:min(1080px, calc(100% - 32px)); margin:32px auto 64px; }}
header {{ display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:22px; }}
h1 {{ margin:0 0 5px; font-size:28px; font-weight:720; }}
.subtitle {{ margin:0; color:var(--muted); font-size:14px; }}
.overall {{ color:var(--green); font-weight:700; white-space:nowrap; }}
.metrics {{ display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); border:1px solid var(--line); background:var(--surface); margin-bottom:28px; }}
.metric {{ padding:18px 20px; border-right:1px solid var(--line); }}
.metric:last-child {{ border-right:0; }}
.metric span {{ display:block; color:var(--muted); font-size:12px; text-transform:uppercase; font-weight:700; }}
.metric strong {{ display:block; margin-top:4px; font-size:25px; }}
.progress {{ height:8px; background:#dfe4ea; margin:10px 0 0; overflow:hidden; }}
.progress div {{ width:{percent}%; height:100%; background:var(--green); }}
section {{ margin-top:26px; }}
.section-title {{ display:flex; align-items:baseline; justify-content:space-between; gap:16px; margin-bottom:8px; }}
h2 {{ margin:0; font-size:18px; }}
.section-title span {{ color:var(--muted); font-size:13px; }}
table {{ width:100%; border-collapse:collapse; background:var(--surface); border:1px solid var(--line); table-layout:fixed; }}
th,td {{ padding:12px 14px; text-align:left; border-bottom:1px solid var(--line); vertical-align:top; }}
th {{ color:var(--muted); background:#f9fafb; font-size:12px; text-transform:uppercase; }}
th:first-child,td:first-child {{ width:94px; }}
th:last-child,td:last-child {{ width:110px; text-align:right; }}
tr:last-child td {{ border-bottom:0; }}
code {{ display:block; margin-top:3px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }}
.status {{ display:inline-block; min-width:52px; padding:3px 7px; text-align:center; font-size:11px; font-weight:800; border:1px solid currentColor; }}
.pass {{ color:var(--green); background:var(--green-bg); }}
.fail,.error {{ color:var(--red); background:var(--red-bg); }}
.skip {{ color:var(--amber); background:var(--amber-bg); }}
.duration {{ color:var(--muted); font-variant-numeric:tabular-nums; font-size:13px; }}
details {{ margin-top:8px; color:var(--red); }}
pre {{ white-space:pre-wrap; overflow-wrap:anywhere; background:var(--red-bg); padding:10px; }}
@media (max-width:700px) {{ header {{ align-items:start; flex-direction:column; }} .metrics {{ grid-template-columns:1fr 1fr; }} .metric:nth-child(2) {{ border-right:0; }} .metric:nth-child(-n+2) {{ border-bottom:1px solid var(--line); }} th:last-child,td:last-child {{ display:none; }} main {{ width:min(100% - 20px,1080px); margin-top:20px; }} }}
</style>
</head>
<body><main>
<header><div><h1>Edit Topology test report</h1><p class="subtitle">Generated {html.escape(generated)} from the live unittest suite</p></div><div class="overall">{overall}</div></header>
<div class="metrics">
  <div class="metric"><span>Passed</span><strong>{passed}/{total}</strong><div class="progress"><div></div></div></div>
  <div class="metric"><span>Pass rate</span><strong>{percent}%</strong></div>
  <div class="metric"><span>Failures</span><strong>{counts['FAIL'] + counts['ERROR']}</strong></div>
  <div class="metric"><span>Runtime</span><strong>{elapsed:.3f}s</strong></div>
</div>
{_render_rows(records)}
</main></body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("edit_topology/examples/generated/test_results.html"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("edit_topology/examples/generated/test_results.json"),
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.discover(
        str(WORKSPACE_ROOT / "edit_topology" / "tests")
    )
    result = RecordingResult()
    started = time.perf_counter()
    suite.run(result)
    elapsed = time.perf_counter() - started

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_report(result.records, elapsed), encoding="utf-8")
    args.json_output.write_text(
        json.dumps(
            {"elapsed_seconds": elapsed, "tests": result.records},
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    counts = _status_counts(result.records)
    print(
        f"{counts['PASS']}/{len(result.records)} passed in {elapsed:.3f}s; "
        f"report: {args.output}"
    )
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
