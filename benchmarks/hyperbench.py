"""Run hyperresearch on official DeepResearch Bench tasks and export for official scoring.

This script only RUNS the agent and EXPORTS reports. Tasks and scoring come
unchanged from the official repos:
  DRB    https://github.com/Ayanami0730/deep_research_bench   data/prompt_data/query.jsonl
  DRB II https://github.com/imlrz/DeepResearch-Bench-II       tasks_and_rubrics.jsonl

  # 1. run one arm (resumable: finished tasks are skipped)
  python benchmarks/hyperbench.py run --bench drb2 --tasks <tasks_and_rubrics.jsonl> \
      --ids 3,17,42 --arm baseline --out bench-runs -- --permission-mode bypassPermissions
  # 2. export into the official evaluator's input layout
  python benchmarks/hyperbench.py export --bench drb2 --tasks <...> --arm baseline --out bench-runs
     drb  -> bench-runs/export/<arm>.jsonl            ({"id","prompt","article"} per line)
     drb2 -> bench-runs/export/drb2/<arm>/idx-<N>.md  (pass bench-runs/export/drb2 as --pdf_dir)

Arms differ by exactly the flags given: --engine (websearch|exa|hybrid) and
--citations (llm: pipeline writes [N] itself; mechanical: pipeline keeps
[[note-id]] and `hyperresearch citecheck render` numbers them from the vault).
Every run records git SHA, arm settings, wall time and Claude's cost/usage in meta.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ID_KEY = {"drb": "id", "drb2": "idx"}
HPR = [sys.executable, "-m", "hyperresearch"]


def load_tasks(bench: str, path: Path, ids: list[str] | None, lang: str | None, limit: int | None) -> list[dict]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = {"id": str(row[ID_KEY[bench]]), "prompt": row["prompt"], "language": row.get("language")}
        if ids and task["id"] not in ids:
            continue
        if lang and task["language"] != lang:
            continue
        tasks.append(task)
    missing = set(ids or []) - {t["id"] for t in tasks}
    if missing:
        raise SystemExit(f"task ids not found in {path}: {sorted(missing)}")
    return tasks[:limit] if limit else tasks


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _hpr(args: list[str], cwd: Path) -> None:
    subprocess.run([*HPR, *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")


def run_task(task: dict, arm_dir: Path, opts: argparse.Namespace) -> dict:
    work = arm_dir / f"task-{task['id']}"
    article = work / "article.md"
    if article.exists():
        return {"id": task["id"], "status": "skipped (done)"}
    work.mkdir(parents=True, exist_ok=True)

    if not (work / ".hyperresearch").exists():
        _hpr(["init", ".", "--json"], work)
    _hpr(["config", "set", "web.search_engine", opts.engine], work)
    _hpr(["install", ".", "--profile", opts.profile, "--json"], work)  # renders prompts for this engine

    research = work / "research"
    research.mkdir(exist_ok=True)
    # The router reads prompt.txt as the canonical query (no shell quoting of long prompts).
    (research / "prompt.txt").write_text(task["prompt"], encoding="utf-8")
    style = "inline" if opts.citations == "llm" else "wikilink"
    (research / "wrapper_contract.json").write_text(
        json.dumps({"citation_style": style}, indent=2), encoding="utf-8")

    cmd = ["claude", "-p", "/hyperresearch Research the query in research/prompt.txt.",
           "--output-format", "json", *opts.claude_args]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=work, capture_output=True, text=True, encoding="utf-8",
                              timeout=opts.timeout_h * 3600)
        exit_code, stdout, err = proc.returncode, proc.stdout, proc.stderr[-4000:]
    except subprocess.TimeoutExpired:
        exit_code, stdout, err = "timeout", "", ""
    wall = time.monotonic() - t0
    (work / "claude_result.json").write_text(stdout or "", encoding="utf-8")

    reports = sorted((research / "notes").glob("final_report_*.md"), key=lambda p: p.stat().st_mtime)
    meta = {
        "id": task["id"], "arm": opts.arm, "engine": opts.engine, "citations": opts.citations,
        "profile": opts.profile, "git_sha": _git_sha(), "exit_code": exit_code,
        "wall_minutes": round(wall / 60, 1), "stderr_tail": err,
    }
    try:
        result = json.loads(stdout)
        meta.update({k: result.get(k) for k in ("total_cost_usd", "num_turns", "usage", "is_error")})
    except (json.JSONDecodeError, TypeError):
        pass
    if reports:
        report = reports[-1]
        meta["report"] = str(report.relative_to(work))
        if opts.citations == "mechanical":
            proc = subprocess.run([*HPR, "citecheck", "render", str(report), "--out", str(article), "--json"],
                                  cwd=work, capture_output=True, text=True, encoding="utf-8")
            meta["render"] = proc.stdout[-2000:]
        else:
            shutil.copyfile(report, article)
        meta["words"] = len(article.read_text(encoding="utf-8").split()) if article.exists() else 0
    meta["status"] = "ok" if article.exists() else "no report"
    (work / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"id": task["id"], "status": meta["status"], "minutes": meta["wall_minutes"]}


def cmd_run(opts: argparse.Namespace) -> None:
    if not shutil.which("claude"):
        raise SystemExit("claude CLI not on PATH")
    if opts.engine != "websearch" and not os.environ.get("EXA_API_KEY"):
        raise SystemExit(f"--engine {opts.engine} needs EXA_API_KEY in the environment")
    tasks = load_tasks(opts.bench, Path(opts.tasks), opts.ids, opts.lang, opts.limit)
    arm_dir = Path(opts.out) / opts.bench / opts.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(tasks)} task(s) -> {arm_dir}")
    with ThreadPoolExecutor(max_workers=opts.jobs) as pool:
        for res in pool.map(lambda t: run_task(t, arm_dir, opts), tasks):
            print(json.dumps(res, ensure_ascii=False), flush=True)


def cmd_export(opts: argparse.Namespace) -> None:
    tasks = {t["id"]: t for t in load_tasks(opts.bench, Path(opts.tasks), opts.ids, None, None)}
    arm_dir = Path(opts.out) / opts.bench / opts.arm
    export = Path(opts.out) / "export"
    done = {p.parent.name.removeprefix("task-"): p for p in arm_dir.glob("task-*/article.md")}
    rows = [(tid, done[tid]) for tid in tasks if tid in done]
    if opts.bench == "drb":
        export.mkdir(parents=True, exist_ok=True)
        with open(export / f"{opts.arm}.jsonl", "w", encoding="utf-8") as f:
            for tid, path in rows:
                f.write(json.dumps({"id": int(tid), "prompt": tasks[tid]["prompt"],
                                    "article": path.read_text(encoding="utf-8")}, ensure_ascii=False) + "\n")
    else:
        dest = export / "drb2" / opts.arm
        dest.mkdir(parents=True, exist_ok=True)
        for tid, path in rows:
            shutil.copyfile(path, dest / f"idx-{tid}.md")
    print(f"exported {len(rows)}/{len(tasks)} task(s); missing: {sorted(set(tasks) - set(done))}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    claude_args = []
    if "--" in argv:  # everything after -- goes to `claude -p`
        claude_args = argv[argv.index("--") + 1:]
        argv = argv[: argv.index("--")]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "export"):
        p = sub.add_parser(name)
        p.add_argument("--bench", choices=("drb", "drb2"), required=True)
        p.add_argument("--tasks", required=True, help="official query.jsonl / tasks_and_rubrics.jsonl")
        p.add_argument("--ids", type=lambda s: s.split(","), help="comma-separated task ids (drb id / drb2 idx)")
        p.add_argument("--arm", required=True, help="label for this experiment arm")
        p.add_argument("--out", default="bench-runs")
    run = sub.choices["run"]
    run.add_argument("--lang", choices=("en", "zh"))
    run.add_argument("--limit", type=int)
    run.add_argument("--engine", choices=("websearch", "exa", "hybrid"), default="websearch")
    run.add_argument("--citations", choices=("llm", "mechanical"), default="llm")
    run.add_argument("--profile", default="full")
    run.add_argument("--jobs", type=int, default=1, help="tasks in parallel (mind Claude rate limits)")
    run.add_argument("--timeout-h", type=float, default=4.0)
    opts = ap.parse_args(argv)
    opts.claude_args = claude_args
    (cmd_run if opts.cmd == "run" else cmd_export)(opts)


if __name__ == "__main__":
    main()
