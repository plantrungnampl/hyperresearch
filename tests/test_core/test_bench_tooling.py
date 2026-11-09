"""Benchmark tooling: Exa discovery wiring, citation renderer, claim ledger, harness I/O."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hyperresearch.core import hooks
from hyperresearch.core.citecheck import render_inline
from hyperresearch.core.claims import claim_coverage, ingest_claims_file
from hyperresearch.core.hooks import _read_skill_source
from hyperresearch.core.render import build_render_context, render_prompt

STEP2 = "hyperresearch-2-width-sweep.md"


def _cfg(tmp_path: Path, engine: str) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(f'[web]\nsearch_engine = "{engine}"\n', encoding="utf-8")
    return cfg


class TestSearchEngineRendering:
    def test_default_engine_adds_no_exa_text(self):
        ctx = build_render_context(None)
        assert ctx["search_engine"] == "websearch"
        assert "search-web" not in render_prompt(_read_skill_source(STEP2), ctx)
        assert "search-web" not in render_prompt(hooks.RESEARCHER_AGENT, ctx)

    @pytest.mark.parametrize("engine", ["exa", "hybrid"])
    def test_exa_engines_render_discovery_instructions(self, tmp_path, engine):
        ctx = build_render_context(_cfg(tmp_path, engine))
        step2 = render_prompt(_read_skill_source(STEP2), ctx)
        assert "$HPR search-web" in step2
        assert ("BOTH WebSearch" in step2) == (engine == "hybrid")
        assert "search-web --similar" in render_prompt(hooks.RESEARCHER_AGENT, ctx)

    def test_unknown_engine_fails_render(self, tmp_path):
        with pytest.raises(ValueError, match="search_engine"):
            build_render_context(_cfg(tmp_path, "bing"))


class TestExaProvider:
    @pytest.fixture
    def client(self, monkeypatch):
        import exa_py

        client = MagicMock()
        client.headers = {}
        client.search.return_value = SimpleNamespace(results=[])
        client.find_similar.return_value = SimpleNamespace(results=[])
        monkeypatch.setattr(exa_py, "Exa", MagicMock(return_value=client))
        monkeypatch.setenv("EXA_API_KEY", "k")
        return client

    def test_search_passes_category_and_date(self, client):
        from hyperresearch.web.exa_provider import ExaProvider

        ExaProvider().search("q", max_results=3, category="research paper",
                             start_published_date="2023-01-01")
        kwargs = client.search.call_args.kwargs
        assert kwargs["category"] == "research paper"
        assert kwargs["start_published_date"] == "2023-01-01"
        assert kwargs["num_results"] == 3 and kwargs["type"] == "auto"

    def test_find_similar_excludes_source_domain(self, client):
        from hyperresearch.web.exa_provider import ExaProvider

        ExaProvider().find_similar("https://example.org/paper", max_results=4)
        args, kwargs = client.find_similar.call_args
        assert args == ("https://example.org/paper",)
        assert kwargs["exclude_source_domain"] is True and "type" not in kwargs


class TestRenderInline:
    def _with_urls(self, vault):
        db = vault.db
        db.execute("UPDATE notes SET source='https://a.example/x' WHERE id='python-async-patterns'")
        db.execute("UPDATE notes SET source='https://b.example/y' WHERE id='rust-ownership'")
        return db

    def test_numbers_groups_and_builds_sources(self, seeded_vault):
        db = self._with_urls(seeded_vault)
        text = ("Async helps [[python-async-patterns]]. Both matter [[rust-ownership]], "
                "[[python-async-patterns]]. Again [[rust-ownership]].\n")
        out, info = render_inline(text, db)
        assert "Async helps [1]." in out
        assert "Both matter [1, 2]." in out
        assert "Again [2]." in out
        assert out.rstrip().endswith("[2] Rust Ownership. https://b.example/y")
        assert "[1] Python Async Patterns. https://a.example/x" in out
        assert info == {"sources": 2, "unresolved": []}

    def test_links_without_url_stay_and_are_reported(self, seeded_vault):
        db = self._with_urls(seeded_vault)
        out, info = render_inline("See [[concurrency]] and [[python-async-patterns]].", db)
        assert "[[concurrency]]" in out and "[1]" in out
        assert info["unresolved"] == ["concurrency"]

    def test_refuses_already_rendered_report(self, seeded_vault):
        with pytest.raises(ValueError):
            render_inline("x [1]\n\n## Sources\n\n[1] t. https://u\n", seeded_vault.db)


class TestClaimCoverage:
    def test_ledger_counts_quantified_claims_per_stage(self, seeded_vault, tmp_path):
        p = tmp_path / "claims-python-async-patterns.json"
        p.write_text(json.dumps([
            {"claim": "Throughput rose 37.5%", "numbers": ["37.5%"], "confidence": "high"},
            {"claim": "Latency fell to 1,250 ms in 2024", "numbers": ["1,250 ms", "2024"]},
            {"claim": "Only a year", "numbers": ["2024"]},  # years alone are not significant
            {"claim": "No numbers here"},
        ]), encoding="utf-8")
        ingest_claims_file(seeded_vault.db, p, vault_tag="t1")
        seeded_vault.db.commit()

        res = claim_coverage(seeded_vault.db, "t1", {
            "draft": "Throughput rose 37.5% and latency hit 1250 ms.",
            "final": "Throughput rose 37.5%.",
        })
        assert res["quantified_claims"] == 2
        assert res["stages"]["draft"]["covered"] == 2
        assert res["stages"]["final"]["covered"] == 1
        assert res["stages"]["final"]["uncovered"][0]["claim"].startswith("Latency")


def _load_harness():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "hyperbench.py"
    spec = importlib.util.spec_from_file_location("hyperbench", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestHarnessIO:
    def test_export_matches_official_layouts(self, tmp_path):
        hb = _load_harness()
        drb = tmp_path / "query.jsonl"
        drb.write_text('{"id": 7, "prompt": "P7", "language": "en"}\n'
                       '{"id": 8, "prompt": "P8", "language": "zh"}\n', encoding="utf-8")
        drb2 = tmp_path / "tasks.jsonl"
        drb2.write_text('{"id": "task1", "idx": "3", "prompt": "Q3", "language": "en"}\n', encoding="utf-8")
        out = tmp_path / "runs"
        for bench, tid in (("drb", "7"), ("drb2", "3")):
            d = out / bench / "base" / f"task-{tid}"
            d.mkdir(parents=True)
            (d / "article.md").write_text(f"report {tid}", encoding="utf-8")

        hb.main(["export", "--bench", "drb", "--tasks", str(drb), "--arm", "base", "--out", str(out)])
        hb.main(["export", "--bench", "drb2", "--tasks", str(drb2), "--arm", "base", "--out", str(out)])

        rows = [json.loads(line) for line in (out / "export" / "base.jsonl").read_text(encoding="utf-8").splitlines()]
        assert rows == [{"id": 7, "prompt": "P7", "article": "report 7"}]
        assert (out / "export" / "drb2" / "base" / "idx-3.md").read_text(encoding="utf-8") == "report 3"

    def test_unknown_task_id_is_an_error(self, tmp_path):
        hb = _load_harness()
        q = tmp_path / "q.jsonl"
        q.write_text('{"id": 1, "prompt": "p"}\n', encoding="utf-8")
        with pytest.raises(SystemExit, match="not found"):
            hb.load_tasks("drb", q, ["99"], None, None)
