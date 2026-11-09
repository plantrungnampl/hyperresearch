"""search-web — URL discovery through Exa. Returns candidates; saves nothing.

Discovery only: agents pick URLs from the result and run `hpr fetch` on them,
so every source still lands as a full-text note through the normal fetch path.
"""

from __future__ import annotations

import typer

from hyperresearch.cli._output import console, output
from hyperresearch.models.output import error, success


def search_web(
    query: str = typer.Argument("", help="Search query (omit with --similar)"),
    similar: str | None = typer.Option(None, "--similar", help="Find pages similar to this URL instead"),
    category: str | None = typer.Option(None, "--category", help='Exa category, e.g. "research paper", "news", "pdf"'),
    since: str | None = typer.Option(None, "--since", help="Only results published on/after YYYY-MM-DD"),
    num: int = typer.Option(10, "--num", "-n", help="Max results"),
    run_tag: str | None = typer.Option(None, "--run", help="vault_tag: log this call to the run's events"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON output"),
) -> None:
    """Discover candidate URLs via Exa (needs EXA_API_KEY). Does not fetch or save."""
    if not query and not similar:
        output(error("give a QUERY or --similar URL", "BAD_ARGS"), json_mode=json_output)
        raise typer.Exit(1)
    try:
        from hyperresearch.web.exa_provider import ExaProvider

        prov = ExaProvider(text_max_characters=1000)
        if similar:
            results = prov.find_similar(similar, max_results=num)
        else:
            results = prov.search(query, max_results=num, category=category, start_published_date=since)
    except Exception as e:  # missing key/SDK, network, API error: agent falls back to WebSearch
        output(error(f"exa search failed: {e}", "SEARCH_FAILED"), json_mode=json_output)
        raise typer.Exit(1)

    if run_tag:
        try:
            from hyperresearch.core.runs import record_event
            from hyperresearch.core.vault import Vault

            record_event(Vault.discover(), run_tag, {
                "type": "search", "engine": "exa", "mode": "similar" if similar else "query",
                "results": len(results),
            })
        except Exception:
            pass  # telemetry must never block discovery

    rows = [
        {
            "url": r.url,
            "title": r.title,
            "published": r.metadata.get("published_date"),
            "highlights": r.metadata.get("highlights", [])[:3],
        }
        for r in results
    ]
    if json_output:
        output(success({"results": rows}, count=len(rows)), json_mode=True)
    else:
        for r in rows:
            console.print(f"  [cyan]{r['url']}[/] {r['title']}")
