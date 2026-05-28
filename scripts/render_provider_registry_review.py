from __future__ import annotations

import html
import sys
from datetime import datetime, timezone
from pathlib import Path

from paper_collect import provider_registry


def render(output_path: Path | str | None = None) -> Path:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    snapshot = provider_registry.provider_registry_snapshot(generated_at=generated_at)
    output = Path(output_path or Path("provider_registry_review.html"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_html(snapshot), encoding="utf-8")
    return output


def _html(snapshot: dict) -> str:
    entries = snapshot["provider_registry_entries"]
    active = [entry for entry in entries if entry["registry_state"] == "active"]
    reference = [entry for entry in entries if entry["registry_state"] == "reference"]
    deferred = [entry for entry in entries if entry["registry_state"] == "deferred"]
    probe = [entry for entry in entries if entry["registry_state"] == "needs_live_probe"]
    rows = "\n".join(_entry_row(entry) for entry in entries)
    active_examples = _status_venue_summary(active)
    reference_examples = _status_venue_summary(reference)
    deferred_examples = _status_venue_summary(deferred)
    probe_examples = _status_venue_summary(probe)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Paper Collect Provider Registry Review</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f242b;
      --muted: #5d6775;
      --line: #d9dee5;
      --paper: #f8fafb;
      --surface: #ffffff;
      --active: #1d7a58;
      --reference: #2f6fb2;
      --deferred: #a06100;
      --probe: #8a4fa3;
      --none: #9b3d45;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
      line-height: 1.5;
    }}
    main {{
      width: min(1440px, calc(100vw - 40px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(320px, .7fr);
      gap: 24px;
      align-items: end;
      padding-bottom: 22px;
      border-bottom: 2px solid var(--ink);
    }}
    h1 {{
      margin: 0;
      max-width: 820px;
      font-size: clamp(34px, 5vw, 64px);
      line-height: .98;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 22px;
    }}
    p {{ margin: 0; color: var(--muted); }}
    code {{
      padding: 1px 5px;
      border: 1px solid var(--line);
      background: #f3f5f7;
      border-radius: 4px;
      font-family: Consolas, "Cascadia Mono", monospace;
      font-size: .92em;
    }}
    .eyebrow {{
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 800;
      text-transform: uppercase;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .stat {{
      min-height: 82px;
      padding: 13px;
      border: 1px solid var(--line);
      border-left: 5px solid var(--reference);
      background: var(--surface);
    }}
    .stat strong {{
      display: block;
      color: var(--muted);
      font-size: 13px;
    }}
    .stat span {{
      display: block;
      margin-top: 4px;
      font-size: 22px;
      font-weight: 800;
    }}
    section {{
      padding: 26px 0;
      border-bottom: 1px solid var(--line);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-top: 16px;
    }}
    .summary-card {{
      min-height: 150px;
      padding: 16px;
      border: 1px solid var(--line);
      background: var(--surface);
    }}
    .summary-card.active {{ border-top: 5px solid var(--active); }}
    .summary-card.reference {{ border-top: 5px solid var(--reference); }}
    .summary-card.deferred {{ border-top: 5px solid var(--deferred); }}
    .summary-card.probe {{ border-top: 5px solid var(--probe); }}
    .summary-card strong {{ display: block; margin-bottom: 10px; }}
    .answer-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .answer-card {{
      min-height: 168px;
      padding: 16px;
      border: 1px solid var(--line);
      background: var(--surface);
    }}
    .answer-card h3 {{
      margin: 0 0 8px;
      font-size: 16px;
    }}
    .answer-card p {{
      margin-top: 8px;
      font-size: 13px;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      background: #fbfcfd;
      font-size: 12px;
      font-weight: 700;
      color: #374151;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--surface);
    }}
    table {{
      width: 100%;
      min-width: 1320px;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 12px 13px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #edf1f4;
      color: #2f3742;
      z-index: 1;
    }}
    tr:last-child td {{ border-bottom: none; }}
    .provider {{
      min-width: 170px;
      font-weight: 800;
    }}
    .status {{
      display: inline-block;
      min-width: 92px;
      padding: 3px 8px;
      border-radius: 999px;
      color: white;
      text-align: center;
      font-weight: 800;
      font-size: 12px;
    }}
    .status.active {{ background: var(--active); }}
    .status.reference {{ background: var(--reference); }}
    .status.deferred {{ background: var(--deferred); }}
    .status.needs_live_probe {{ background: var(--probe); }}
    .status.unsupported {{ background: var(--none); }}
    .muted {{ color: var(--muted); }}
    footer {{
      padding-top: 22px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      header,
      .summary-grid,
      .answer-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Paper Collect / Source Provider Registry</div>
        <h1>Provider 覆盖与接入状态审阅表</h1>
      </div>
      <aside class="stats" aria-label="registry status">
        <div class="stat"><strong>Registry rows</strong><span>{len(entries)}</span></div>
        <div class="stat"><strong>Active runtime providers</strong><span>{len(active)}</span></div>
        <div class="stat"><strong>Reference rows</strong><span>{len(reference)}</span></div>
        <div class="stat"><strong>Deferred / probe rows</strong><span>{len(deferred) + len(probe)}</span></div>
      </aside>
    </header>
    <section>
      <h2>老板视角的一句话</h2>
      <p>当前真正执行的是 <code>{_join(snapshot["active_provider_ids"])}</code>。其他 provider 已登记覆盖、venue、access 和 provenance 能力，用于回答“能查哪些来源 / 后续该接哪些数据库”，但不会在 Phase 1 静默变成 active runtime provider。</p>
      <div class="answer-grid">
        <article class="answer-card">
          <h3>当前能实际执行什么</h3>
          <div class="pill-row">{active_examples}</div>
          <p>这些来自 active runtime provider。它们说明本轮 search 能看到的 venue evidence 形态，不等于 exhaustive venue coverage 或 full-text reachability。</p>
        </article>
        <article class="answer-card">
          <h3>下一波最该接什么</h3>
          <div class="pill-row">{reference_examples}</div>
          <p>Reference rows 已有研究证据，但 Phase 1 不自动执行。Phase 2 应把 Crossref、DBLP、DOAJ、PubMed/PMC/Europe PMC 等逐波提升为 active provider。</p>
        </article>
        <article class="answer-card">
          <h3>只能作为后续 / 补充</h3>
          <div class="pill-row">{deferred_examples}</div>
          <p>Deferred rows 主要适合 citation graph、repository、OA/full-text enrichment 或 resolver，不应在 Phase 1 被说成已覆盖。</p>
        </article>
        <article class="answer-card">
          <h3>仍需 live probe</h3>
          <div class="pill-row">{probe_examples}</div>
          <p>未被证据确认的 venue 不进入 observed examples。比如 ICLR 在本轮研究中保守保持 unconfirmed，不能硬写成已覆盖。</p>
        </article>
      </div>
      <div class="summary-grid">
        <article class="summary-card active">
          <strong>Active</strong>
          <div class="pill-row">{_pills(entry["display_name"] for entry in active)}</div>
        </article>
        <article class="summary-card reference">
          <strong>Reference</strong>
          <div class="pill-row">{_pills(entry["display_name"] for entry in reference)}</div>
        </article>
        <article class="summary-card deferred">
          <strong>Deferred</strong>
          <div class="pill-row">{_pills(entry["display_name"] for entry in deferred)}</div>
        </article>
        <article class="summary-card probe">
          <strong>Needs Live Probe</strong>
          <div class="pill-row">{_pills(entry["display_name"] for entry in probe)}</div>
        </article>
      </div>
    </section>
    <section>
      <h2>Provider registry table</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Provider</th>
              <th>Status</th>
              <th>Role</th>
              <th>Coverage scope</th>
              <th>Venue coverage</th>
              <th>Search surfaces</th>
              <th>Filters</th>
              <th>Result / provenance</th>
              <th>Stage usage</th>
              <th>Access</th>
              <th>Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows}
          </tbody>
        </table>
      </div>
    </section>
    <footer>
      Generated from <code>paper_collect.provider_registry.provider_registry_snapshot()</code> at {html.escape(str(snapshot["generated_at"]))}. This HTML is a review surface; the Python registry remains the executable static table.
    </footer>
  </main>
</body>
</html>
"""


def _entry_row(entry: dict) -> str:
    venue = entry["venue_coverage"]
    filters = entry["supported_filters"]
    venue_text = (
        f"<strong>{_e(venue['kind'])}</strong><br>"
        f"searchable={_e(str(venue['searchable']).lower())}; "
        f"corpus_limited={_e(str(venue['corpus_limited']).lower())}<br>"
        f"<span class=\"muted\">{_join(venue.get('examples', []))}</span>"
    )
    filter_text = (
        f"implemented: {_code_list(filters.get('implemented', []))}<br>"
        f"documented: {_code_list(filters.get('documented_reference', []))}<br>"
        f"future: {_code_list(filters.get('future_candidates', []))}"
    )
    result_text = (
        f"fields: {_join(entry['result_fields'])}<br>"
        f"<span class=\"muted\">provenance: {_join(entry['provenance_capabilities'])}</span>"
    )
    return f"""<tr>
  <td class="provider">{_e(entry["display_name"])}<br><span class="muted"><code>{_e(entry["provider_id"])}</code></span></td>
  <td><span class="status {_e(entry["registry_state"])}">{_e(entry["registry_state"])}</span></td>
  <td>{_join(entry["roles"])}</td>
  <td>{_join(entry["coverage_scope"])}</td>
  <td>{venue_text}</td>
  <td>{_code_list(entry["supported_search_surfaces"])}</td>
  <td>{filter_text}</td>
  <td>{result_text}</td>
  <td>{_code_list(entry["stage_usage"])}</td>
  <td>{_code_list(entry["access_model"]["tags"])}</td>
  <td>{_join(entry["evidence_refs"])}</td>
</tr>"""


def _e(value: str) -> str:
    return html.escape(value)


def _join(values: object) -> str:
    if not values:
        return "-"
    if isinstance(values, str):
        return _e(values)
    return ", ".join(_e(str(value)) for value in values)


def _code_list(values: object) -> str:
    if not values:
        return "-"
    if isinstance(values, str):
        values = [values]
    return " ".join(f"<code>{_e(str(value))}</code>" for value in values)


def _pills(values: object) -> str:
    return "".join(f"<span class=\"pill\">{_e(str(value))}</span>" for value in values) or "<span class=\"pill\">-</span>"


def _status_venue_summary(entries: list[dict]) -> str:
    examples: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for example in entry.get("venue_coverage", {}).get("examples", []):
            key = str(example).casefold()
            if key in seen:
                continue
            seen.add(key)
            examples.append(str(example))
            if len(examples) >= 14:
                return _pills(examples)
    return _pills(examples)


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    print(render(target))
