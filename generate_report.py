"""
Research-style HTML report generator for DSN_NA2Q experiments.

Reads aggregated.npz files produced by aggregate_results.py and generates
a self-contained HTML report with learning curves, confidence bands, and
a results table suitable for a research paper appendix.

Usage:
    python generate_report.py --scenario 1 --seeds 5
    python generate_report.py --scenario 1 2 --seeds 5 --output reports/report.html
"""

import argparse
import os
import json
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_aggregated(algo: str, scenario: int) -> dict | None:
    if algo == "na2q":
        path = os.path.join("na2q", "checkpoints", f"scenario{scenario}", "aggregated.npz")
    else:
        path = os.path.join("hitmac", "checkpoints", f"scenario{scenario}", "aggregated.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k].tolist() for k in d.files}


def summary_stats(agg: dict) -> dict:
    """Compute paper-table stats from aggregated data."""
    cov_mean = np.array(agg["coverage_mean"])
    cov_std  = np.array(agg["coverage_std"])
    n        = len(cov_mean)

    scale = 100.0 if cov_mean.max() <= 1.0 else 1.0
    tail_mean = cov_mean[max(0, n - 500):] * scale
    tail_std  = cov_std [max(0, n - 500):] * scale
    best      = cov_mean.max() * scale

    # Episode where rolling mean first crosses 50%
    threshold = 50.0
    conv_ep = None
    window = 100
    for i in range(window - 1, n):
        w = cov_mean[i - window + 1: i + 1] * scale
        if w.mean() >= threshold:
            conv_ep = agg["episodes"][i]
            break

    return {
        "final_mean":  round(float(tail_mean.mean()), 1),
        "final_std":   round(float(tail_std .mean()), 1),
        "best":        round(float(best), 1),
        "conv_ep":     int(conv_ep) if conv_ep else None,
        "n_seeds":     int(agg.get("n_seeds", [1])[0] if isinstance(agg.get("n_seeds"), list) else agg.get("n_seeds", 1)),
    }


# ---------------------------------------------------------------------------
# Downsampling for chart JSON (keep ≤ 500 points per series)
# ---------------------------------------------------------------------------

def downsample(xs, ys, n=500):
    if len(xs) <= n:
        return xs, ys
    idx = np.round(np.linspace(0, len(xs) - 1, n)).astype(int)
    return [xs[i] for i in idx], [round(float(ys[i]), 3) for i in idx]


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DSN_NA2Q — Research Report</title>
<script src="https://code.highcharts.com/highcharts.js"></script>
<style>
  :root {
    --na2q:   #4f46e5;
    --hitmac: #d97706;
    --na2q-bg:   rgba(79,70,229,0.08);
    --hitmac-bg: rgba(217,119,6,0.08);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Georgia', serif; background: #f8f9fa; color: #1a1a2e; line-height: 1.6; }
  .page { max-width: 960px; margin: 0 auto; padding: 40px 24px; }

  h1 { font-size: 1.8rem; font-weight: 700; margin-bottom: 4px; }
  h2 { font-size: 1.15rem; font-weight: 700; margin: 32px 0 12px; padding-bottom: 6px;
       border-bottom: 2px solid #e2e8f0; }
  .meta { font-size: 0.85rem; color: #64748b; margin-bottom: 32px; }

  /* Results table */
  .results-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  .results-table th { background: #1e293b; color: #fff; padding: 10px 16px; text-align: left; font-weight: 600; }
  .results-table td { padding: 10px 16px; border-bottom: 1px solid #e2e8f0; }
  .results-table tr:last-child td { border-bottom: none; }
  .results-table tr:nth-child(even) td { background: #f1f5f9; }
  .val { font-weight: 700; }
  .std { color: #64748b; font-size: 0.85em; }
  .na2q-name  { color: var(--na2q);   font-weight: 700; }
  .hitmac-name{ color: var(--hitmac); font-weight: 700; }
  .best { background: #ecfdf5 !important; }
  .na-val { color: #94a3b8; font-style: italic; }

  /* Charts */
  .chart-card { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
                padding: 24px; margin-bottom: 20px; }
  .chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .chart-title { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.06em;
                 text-transform: uppercase; color: #64748b; margin-bottom: 12px; }

  /* Legend */
  .legend { display: flex; gap: 24px; margin-bottom: 12px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 6px; font-size: 0.8rem; }
  .legend-swatch { width: 28px; height: 3px; border-radius: 2px; }

  footer { margin-top: 48px; font-size: 0.78rem; color: #94a3b8; text-align: center; }
</style>
</head>
<body>
<div class="page">

  <h1>DSN Coverage — Experiment Report</h1>
  <p class="meta">Generated: {date} &nbsp;·&nbsp; Seeds per algorithm: {n_seeds} &nbsp;·&nbsp; Scenarios: {scenarios}</p>

  <h2>1. Results Summary</h2>
  <table class="results-table">
    <thead>
      <tr>
        <th>Algorithm</th>
        <th>Scenario</th>
        <th>Final Coverage (%)</th>
        <th>Best Mean Coverage (%)</th>
        <th>Ep. to 50% Cov.</th>
        <th>Seeds</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <h2>2. Learning Curves</h2>
  <div class="legend">
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--na2q)"></div><span>NA²Q</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:var(--hitmac)"></div><span>HiT-MAC</span>
    </div>
    <div class="legend-item">
      <div class="legend-swatch" style="background:#94a3b8;opacity:.5;height:8px;border-radius:4px"></div>
      <span>Mean ± std (shaded)</span>
    </div>
  </div>
  {chart_sections}

  <footer>
    DSN_NA2Q Research Report &nbsp;·&nbsp; Auto-generated by generate_report.py
  </footer>
</div>

<script>
const DATA = {chart_data_json};

Object.entries(DATA).forEach(([scenarioKey, scenarioData]) => {{
  ['reward', 'coverage'].forEach(metric => {{
    const el = document.getElementById(`chart-${{scenarioKey}}-${{metric}}`);
    if (!el) return;

    const series = [];
    const colors = {{ na2q: '#4f46e5', hitmac: '#d97706' }};

    Object.entries(scenarioData).forEach(([algo, d]) => {{
      if (!d[metric + '_mean']) return;
      const col = colors[algo] || '#888';
      const eps = d.episodes;
      const mean = d[metric + '_mean'];
      const hi   = metric + '_max' in d ? d[metric + '_max'] : mean.map((v,i) => v + (d[metric+'_std'][i]||0));
      const lo   = metric + '_min' in d ? d[metric + '_min'] : mean.map((v,i) => v - (d[metric+'_std'][i]||0));
      const scale = metric === 'coverage' && mean[mean.length-1] <= 1.0 ? 100 : 1;

      // band
      series.push({{
        name: algo + ' band',
        type: 'arearange',
        data: eps.map((x,i) => [x, lo[i]*scale, hi[i]*scale]),
        color: col, fillOpacity: 0.12, lineWidth: 0,
        marker: {{ enabled: false }}, showInLegend: false, enableMouseTracking: false,
      }});
      // mean line
      series.push({{
        name: algo.toUpperCase(),
        data: eps.map((x,i) => [x, Math.round(mean[i]*scale*10)/10]),
        color: col, lineWidth: 2, marker: {{ enabled: false }},
      }});
    }});

    Highcharts.chart(el.id, {{
      chart: {{ type: 'line', height: 260, backgroundColor: 'transparent', animation: false }},
      title: {{ text: null }},
      credits: {{ enabled: false }},
      xAxis: {{ title: {{ text: 'Episode' }}, type: 'linear' }},
      yAxis: {{ title: {{ text: metric === 'coverage' ? 'Coverage %' : 'Reward' }},
               gridLineColor: 'rgba(0,0,0,0.05)',
               ...(metric === 'coverage' ? {{ max: 100 }} : {{}}) }},
      tooltip: {{ shared: true, valueDecimals: 1,
                  valueSuffix: metric === 'coverage' ? '%' : '' }},
      legend: {{ enabled: true }},
      series,
    }});
  }});
}});
</script>
</body>
</html>
"""

CHART_SECTION_TEMPLATE = """
  <div class="chart-card">
    <div class="chart-title">Scenario {scenario} — {label}</div>
    <div class="chart-grid">
      <div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:6px">REWARD</div>
        <div id="chart-scenario{scenario}-reward"></div>
      </div>
      <div>
        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:6px">COVERAGE %</div>
        <div id="chart-scenario{scenario}-coverage"></div>
      </div>
    </div>
  </div>
"""

SCENARIO_LABELS = {1: "5 sensors · 6 targets", 2: "50 sensors · 60 targets"}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=int, nargs="+", default=[1])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=str, default="reports/report.html")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    algos = ["na2q", "hitmac"]
    algo_labels = {"na2q": "NA²Q", "hitmac": "HiT-MAC"}
    algo_classes = {"na2q": "na2q-name", "hitmac": "hitmac-name"}

    table_rows = []
    chart_sections = []
    chart_data = {}

    for scenario in args.scenario:
        label = SCENARIO_LABELS.get(scenario, f"Scenario {scenario}")
        chart_sections.append(CHART_SECTION_TEMPLATE.format(scenario=scenario, label=label))
        chart_data[f"scenario{scenario}"] = {}

        for algo in algos:
            agg = load_aggregated(algo, scenario)
            if agg is None:
                row = f"""<tr>
                  <td class="{algo_classes[algo]}">{algo_labels[algo]}</td>
                  <td>{scenario}</td>
                  <td class="na-val" colspan="4">Not trained yet</td>
                </tr>"""
                table_rows.append(row)
                continue

            stats = summary_stats(agg)
            conv  = f"{stats['conv_ep']:,}" if stats["conv_ep"] else "N/A"
            best_cls = "best" if algo == "na2q" else ""  # highlight NA2Q if best

            row = f"""<tr class="{best_cls}">
              <td class="{algo_classes[algo]}">{algo_labels[algo]}</td>
              <td>{scenario}</td>
              <td><span class="val">{stats['final_mean']}</span> <span class="std">± {stats['final_std']}</span></td>
              <td><span class="val">{stats['best']}</span></td>
              <td>{conv}</td>
              <td>{stats['n_seeds']}</td>
            </tr>"""
            table_rows.append(row)

            # Downsample for chart
            eps = agg["episodes"]
            entry = {"episodes": eps}
            for metric in ("reward", "coverage"):
                for stat in ("mean", "std", "min", "max"):
                    key = f"{metric}_{stat}"
                    if key in agg:
                        xs, ys = downsample(eps, agg[key])
                        entry[f"episodes"] = xs
                        entry[key] = ys
            chart_data[f"scenario{scenario}"][algo] = entry

    html = HTML_TEMPLATE.format(
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        n_seeds=args.seeds,
        scenarios=", ".join(str(s) for s in args.scenario),
        table_rows="\n".join(table_rows),
        chart_sections="\n".join(chart_sections),
        chart_data_json=json.dumps(chart_data),
    )

    with open(args.output, "w") as f:
        f.write(html)

    print(f"Report saved → {args.output}")
    print(f"Open in browser: open {args.output}")


if __name__ == "__main__":
    main()
