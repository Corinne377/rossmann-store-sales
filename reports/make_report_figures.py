"""Generate portable SVG charts for the README using committed report data."""
from __future__ import annotations
from html import escape
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NAVY = "#243447"
BLUE = "#1f77b4"
GRAY = "#96a1ad"
GRID = "#d9dfe5"
MUTED = "#52606d"


def svg_start(width, height, title, desc):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Arial,sans-serif;fill:#243447}.muted{fill:#52606d;font-size:13px}.title{font-size:19px;font-weight:600}.value{font-size:15px;font-weight:600}</style>',
    ]


def comparison_chart():
    baseline = json.loads((ROOT / "reports/metrics-calendar.json").read_text())
    strong = json.loads((ROOT / "reports/metrics-history.json").read_text())
    values = [baseline["validation_rmspe"], strong["validation_rmspe"]]
    reduction = 100 * (1 - values[1] / values[0])
    w, h = 900, 300
    x0, x1 = 290, 785
    xmax = 0.20
    scale = (x1 - x0) / xmax
    out = svg_start(w, h, "Forward validation RMSPE model comparison", "Lower is better. History-feature CatBoost scores 0.1611 compared with 0.1753 for the calendar-only baseline, an 8.1 percent reduction.")
    out.append('<text class="title" x="48" y="38">Same 42-day forward holdout · lower is better</text>')
    for tick in range(0, 21, 5):
        x = x0 + tick / 100 * scale
        out.append(f'<line x1="{x:.1f}" y1="76" x2="{x:.1f}" y2="230" stroke="{GRID}"/>')
        out.append(f'<text class="muted" x="{x:.1f}" y="254" text-anchor="middle">{tick/100:.2f}</text>')
    labels = ["Calendar CatBoost baseline", "History-feature CatBoost"]
    colors = [GRAY, BLUE]
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = 105 + i * 75
        out.append(f'<text x="48" y="{y+7}" font-size="15">{escape(label)}</text>')
        out.append(f'<rect x="{x0}" y="{y-14}" width="{value*scale:.1f}" height="30" rx="3" fill="{color}"/>')
        out.append(f'<text class="value" x="{x0+value*scale+10:.1f}" y="{y+6}">{value:.4f}</text>')
    out.append(f'<text x="785" y="286" text-anchor="end" fill="{BLUE}" font-size="14" font-weight="600">{reduction:.1f}% lower RMSPE</text>')
    out.append('</svg>')
    (OUT / "model-comparison.svg").write_text("\n".join(out))


def weekly_chart():
    frames = []
    for filename, name in [("validation-calendar-predictions.csv", "baseline"), ("validation-history-predictions.csv", "history")]:
        df = pd.read_csv(ROOT / "reports" / filename, parse_dates=["Date"])
        df.loc[df.Open.eq(0), "PredictedSales"] = 0
        first = df.Date.min()
        df["week"] = ((df.Date - first).dt.days // 7).astype(int) + 1
        totals = df.groupby("week").agg(actual=("ActualSales", "sum"), predicted=("PredictedSales", "sum"))
        frames.append((name, totals))
    actual = frames[0][1]["actual"].to_numpy() / 1e6
    baseline = frames[0][1]["predicted"].to_numpy() / 1e6
    history = frames[1][1]["predicted"].to_numpy() / 1e6
    w, h = 900, 370
    x0, x1, y0, y1 = 80, 850, 90, 290
    vmax = max(max(actual), max(baseline), max(history)) * 1.08
    def xcoord(i): return x0 + i * (x1-x0) / max(1, len(actual)-1)
    def ycoord(v): return y1 - v / vmax * (y1-y0)
    out = svg_start(w, h, "Weekly actual and predicted sales", "Six weekly sales totals on the forward validation window, in millions. The history-feature model tracks actual totals more closely than the calendar baseline.")
    out.append('<text class="title" x="48" y="38">Weekly sales totals on the forward holdout</text>')
    for i in range(5):
        v = vmax * i / 4
        y = ycoord(v)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="{GRID}"/>')
        out.append(f'<text class="muted" x="{x0-12}" y="{y+4:.1f}" text-anchor="end">{v:.1f}</text>')
    def series(vals, color, marker, label):
        coords = [(xcoord(i), ycoord(v)) for i,v in enumerate(vals)]
        path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i,(x,y) in enumerate(coords))
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x,y in coords:
            if marker == "circle": out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
            else: out.append(f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{color}"/>')
        return label
    series(actual, NAVY, "circle", "Actual")
    series(baseline, GRAY, "square", "Calendar baseline")
    series(history, BLUE, "circle", "History-feature model")
    for i in range(len(actual)):
        out.append(f'<text class="muted" x="{xcoord(i):.1f}" y="{y1+23}" text-anchor="middle">{i+1}</text>')
    for i,(label,color) in enumerate([("Actual",NAVY),("Calendar baseline",GRAY),("History-feature model",BLUE)]):
        x = 190 + i*230
        out.append(f'<line x1="{x}" y1="{h-27}" x2="{x+24}" y2="{h-27}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<text class="muted" x="{x+32}" y="{h-22}">{label}</text>')
    out.append('<text class="muted" x="465" y="335" text-anchor="middle">Holdout week</text>')
    out.append('<text class="muted" transform="translate(23 195) rotate(-90)" text-anchor="middle">Total sales (millions)</text>')
    out.append('</svg>')
    (OUT / "weekly-forecast.svg").write_text("\n".join(out))


def importance_chart():
    path = ROOT / "reports/feature-importance.csv"
    if not path.exists(): return
    imp = pd.read_csv(path).nlargest(12, "importance").sort_values("importance")
    w, h = 900, 410
    left, right = 300, 835
    top, rowh = 74, 25
    maxval = float(imp.importance.max())
    out = svg_start(w, h, "Most influential model features", "Top twelve CatBoost prediction-value importances from the full-data trained model. Past sales history features rank among the strongest signals.")
    out.append('<text class="title" x="48" y="38">Most influential features</text>')
    for i, row in enumerate(imp.itertuples(index=False)):
        y = top + i * rowh
        out.append(f'<text x="48" y="{y+5}" font-size="13">{escape(str(row.feature))}</text>')
        bw = max(0, float(row.importance) / maxval * (right-left))
        out.append(f'<rect x="{left}" y="{y-10}" width="{bw:.1f}" height="17" rx="2" fill="{BLUE}"/>')
        out.append(f'<text class="muted" x="{left+bw+7:.1f}" y="{y+4}">{float(row.importance):.1f}</text>')
    out.append('<text class="muted" x="570" y="394" text-anchor="middle">Prediction-value importance</text>')
    out.append('</svg>')
    (OUT / "feature-importance.svg").write_text("\n".join(out))


comparison_chart()
weekly_chart()
importance_chart()
print(f"Wrote SVG charts to {OUT}")
