from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

OUTPUT_HTML = "benchmark_report.html"
BASELINE_NAME: Optional[str] = None


# -------------------- Loading --------------------


def load_results(pattern: str = "benchmark_results*.json") -> Dict[str, Any]:
    files = sorted(glob.glob(f"data/{pattern}"))
    data: Dict[str, Any] = {}

    for f in files:
        with open(f) as fh:
            content = json.load(fh)
            name = Path(f).stem.replace("benchmark_results_", "")
            data[name] = content

    return data


# -------------------- Metrics --------------------


def collect_metrics(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for system, content in data.items():
        benches = content.get("benchmarks", {})

        for category, sub in benches.items():
            for name, values in sub.items():
                key = f"{category}.{name}"
                metrics.setdefault(key, {})

                if isinstance(values, dict) and "mean" in values:
                    metrics[key][system] = {
                        "mean": values["mean"],
                        "std": values.get("std", 0),
                        **values,
                    }

    return metrics


# -------------------- Ratios --------------------


def compute_ratios(
    metrics: Dict[str, Any], baseline: str
) -> Dict[str, Dict[str, Optional[float]]]:
    ratios: Dict[str, Dict[str, Optional[float]]] = {}

    for metric, systems in metrics.items():
        if baseline not in systems:
            continue

        base = systems[baseline]["mean"]
        ratios[metric] = {}

        for sys, vals in systems.items():
            ratios[metric][sys] = vals["mean"] / base if base else None

    return ratios


# -------------------- Normalization --------------------


def normalize_metrics(metrics: Dict[str, Any]) -> Dict[str, Dict[str, Optional[float]]]:
    normalized: Dict[str, Dict[str, Optional[float]]] = {}

    for metric, systems in metrics.items():
        normalized[metric] = {}
        for sys, vals in systems.items():
            t = vals["mean"]
            normalized[metric][sys] = (1.0 / t) if t else None

    return normalized


# -------------------- Physical Units --------------------


def compute_physical_units(
    metrics: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[float]]]:
    physical: Dict[str, Dict[str, Optional[float]]] = {}

    for metric, systems in metrics.items():
        physical[metric] = {}

        for sys, vals in systems.items():
            t = vals.get("mean")
            if not t or t <= 0:
                physical[metric][sys] = None
                continue

            if "flops" in vals:
                physical[metric][sys] = vals["flops"] / t / 1e9  # GFLOPS
            elif "bytes" in vals:
                physical[metric][sys] = vals["bytes"] / t / 1e6  # MB/s
            elif "samples" in vals:
                physical[metric][sys] = vals["samples"] / t  # samples/s
            else:
                physical[metric][sys] = None

    return physical


# -------------------- Diagnosis (with GPU detection) --------------------


def diagnose(
    metrics: Dict[str, Any], ratios: Dict[str, Any], baseline: str
) -> List[str]:
    messages: List[str] = []

    # ratio-based bottlenecks
    for metric, systems in ratios.items():
        for sys, ratio in systems.items():
            if sys == baseline or ratio is None:
                continue

            if ratio > 2.0:
                messages.append(
                    f"{sys}: {metric} is {ratio:.2f}x slower → strong bottleneck"
                )
            elif ratio > 1.5:
                messages.append(
                    f"{sys}: {metric} is {ratio:.2f}x slower → moderate bottleneck"
                )

    # category grouping
    categories = {"cpu": [], "disk": [], "memory": [], "ml": [], "torch": []}

    for metric, systems in ratios.items():
        for sys, ratio in systems.items():
            if sys == baseline or ratio is None:
                continue
            for cat in categories:
                if metric.startswith(cat) and ratio > 1.5:
                    categories[cat].append((sys, ratio))

    for cat, vals in categories.items():
        if vals:
            systems_str = ", ".join([f"{s} ({r:.1f}x)" for s, r in vals])
            messages.append(f"Likely {cat.upper()} bottleneck: {systems_str}")

    # ---------------- GPU underutilization detection ----------------
    def get_metric(metrics: dict, key_candidates: list[str]) -> str | None:
        """Return first matching metric key."""
        for k in key_candidates:
            if k in metrics:
                return k
        return None

    cpu_key = get_metric(metrics, ["torch.cpu", "torch_cpu", "cpu"])
    gpu_key = get_metric(metrics, ["torch.cuda", "torch_cuda", "cuda"])

    if cpu_key and gpu_key:
        cpu_vals = metrics[cpu_key]
        gpu_vals = metrics[gpu_key]

        for system in gpu_vals.keys():
            if system not in cpu_vals:
                continue

            cpu_entry = cpu_vals[system]
            gpu_entry = gpu_vals[system]

            # ---------------- compute GFLOPS safely ----------------
            cpu_gflops = None
            gpu_gflops = None

            if cpu_entry.get("mean") and cpu_entry.get("flops"):
                cpu_gflops = cpu_entry["flops"] / cpu_entry["mean"] / 1e9

            if gpu_entry.get("mean") and gpu_entry.get("flops"):
                gpu_gflops = gpu_entry["flops"] / gpu_entry["mean"] / 1e9

            # ---------------- analysis ----------------
            if cpu_gflops and gpu_gflops:
                speedup = gpu_gflops / cpu_gflops if cpu_gflops > 0 else None

                if speedup is None:
                    continue

                # ---- underutilization rules ----
                if speedup < 1.2:
                    messages.append(
                        f"🚨 Severe GPU underutilization on {system}: "
                        f"GPU ≈ {speedup:.2f}x CPU (likely too small batch or memory-bound)"
                    )

                elif speedup < 1.5:
                    messages.append(
                        f"⚠️ Moderate GPU underutilization on {system}: "
                        f"GPU only {speedup:.2f}x faster than CPU"
                    )

                elif speedup < 3.0:
                    messages.append(
                        f"ℹ️ Weak GPU scaling on {system}: "
                        f"{speedup:.2f}x speedup (check batch size / kernel saturation)"
                    )

    return messages


# -------------------- System table --------------------


def system_table(data: Dict[str, Any]) -> str:
    rows = []
    for name, content in data.items():
        sys = content.get("system", {})
        rows.append(f"""
        <tr>
            <td>{name}</td>
            <td>{sys.get("cpu_count")}</td>
            <td>{sys.get("memory_total_gb"):.1f}</td>
            <td>{sys.get("platform")}</td>
        </tr>
        """)

    return f"""
    <h2>System Info</h2>
    <table border=1 cellpadding=5>
        <tr><th>System</th><th>CPU</th><th>RAM (GB)</th><th>Platform</th></tr>
        {"".join(rows)}
    </table>
    """


# -------------------- HTML --------------------


def generate_html(
    metrics: Dict[str, Any],
    ratios: Dict[str, Any],
    normalized: Dict[str, Any],
    physical: Dict[str, Any],
    data: Dict[str, Any],
    baseline: str,
    diagnosis: List[str],
) -> str:

    html = ["<html><head>"]
    html.append("""
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
    body { font-family: Arial; margin: 40px; }
    canvas { max-width: 900px; margin-bottom: 60px; }
    table { margin-bottom: 40px; }
    .diag { background: #f5f5f5; padding: 15px; margin-bottom: 30px; }
    </style>
    """)
    html.append("</head><body>")

    html.append("<h1>Benchmark Comparison Report</h1>")
    html.append(f"<p><b>Baseline:</b> {baseline}</p>")

    html.append(system_table(data))

    html.append("<div class='diag'><h2>Automatic Diagnosis</h2>")
    for msg in diagnosis:
        html.append(f"<p>⚠️ {msg}</p>")
    html.append("</div>")

    chart_id = 0

    for metric, systems in metrics.items():
        labels = list(systems.keys())
        means = [systems[s]["mean"] for s in labels]
        norm_vals = [normalized[metric][s] for s in labels]
        phys_vals = [physical[metric][s] for s in labels]
        ratio_vals = [ratios.get(metric, {}).get(s) for s in labels]

        html.append(f"<h3>{metric}</h3>")
        html.append(f'<canvas id="chart{chart_id}"></canvas>')

        html.append("<script>")
        html.append(f"""
        new Chart(document.getElementById('chart{chart_id}'), {{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [
                    {{ label: 'Time (s)', data: {means} }},
                    {{ label: 'Throughput (1/s)', data: {norm_vals} }},
                    {{ label: 'Physical Units', data: {phys_vals} }}
                ]
            }},
            options: {{
                plugins: {{
                    tooltip: {{
                        callbacks: {{
                            afterLabel: function(ctx) {{
                                const ratios = {ratio_vals};
                                return 'Ratio vs baseline: ' + ratios[ctx.dataIndex];
                            }}
                        }}
                    }}
                }}
            }}
        }});
        """)
        html.append("</script>")

        chart_id += 1

    html.append("</body></html>")
    return "\n".join(html)


# -------------------- Main --------------------


def main() -> None:
    data = load_results()

    if not data:
        print("No benchmark files found.")
        return

    baseline = BASELINE_NAME or list(data.keys())[0]

    metrics = collect_metrics(data)
    ratios = compute_ratios(metrics, baseline)
    normalized = normalize_metrics(metrics)
    physical = compute_physical_units(metrics)
    diagnosis = diagnose(metrics, ratios, baseline)

    html = generate_html(
        metrics, ratios, normalized, physical, data, baseline, diagnosis
    )

    with open(OUTPUT_HTML, "w") as f:
        f.write(html)

    print(f"Report generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
