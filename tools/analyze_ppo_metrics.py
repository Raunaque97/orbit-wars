from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_RUN_DIR = Path("rl/checkpoints/ppo_psro_v1")
DEFAULT_LAST = 20

KL_WARN = 0.20
KL_BAD = 1.00
CLIP_WARN = 0.30
CLIP_BAD = 0.50
EV_WARN_AFTER_ITER = 10
EV_WARN = 0.30
CACHE_HIT_WARN = 0.99
INVALID_RATE_WARN = 0.10


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
    return rows


def _stats(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row.get("stats", {}))


def _value(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = _stats(row).get(key, default)
    if value is None:
        return default
    return float(value)


def _invalid_rate(row: dict[str, Any]) -> float:
    stats = _stats(row)
    transitions = max(1.0, float(stats.get("transitions", 0) or 0))
    invalid = sum(int(v) for v in dict(stats.get("invalid_counts", {})).values())
    return invalid / transitions


def _metric_for_warning(row: dict[str, Any], per_decision_key: str, joint_key: str) -> float:
    stats = _stats(row)
    if per_decision_key in stats:
        return _value(row, per_decision_key)
    return _value(row, joint_key)


def _cache_hit_rate(row: dict[str, Any]) -> float:
    stats = _stats(row)
    calls = max(1.0, float(stats.get("feature_calls", 0) or 0))
    hits = float(stats.get("delay_cache_hits", 0) or 0)
    return hits / calls


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return mean(_value(row, key) for row in rows)


def _population_line(row: dict[str, Any]) -> str:
    population = row.get("opponent_population", [])
    if not isinstance(population, list):
        return ""
    parts = []
    for entry in population:
        parts.append(
            f"{entry.get('name')}:{entry.get('learner_win_rate')}@{entry.get('sampling_weight')}"
        )
    return ", ".join(parts)


def _warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    for row in rows:
        iteration = int(row.get("iteration", _stats(row).get("iteration", 0)))
        kl = _metric_for_warning(row, "approx_kl_per_decision", "approx_kl")
        clip = _metric_for_warning(row, "clip_fraction_per_decision", "clip_fraction")
        ev = _value(row, "explained_variance")
        cache_hit = _cache_hit_rate(row)
        invalid_rate = _invalid_rate(row)
        if kl >= KL_BAD:
            warnings.append(f"iter {iteration}: KL is very high ({kl:.4f})")
        elif kl >= KL_WARN:
            warnings.append(f"iter {iteration}: KL is high ({kl:.4f})")
        if clip >= CLIP_BAD:
            warnings.append(f"iter {iteration}: clip fraction is very high ({clip:.3f})")
        elif clip >= CLIP_WARN:
            warnings.append(f"iter {iteration}: clip fraction is high ({clip:.3f})")
        if iteration >= EV_WARN_AFTER_ITER and ev < EV_WARN:
            warnings.append(f"iter {iteration}: explained variance is low ({ev:.3f})")
        if cache_hit < CACHE_HIT_WARN:
            warnings.append(f"iter {iteration}: delay cache hit rate is low ({cache_hit:.3f})")
        if invalid_rate > INVALID_RATE_WARN:
            warnings.append(f"iter {iteration}: invalid action rate is high ({invalid_rate:.3f})")
    return warnings


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "iteration",
        "win_rate",
        "rolling_train_win_rate",
        "mean_reward",
        "mean_length",
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "explained_variance",
        "approx_kl",
        "clip_fraction",
        "approx_kl_per_decision",
        "clip_fraction_per_decision",
        "mean_action_terms",
        "transitions",
        "cache_hit_rate",
        "invalid_rate",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(columns) + "\n")
        for row in rows:
            stats = _stats(row)
            values: dict[str, Any] = {
                "iteration": row.get("iteration", stats.get("iteration")),
                "cache_hit_rate": _cache_hit_rate(row),
                "invalid_rate": _invalid_rate(row),
            }
            values.update(stats)
            fh.write(",".join(str(values.get(column, "")) for column in columns) + "\n")


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    import matplotlib.pyplot as plt

    iterations = [int(row.get("iteration", _stats(row).get("iteration", 0))) for row in rows]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(iterations, [_value(row, "win_rate") for row in rows], label="win")
    axes[0].plot(
        iterations,
        [_value(row, "rolling_train_win_rate") for row in rows],
        label="rolling",
    )
    axes[0].set_ylabel("win rate")
    axes[0].legend()
    axes[1].plot(iterations, [_value(row, "explained_variance") for row in rows])
    axes[1].axhline(0.9, color="green", linestyle="--", linewidth=1)
    axes[1].set_ylabel("EV")
    axes[2].plot(iterations, [_value(row, "approx_kl") for row in rows], label="KL")
    axes[2].axhline(KL_WARN, color="orange", linestyle="--", linewidth=1)
    axes[2].axhline(KL_BAD, color="red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("KL")
    axes[3].plot(iterations, [_value(row, "clip_fraction") for row in rows])
    axes[3].axhline(CLIP_WARN, color="orange", linestyle="--", linewidth=1)
    axes[3].axhline(CLIP_BAD, color="red", linestyle="--", linewidth=1)
    axes[3].set_ylabel("clip frac")
    axes[3].set_xlabel("iteration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)


def _polyline(
    rows: list[dict[str, Any]],
    *,
    key: str,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> str:
    if not rows:
        return ""
    if len(rows) == 1:
        x_values = [x0 + width / 2.0]
    else:
        x_values = [x0 + width * idx / (len(rows) - 1) for idx in range(len(rows))]
    denom = max(1e-9, y_max - y_min)
    points = []
    for x, row in zip(x_values, rows):
        value = max(y_min, min(y_max, _value(row, key)))
        y = y0 + height - ((value - y_min) / denom) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _polyline_for_values(
    values: list[float],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> str:
    if not values:
        return ""
    if len(values) == 1:
        x_values = [x0 + width / 2.0]
    else:
        x_values = [x0 + width * idx / (len(values) - 1) for idx in range(len(values))]
    denom = max(1e-9, y_max - y_min)
    points = []
    for x, value in zip(x_values, values):
        clipped = max(y_min, min(y_max, value))
        y = y0 + height - ((clipped - y_min) / denom) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _write_entropy_loss_svg(rows: list[dict[str, Any]], path: Path) -> None:
    width = 980
    chart_h = 150
    gap = 32
    left = 86
    right = 28
    top = 44
    plot_w = width - left - right

    loss_values = {
        "loss": [_value(row, "loss") for row in rows],
        "value_loss": [_value(row, "value_loss") for row in rows],
        "policy_loss": [_value(row, "policy_loss") for row in rows],
    }
    max_loss = max(
        1.0,
        max(max(values) for values in loss_values.values() if values),
    )
    min_policy = min(loss_values["policy_loss"]) if loss_values["policy_loss"] else 0.0
    loss_min = min(0.0, min_policy)
    loss_max = max_loss * 1.05
    latest_iter = rows[-1].get("iteration", _stats(rows[-1]).get("iteration", "?"))
    latest = rows[-1]
    recent = rows[-20:]
    charts = [
        (
            "entropy",
            "entropy",
            [_value(row, "entropy") for row in rows],
            0.0,
            max(1.0, max(_value(row, "entropy") for row in rows) * 1.08),
            [("entropy", "#2563eb")],
        ),
        (
            "loss",
            "loss values",
            [],
            loss_min,
            loss_max,
            [
                ("loss", "#dc2626"),
                ("value_loss", "#9333ea"),
                ("policy_loss", "#0f766e"),
            ],
        ),
    ]
    height = top + len(charts) * chart_h + (len(charts) - 1) * gap + 70
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="24" font-family="monospace" font-size="16" fill="#111827">Entropy and loss through iter {latest_iter}</text>',
        (
            f'<text x="{left}" y="40" font-family="monospace" font-size="12" fill="#4b5563">'
            f'latest entropy={_value(latest, "entropy"):.3f}, loss={_value(latest, "loss"):.3f}, '
            f'value_loss={_value(latest, "value_loss"):.3f}, policy_loss={_value(latest, "policy_loss"):.3f}; '
            f'last20 entropy={_mean(recent, "entropy"):.3f}, loss={_mean(recent, "loss"):.3f}'
            "</text>"
        ),
    ]
    for idx, (_key, label, values, y_min, y_max, series) in enumerate(charts):
        y0 = top + idx * (chart_h + gap)
        parts.extend(
            [
                f'<text x="16" y="{y0 + 16}" font-family="monospace" font-size="13" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{chart_h}" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<line x1="{left}" y1="{y0 + chart_h}" x2="{left + plot_w}" y2="{y0 + chart_h}" stroke="#9ca3af"/>',
                f'<text x="{left - 58}" y="{y0 + 12}" font-family="monospace" font-size="11" fill="#6b7280">{y_max:.2f}</text>',
                f'<text x="{left - 58}" y="{y0 + chart_h}" font-family="monospace" font-size="11" fill="#6b7280">{y_min:.2f}</text>',
            ]
        )
        if values:
            points = _polyline_for_values(
                values,
                x0=left,
                y0=y0,
                width=plot_w,
                height=chart_h,
                y_min=y_min,
                y_max=y_max,
            )
            parts.append(
                f'<polyline fill="none" stroke="{series[0][1]}" stroke-width="2.2" points="{points}"/>'
            )
        else:
            legend_x = left + 12
            for series_idx, (series_key, color) in enumerate(series):
                series_values = loss_values[series_key]
                points = _polyline_for_values(
                    series_values,
                    x0=left,
                    y0=y0,
                    width=plot_w,
                    height=chart_h,
                    y_min=y_min,
                    y_max=y_max,
                )
                parts.append(
                    f'<polyline fill="none" stroke="{color}" stroke-width="2.0" points="{points}"/>'
                )
                lx = legend_x + series_idx * 150
                parts.extend(
                    [
                        f'<line x1="{lx}" y1="{y0 + 18}" x2="{lx + 28}" y2="{y0 + 18}" stroke="{color}" stroke-width="2.0"/>',
                        f'<text x="{lx + 34}" y="{y0 + 22}" font-family="monospace" font-size="12" fill="#374151">{series_key}</text>',
                    ]
                )
    if rows:
        first_iter = rows[0].get("iteration", _stats(rows[0]).get("iteration", "?"))
        parts.append(
            f'<text x="{left}" y="{height - 24}" font-family="monospace" font-size="12" fill="#6b7280">x-axis: iterations {first_iter}..{latest_iter}</text>'
        )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_svg(rows: list[dict[str, Any]], path: Path) -> None:
    charts = [
        ("win_rate", "win rate", 0.0, 1.0, "#2563eb"),
        ("explained_variance", "explained variance", 0.0, 1.0, "#16a34a"),
        ("approx_kl_per_decision", "KL per decision", 0.0, 0.02, "#dc2626"),
        ("clip_fraction_per_decision", "clip fraction per decision", 0.0, 0.10, "#9333ea"),
    ]
    width = 980
    chart_h = 155
    gap = 32
    left = 82
    right = 24
    top = 36
    plot_w = width - left - right
    height = top + len(charts) * chart_h + (len(charts) - 1) * gap + 42
    latest_iter = rows[-1].get("iteration", _stats(rows[-1]).get("iteration", "?"))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="22" font-family="monospace" font-size="16" fill="#111827">PPO metrics through iter {latest_iter}</text>',
    ]
    for idx, (key, label, y_min, y_max, color) in enumerate(charts):
        y0 = top + idx * (chart_h + gap)
        parts.extend(
            [
                f'<text x="16" y="{y0 + 16}" font-family="monospace" font-size="13" fill="#111827">{label}</text>',
                f'<rect x="{left}" y="{y0}" width="{plot_w}" height="{chart_h}" fill="#f9fafb" stroke="#d1d5db"/>',
                f'<line x1="{left}" y1="{y0 + chart_h}" x2="{left + plot_w}" y2="{y0 + chart_h}" stroke="#9ca3af"/>',
                f'<text x="{left - 48}" y="{y0 + 12}" font-family="monospace" font-size="11" fill="#6b7280">{y_max:g}</text>',
                f'<text x="{left - 48}" y="{y0 + chart_h}" font-family="monospace" font-size="11" fill="#6b7280">{y_min:g}</text>',
            ]
        )
        if key == "explained_variance":
            guide_y = y0 + chart_h - ((0.9 - y_min) / max(1e-9, y_max - y_min)) * chart_h
            parts.append(
                f'<line x1="{left}" y1="{guide_y:.1f}" x2="{left + plot_w}" y2="{guide_y:.1f}" stroke="#22c55e" stroke-dasharray="4 4"/>'
            )
        points = _polyline(
            rows,
            key=key,
            x0=left,
            y0=y0,
            width=plot_w,
            height=chart_h,
            y_min=y_min,
            y_max=y_max,
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.2" points="{points}"/>'
        )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--last", type=int, default=DEFAULT_LAST)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument("--entropy-loss-svg", type=Path, default=None)
    args = parser.parse_args()

    metrics_path = args.metrics or args.run_dir / "metrics.jsonl"
    rows = _load_rows(metrics_path)
    if not rows:
        print(f"no rows in {metrics_path}")
        return

    recent = rows[-max(1, args.last) :]
    latest = rows[-1]
    latest_stats = _stats(latest)
    print(f"metrics={metrics_path}")
    print(
        f"rows={len(rows)} latest_iter={latest.get('iteration', latest_stats.get('iteration'))}"
    )
    print(
        "latest "
        f"win={_value(latest, 'win_rate'):.3f} "
        f"rolling={_value(latest, 'rolling_train_win_rate'):.3f} "
        f"reward={_value(latest, 'mean_reward'):.3f} "
        f"len={_value(latest, 'mean_length'):.1f} "
        f"ev={_value(latest, 'explained_variance'):.3f} "
        f"kl={_value(latest, 'approx_kl'):.5f} "
        f"clip={_value(latest, 'clip_fraction'):.3f} "
        f"kl_dec={_value(latest, 'approx_kl_per_decision'):.5f} "
        f"clip_dec={_value(latest, 'clip_fraction_per_decision'):.3f} "
        f"terms={_value(latest, 'mean_action_terms'):.1f} "
        f"cache={_cache_hit_rate(latest):.3f} "
        f"invalid={_invalid_rate(latest):.3f}"
    )
    print(
        f"last_{len(recent)} "
        f"win={_mean(recent, 'win_rate'):.3f} "
        f"reward={_mean(recent, 'mean_reward'):.3f} "
        f"ev={_mean(recent, 'explained_variance'):.3f} "
        f"kl={_mean(recent, 'approx_kl'):.5f} "
        f"clip={_mean(recent, 'clip_fraction'):.3f} "
        f"kl_dec={_mean(recent, 'approx_kl_per_decision'):.5f} "
        f"clip_dec={_mean(recent, 'clip_fraction_per_decision'):.3f}"
    )
    population = _population_line(latest)
    if population:
        print(f"population {population}")

    warnings = _warnings(rows)
    if warnings:
        print("warnings:")
        for warning in warnings[-30:]:
            print(f"- {warning}")
    else:
        print("warnings: none")

    if args.csv is not None:
        _write_csv(rows, args.csv)
        print(f"wrote_csv={args.csv}")
    if args.plot is not None:
        try:
            _plot(rows, args.plot)
            print(f"wrote_plot={args.plot}")
        except ModuleNotFoundError as exc:
            raise SystemExit("matplotlib is not installed; use --csv instead") from exc
    if args.svg is not None:
        _write_svg(rows, args.svg)
        print(f"wrote_svg={args.svg}")
    if args.entropy_loss_svg is not None:
        _write_entropy_loss_svg(rows, args.entropy_loss_svg)
        print(f"wrote_entropy_loss_svg={args.entropy_loss_svg}")


if __name__ == "__main__":
    main()
