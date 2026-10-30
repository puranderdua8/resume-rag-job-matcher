"""Percentiles over warm runs, and their macro average across JDs."""

from __future__ import annotations

import statistics


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return quantiles[max(0, min(98, pct - 1))]


def macro_average_latency(
    per_jd_latency: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    stages = next(iter(per_jd_latency.values()))
    n = len(per_jd_latency)
    return {
        stage: {
            "p50_ms": sum(d[stage]["p50_ms"] for d in per_jd_latency.values()) / n,
            "p95_ms": sum(d[stage]["p95_ms"] for d in per_jd_latency.values()) / n,
        }
        for stage in stages
    }
