from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Metric:
    count: int = 0
    total_s: float = 0.0
    max_s: float = 0.0

    def add(self, value_s: float):
        self.count += 1
        self.total_s += value_s
        if value_s > self.max_s:
            self.max_s = value_s

    def as_dict(self):
        avg_s = self.total_s / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_s": self.total_s,
            "avg_s": avg_s,
            "max_s": self.max_s,
        }


@dataclass
class GpuProfiler:
    mode: str = "off"
    output_path: str | None = None
    logger: object | None = None
    run_label: str | None = None
    _metrics: dict = field(default_factory=lambda: {
        "stage1": _Metric(),
        "queue_wait": _Metric(),
        "transition_drain": _Metric(),
        "copy": _Metric(),
        "kernel": _Metric(),
        "cache_upload": _Metric(),
    })
    _events_completed: int = 0
    _flush_count: int = 0

    @classmethod
    def from_dsparms(cls, dsparms, logger=None, run_label=None):
        return cls(
            mode=getattr(dsparms, "gpu_profile", "off"),
            output_path=getattr(dsparms, "gpu_profile_output", None),
            logger=logger,
            run_label=run_label,
        )

    @property
    def enabled(self):
        return self.mode != "off"

    def record_stage1(self, dt_s: float):
        self._record("stage1", dt_s)

    def record_queue_wait(self, dt_s: float):
        self._record("queue_wait", dt_s)

    def record_transition_drain(self, dt_s: float):
        self._record("transition_drain", dt_s)

    def record_copy(self, dt_s: float):
        self._record("copy", dt_s)

    def record_kernel(self, dt_s: float):
        self._record("kernel", dt_s)

    def record_cache_upload(self, dt_s: float):
        self._record("cache_upload", dt_s)

    def record_event_completed(self):
        if self.enabled:
            self._events_completed += 1

    def flush_summary(self):
        if not self.enabled:
            return None

        summary = {
            "run": self.run_label,
            "mode": self.mode,
            "events_completed": self._events_completed,
            "metrics": {name: metric.as_dict() for name, metric in self._metrics.items()},
        }

        if self.logger is not None:
            self.logger.info(
                "gpu profile summary run=%s events=%d stage1_avg_ms=%.3f queue_avg_ms=%.3f drain_avg_ms=%.3f",
                self.run_label,
                self._events_completed,
                1e3 * summary["metrics"]["stage1"]["avg_s"],
                1e3 * summary["metrics"]["queue_wait"]["avg_s"],
                1e3 * summary["metrics"]["transition_drain"]["avg_s"],
            )

        if self.output_path:
            self._flush_count += 1
            payload = dict(summary)
            payload["flush_count"] = self._flush_count
            path = Path(self.output_path)
            if self.mode == "trace":
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(payload, sort_keys=True) + "\n")
            else:
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        return summary

    def _record(self, name: str, dt_s: float):
        if not self.enabled:
            return
        metric = self._metrics.get(name)
        if metric is None:
            metric = self._metrics[name] = _Metric()
        metric.add(dt_s)
