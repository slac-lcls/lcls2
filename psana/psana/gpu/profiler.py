from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class _Metric:
    count: int = 0
    total_s: float = 0.0
    min_s: float | None = None
    max_s: float = 0.0

    def add(self, value_s: float):
        self.count += 1
        self.total_s += value_s
        if self.min_s is None or value_s < self.min_s:
            self.min_s = value_s
        if value_s > self.max_s:
            self.max_s = value_s

    def as_dict(self):
        avg_s = self.total_s / self.count if self.count else 0.0
        return {
            "count": self.count,
            "total_s": self.total_s,
            "avg_s": avg_s,
            "min_s": self.min_s if self.min_s is not None else 0.0,
            "max_s": self.max_s,
        }


@dataclass
class _TransferMetric:
    count: int = 0
    total_bytes: int = 0
    total_s: float = 0.0
    total_rate_Bps: float = 0.0
    min_bytes: int | None = None
    max_bytes: int = 0
    min_rate_Bps: float | None = None
    max_rate_Bps: float = 0.0

    def add(self, size_bytes: int, dt_s: float):
        if size_bytes <= 0 or dt_s <= 0:
            return
        rate_Bps = float(size_bytes) / float(dt_s)
        self.count += 1
        self.total_bytes += int(size_bytes)
        self.total_s += float(dt_s)
        self.total_rate_Bps += rate_Bps
        if self.min_bytes is None or size_bytes < self.min_bytes:
            self.min_bytes = int(size_bytes)
        if size_bytes > self.max_bytes:
            self.max_bytes = int(size_bytes)
        if self.min_rate_Bps is None or rate_Bps < self.min_rate_Bps:
            self.min_rate_Bps = rate_Bps
        if rate_Bps > self.max_rate_Bps:
            self.max_rate_Bps = rate_Bps

    def as_dict(self):
        avg_bytes = self.total_bytes / self.count if self.count else 0.0
        avg_rate_Bps = self.total_rate_Bps / self.count if self.count else 0.0
        total_rate_Bps = self.total_bytes / self.total_s if self.total_s > 0 else 0.0
        return {
            "count": self.count,
            "total_bytes": self.total_bytes,
            "avg_bytes": avg_bytes,
            "min_bytes": self.min_bytes if self.min_bytes is not None else 0,
            "max_bytes": self.max_bytes,
            "avg_rate_Bps": avg_rate_Bps,
            "min_rate_Bps": self.min_rate_Bps if self.min_rate_Bps is not None else 0.0,
            "max_rate_Bps": self.max_rate_Bps,
            "total_rate_Bps": total_rate_Bps,
            "total_s": self.total_s,
        }


@dataclass
class GpuProfiler:
    mode: str = "off"
    output_path: str | None = None
    logger: object | None = None
    run_label: str | None = None
    emit_summary: bool = True
    _metrics: dict = field(default_factory=lambda: {
        "initialization": _Metric(),
        "event_loop_wall": _Metric(),
        "stage1": _Metric(),
        "queue_wait": _Metric(),
        "transition_drain": _Metric(),
        "copy": _Metric(),
        "kernel": _Metric(),
        "cache_upload": _Metric(),
    })
    _transfer: _TransferMetric = field(default_factory=_TransferMetric)
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

    def record_initialization(self, dt_s: float):
        self._record("initialization", dt_s)

    def record_event_loop_wall(self, dt_s: float):
        self._record("event_loop_wall", dt_s)

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

    def record_transfer(self, size_bytes: int, dt_s: float):
        if self.enabled:
            self._transfer.add(size_bytes, dt_s)

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
            "transfer": self._transfer.as_dict(),
        }

        if self.logger is not None and self.emit_summary:
            loop_total_s = summary["metrics"]["event_loop_wall"]["total_s"]
            events_per_s = (
                float(self._events_completed) / loop_total_s
                if loop_total_s > 0.0
                else 0.0
            )
            self.logger.info(
                "gpu profile summary run=%s events=%d",
                self.run_label,
                self._events_completed,
            )
            self.logger.info(
                "gpu profile cpu_wall_s init=%.3f loop=%.3f rate_evt_s=%.3f",
                summary["metrics"]["initialization"]["total_s"],
                loop_total_s,
                events_per_s,
            )
            for stat_name, field_name in (
                ("avg_s", "avg_s"),
                ("min_s", "min_s"),
                ("max_s", "max_s"),
                ("total_s", "total_s"),
            ):
                self.logger.info(
                    "gpu profile %s stage1=%.3f queue=%.3f drain=%.3f copy=%.3f kernel=%.3f cache_upload=%.3f transfer_size_mib=%.3f transfer_rate_mib_s=%.3f",
                    stat_name,
                    summary["metrics"]["stage1"][field_name],
                    summary["metrics"]["queue_wait"][field_name],
                    summary["metrics"]["transition_drain"][field_name],
                    summary["metrics"]["copy"][field_name],
                    summary["metrics"]["kernel"][field_name],
                    summary["metrics"]["cache_upload"][field_name],
                    _transfer_stat(summary["transfer"], "bytes", stat_name) / (1024.0 * 1024.0),
                    _transfer_stat(summary["transfer"], "rate_Bps", stat_name) / (1024.0 * 1024.0),
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


def _transfer_stat(transfer, quantity, stat_name):
    field_map = {
        ("bytes", "avg_s"): "avg_bytes",
        ("bytes", "min_s"): "min_bytes",
        ("bytes", "max_s"): "max_bytes",
        ("bytes", "total_s"): "total_bytes",
        ("rate_Bps", "avg_s"): "avg_rate_Bps",
        ("rate_Bps", "min_s"): "min_rate_Bps",
        ("rate_Bps", "max_s"): "max_rate_Bps",
        ("rate_Bps", "total_s"): "total_rate_Bps",
    }
    return transfer[field_map[(quantity, stat_name)]]
