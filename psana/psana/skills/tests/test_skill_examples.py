"""Offline checks for documented commands and launcher time assumptions in
the psana-daq skill suite.

These extract the literal fenced command block from a shipped SKILL.md by
marker string and execute it against synthetic fixtures or a loopback HTTP
server, so documented behavior and tested behavior can't drift apart
silently (the pattern this file follows was first demonstrated against an
earlier version of these skills in lcls2 PR #131).

Run: python -m unittest discover -s psana/psana/skills/tests -v
Requires bash, zstd, and curl on PATH; uses only synthetic files and
loopback HTTP — no network access, no DAQ, no production service calls.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

SKILLS = Path(__file__).resolve().parents[1]
REPO = SKILLS.parents[2]
DAQ_LOGS_SCRIPT = SKILLS / "psana-daq-logs" / "scripts" / "daq-logs.py"


def bash_block(skill, marker):
    """Return the first ```bash fenced block in <skill>/SKILL.md containing
    marker."""
    text = (SKILLS / skill / "SKILL.md").read_text()
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        if marker in block:
            return block
    raise AssertionError(f"no bash block containing {marker!r} found in {skill}/SKILL.md")


# ---------------------------------------------------------------------------
# psana-daq-logs/scripts/daq-logs.py — grep subcommand
# ---------------------------------------------------------------------------


class DaqLogsGrepTests(unittest.TestCase):
    """Exercises the real script, not a reimplementation, against synthetic
    plain + zstd-compressed fixtures."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name)

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(DAQ_LOGS_SCRIPT), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def write_session(self, prefix, filename, content, compress=False):
        plain = self.log_dir / f"{prefix}_{filename}"
        plain.write_text(content)
        if compress:
            zipped = Path(str(plain) + ".zst")
            subprocess.run(["zstd", "-q", "-f", str(plain), "-o", str(zipped)], check=True)
            plain.unlink()
            return zipped
        return plain

    def test_plain_and_compressed_counts(self):
        noise = "<C> Inadequate RTPRIO limit: got 0, require 99\n"
        error = "<E> synthetic read failure\n"
        cases = [
            ("rtprio-only", noise, 0),
            ("real-only", error, 1),
            ("mixed", noise + error + "<C> synthetic stop\n", 2),
            ("no-errors", "<I> ready\n<W> waiting\n", 0),
            ("empty", "", 0),
            ("repeated", error * 50, 50),
        ]
        for name, content, expected in cases:
            for compress in (False, True):
                with self.subTest(case=name, compressed=compress):
                    log_dir = Path(tempfile.mkdtemp())
                    self.addCleanup(lambda d=log_dir: subprocess.run(["rm", "-rf", str(d)]))
                    prefix = "15_15:31:17"
                    f = log_dir / f"{prefix}_host:{name} with spaces.log"
                    f.write_text(content)
                    if compress:
                        zipped = Path(str(f) + ".zst")
                        subprocess.run(["zstd", "-q", "-f", str(f), "-o", str(zipped)], check=True)
                        f.unlink()
                    result = self.run_script("grep", str(log_dir), prefix, "--count", "--json")
                    self.assertEqual(result.returncode, 0, result.stderr)
                    out = json.loads(result.stdout)
                    self.assertEqual(out["total"], expected)

    def test_corrupt_zst_is_not_a_successful_empty_read(self):
        prefix = "15_15:31:17"
        corrupt = self.log_dir / f"{prefix}_host:corrupt.log.zst"
        corrupt.write_bytes(b"not zstd data")
        result = self.run_script("grep", str(self.log_dir), prefix, "--count", "--json")
        # The script itself always exits 0 (it treats a decompression
        # failure as "no matches from this file", warned on stderr) --
        # the behavior under test is that the warning is emitted and the
        # count does NOT silently include fabricated matches.
        self.assertEqual(result.returncode, 0)
        self.assertIn("warning", result.stderr)
        out = json.loads(result.stdout)
        self.assertEqual(out["total"], 0)

    def test_filter_then_exclude_order(self):
        prefix = "15_15:31:17"
        content = (
            "<E> real error one\n"
            "<C> Inadequate RTPRIO limit: got 0, require 99\n"
            "<E> real error two\n"
        )
        (self.log_dir / f"{prefix}_host:x.log").write_text(content)
        result = self.run_script(
            "grep", str(self.log_dir), prefix, "--full", "--json",
            "--pattern", "<[EC]>", "--exclude", "Inadequate RTPRIO",
        )
        out = json.loads(result.stdout)
        lines = [r["line"] for r in out["results"]]
        self.assertEqual(len(lines), 2)
        self.assertTrue(all("RTPRIO" not in l for l in lines))

    def test_component_glob_restricts_files(self):
        prefix = "15_15:31:17"
        (self.log_dir / f"{prefix}_drp-srcf-mon001:teb0.log").write_text("<E> teb error\n")
        (self.log_dir / f"{prefix}_drp-srcf-mon001:control.log").write_text("<E> control error\n")
        result = self.run_script(
            "grep", str(self.log_dir), prefix, "--full", "--json", "--component", "teb*"
        )
        out = json.loads(result.stdout)
        self.assertEqual(len(out["results"]), 1)
        self.assertIn("teb0", out["results"][0]["file"])


# ---------------------------------------------------------------------------
# psana-daq-logs/scripts/daq-logs.py — list subcommand (session selection)
# ---------------------------------------------------------------------------


class DaqLogsListTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # <YYYY>/<MM> structure, as the script expects for date anchoring
        self.year_dir = Path(self.tmp.name) / "2026" / "09"
        self.year_dir.mkdir(parents=True)

    def run_script(self, *args):
        return subprocess.run(
            [sys.executable, str(DAQ_LOGS_SCRIPT), "list", str(self.year_dir), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def make_session(self, prefix, mtime=None, content=""):
        f = self.year_dir / f"{prefix}_host:control.log"
        f.write_text(content)
        if mtime is not None:
            ts = mtime.timestamp()
            os.utime(str(f), (ts, ts))
        return f

    def test_interval_overlap_not_same_day_mtime(self):
        # Session started 2026-09-16 (yesterday relative to the window
        # below) but is STILL WRITING on 2026-09-17 -- must be included
        # in a "2026-09-17" window even though it didn't start that day.
        self.make_session("16_23:50:00", mtime=datetime(2026, 9, 17, 2, 0, 0))
        # Session entirely before the window -- must be excluded.
        self.make_session("10_08:00:00", mtime=datetime(2026, 9, 10, 8, 5, 0))
        result = self.run_script(
            "--start", "2026-09-17T00:00:00", "--end", "2026-09-18T00:00:00", "--json"
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads(result.stdout)
        prefixes = {r["prefix"] for r in rows}
        self.assertIn("16_23:50:00", prefixes)
        self.assertNotIn("10_08:00:00", prefixes)

    def test_month_rollover_flagged(self):
        # Started 2026-09-30, still writing into what the directory's own
        # YYYY/MM would otherwise suggest is a different period.
        self.make_session("30_23:00:00", mtime=datetime(2026, 10, 1, 1, 0, 0))
        result = self.run_script(
            "--start", "2026-09-30T00:00:00", "--end", "2026-10-01T06:00:00", "--json"
        )
        rows = json.loads(result.stdout)
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["spans_outside_dir_date"])  # 30 == prefix day

    def test_error_count_uses_canonical_filter(self):
        self.make_session(
            "17_10:00:00",
            mtime=datetime(2026, 9, 17, 10, 5, 0),
            content="<C> Inadequate RTPRIO limit: got 0, require 99\n<E> real failure\n",
        )
        result = self.run_script(
            "--start", "2026-09-17T00:00:00", "--end", "2026-09-18T00:00:00", "--json"
        )
        rows = json.loads(result.stdout)
        self.assertEqual(rows[0]["error_count"], 1)  # RTPRIO excluded
        self.assertIn("real failure", rows[0]["first_error"])


# ---------------------------------------------------------------------------
# psana-configdb — get_history curl example
# ---------------------------------------------------------------------------


class ConfigDbHistoryRequestTests(unittest.TestCase):
    def request(self, status):
        seen = {}

        class Receiver(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["method"] = self.command
                seen["path"] = self.path
                seen["type"] = self.headers.get("Content-Type")
                seen["body"] = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"success": true, "value": []}')

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Receiver)
        server.timeout = 5
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            command = bash_block("psana-configdb", "curl --request GET")
            command = command.replace(
                "https://pswww.slac.stanford.edu/ws/configdb/ws/configDB",
                "http://127.0.0.1:%d" % server.server_port,
            )
            result = subprocess.run(
                ["bash", "-c", command],
                env=dict(os.environ, NO_PROXY="127.0.0.1", no_proxy="127.0.0.1"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        finally:
            worker.join(timeout=6)
            server.server_close()
        return result, seen

    def test_get_sends_json_body_not_query_string(self):
        result, seen = self.request(200)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(seen["method"], "GET")
        self.assertEqual(seen["path"], "/get_history/xpp/BEAM/timing_0/")
        self.assertEqual(seen["type"], "application/json")
        self.assertEqual(json.loads(seen["body"]), ["detName:RO"])

    def test_http_failure_is_nonzero_exit(self):
        result, _ = self.request(503)
        self.assertNotEqual(result.returncode, 0)


# ---------------------------------------------------------------------------
# Launcher time assumptions (psdaq/psdaq/slurm/utils.py) -- underpins
# psana-daq-logs' session-resolution claims about rollover/DST/mtime
# ---------------------------------------------------------------------------


class LauncherEvidenceTests(unittest.TestCase):
    """Exercises the real SbatchManager class with mocked clock/env; no
    Slurm calls are made (output= is passed explicitly)."""

    def setUp(self):
        path = REPO / "psdaq/psdaq/slurm/utils.py"
        spec = importlib.util.spec_from_file_location("skill_launcher_fixture", str(path))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def launch(self, when):
        with patch.object(self.module, "datetime") as clock, patch.dict(
            os.environ, {"USER": "testopr", "HOME": self.tmp.name}
        ):
            clock.now.return_value = when
            return self.module.SbatchManager(
                "fixture.py", 0, 0, 0, False, False, output=self.tmp.name
            )

    def test_midnight_month_and_year_rollover_retain_launch_directory(self):
        cases = [
            (datetime(2026, 1, 10, 23, 58), datetime(2026, 1, 11, 0, 5)),
            (datetime(2026, 1, 31, 23, 58), datetime(2026, 2, 1, 0, 5)),
            (datetime(2025, 12, 31, 23, 58), datetime(2026, 1, 1, 0, 5)),
        ]
        for start, later in cases:
            with self.subTest(start=start):
                old = self.launch(start)
                new = self.launch(later)
                self.assertTrue(old.output_path.endswith(start.strftime("%Y/%m")))
                self.assertEqual(old.output_prefix_datetime, start.strftime("%d_%H:%M:%S"))
                self.assertTrue(new.output_path.endswith(later.strftime("%Y/%m")))
                # old launch's log can still be written after the next launch starts
                log = Path(old.output_path) / (
                    old.output_prefix_datetime + "_fixture:control.log"
                )
                log.write_text("synthetic ongoing activity\n")
                os.utime(str(log), (later.timestamp(), later.timestamp()))
                self.assertEqual(log.stat().st_mtime, later.timestamp())

    def test_prefix_alone_does_not_identify_month_or_launch(self):
        a = self.launch(datetime(2026, 1, 1, 10))
        b = self.launch(datetime(2026, 2, 1, 10))
        c = self.launch(datetime(2026, 1, 1, 10))
        self.assertEqual(a.output_prefix_datetime, b.output_prefix_datetime)
        self.assertNotEqual(a.output_path, b.output_path)
        self.assertEqual(a.output_prefix_datetime, c.output_prefix_datetime)
        self.assertEqual(a.output_path, c.output_path)

    def test_prefix_omits_dst_offset(self):
        # Two distinct instants at a repeated local wall time (the DST
        # fold) produce the identical prefix -- psana-daq-logs' SKILL.md
        # instructs treating this as ambiguous rather than silently
        # normalizing it.
        a = self.launch(datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-7))))
        b = self.launch(datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-8))))
        self.assertEqual(a.output_prefix_datetime, b.output_prefix_datetime)
        self.assertEqual(a.output_path, b.output_path)

    def test_mtime_can_change_without_changing_content(self):
        launch = self.launch(datetime(2025, 12, 31, 23, 58))
        log = Path(launch.output_path) / (
            launch.output_prefix_datetime + "_fixture:control.log"
        )
        log.write_text("2026-01-01T00:01:00Z BeginRun 42\n")
        before = log.read_bytes()
        copied_time = datetime(2026, 1, 3).timestamp()
        os.utime(str(log), (copied_time, copied_time))
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(log.stat().st_mtime, copied_time)


if __name__ == "__main__":
    unittest.main()
