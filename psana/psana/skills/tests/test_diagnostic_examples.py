"""Offline checks for documented commands and launcher time assumptions.

Run: python -m unittest discover -s psana/psana/skills/tests -v
Requires Bash, awk, curl and zstd; uses only synthetic files and loopback HTTP.
"""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

SKILLS = Path(__file__).resolve().parents[1]
REPO = SKILLS.parents[2]


def bash_block(skill, marker):
    text = (SKILLS / skill / 'SKILL.md').read_text()
    return next(block for block in re.findall(r'```bash\n(.*?)```', text, re.S)
                if marker in block)


class LogExampleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.command = bash_block('psana-daq-logs', 'count_daq_errors')

    def count(self, path):
        env = dict(os.environ, log_file=str(path))
        return subprocess.run(['bash', '-o', 'pipefail', '-c', self.command],
                              env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)

    def test_plain_and_compressed_counts(self):
        noise = '<C> Inadequate RTPRIO limit: got 0, require 99\n'
        error = '<E> synthetic read failure\n'
        cases = [('rtprio-only', noise, 0), ('real-only', error, 1),
                 ('mixed', noise + error + '<C> synthetic stop\n', 2),
                 ('no-errors', '<I> ready\n<W> waiting\n', 0),
                 ('empty', '', 0), ('repeated', error * 50, 50)]
        for name, content, expected in cases:
            plain = self.directory / (name + ' with spaces.log')
            plain.write_text(content)
            zipped = Path(str(plain) + '.zst')
            subprocess.run(['zstd', '-q', str(plain), '-o', str(zipped)], check=True)
            for path in (plain, zipped):
                with self.subTest(case=name, format=path.suffix):
                    result = self.count(path)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, str(expected) + '\n')

    def test_unreadable_or_corrupt_is_not_successful_zero(self):
        corrupt = self.directory / 'corrupt.log.zst'
        corrupt.write_bytes(b'not zstd data')
        for path in (self.directory / 'absent.log', corrupt):
            with self.subTest(path=path.name):
                self.assertNotEqual(self.count(path).returncode, 0)

    def test_excerpt_line_numbers_survive_decompression(self):
        plain = self.directory / 'excerpt.log'
        plain.write_text('<I> ready\n<C> Inadequate RTPRIO\n<E> synthetic failure\n')
        zipped = Path(str(plain) + '.zst')
        subprocess.run(['zstd', '-q', str(plain), '-o', str(zipped)], check=True)
        self.command = self.command.replace(
            "'/<[EC]>/ && !/Inadequate RTPRIO/ {n++} END {print n+0}'",
            "'/<[EC]>/ && !/Inadequate RTPRIO/ {print NR \":\" $0}'")
        self.assertEqual(self.count(zipped).stdout, '3:<E> synthetic failure\n')


class HistoryRequestTests(unittest.TestCase):
    def request(self, status):
        seen = {}

        class Receiver(BaseHTTPRequestHandler):
            def do_GET(self):
                seen['method'] = self.command
                seen['path'] = self.path
                seen['type'] = self.headers.get('Content-Type')
                seen['body'] = self.rfile.read(int(self.headers.get('Content-Length', 0)))
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"success": true, "value": []}')

            def log_message(self, *args):
                pass

        server = HTTPServer(('127.0.0.1', 0), Receiver)
        server.timeout = 5
        worker = threading.Thread(target=server.handle_request)
        worker.start()
        try:
            command = bash_block('psana-configdb', 'curl --fail')
            command = command.replace(
                'https://pswww.slac.stanford.edu/ws/configdb/ws/configDB',
                'http://127.0.0.1:%d' % server.server_port)
            command = command.replace('<hutch>/<alias>/<device>', 'test/BEAM/det_0')
            result = subprocess.run(['bash', '-c', command],
                                    env=dict(os.environ, NO_PROXY='127.0.0.1', no_proxy='127.0.0.1'),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        finally:
            worker.join(timeout=6)
            server.server_close()
        return result, seen

    def test_get_json_body_not_query_string(self):
        result, seen = self.request(200)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(seen['method'], 'GET')
        self.assertEqual(seen['path'], '/get_history/test/BEAM/det_0/')
        self.assertEqual(seen['type'], 'application/json')
        self.assertEqual(json.loads(seen['body']), ['detName:RO'])

    def test_http_failure_is_nonzero(self):
        result, _ = self.request(503)
        self.assertNotEqual(result.returncode, 0)


class LauncherEvidenceTests(unittest.TestCase):
    """Exercise actual launcher code using synthetic dates/files; no Slurm calls."""
    def setUp(self):
        path = REPO / 'psdaq/psdaq/slurm/utils.py'
        spec = importlib.util.spec_from_file_location('skill_launcher_fixture', str(path))
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def launch(self, when):
        with patch.object(self.module, 'datetime') as clock, patch.dict(
                os.environ, {'USER': 'testopr', 'HOME': self.tmp.name}):
            clock.now.return_value = when
            return self.module.SbatchManager('fixture.py', 0, 0, 0, False, False,
                                             output=self.tmp.name)

    def test_midnight_month_and_year_rollover_retain_launch_directory(self):
        cases = [(datetime(2026, 1, 10, 23, 58), datetime(2026, 1, 11, 0, 5)),
                 (datetime(2026, 1, 31, 23, 58), datetime(2026, 2, 1, 0, 5)),
                 (datetime(2025, 12, 31, 23, 58), datetime(2026, 1, 1, 0, 5))]
        for start, later in cases:
            with self.subTest(start=start):
                old = self.launch(start)
                new = self.launch(later)
                self.assertTrue(old.output_path.endswith(start.strftime('%Y/%m')))
                self.assertEqual(old.output_prefix_datetime, start.strftime('%d_%H:%M:%S'))
                self.assertTrue(new.output_path.endswith(later.strftime('%Y/%m')))
                # Old launch's log can still be written after the next launch.
                log = Path(old.output_path) / (old.output_prefix_datetime + '_fixture:control.log')
                log.write_text('synthetic ongoing activity\n')
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
        # Two distinct instants at a repeated wall time produce the same prefix.
        from datetime import timedelta, timezone
        a = self.launch(datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-7))))
        b = self.launch(datetime(2026, 11, 1, 1, 30, tzinfo=timezone(timedelta(hours=-8))))
        self.assertEqual(a.output_prefix_datetime, b.output_prefix_datetime)
        self.assertEqual(a.output_path, b.output_path)

    def test_mtime_can_change_without_changing_evidence(self):
        launch = self.launch(datetime(2025, 12, 31, 23, 58))
        log = Path(launch.output_path) / (launch.output_prefix_datetime + '_fixture:control.log')
        log.write_text('2026-01-01T00:01:00Z BeginRun 42\n')
        before = log.read_bytes()
        copied_time = datetime(2026, 1, 3).timestamp()
        os.utime(str(log), (copied_time, copied_time))
        self.assertEqual(log.read_bytes(), before)
        self.assertEqual(log.stat().st_mtime, copied_time)


if __name__ == '__main__':
    unittest.main()
