"""Focused failure/rollover checks; the real pilot gets a separate full validator."""
import struct
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from xtc_fixture import HEADER, children, from_ns, insert_times, to_ns, file_sha
from verify_pilot import compare_allowed
import pilot_admin


class FixtureTests(unittest.TestCase):
    def test_second_rollover(self):
        left = from_ns(7_999_999_998)
        right = from_ns(8_000_000_021)
        times = insert_times(left, right, 4)
        self.assertEqual([to_ns(t) for t in times],
                         [8_000_000_003, 8_000_000_008, 8_000_000_013, 8_000_000_018])
        self.assertTrue(all(t >> 32 == 8 and t & 0xffffffff < 1_000_000_000 for t in times))

    def test_tight_gap(self):
        self.assertEqual(insert_times(100, 1914, 4), [553, 1006, 1459, 1912])

    def test_insufficient_gap(self):
        with self.assertRaisesRegex(ValueError, "No space"):
            insert_times(100, 104, 4)

    def test_unnormalized_timestamp(self):
        with self.assertRaisesRegex(ValueError, "Unnormalized"):
            to_ns((7 << 32) | 1_000_000_000)

    def test_truncated_child(self):
        data = HEADER.pack(1, 12 << 24, 0, 0, 0, 24) + struct.pack("<IHHI", 0, 0, 3, 16)
        with self.assertRaisesRegex(ValueError, "Bad child extent"):
            list(children(data))

    def test_reject_control_change(self):
        original = HEADER.pack(1, (9 << 24) | 3, 5, 1, 0, 12)
        changed = HEADER.pack(2, (9 << 24) | 4, 5, 1, 0, 12)
        with self.assertRaisesRegex(ValueError, "Unapproved"):
            compare_allowed(original, changed, 9, 0, {}, {}, 255)

    def test_preserve_damage_with_new_timestamp(self):
        original = HEADER.pack(1, (9 << 24) | 3, 5, 1, 0, 12)
        changed = HEADER.pack(2, (9 << 24) | 3, 5, 1, 0, 12)
        compare_allowed(original, changed, 9, 0, {}, {}, 255)

    def test_missing_smd_directory_rejected_before_auth(self):
        with tempfile.TemporaryDirectory() as name, patch("pilot_admin.DEST", Path(name)), \
                patch("krtc.KerberosTicket", side_effect=AssertionError("Authentication must not run")):
            with self.assertRaisesRegex(ValueError, "Missing or unwritable"):
                pilot_admin.preflight()

    def test_unwritable_directory_rejected_before_auth(self):
        with tempfile.TemporaryDirectory() as name, patch("pilot_admin.DEST", Path(name)), \
                patch("pilot_admin.os.access", return_value=False), \
                patch("krtc.KerberosTicket", side_effect=AssertionError("Authentication must not run")):
            (Path(name) / "smalldata").mkdir()
            with self.assertRaisesRegex(ValueError, "Missing or unwritable"):
                pilot_admin.preflight()

    def test_provenance_rejected_before_publication(self):
        code = Path(__file__).parent
        commit = "a" * 40
        for bad in ("generator", "validator", "missing_validator"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as name:
                dest = Path(name)
                staging = dest / ".synthetic-mfx-r0001-test"
                staging.mkdir()
                generation = {"code_sha256": {n: file_sha(code / n) for n in ("make_pilot.py", "xtc_fixture.py")}}
                if bad == "generator":
                    generation["code_sha256"]["make_pilot.py"] = "wrong"
                (staging / "generation.json").write_text(json.dumps(generation))
                report = {"status": "pass", "generation_sha256": file_sha(staging / "generation.json"),
                          "code_sha256": {n: file_sha(code / n) for n in ("verify_pilot.py", "xtc_fixture.py")}}
                if bad == "validator":
                    report["code_sha256"]["verify_pilot.py"] = "wrong"
                elif bad == "missing_validator":
                    del report["code_sha256"]
                (dest / "report.json").write_text(json.dumps(report))
                (dest / "review.json").write_text(json.dumps({"verdict": "pass", "reviewed_commit": commit, "findings": []}))
                def git_blob(command):
                    return (code / command[-1].split("/")[-1]).read_bytes()
                with patch("pilot_admin.DEST", dest), \
                        patch("pilot_admin.subprocess.check_output", side_effect=git_blob), \
                        patch("pilot_admin.preflight", side_effect=AssertionError("Network must not run")), \
                        patch("pilot_admin.os.link") as link:
                    with self.assertRaisesRegex(ValueError, "Unreviewed .* provenance"):
                        pilot_admin.publish(staging, dest / "report.json", dest / "review.json", commit, dest / "journal")
                    link.assert_not_called()
                    self.assertFalse((dest / "journal").exists())


if __name__ == "__main__":
    unittest.main()
