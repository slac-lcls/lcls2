"""
datadev Node Status Monitor
Checks datadev-related files on drp-srcf-cmp001..080 via SSH.

Checks per node:
  1. /proc/datadev_*          - kernel module loaded?
  2. /usr/local/sbin/datadev.ko  - module file present?
  3. /etc/systemd/system/datadev.service  - service file present?
  4. datadev.service contains cfgRxCount?
  5. ExecStart line from datadev.service
"""

import re
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QStatusBar,
    QProgressBar,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QBrush, QColor, QFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NODE_PREFIX = "drp-srcf-cmp"
NODE_RANGE = range(1, 81)  # 001 .. 080
MAX_WORKERS = 5  # parallel SSH connections
SSH_TIMEOUT = 8  # seconds per node

# Remote bash snippet – all checks in one SSH round-trip
CHECK_SCRIPT = r"""
set -o pipefail

# 1) /proc/datadev_*
proc_list=$(ls /proc/datadev_* 2>/dev/null | tr '\n' ',' | sed 's/,$//')
if [ -n "$proc_list" ]; then
    echo "PROC:${proc_list}"
else
    echo "PROC:NONE"
fi

# 2) /usr/local/sbin/datadev.ko
if [ -f /usr/local/sbin/datadev.ko ]; then echo "KO:YES"; else echo "KO:NO"; fi

# 3-5) datadev.service
SVC=/etc/systemd/system/datadev.service
if [ -f "$SVC" ]; then
    echo "SVC:YES"
    if grep -q cfgRxCount "$SVC" 2>/dev/null; then echo "CFG:YES"; else echo "CFG:NO"; fi
    execline=$(grep -m1 '^ExecStart=' "$SVC" 2>/dev/null)
    echo "EXEC:${execline}"
else
    echo "SVC:NO"
    echo "CFG:N/A"
    echo "EXEC:N/A"
fi
"""

# ---------------------------------------------------------------------------
# Ansible inventory
# ---------------------------------------------------------------------------
HOSTS_FILE = "/sdf/home/p/psrel/git/daqnodeconfig/ansible/hosts"


def _expand_range(pattern: str) -> list:
    """Expand 'prefix[01:10]' → ['prefix01', …, 'prefix10']."""
    m = re.match(r"^([^\[]+)\[(\d+):(\d+)\]$", pattern)
    if not m:
        return [pattern]
    prefix, start_s, end_s = m.group(1), m.group(2), m.group(3)
    width = max(len(start_s), len(end_s))
    return [f"{prefix}{i:0{width}d}" for i in range(int(start_s), int(end_s) + 1)]


def parse_ansible_hosts(path: str) -> dict:
    """
    Parse an Ansible INI-style inventory file.
    Returns {hostname: [group1, group2, …]} ordered by first appearance.
    """
    groups: dict = {}
    current_group = None
    try:
        with open(path) as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("["):
                    # strip inline comment from header
                    ci = line.find("#")
                    if ci > 0:
                        line = line[:ci].strip()
                    if line.endswith("]"):
                        current_group = line[1:-1].strip()
                        groups.setdefault(current_group, [])
                    continue
                # host line – strip inline comment
                ci = line.find("#")
                if ci >= 0:
                    line = line[:ci].strip()
                if line and current_group is not None:
                    for h in _expand_range(line):
                        groups[current_group].append(h)
    except OSError:
        pass
    # invert to hostname → [groups]
    host_groups: dict = {}
    for grp, hosts in groups.items():
        for h in hosts:
            host_groups.setdefault(h, []).append(grp)
    return host_groups


# ---------------------------------------------------------------------------
# Column indices
# ---------------------------------------------------------------------------
COL_NODE = 0
COL_GROUPS = 1
COL_PROC = 2
COL_KO = 3
COL_SVC = 4
COL_CFG = 5
COL_EXEC = 6
NUM_COLS = 7

HEADERS = [
    "Node",
    "Groups",
    "/proc/datadev_*",
    "datadev.ko",
    "datadev.service",
    "cfgRxCount",
    "ExecStart",
]

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
GREEN = QColor("#c8f5c8")
RED = QColor("#f5c8c8")
GRAY = QColor("#cccccc")
AMBER = QColor("#f5e8c8")


# ---------------------------------------------------------------------------
# SSH worker
# ---------------------------------------------------------------------------


def check_node(hostname: str) -> dict:
    """Run all checks on one node over SSH. Returns a result dict."""
    result = {
        "hostname": hostname,
        "reachable": False,
        "proc": "—",
        "proc_names": "",
        "ko": "—",
        "svc": "—",
        "cfg": "—",
        "exec": "—",
    }

    try:
        proc = subprocess.run(
            [
                "ssh",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "LogLevel=ERROR",
                hostname,
                "bash -s",
            ],
            input=CHECK_SCRIPT,
            capture_output=True,
            text=True,
            timeout=SSH_TIMEOUT,
        )

        # If stderr only (no stdout) we treat as unreachable
        if not proc.stdout.strip():
            result["exec"] = proc.stderr.strip()[:80] or "no output"
            return result

        result["reachable"] = True

        for line in proc.stdout.splitlines():
            if line.startswith("PROC:"):
                val = line[5:]
                if val == "NONE":
                    result["proc"] = "None"
                else:
                    names = [p.replace("/proc/", "") for p in val.split(",") if p]
                    result["proc"] = ", ".join(names)
                    result["proc_names"] = val
            elif line.startswith("KO:"):
                result["ko"] = line[3:]
            elif line.startswith("SVC:"):
                result["svc"] = line[4:]
            elif line.startswith("CFG:"):
                result["cfg"] = line[4:]
            elif line.startswith("EXEC:"):
                result["exec"] = line[5:]

    except subprocess.TimeoutExpired:
        result["exec"] = "timeout"
    except Exception as exc:
        result["exec"] = str(exc)[:80]

    return result


# ---------------------------------------------------------------------------
# Qt worker object (runs in a QThread)
# ---------------------------------------------------------------------------


class NodeChecker(QObject):
    result_ready = pyqtSignal(dict)
    progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, hostnames: list):
        super().__init__()
        self._hostnames = hostnames
        self._running = True

    def run(self):
        total = len(self._hostnames)
        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
            futures = {exe.submit(check_node, h): h for h in self._hostnames}
            for future in as_completed(futures):
                if not self._running:
                    break
                self.result_ready.emit(future.result())
                done += 1
                self.progress.emit(int(done * 100 / total))
        self.finished.emit()

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Helper – styled table cell
# ---------------------------------------------------------------------------


def make_item(text: str, align=Qt.AlignCenter, tooltip: str = "") -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setTextAlignment(align)
    if tooltip:
        item.setToolTip(tooltip)
    return item


def colour_item(item: QTableWidgetItem, colour: QColor):
    item.setBackground(colour)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("datadev Node Status Monitor")
        self.resize(1500, 900)

        self._hostnames = [f"{NODE_PREFIX}{i:03d}" for i in NODE_RANGE]
        self._row_map = {h: i for i, h in enumerate(self._hostnames)}
        self._host_groups = parse_ansible_hosts(HOSTS_FILE)

        self._checker = None
        self._checker_thread = None

        self._build_ui()
        self._populate_table()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ---- top toolbar ----
        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self._refresh_btn = QPushButton("Refresh All")
        self._refresh_btn.setFixedWidth(110)
        self._refresh_btn.clicked.connect(self._start_refresh)
        toolbar.addWidget(self._refresh_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        self._progress.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        toolbar.addWidget(self._progress)

        self._ts_label = QLabel("Not yet refreshed")
        self._ts_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        toolbar.addWidget(self._ts_label)

        root.addLayout(toolbar)

        # ---- legend ----
        legend = QHBoxLayout()
        for colour, label in [
            (GREEN, "OK / present"),
            (RED, "Missing / No"),
            (GRAY, "Unreachable"),
            (AMBER, "N/A"),
        ]:
            swatch = QLabel("  ")
            swatch.setFixedSize(18, 18)
            swatch.setStyleSheet(f"background:{colour.name()}; border:1px solid #888;")
            legend.addWidget(swatch)
            legend.addWidget(QLabel(label))
            legend.addSpacing(16)
        legend.addStretch()
        root.addLayout(legend)

        # ---- table ----
        self._table = QTableWidget()
        self._table.setColumnCount(NUM_COLS)
        self._table.setHorizontalHeaderLabels(HEADERS)
        self._table.setSortingEnabled(True)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setAlternatingRowColors(False)
        self._table.setWordWrap(True)
        self._table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        hdr = self._table.horizontalHeader()
        for col in range(NUM_COLS - 1):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(COL_EXEC, QHeaderView.Stretch)

        # Monospace font for ExecStart
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.Monospace)
        mono.setPointSize(8)
        self._table.setFont(mono)

        root.addWidget(self._table)

        # ---- status bar ----
        self._status = QStatusBar()
        self.setStatusBar(self._status)

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self):
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(self._hostnames))
        for row, hostname in enumerate(self._hostnames):
            self._table.setItem(row, COL_NODE, make_item(hostname))
            # Groups column – static, filled once from the inventory file
            grps = self._host_groups.get(hostname, [])
            grp_text = "\n".join(grps) if grps else "—"
            grp_item = make_item(
                grp_text, align=Qt.AlignLeft | Qt.AlignVCenter, tooltip="\n".join(grps)
            )
            self._table.setItem(row, COL_GROUPS, grp_item)
            # remaining dynamic columns
            for col in range(COL_PROC, NUM_COLS):
                self._table.setItem(row, col, make_item("—"))
        self._table.setSortingEnabled(True)

    def _reset_pending(self):
        """Mark all data cells as '…' while a refresh is running (skip Groups)."""
        self._table.setSortingEnabled(False)
        for row in range(self._table.rowCount()):
            for col in range(COL_PROC, NUM_COLS):  # skip COL_GROUPS
                item = self._table.item(row, col)
                item.setText("…")
                item.setBackground(QBrush())
                item.setToolTip("")
        self._table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # Refresh logic
    # ------------------------------------------------------------------

    def _start_refresh(self):
        if self._checker_thread and self._checker_thread.isRunning():
            return

        self._refresh_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setValue(0)
        self._status.showMessage("Checking nodes …")
        self._reset_pending()

        self._checker = NodeChecker(self._hostnames)
        self._checker_thread = QThread()
        self._checker.moveToThread(self._checker_thread)

        self._checker_thread.started.connect(self._checker.run)
        self._checker.result_ready.connect(self._update_row)
        self._checker.progress.connect(self._progress.setValue)
        self._checker.finished.connect(self._on_done)

        self._checker_thread.start()

    def _on_done(self):
        self._checker_thread.quit()
        self._checker_thread.wait()
        self._refresh_btn.setEnabled(True)
        self._progress.setVisible(False)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._ts_label.setText(f"Last refresh: {now}")

        # Tally for status bar
        reachable = ok_svc = ok_cfg = 0
        for row in range(self._table.rowCount()):
            node_item = self._table.item(row, COL_NODE)
            if not node_item:
                continue
            h = node_item.text()
            r = self._row_map.get(h, -1)
            if r < 0:
                continue
            svc_item = self._table.item(row, COL_SVC)
            cfg_item = self._table.item(row, COL_CFG)
            ko_item = self._table.item(row, COL_KO)
            if ko_item and ko_item.text() not in ("—", "…", "unreachable"):
                reachable += 1
            if svc_item and svc_item.text() == "YES":
                ok_svc += 1
            if cfg_item and cfg_item.text() == "YES":
                ok_cfg += 1

        self._status.showMessage(
            f"Done {now}  |  {reachable}/{len(self._hostnames)} reachable  |  "
            f"{ok_svc} with service  |  {ok_cfg} with cfgRxCount"
        )

    # ------------------------------------------------------------------
    # Row update (called from worker thread via Qt signal)
    # ------------------------------------------------------------------

    def _update_row(self, result: dict):
        hostname = result["hostname"]
        # Find the current visual row for this hostname
        row = self._find_row(hostname)
        if row < 0:
            return

        self._table.setSortingEnabled(False)

        if not result["reachable"]:
            for col in range(1, NUM_COLS):
                item = self._table.item(row, col)
                item.setText("unreachable" if col == COL_KO else "—")
                colour_item(item, GRAY)
            # Put error hint in ExecStart column
            item = self._table.item(row, COL_EXEC)
            item.setText(result.get("exec", ""))
            self._table.setSortingEnabled(True)
            return

        # /proc/datadev_*
        proc_text = result["proc"]
        proc_item = self._table.item(row, COL_PROC)
        proc_item.setText(proc_text)
        if proc_text == "None":
            colour_item(proc_item, RED)
            proc_item.setToolTip("No /proc/datadev_* entries found")
        else:
            colour_item(proc_item, GREEN)
            proc_item.setToolTip(result.get("proc_names", ""))

        # datadev.ko
        ko_item = self._table.item(row, COL_KO)
        ko_item.setText(result["ko"])
        colour_item(ko_item, GREEN if result["ko"] == "YES" else RED)

        # datadev.service
        svc_item = self._table.item(row, COL_SVC)
        svc_item.setText(result["svc"])
        colour_item(svc_item, GREEN if result["svc"] == "YES" else RED)

        # cfgRxCount
        cfg_item = self._table.item(row, COL_CFG)
        cfg_item.setText(result["cfg"])
        if result["cfg"] == "YES":
            colour_item(cfg_item, GREEN)
        elif result["cfg"] == "N/A":
            colour_item(cfg_item, AMBER)
        else:
            colour_item(cfg_item, RED)

        # ExecStart – keep default background (no custom colour)
        exec_item = self._table.item(row, COL_EXEC)
        exec_item.setText(result["exec"])
        exec_item.setBackground(QBrush())  # clear any leftover colour
        exec_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        exec_item.setToolTip(result["exec"])

        self._table.setSortingEnabled(True)

    def _find_row(self, hostname: str) -> int:
        """Return the current visual row index for hostname (table may be sorted)."""
        for row in range(self._table.rowCount()):
            item = self._table.item(row, COL_NODE)
            if item and item.text() == hostname:
                return row
        return -1

    def closeEvent(self, event):
        if self._checker:
            self._checker.stop()
        if self._checker_thread:
            self._checker_thread.quit()
            self._checker_thread.wait(2000)
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Slightly larger base font
    font = app.font()
    font.setPointSize(9)
    app.setFont(font)

    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
