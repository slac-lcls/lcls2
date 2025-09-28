import argparse
import time
import sys
from typing import TypedDict

import epics
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtCore import Qt, QThread
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QMessageBox, QProgressBar


class TimeVarDict(TypedDict):
    status: int
    severity: int
    timestamp: float
    posixseconds: float
    nanoseconds: int


class BlockerPVMonitor(QThread):
    trippedSignal = Signal(bool)
    npixOverThreshSignal = Signal(int)  # Threshold pixel count before "tripping"
    npixOverSignal = Signal(int)  # Number of pixels exceeding ADU threshold
    aduThreshSignal = Signal(int)  # ADU threshold to consider pixel "hot"

    def __init__(self, base_pv: str):
        super().__init__()

        self._base_pv: str = base_pv
        self._npix_ot_thresh: int = 0
        self._npix_ot: int = 0
        self._adu_thresh: int = 0

        self._running: bool = True
        self._should_report: bool = True

    def run(self):
        t0: float = time.monotonic()
        last_tripped_ts: float = -1
        blocked_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:BLOCKED")
        npix_ot_thresh_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:NPIX")
        npix_ot_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:NPIX_OT")
        adu_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:ADU")
        while self._running:
            if blocked_pv.connected:
                time_vars: TimeVarDict = blocked_pv.get_timevars()
                curr_ts: float = time_vars["timestamp"]
                npix_ot: int = npix_ot_pv.get()
                npix_ot_thresh: int = npix_ot_thresh_pv.get()
                adu: int = adu_pv.get()
                # Set thresholds first - may get division by zero otherwise
                if npix_ot_thresh != self._npix_ot_thresh:
                    self._npix_ot_thresh = npix_ot_thresh
                    self.npixOverThreshSignal.emit(npix_ot_thresh)
                if adu != self._adu_thresh:
                    self._adu_thresh = adu
                    self.aduThreshSignal.emit(adu)
                if npix_ot != self._npix_ot:
                    self._npix_ot = npix_ot
                    self.npixOverSignal.emit(npix_ot)
                if last_tripped_ts == -1:
                    last_tripped_ts = curr_ts
                elif last_tripped_ts != curr_ts and self._should_report:
                    self.trippedSignal.emit(True)
                    # Prevent generation of multiple dialogs
                    self._should_report = False

                tnow: float
                if (tnow := time.monotonic()) > (t0 + 60):
                    print(f"ADU Threshold: {adu}, Pixels Over Threshold: {npix_ot}")
                    t0 = tnow

    @Slot(bool)
    def report(self, should_report: bool) -> None:
        self._should_report = should_report

    @Slot()
    def exit(self) -> None:
        self._running = False


class MonitorQObject(QMainWindow):
    reportSignal = Signal(bool)
    exitSignal = Signal()

    def __init__(self, base_pv: str, detname: str):
        super().__init__()

        self._detname: str = detname

        # Setup layout and windowing
        ############################
        ## Main window geometry
        self.setGeometry(50, 50, 500, 150)
        self.setWindowTitle(f"{detname} Detector Protection Monitor")

        self.setMinimumSize(500, 150)
        self.setMaximumSize(500, 150)

        ## Description of progress bar
        info_label_text: str = (
            "Progress bar of number of pixels over ADU threshold as percentage of "
            "the maximum number of hot pixels."
        )

        self._info_label_font: QFont = QFont()
        self._info_label_font.setFamily("Verdana")
        self._info_label_font.setPointSize(12)

        self._info_label: QLabel = QLabel(info_label_text, self)
        self._info_label.setFont(self._info_label_font)
        self._info_label.setAlignment(Qt.AlignCenter)
        self._info_label.setWordWrap(True)
        self._info_label.setMinimumSize(500, 50)
        self._info_label.setMaximumSize(500, 50)
        self._info_label.move(0, 30)
        self.setCentralWidget(self._info_label)  # Need to set to work properly

        ## Progress bar - show pixels over threshold as percentage before of
        ##                the number of hot pixels threshold before "tripping"
        self._pix_over_threshold_bar: QProgressBar = QProgressBar(self)
        self._pix_over_threshold_bar.setGeometry(80, 80, 340, 20)
        self._pix_over_threshold_bar.setMaximum(100)
        ############################

        # Setup monitoring
        ##################
        self._monitorThread: BlockerPVMonitor = BlockerPVMonitor(base_pv=base_pv)
        ## Connect signal to display dialog of detector protection closes pulse picker
        self._monitorThread.trippedSignal.connect(self.detector_tripped)

        ## Setup signals for progress bar

        ### Update progress bar each time NPIX over threshold changes
        self._monitorThread.npixOverSignal.connect(self.update_progress_bar)
        ### Update the stored thresholds if those PVs change
        self._monitorThread.npixOverThreshSignal.connect(
            self.update_npix_over_threshold
        )
        self._monitorThread.aduThreshSignal.connect(self.update_adu_threshold)

        self.reportSignal.connect(self._monitorThread.report)
        self.exitSignal.connect(self._monitorThread.exit)
        ###################

        self._npix_ot: int = 0
        self._npix_ot_thresh: int = 0
        self._adu_thresh: int = 0

        self._monitorThread.start()
        self.show()

    @Slot(int)
    def update_progress_bar(self, new_pix_over_threshold: int) -> None:
        """Update the number of pixels over threshold."""
        self._npix_ot = new_pix_over_threshold
        percentage: float = (self._npix_ot / self._npix_ot_thresh) * 100
        self._pix_over_threshold_bar.setValue(int(percentage))

    @Slot(int)
    def update_adu_threshold(self, new_adu_threshold: int) -> None:
        """Update the ADU threshold."""
        self._adu_thresh = new_adu_threshold

    @Slot(int)
    def update_npix_over_threshold(self, new_npix_over_threshold: int) -> None:
        """Update the number of hot pixels threshold."""
        self._npix_ot_thresh = new_npix_over_threshold

    @Slot(bool)
    def detector_tripped(self, tripped: bool) -> None:
        if tripped:
            # Redundant - the thread sets the bool False already, but just in case...
            self.reportSignal.emit(False)
            print(
                f"DETECTOR TRIPPED! ADU: {self._adu_thresh}, Pixels Over Threshold: {self._npix_ot}"
            )

            title: str = f"{self._detname} Tripped"
            message: str = f"{self._npix_ot} pixels over threshold {self._adu_thresh}"

            msg_box_btn_choice: int = QMessageBox.Ok
            msg_box: QMessageBox = QMessageBox.critical(
                None, title, message, msg_box_btn_choice
            )
            # Re-enable generation of dialogs
            self.reportSignal.emit(True)

    def exit(self) -> None:
        self.exitSignal.emit()
        self._monitorThread.quit()
        self._monitorThread.wait()


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="detector_protection_monitor",
        description="Monitor DAQ hot-pixel detector protection.",
        epilog="",
    )
    parser.add_argument(
        "-b",
        "--base_pv",
        type=str,
        default="MFX:JF16M:BLOCKER",
        help="Base PV for the detector protection IOC.",
    )
    parser.add_argument(
        "-d",
        "--detector",
        type=str,
        default="Jungfrau",
        help="Name of detector. Used for dialogs only.",
    )

    args: argparse.Namespace = parser.parse_args()
    app: QApplication = QApplication([])
    mon_obj: MonitorQObject = MonitorQObject(
        base_pv=args.base_pv, detname=args.detector
    )
    app.aboutToQuit.connect(mon_obj.exit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

