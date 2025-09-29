import argparse
import os
import time
import signal
import sys
from typing import TypedDict

import epics
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtCore import pyqtSlot as Slot
from PyQt5.QtCore import Qt, QThread, QTimer
from PyQt5.QtGui import QFont, QIcon, QCursor
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QStyle,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)


class TimeVarDict(TypedDict):
    status: int
    severity: int
    timestamp: float
    posixseconds: float
    nanoseconds: int


class BlockerPVMonitor(QThread):
    protectionActivatedSignal = Signal(bool, int, int)
    """Indicates the pulse picker has closed due to protection activation."""
    npixOverThreshSignal = Signal(int)
    """Reports the threshold for # of hot pixels before protection activation."""
    npixOverSignal = Signal(int)
    """Reports the current # of hot pixels."""
    aduThreshSignal = Signal(int)
    """Reports the ADU threshold to consider a pixel hot."""

    def __init__(self, base_pv: str) -> None:
        """Montior thread which checks blocker PVs.

        Information on protection status and current thresholds is reported via
        various signals.

        Args:
            base_pv (str): The base PV for the detector protection IOC.
        """
        super().__init__()

        self._base_pv: str = base_pv
        self._npix_ot_thresh: int = 0
        self._npix_ot: int = 0
        self._adu_thresh: int = 0

        self._running: bool = True
        self._should_report: bool = True

    def run(self) -> None:
        t0: float = time.monotonic()
        last_tripped_ts: float = -1
        blocked_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:BLOCKED")
        npix_ot_thresh_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:NPIX")
        npix_ot_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:NPIX_OT")
        adu_thresh_pv: epics.pv.PV = epics.PV(f"{self._base_pv}:ADU")
        while self._running:
            if blocked_pv.connected:
                time_vars: TimeVarDict = blocked_pv.get_timevars()
                curr_ts: float = time_vars["timestamp"]
                npix_ot: int = npix_ot_pv.get()
                npix_ot_thresh: int = npix_ot_thresh_pv.get()
                adu_thresh: int = adu_thresh_pv.get()
                # Set thresholds first - may get division by zero otherwise
                if npix_ot_thresh != self._npix_ot_thresh:
                    self._npix_ot_thresh = npix_ot_thresh
                    self.npixOverThreshSignal.emit(npix_ot_thresh)
                if adu_thresh != self._adu_thresh:
                    self._adu_thresh = adu_thresh
                    self.aduThreshSignal.emit(adu_thresh)
                if npix_ot != self._npix_ot:
                    self._npix_ot = npix_ot
                    self.npixOverSignal.emit(npix_ot)
                if last_tripped_ts == -1:
                    last_tripped_ts = curr_ts
                elif last_tripped_ts != curr_ts and self._should_report:
                    self.protectionActivatedSignal.emit(True, npix_ot, adu_thresh)
                    # Prevent generation of multiple dialogs
                    self._should_report = False

                tnow: float
                if (tnow := time.monotonic()) > (t0 + 60):
                    print(
                        f"ADU Threshold: {adu_thresh}, Pixels Over Threshold: {npix_ot}"
                    )
                    t0 = tnow
        print("[BlockerPVMonitor] Thread exited.")

    @Slot(bool)
    def report(self, should_report: bool) -> None:
        """Slot to disable reporting on protection activations.

        Args:
            should_report (bool): Whether the main thread is ready to receive new
                reports of detector protection activations.
        """
        self._should_report = should_report

    @Slot()
    def exit(self) -> None:
        print("[BlockerPVMonitor] Thread exiting.")
        self._running = False


class MonitorProgressBar(QWidget):
    reportSignal = Signal(bool)
    exitSignal = Signal()

    def __init__(self, detname: str) -> None:
        """A window containing a progress bar of the pixels over threshold.

        Args:
            detname (str): The detector name. Used only for window formatting.
        """
        super().__init__()

        # Setup layout and windowing
        ############################
        ## Window geometry
        self.setWindowTitle(f"{detname} Detector Protection Monitor")
        self._layout: QVBoxLayout = QVBoxLayout()
        self._layout.setSpacing(5)
        self.setLayout(self._layout)

        self.resize(500, 150)
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

        ## Progress bar - show pixels over threshold ("hot") as percentage of the
        ##                number of hot pixels before "tripping"
        self._pix_over_threshold_bar: QProgressBar = QProgressBar(self)
        self._pix_over_threshold_bar.setGeometry(80, 80, 340, 20)
        self._pix_over_threshold_bar.setMaximum(100)
        ############################

        # Initialize the main variables
        ## The number of pixels over the ADU threshold ("hot" pixels)
        self._npix_ot: int = 0
        ## The hot pixel count threshold. If you have more "hot" pixels than this
        ## you "trip", i.e. the pulse picker closes
        self._npix_ot_thresh: int = 0
        ## The ADU threshold to consider a pixel "hot"
        self._adu_thresh: int = 0

        self.show()

    @Slot(int)
    def update_progress_bar(self, new_pix_over_threshold: int) -> None:
        """Update the number of pixels over threshold.

        Args:
            new_pix_over_threshold (int): The current count of pixels over the
                ADU threshold.
        """
        self._npix_ot = new_pix_over_threshold
        percentage: float = (self._npix_ot / self._npix_ot_thresh) * 100
        self._pix_over_threshold_bar.setValue(int(percentage))

    @Slot(int)
    def update_adu_threshold(self, new_adu_threshold: int) -> None:
        """Update the ADU threshold.

        Args:
            new_adu_threshold (int): The new ADU threshold that has been set to
                consider a pixel a "hot" pixel.
        """
        self._adu_thresh = new_adu_threshold

    @Slot(int)
    def update_npix_over_threshold(self, new_npix_over_threshold: int) -> None:
        """Update the number of hot pixels threshold.

        Args:
            new_npix_over_threshold (int): The new threshold count of "hot" pixels
                that is used before activating the detector protection.
        """
        self._npix_ot_thresh = new_npix_over_threshold

    def detector_protection_acknowledge(self) -> None:
        print("It was acknowledged that the DAQ was unlatched.")


class MonitorMW(QMainWindow):
    reportSignal = Signal(bool)
    exitSignal = Signal()

    def __init__(self, base_pv: str, detname: str):
        """The main window for monitoring the detector protection.

        This main window is hidden. It spawns a separate window with a progress
        bar. When the detector protection activates, it will additionally spawn
        an warning dialog indicting so.

        Args:
            base_pv (str): The base PV for the detector protection IOC.

            detname (str): The detector name. Used only for window formatting.
        """
        super().__init__()

        self._detname: str = detname

        self._progress_bar: MonitorProgressBar = MonitorProgressBar(detname=detname)
        # Setup monitoring
        ##################
        self._monitorThread: BlockerPVMonitor = BlockerPVMonitor(base_pv=base_pv)
        ## Connect signal to display dialog of detector protection closing pulse
        ## picker -> launches an error window
        self._monitorThread.protectionActivatedSignal.connect(
            self.detector_protection_activated
        )

        ## Setup signals for progress bar
        ### Update progress bar each time the number of pixels over threshold changes
        self._monitorThread.npixOverSignal.connect(
            self._progress_bar.update_progress_bar
        )
        ### Update the stored thresholds if those PVs change
        self._monitorThread.npixOverThreshSignal.connect(
            self._progress_bar.update_npix_over_threshold
        )
        self._monitorThread.aduThreshSignal.connect(
            self._progress_bar.update_adu_threshold
        )

        ## Setup signals for maintenance/cleanup
        ### Signal for disabling/enabling the monitoring thread from launching
        ### more dialog windows via connected slots
        self.reportSignal.connect(self._monitorThread.report)
        ### Signal for propagating that the application is closing and that
        ### any appropriate cleanup actions should be taken
        self.exitSignal.connect(self._monitorThread.exit)
        ###################

        self._monitorThread.start()
        self.hide()

    @Slot(bool, int, int)
    def detector_protection_activated(
        self, activated: bool, npix_ot: int, adu_thresh: int
    ) -> None:
        """Slot for detector protection activations.

        If the detector protection has activated a new warning dialog will be
        spawned.

        Args:
            activated (bool): Whether the protection activated. In general this
                should always be True. If it is false, this method does nothing.

            npix_ot (int): The number of pixels over the ADU threshold when the
                protection mechanism was activated.

            adu_thresh (int): The ADU threshold to consider a pixel "hot" when the
                protection mechanism was activated.
        """
        if activated:
            # Redundant - the thread sets the bool False already, but just in case...
            self.reportSignal.emit(False)
            print(
                "DETECTOR PROTECTION ACTIVATED! Pulse picker closed. "
                f"ADU: {adu_thresh}, Pixels Over Threshold: {npix_ot}"
            )

            title: str = f"{self._detname} Tripped"
            message: str = f"{npix_ot} pixels over threshold {adu_thresh}"

            msg_box: QMessageBox = QMessageBox()
            msg_box.setWindowTitle(title)
            msg_box.setText(message)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box_btn_choice: QMessageBox.ButtonRole = QMessageBox.ActionRole
            msg_box.addButton("Dismiss.", msg_box_btn_choice)
            msg_box.exec_()
            # Re-enable generation of dialogs
            self.reportSignal.emit(True)

    def exit(self) -> None:
        if not self._monitorThread.isRunning():
            return
        print("[MonitorMW] App waiting for thread")
        self.exitSignal.emit()
        self._monitorThread.quit()
        self._monitorThread.wait()
        print("[MonitorMW] App exiting.")


class MonitorTrayIcon(QSystemTrayIcon):
    def __init__(self, app: QApplication) -> None:
        """A simple tray icon for the detector protection monitor.

        This is used as a visible indication the app is running even if there
        are no visible windows.
        """
        icon: QIcon = app.style().standardIcon(QStyle.SP_ComputerIcon)
        super().__init__(icon, app)

        self._menu: QMenu = QMenu()
        self._exit_action: QAction = QAction("Exit")
        self._exit_action.triggered.connect(app.quit)
        self._menu.addAction(self._exit_action)

        self.activated.connect(self.tray_activated)

        self.setContextMenu(self._menu)
        self.setToolTip("Detector Protection Monitor")

    @Slot(QSystemTrayIcon.ActivationReason)
    def tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self._menu.exec_(QCursor.pos())

    def show(self) -> None:
        super().show()
        self.showMessage(
            "Monitor Running",
            "The detector protection monitor is running in the background.",
            QSystemTrayIcon.Information,
            5000,
        )


def main() -> None:
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
    app.setQuitOnLastWindowClosed(False)

    mon_obj: MonitorMW = MonitorMW(base_pv=args.base_pv, detname=args.detector)
    app.aboutToQuit.connect(mon_obj.exit)
    mon_tray_icon: MonitorTrayIcon = MonitorTrayIcon(app)
    mon_tray_icon.show()

    # Allow exit on Ctrl-C even without windows
    ###########################################
    def sigint_handler(*args) -> None:
        mon_obj.exit()
        app.quit()
        # Need this... EPICS seemingly spawns more threads causing segfaults
        os._exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    timer: QTimer = QTimer()
    timer.setInterval(100)
    timer.timeout.connect(lambda: None)
    timer.start()
    ###########################################
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

