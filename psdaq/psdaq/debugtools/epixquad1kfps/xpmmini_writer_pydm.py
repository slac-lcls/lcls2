"""PyDM top window for the ePixQuad XpmMini writer helper."""

import os

from pydm import Display
from qtpy.QtWidgets import QTabWidget, QVBoxLayout

from pyrogue.pydm.widgets import DebugTree
from pyrogue.pydm.widgets import SystemWindow


class DefaultTop(Display):
    def __init__(self, parent=None, args=None, macros=None):
        super().__init__(parent=parent, args=args or [], macros=macros)

        self.sizeX = 1000
        self.sizeY = 1000
        self.title = None

        for arg in args or []:
            if "sizeX=" in arg:
                self.sizeX = int(arg.split("=")[1])
            if "sizeY=" in arg:
                self.sizeY = int(arg.split("=")[1])
            if "title=" in arg:
                self.title = arg.split("=")[1].strip("'")

        if self.title is None:
            self.title = f"Rogue Server: {os.getenv('ROGUE_SERVERS')}"

        self.setWindowTitle(self.title)

        layout = QVBoxLayout()
        self.setLayout(layout)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        tabs.addTab(SystemWindow(parent=None, init_channel="rogue://0/root"), "C1100 System")
        tabs.addTab(DebugTree(parent=None, init_channel="rogue://0/root"), "C1100 Debug")
        tabs.addTab(SystemWindow(parent=None, init_channel="rogue://1/root"), "Camera System")
        tabs.addTab(DebugTree(parent=None, init_channel="rogue://1/root"), "Camera Debug")

        tabs.setCurrentIndex(2)
        self.resize(self.sizeX, self.sizeY)

    def ui_filepath(self):
        return None
