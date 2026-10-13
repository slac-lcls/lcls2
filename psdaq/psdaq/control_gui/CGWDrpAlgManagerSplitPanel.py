"""
Main window panel for configuring DRP algorithms and assigning config to segments.

Classes:
    CGWDrpAlgManagerSplitPanel: The split detector hierarchy/algorithm config panel.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QLabel,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QMessageBox,
    QSplitter,
    QScrollArea,
)
from PyQt5.QtCore import Qt

from psdaq.configdb.alg_parameter_manager import DrpAlgParamManager
from psdaq.configdb.alg_parameter_validator import (
    typerange,
    validate_parameters_against_schema,
    AlgValidationError,
)
from psdaq.configdb.configdb_endpoints import normalize_version

_drpalgos_error: Optional[ImportError] = None
try:
    from drpalgos.adapters import (
        ParamAdapter,
        load_adapter,
        verify_adapter_against_schema,
    )
    from drpalgos.geometry import DetectorGeometry, get_geometry
    from drpalgos.presets import (
        RESERVED_PREFIX,
        GroupKey,
        PresetGroup,
        encode_preset_name,
        group_presets,
        new_preset_group,
    )
except ImportError as err:
    _drpalgos_error = err
    RESERVED_PREFIX = "__PLG"

    def group_presets(presets):
        return list(presets), {}


from psdaq.control_gui.drp_alg_plugins import DrpAlgParamPlugin, load_plugin
from psdaq.control_gui.CGJsonUtils import dict_platform

logger: logging.Logger = logging.getLogger(__name__)


_INT_T = {
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "integer",
}
_FLT_T = {"FLOAT", "DOUBLE", "number"}
_I32 = (-(2**31), 2**31 - 1)


def _int_from_text(w: QLineEdit) -> int:
    """Read an integer from a widget.

    If empty provide an error.
    """
    text: str = w.text().strip()
    if not text:
        raise ValueError("no value entered")
    return int(text)


def make_param_widget(spec):
    """Create the appropriate widget for a specific parameter.

    Returns (widget, getter) for interacting with the parameter's configruation.

    Returns:
        widget: The widget for editing the parameter

        getter: A function which when called as getter() yields the typed value.
    """
    p_type: str = spec.get("type", "string")
    default: str = spec.get("default")
    if p_type in ("boolEnum", "boolean"):  # before enum: boolEnum has enum
        widget_chk: QCheckBox = QCheckBox()
        widget_chk.setChecked(bool(default))
        return widget_chk, widget_chk.isChecked
    if "enum" in spec:
        opts = list(spec["enum"])
        widget_combo = QComboBox()
        widget_combo.addItems([str(o) for o in opts])
        if default in opts:
            widget_combo.setCurrentIndex(opts.index(default))
        return (
            widget_combo,
            lambda: opts[widget_combo.currentIndex()],
        )  # original typed value
    if p_type in _INT_T:
        low: int
        high: int
        # Check default range for int32_t first
        low, high = typerange.get(p_type, _I32)

        # Schema allows further restricting range of values
        low, high = spec.get("minimum", low), spec.get("maximum", high)

        # Use QSpinBox (int32_t)
        if _I32[0] <= low and high <= _I32[1]:
            widget_spin = QSpinBox()
            widget_spin.setRange(low, high)
            if default is not None:
                widget_spin.setValue(int(default))
            return widget_spin, widget_spin.value
        widget_line = QLineEdit("" if default is None else str(default))
        return widget_line, lambda: _int_from_text(widget_line)

    if p_type in _FLT_T:
        widget_dbl_spin = QDoubleSpinBox()
        widget_dbl_spin.setDecimals(6)
        widget_dbl_spin.setRange(spec.get("minimum", -1e12), spec.get("maximum", 1e12))
        if default is not None:
            widget_dbl_spin.setValue(float(default))
        return widget_dbl_spin, widget_dbl_spin.value

    if p_type == "array":
        widget_line = QLineEdit(json.dumps(default) if default is not None else "[]")
        return widget_line, lambda: json.loads(widget_line.text())

    widget_line = QLineEdit("" if default is None else str(default))
    return widget_line, widget_line.text


def set_param_widget(widget, spec, value) -> None:
    if isinstance(widget, QCheckBox):
        widget.setChecked(bool(value))
    elif isinstance(widget, QComboBox):
        opts = list(spec["enum"])
        if value in opts:
            widget.setCurrentIndex(opts.index(value))
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        widget.setValue(value)
    elif isinstance(widget, QLineEdit):
        widget.setText(json.dumps(value) if spec.get("type") == "array" else str(value))


class CGWDrpAlgManagerSplitPanel(QWidget):
    """
    Main window panel for configuring DRP algorithms and assigning config to segments.

    The panel is split into two halves organized into a:
    - A left-side tree of detectors with a hierarchy of segments inside
    - A right-side algorithm configuration panel

    The overall organization looks (or... is intended to look) like:

    +--------------------------------------------------------------------------------+
    | DRP ALGORITHM MANAGER                                                          |
    +--------------------------------------------------------------------------------+
    | Partition: [ BEAM  v ]                                                         |
    +-------------------------------------------+------------------------------------+
    | DETECTOR / SEGMENT TREE                   | INSPECTOR: epixuhr3x2 (Segs: 0, 1) |
    +-------------------------------------------+------------------------------------+
    |  [x] epixuhr3x2 (4 segments)    [<Mixed> ]| [x] Enable Algorithm for Selected  |
    |   ├── [x] epixuhr3x2_0          [Binning ]|                                    |
    |   ├── [x] epixuhr3x2_1          [Binning ]| Algorithm: [ Binning           v ] |
    |   ├── [ ] epixuhr3x2_2          [Disabled]| Version:   [ v1                v ] |
    |   └── [ ] epixuhr3x2_3          [Disabled]| Preset:    [ Std Low Noise     v ] |
    |  [ ] jungfrau16m (2 segments)   [Disabled]|                                    |
    |   ├── [ ] jungfrau16m_0         [Disabled]| --- Parameters (Binning v1) ------ |
    |   ├── [ ] jungfrau16m_1         [Disabled]| Field:           [ PIX_VALUE   v ] |
    |   ├── [ ] jungfrau16m_2         [Disabled]| Operation:       [ THRESHOLD   v ] |
    |   ├── [ ] jungfrau16m_3         [Disabled]| Threshold Value: [ 15.0          ] |
    |   └── [ ] jungfrau16m_4         [Disabled]|                                    |
    |  [x] uhrMini_0 (1 segment)      [ Thresh ]| ---------------------------------- |
    |                                           | [ ] Save as new preset:            |
    |                                           |     [ My Custom Run Preset       ] |
    |                                           |                                    |
    |                                           | [ Apply to Checked Segs (2)      ] |
    +-------------------------------------------+------------------------------------+
    """

    MIXED_DISP_ID: ClassVar[str] = "<Mixed>"
    """Label displayed at detector level if segments have different alg selections."""
    CUSTOM: ClassVar[str] = "-- Custom --"
    """For parameter sets not part of a preset."""
    DEFAULT_PRESET: ClassVar[str] = "Default"
    """Label (and selection) for the registered defaults. Loaded and displayed if present at startup."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        confdb: Optional[str] = None,
        hutch: str = "xpp",
        alias: str = "BEAM",
        detector_tree_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.confdb = confdb
        self.hutch: str = hutch
        self.alias: str = alias
        self.alg_man: DrpAlgParamManager = DrpAlgParamManager.from_configdb_client(
            confdb, hutch=hutch
        )
        self.detector_tree_data = detector_tree_data or self.devices_from_configdb()

        self.algs: Dict[str, List[str]] = {}
        self.presets: List[Dict[str, Any]] = []
        self.current_schema: Dict[str, Any] = {}
        self.field_widgets = {}  # name -> (widget, getter, spec)
        self.is_updating_checkboxes: bool = False

        self.gui_plugin: Dict[str, Any] = {}
        self.plugin: Optional[DrpAlgParamPlugin] = None
        self.adapter: Optional[ParamAdapter] = None
        self.plugin_dirty: bool = False  # User edited the intent since it was loaded
        self.lbl_plugin_problems: Optional[QLabel] = None
        self.preset_groups: Dict[GroupKey, PresetGroup] = {}
        self.preset_entries: List[Tuple[str, Any]] = (
            []
        )  # Per combo item, see refresh_presets

        self.setWindowTitle("DRP Algorithm Manager")
        self.resize(950, 550)
        self.init_ui()

    def devices_from_partition(self) -> Dict[str, List[int]]:
        """Detector segments of the active DRPs in the currently allocated partition."""
        tree: Dict[str, List[int]] = {}
        dpf: Any = dict_platform()

        platform: Dict[str, Any]
        if isinstance(dpf, dict):
            platform = dpf
        else:
            platform = {}

        for proc in platform.get("drp", {}).values():
            if proc.get("active") != 1:
                continue
            alias: str = proc.get("proc_info", {}).get("alias", "")
            det, _, seg = alias.rpartition("_")
            if det and seg.isdigit():
                tree.setdefault(det, []).append(int(seg))

        return {det: sorted(segs) for det, segs in tree.items()}

    def refresh_tree(self) -> None:
        partition: Dict[str, List[int]] = self.devices_from_partition()
        if self.chk_all_devices.isChecked() or not partition:
            self.detector_tree_data = self.devices_from_configdb()
        else:
            self.detector_tree_data = partition

        self.populate_tree()
        self.apply_filter(self.txt_filter.text())

    def apply_filter(self, text: str) -> None:
        text = text.lower()
        root = self.tree.invisibleRootItem()

        for i in range(root.childCount()):
            parent = root.child(i)
            name: str = parent.data(0, Qt.UserRole)["name"].lower()
            parent.setHidden(bool(text) and text not in name)

    def init_ui(self) -> None:
        main_layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(main_layout)

        # View split into detector/segment on left, and algorithm parameters on right
        splitter: QSplitter = QSplitter(Qt.Horizontal)

        # Detector/segment tree selector
        tree_group: QGroupBox = QGroupBox("Partition Detectors && Segments")
        tree_layout: QVBoxLayout = QVBoxLayout()

        # Setup a split between dets/segs in the partition and all in hutch/alias
        tree_ctrl_layout: QHBoxLayout = QHBoxLayout()
        self.txt_filter: QLineEdit = QLineEdit()
        self.txt_filter.setPlaceholderText("Filter detectors...")
        self.txt_filter.textChanged.connect(self.apply_filter)
        self.chk_all_devices: QCheckBox = QCheckBox("Show all in configdb")
        self.chk_all_devices.toggled.connect(self.refresh_tree)
        self.btn_refresh: QPushButton = QPushButton("Refresh")
        self.btn_refresh.clicked.connect(self.refresh_tree)
        tree_ctrl_layout.addWidget(self.txt_filter)
        tree_ctrl_layout.addWidget(self.chk_all_devices)
        tree_ctrl_layout.addWidget(self.btn_refresh)
        tree_layout.addLayout(tree_ctrl_layout)

        self.tree: QTreeWidget = QTreeWidget()
        self.tree.setHeaderLabels(["Detector / Segment", "Current Alg"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnCount(2)

        # Tree Signals
        self.tree.itemSelectionChanged.connect(self.on_tree_selection_changed)
        self.tree.itemChanged.connect(self.on_tree_item_changed)

        tree_layout.addWidget(self.tree)
        tree_group.setLayout(tree_layout)
        splitter.addWidget(tree_group)

        inspector_group: QGroupBox = QGroupBox("Inspector && Algorithm Config")
        inspector_layout: QVBoxLayout = QVBoxLayout()

        self.lbl_selected_target: QLabel = QLabel("Selected Target: None")
        self.lbl_selected_target.setStyleSheet(
            "font-weight: bold; font-size: 13px; color: #2c3e50;"
        )
        inspector_layout.addWidget(self.lbl_selected_target)

        # Algorithm selection controls
        alg_select_layout: QFormLayout = QFormLayout()

        self.chk_enable: QCheckBox = QCheckBox("Enable DRP Algorithm")
        self.chk_enable.setChecked(True)
        alg_select_layout.addRow("", self.chk_enable)

        self.combo_alg: QComboBox = QComboBox()
        self.combo_alg.addItems(["Binning", "Threshold"])
        self.combo_alg.currentTextChanged.connect(self.on_alg_changed)
        alg_select_layout.addRow("Algorithm:", self.combo_alg)

        self.combo_ver: QComboBox = QComboBox()
        self.combo_ver.addItems(["v1", "v2"])
        self.combo_ver.currentTextChanged.connect(self.load_schema)
        alg_select_layout.addRow("Version:", self.combo_ver)

        self.combo_preset: QComboBox = QComboBox()
        self.combo_preset.addItems(
            ["Standard Low Noise", "High Throughput", "-- Custom --"]
        )
        self.combo_preset.currentTextChanged.connect(self.on_preset_changed)
        alg_select_layout.addRow("Preset:", self.combo_preset)

        inspector_layout.addLayout(alg_select_layout)

        param_scroll: QScrollArea = QScrollArea()
        param_scroll.setWidgetResizable(True)
        self.param_widget: QWidget = QWidget()
        self.param_form_layout: QFormLayout = QFormLayout()
        self.param_widget.setLayout(self.param_form_layout)
        param_scroll.setWidget(self.param_widget)

        param_box: QGroupBox = QGroupBox("Algorithm Parameters (Dynamic Schema)")
        param_box_layout: QVBoxLayout = QVBoxLayout()
        param_box_layout.addWidget(param_scroll)
        param_box.setLayout(param_box_layout)

        inspector_layout.addWidget(param_box)

        # Save Preset Option
        preset_save_layout: QHBoxLayout = QHBoxLayout()
        self.chk_save_preset: QCheckBox = QCheckBox("Save as new preset:")
        self.txt_preset_name: QLineEdit = QLineEdit()
        self.txt_preset_name.setPlaceholderText("e.g. Low Noise Run 42")
        preset_save_layout.addWidget(self.chk_save_preset)
        preset_save_layout.addWidget(self.txt_preset_name)
        inspector_layout.addLayout(preset_save_layout)

        # Apply Button
        self.btn_apply: QPushButton = QPushButton("Apply Changes to Checked Segments")
        self.btn_apply.setStyleSheet(
            "font-weight: bold; background-color: #27ae60; color: white; padding: 6px;"
        )
        self.btn_apply.clicked.connect(self.on_apply_changes)
        inspector_layout.addWidget(self.btn_apply)

        inspector_group.setLayout(inspector_layout)
        splitter.addWidget(inspector_group)

        # Leave 40% for detector/segments and 60% for the parameters
        splitter.setSizes([380, 570])
        main_layout.addWidget(splitter)

        # Populate everything initially
        # self.populate_tree()
        self.refresh_tree()
        self.refresh_algorithms()

    def set_alias(self, alias: str) -> None:
        """Switch the config alias being used (e.g. BEAM/NOBEAM)."""
        if alias == self.alias:
            return

        self.alias = alias
        self.refresh_tree()  # resets self.detector_tree_data

    def _refresh_parent_label(self, parent: QTreeWidgetItem) -> None:
        """Set a detector row to its segments' common label, or <Mixed>."""
        labels = {parent.child(j).text(1) for j in range(parent.childCount())}
        parent.setText(1, labels.pop() if len(labels) == 1 else self.MIXED_DISP_ID)

    def populate_tree(self):
        """Populates the QTreeWidget with detectors and their segment children."""
        self.tree.blockSignals(True)
        self.tree.clear()

        # refresh_tree resets self.detector_tree_data (should've been called)
        det_name: str
        segs: List[int]
        for det_name, segs in self.detector_tree_data.items():
            parent: QTreeWidget = QTreeWidgetItem(self.tree)
            parent.setText(0, f"{det_name} ({len(segs)} segs)")
            parent.setText(1, "Binning v1")
            parent.setFlags(
                parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
            )
            parent.setCheckState(0, Qt.Checked)
            parent.setData(0, Qt.UserRole, {"type": "detector", "name": det_name})

            for seg in segs:
                child: QTreeWidgetItem = QTreeWidgetItem(parent)
                seg_id: str = f"{det_name}_{seg}"
                child.setText(0, seg_id)
                det_type: str = ""
                try:
                    doc: Dict[str, Any] = self.segment_config(seg_id)
                    ref: Optional[Dict[str, Any]] = doc.get("drp_alg0")
                    det_type = doc.get("detType:RO", "")
                    txt: str = (
                        f"{ref['alg_name']} {ref['version']}" if ref else "Disabled"
                    )
                    child.setFlags(
                        child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                    )
                    child.setCheckState(0, Qt.Unchecked)
                except Exception as err:
                    ref, txt = None, "<unreachable>"
                    child.setToolTip(0, f"{seg_id}: {err}")
                    child.setDisabled(True)
                child.setText(1, txt)
                child.setData(
                    0,
                    Qt.UserRole,
                    {
                        "type": "segment",
                        "name": seg_id,
                        "det": det_name,
                        "det_type": det_type,
                        "seg": seg,
                        "ref": ref,
                    },
                )
            self._refresh_parent_label(parent)
            parent.setExpanded(True)

        self.tree.blockSignals(False)

    ### For Config Plugins ###
    ##########################

    def detector_geometry(self) -> Tuple[DetectorGeometry, List[int]]:
        """Geometry and segment numbers of the checked segments.

        Returns:
            geom (DetectorGeometry): Geometry of the checked segments' detector.

            segments (List[int]): The checked segment numbers.

        Raises:
            ValueError: If no segments are checked, or they span more than one detector.

            KeyError: If no geometry is known for the detector type.
        """
        data: List[Dict[str, Any]] = [
            item.data(0, Qt.UserRole) for item in self.get_checked_segment_items()
        ]
        if not data:
            raise ValueError("No segments checked")
        dets: List[str] = sorted({d["det"] for d in data})
        if len(dets) > 1:
            raise ValueError(
                f"{self.gui_plugin.get('label', 'This plugin')} configures one detector "
                f"at a time; checked segments span {', '.join(dets)}"
            )
        segs: List[int] = sorted(d["seg"] for d in data)
        return get_geometry(data[0]["det_type"], segs), segs

    def refresh_plugin(self) -> None:
        """Update a plugin's geometry, reloading intent unless already edited.

        A plugin's intent is rebuilt from the parameters of every current checked
        segment that is making use of the algorithm (and more specifically, that
        exact algorithm version.)
        """
        if self.plugin is None or self.adapter is None:
            return
        try:
            geom, _ = self.detector_geometry()
        except (KeyError, ValueError):
            self.plugin.set_geometry(None)
            self.check_plugin()
            return
        self.plugin.set_geometry(geom)

        if not self.plugin_dirty:
            alg_name: str = self.combo_alg.currentText()
            version: str = self.combo_ver.currentText()
            seg_params: Dict[int, Dict[str, Any]] = {}
            for item in self.get_checked_segment_items():
                data: Dict[str, Any] = item.data(0, Qt.UserRole)
                ref: Optional[Dict[str, str]] = data.get("ref")

                if not ref or ref["alg_name"] != alg_name:
                    continue
                if normalize_version(ref["version"]) != normalize_version(version):
                    continue

                try:
                    doc = self.alg_man.get_algorithm_parameters(
                        alg_name, version, ref["params_id"]
                    )
                    seg_params[data["seg"]] = doc["parameters"]
                except Exception as err:
                    logger.warning(f"Could not load {data['name']} parameters: {err}")
            try:
                self.plugin.set_intent(self.adapter.assemble(seg_params, geom))
            except (KeyError, ValueError, TypeError) as err:
                logger.warning(f"Could not rebuild plugin view from parameters: {err}")

        self.check_plugin()

    def check_plugin(self) -> List[str]:
        """Verify the plugin intent via the adaptor.

        Any problems are returned and showed. If a problem includes `Error`, then
        parameter application is blocked. Otherwise, they will only be displayed
        as warnings (for user awareness and confirmation).

        Returns:
            problems (List[str]): A list of problems as flagged by the adapter.
        """
        if self.plugin is None or self.adapter is None:
            return []

        problems: List[str]
        try:
            geom: DetectorGeometry
            segs: List[int]
            geom, segs = self.detector_geometry()

            problems: List[str] = self.adapter.check(
                self.plugin.intent(),
                geom,
                segs,
                self.extract_form_parameters(lenient=True),
            )
        except (KeyError, ValueError) as err:
            problems = [f"Error: {err.args[0] if err.args else err}"]

        if self.lbl_plugin_problems is not None:
            is_error: bool = any(p.startswith("Error:") for p in problems)
            self.lbl_plugin_problems.setText("\n".join(problems))
            self.lbl_plugin_problems.setStyleSheet(
                "color: #c0392b;" if is_error else "color: #b9770e;"
            )
        return problems

    def on_plugin_changed(self) -> None:
        self.plugin_dirty = True
        self.check_plugin()

    #############################
    ### End of Config Plugins ###

    ### Schema and Tree Management ###
    ##################################

    def on_tree_item_changed(self, item, column):
        """Handles parent/child checkbox synchronization and tristate states."""
        if self.is_updating_checkboxes or column != 0:
            return

        self.is_updating_checkboxes = True
        self.tree.blockSignals(True)

        # When a parent node is checked or unchecked must update ALL children
        if item.childCount() > 0:
            state = item.checkState(0)
            if state != Qt.PartiallyChecked:
                for i in range(item.childCount()):
                    item.child(i).setCheckState(0, state)
        # When a child node is checked or unchecked must update the parent
        else:
            parent = item.parent()
            if parent:
                checked_count = sum(
                    1
                    for i in range(parent.childCount())
                    if parent.child(i).checkState(0) == Qt.Checked
                )
                if checked_count == parent.childCount():
                    parent.setCheckState(0, Qt.Checked)
                elif checked_count == 0:
                    parent.setCheckState(0, Qt.Unchecked)
                else:
                    parent.setCheckState(0, Qt.PartiallyChecked)

        self.tree.blockSignals(False)
        self.is_updating_checkboxes = False

        # Also rebuild plugin if segments change
        self.refresh_plugin()

    def on_tree_selection_changed(self):
        """Triggered when user clicks/highlights a row in the tree."""
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        data = item.data(0, Qt.UserRole)

        if data["type"] == "detector":
            self.lbl_selected_target.setText(
                f"Selected Target: Detector '{data['name']}' (All Segments)"
            )
        else:
            self.lbl_selected_target.setText(
                f"Selected Target: Segment '{data['name']}'"
            )

        if data["type"] == "segment":
            self.chk_enable.setChecked(bool(data.get("ref")))
            if data.get("ref"):
                self.load_ref(data["ref"], data["seg"])

    def load_ref(self, ref: Dict[str, str], segment: int) -> None:
        alg: str = ref["alg_name"]
        version: str = ref["version"]
        if alg not in self.algs:
            QMessageBox.warning(self, "Unknown Algorithm", f"{alg} is not registered.")
            return
        self.combo_alg.setCurrentText(alg)  # refills versions + schema if changed
        idx = next(
            (
                i
                for i in range(self.combo_ver.count())
                if normalize_version(self.combo_ver.itemText(i))
                == normalize_version(version)
            ),
            -1,
        )
        if idx < 0:
            QMessageBox.warning(
                self, "Unknown Version", f"{alg} {version} is not registered."
            )
            return
        self.combo_ver.setCurrentIndex(idx)  # reloads schema if changed
        try:
            doc = self.alg_man.get_algorithm_parameters(alg, version, ref["params_id"])
        except Exception as err:
            QMessageBox.critical(self, "ConfigDB Error", f"{alg} {version}: {err}")
            return
        self.fill_form(doc["parameters"], segment)
        self.show_custom_preset()

    def on_alg_or_ver_changed(self) -> None:
        """Re-loads schema and re-builds parameters when Alg or Version changes."""
        self.load_schema()

    def refresh_algorithms(self) -> None:
        try:
            self.algs = {
                a["name"]: a["versions"] for a in self.alg_man.list_algorithms()
            }
        except Exception as err:
            QMessageBox.critical(
                self, "ConfigDB Error", f"Cannot list algorithms: {err}"
            )
            self.algs = {}
        self.combo_alg.blockSignals(True)
        self.combo_alg.clear()
        self.combo_alg.addItems(sorted(self.algs))
        self.combo_alg.blockSignals(False)
        self.on_alg_changed()

    def on_alg_changed(self) -> None:
        self.combo_ver.blockSignals(True)
        self.combo_ver.clear()
        self.combo_ver.addItems(self.algs.get(self.combo_alg.currentText(), []))
        self.combo_ver.blockSignals(False)
        self.load_schema()

    def load_schema(self) -> None:
        """Fetches schema from ConfigDB (or fallback dict) and builds dynamic form."""
        alg_name: str = self.combo_alg.currentText()
        version: str = self.combo_ver.currentText()

        self.current_schema = {}
        self.gui_plugin = {}
        self.presets = []
        if alg_name and version:
            try:
                self.current_schema = self.alg_man.get_algorithm_schema(
                    alg_name, version
                )
                metadata: Dict[str, Any] = self.alg_man.get_algorithm_metadata(
                    alg_name, version
                )
                self.gui_plugin = metadata.get("gui_plugin") or {}
                self.presets = self.alg_man.list_presets(alg_name, version)
            except Exception as err:
                QMessageBox.critical(
                    self, "ConfigDB Error", f"{alg_name} {version}: {err}"
                )
        self.build_dynamic_form()
        self.refresh_presets()
        self.select_default_preset()

    def build_dynamic_form(self) -> None:
        """Generates PyQt5 widgets from current_schema.properties."""
        # Clear existing form
        while self.param_form_layout.count() > 0:
            item = self.param_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.field_widgets.clear()
        self.plugin = None
        self.adapter = None
        self.plugin_dirty = False
        self.lbl_plugin_problems = None

        properties = self.current_schema.get("properties", {})
        required = self.current_schema.get("required", [])

        # A plugin is only usable with an adapter that matches it and the schema.
        # Otherwise fall back to plain form fields for every parameter.
        plugin_cls = load_plugin(self.gui_plugin) if self.gui_plugin else None
        adapter: Optional[ParamAdapter] = None
        if plugin_cls is not None and _drpalgos_error is None:
            adapter = load_adapter(self.gui_plugin)
        problems: List[str] = []
        if self.gui_plugin and _drpalgos_error is not None:
            problems.append(f"GUI plugins unavailable (drpalgos): {_drpalgos_error}")
        elif self.gui_plugin and plugin_cls is None:
            problems.append(
                f"GUI plugin adapter unavailable: {self.gui_plugin.get('adapter')}"
            )
        elif plugin_cls is not None and adapter is not None:
            if adapter.plugin != plugin_cls.intent_type:
                problems.append(
                    f"GUI plugin {plugin_cls.__name__} edits '{plugin_cls.intent_type}' but "
                    f"adapter {type(adapter).__name__} expects '{adapter.plugin}'"
                )
            problems += verify_adapter_against_schema(adapter, self.current_schema)
        if problems:
            for problem in problems:
                self.param_form_layout.addRow(QLabel(problem))
            plugin_cls, adapter = None, None
        self.adapter = adapter

        # Choose appropriate widgets based on the type of each parameter.
        # Parameters the adapter produces per segment are not shown.
        hidden: Tuple[str, ...] = adapter.produces if adapter is not None else ()
        for field_name, spec in properties.items():
            if field_name in hidden:
                continue
            label = spec.get("title", field_name) + (
                " *" if field_name in required else ""
            )
            widget, getter = make_param_widget(spec)
            self.field_widgets[field_name] = (widget, getter, spec)
            self.param_form_layout.addRow(QLabel(label), widget)

        if plugin_cls is not None:
            self.plugin = plugin_cls(self.gui_plugin)
            self.plugin.valuesChanged.connect(self.on_plugin_changed)
            self.lbl_plugin_problems = QLabel()
            self.lbl_plugin_problems.setWordWrap(True)
            plugin_box: QGroupBox = QGroupBox(
                self.gui_plugin.get("label", plugin_cls.__name__)
            )
            plugin_box_layout: QVBoxLayout = QVBoxLayout()
            plugin_box_layout.addWidget(self.plugin)
            plugin_box_layout.addWidget(self.lbl_plugin_problems)
            plugin_box.setLayout(plugin_box_layout)
            self.param_form_layout.addRow(plugin_box)
            self.refresh_plugin()

    def extract_form_parameters(self, lenient: bool = False) -> Dict[str, Any]:
        """Extracts parameter dictionary from form widgets.

        Args:
            lenient (bool): Omit fields whose value cannot be read instead of raising.

        Raises:
            ValueError: Naming the field, if a value cannot be read and not lenient.
        """
        params: Dict[str, Any] = {}
        for name, (_, getter, _) in self.field_widgets.items():
            try:
                params[name] = getter()
            except ValueError as err:
                if not lenient:
                    raise ValueError(f"Parameter '{name}': {err}") from err
        return params

    ######################################
    ### End Schema and Tree Management ###

    ### Preset (and Detector Preset) Handling ###
    #############################################

    def refresh_presets(self) -> None:
        """Fill the preset list from self.presets.

        NOTE: Per-segment (normal, or "plain", presets) are listed directly by name.

        NOTE: Detector presets are listed once per the group, in chronological order
              with newest first. If there is no matching plugin for the detector
              preset, then the members of the detector preset group are listed
              individually.

        The combo item's data is an index to `self.preset_entries`. Those entries
        hold:
          - ("preset", preset_doc)
          - ("group", group_key)
          - (CUSTOM, None)
        """
        plain: List[Dict[str, Any]]
        plain, self.preset_groups = group_presets(self.presets)
        entries: List[Tuple[str, str, Any]] = [
            (p["preset_name"], "preset", p) for p in plain
        ]
        by_newest: List[Tuple[GroupKey, PresetGroup]] = sorted(
            self.preset_groups.items(), key=lambda kv: kv[1].timestamp, reverse=True
        )
        for key, group in by_newest:
            if self.adapter is not None and group.plugin == self.adapter.plugin:
                entries.append((group.label(), "group", key))
                continue
            for seg, preset in sorted(group.members.items()):
                label: str = f"{group.name} [{group.det_type} seg {seg}]"
                entries.append((label, "preset", preset))
        entries.append((self.CUSTOM, self.CUSTOM, None))

        self.preset_entries = [(kind, value) for _, kind, value in entries]
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        for i, (label, _, _) in enumerate(entries):
            self.combo_preset.addItem(label, i)
        self.combo_preset.setCurrentIndex(len(entries) - 1)
        self.combo_preset.blockSignals(False)

    def select_default_preset(self) -> None:
        """Show and load the algorithm's "Default" preset (registered defaults), if any."""
        for i, (kind, value) in enumerate(self.preset_entries):
            if kind == "preset" and value.get("preset_name") == self.DEFAULT_PRESET:
                self.combo_preset.blockSignals(True)
                self.combo_preset.setCurrentIndex(i)
                self.combo_preset.blockSignals(False)
                self.fill_form(value["parameters"])
                return

    def show_custom_preset(self) -> None:
        """Show "-- Custom --" in the preset list without loading anything."""
        self.combo_preset.blockSignals(True)
        self.combo_preset.setCurrentIndex(self.combo_preset.count() - 1)
        self.combo_preset.blockSignals(False)

    def on_preset_changed(self, preset_name: str) -> None:
        """Handle preset selection changes."""
        index: Optional[int] = self.combo_preset.currentData()
        if index is None or not 0 <= index < len(self.preset_entries):
            return
        kind, value = self.preset_entries[index]
        if kind == "preset":
            self.fill_form(value["parameters"])
        elif kind == "group":
            self.load_preset_group(self.preset_groups[value])

    def load_preset_group(self, group: PresetGroup) -> None:
        """Load a detector preset into the form and the plugin.

        The fields come from the lowest-numbered member. The plugin intent is
        then assembled from all the members over the group's geometry. Panels
        are placed in the segment order - applying to a different set of segments
        will map the group's i-th segment to the i-th checked segment.
        """
        checked_types: List[str] = sorted(
            {
                item.data(0, Qt.UserRole)["det_type"]
                for item in self.get_checked_segment_items()
            }
            - {group.det_type}
        )
        if checked_types:
            QMessageBox.warning(
                self,
                "Preset Not Applicable",
                f"'{group.name}' is for {group.det_type}; the checked segments "
                f"are {', '.join(checked_types)}.",
            )
            return

        first: Dict[str, Any] = group.members[min(group.members)]
        self.fill_form(first["parameters"])
        if self.plugin is None or self.adapter is None:
            return

        try:
            group_geom: DetectorGeometry = get_geometry(group.det_type, group.members)
            intent: Dict[str, Any] = self.adapter.assemble(
                {seg: p["parameters"] for seg, p in group.members.items()}, group_geom
            )
        except (KeyError, ValueError, TypeError) as err:
            QMessageBox.warning(self, "Cannot Load Preset", f"{group.name}: {err}")
            return
        self.plugin.set_intent(intent)
        self.plugin_dirty = True  # Loaded but not applied: keep over stored params
        self.check_plugin()

        if not group.complete:
            QMessageBox.warning(
                self,
                "Incomplete Preset",
                f"'{group.name}' was saved for {group.count} segments but only "
                f"segments {sorted(group.members)} were found.",
            )

    #############################################
    ### Preset (and Detector Preset) Handling ###

    ### Form Updates and Applying Changes ###
    #########################################

    def fill_form(self, params: Dict[str, Any], segment: Optional[int] = None) -> None:
        """Fill the form fields from a parameter set.

        NOTE: This does not touch anything done by the plugin. The plugin intent is
              always assembled from the segments which have been checked. Refer to
              the refresh_plugin method.
        """
        for name, value in params.items():
            if name in self.field_widgets:
                widget, _, spec = self.field_widgets[name]
                set_param_widget(widget, spec, value)

        self.check_plugin()

    def get_checked_segment_names(self) -> List[str]:
        """Traverses tree and returns list of all checked segment names."""
        checked_segments = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    data = child.data(0, Qt.UserRole)
                    checked_segments.append(data["name"])
        return checked_segments

    def get_checked_segment_items(self) -> List[QTreeWidgetItem]:
        """Traverses tree and returns list of all checked segment items (Qt objs)."""
        checked_segments = []
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    checked_segments.append(child)
        return checked_segments

    def update_det_tree_alg_col(self, alg_name: str, ver: str) -> None:
        """Update the display of the `Current Algorithm` column for detector segments.

        The tree widget hierarchy is organized as:

        Column 0 (Index 0)       | Column 1 (Index 1)
        -------------------------+-------------------------
        [x] <detector name>      | <Current Alg>

        So, if column 0's check box is applied, column 1 will be updated. For the
        parent group (the row item representing all detector segments), it will
        display as either:
        - <Mixed> if the underlying segments have different algorithms/states.
        - Disabled if everything is disabled
        - <Alg Name> if all the underlying segments have the same algorithm

        - A mixed state:

        Detector / Segment         | Current Alg
        ---------------------------+---------------------------
        [-] epixuhr3x2 (2 segs)      <Mixed>
            [x] epixuhr3x2_0         Threshold 1
            [ ] epixuhr3x2_1         Disabled

        - A Disabled state:

        Detector / Segment         | Current Alg
        ---------------------------+---------------------------
        [x] epixuhr3x2 (2 segs)      Disabled
            [x] epixuhr3x2_0         Disabled
            [x] epixuhr3x2_1         Disabled

        - A single algorithm state

        Detector / Segment         | Current Alg
        ---------------------------+---------------------------
        [x] epixuhr3x2 (2 segs)      Threshold 1
            [x] epixuhr3x2_0         Threshold 1
            [x] epixuhr3x2_1         Threshold 1

        NOTE: The check box selection determines whether an UPDATE is applied to the
              item, NOT whether the algorithm is enabled. I.e. you select items that
              you want to either Enable/Disable or Change the configuration of.

        Args:
            alg_name (str): The name of the new algorithm being applied.

            ver (str): The version identifier for the algorithm parameter schema.
        """
        enabled: bool = self.chk_enable.isChecked()

        self.tree.blockSignals(True)
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    child.setText(1, f"{alg_name} {ver}" if enabled else "Disabled")
            self._refresh_parent_label(parent)

        self.tree.blockSignals(False)

    def on_apply_changes(self) -> None:
        """Validatesparameters and update checked segments in ConfigDB."""
        checked_segs: List[QTreeWidgetItem] = self.get_checked_segment_items()
        if not checked_segs:
            QMessageBox.warning(
                self,
                "No Segments Selected",
                "Please check at least one segment checkbox in the tree.",
            )
            return

        enabled: bool = self.chk_enable.isChecked()
        alg_name: str = self.combo_alg.currentText()
        ver: str = self.combo_ver.currentText()
        preset: str = (
            self.txt_preset_name.text() if self.chk_save_preset.isChecked() else ""
        )
        if preset.startswith(RESERVED_PREFIX):
            QMessageBox.critical(
                self,
                "Invalid Preset Name",
                f"Preset names may not start with '{RESERVED_PREFIX}'.",
            )
            return
        per_segment: bool = enabled and self.adapter is not None
        params: Dict[str, Any] = {}
        seg_params: Dict[int, Dict[str, Any]] = {}
        pid: Optional[str] = None

        if enabled:
            try:
                params = self.extract_form_parameters()
                if per_segment:
                    partitioned = self.partition_plugin(params)
                    if partitioned is None:
                        return
                    seg_params = partitioned
                else:
                    validate_parameters_against_schema(params, self.current_schema)
            except (AlgValidationError, ValueError) as err:
                QMessageBox.critical(self, "Validation Error", str(err))
                return
            if not per_segment:
                try:
                    pid = self.alg_man.save_parameters(
                        alg_name, ver, params, preset_name=preset
                    )
                except Exception as err:
                    QMessageBox.critical(
                        self, "ConfigDB Error", f"Could not save parameters: {err}"
                    )
                    return

        # Per-segment presets are linked as one detector preset by name encoding
        # Refer to `DRP-algos` for the preset management (src/drpalgos/presets.py)
        seg_presets: Dict[int, str] = {}
        if per_segment and preset:
            assert self.adapter is not None
            geom: DetectorGeometry
            geom, _ = self.detector_geometry()

            group: str = new_preset_group()
            ts: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
            try:
                seg_presets = {
                    seg: encode_preset_name(
                        preset,
                        self.adapter.plugin,
                        geom.det_type,
                        group,
                        seg,
                        len(seg_params),
                        ts,
                    )
                    for seg in seg_params
                }
            except ValueError as err:
                QMessageBox.critical(self, "Invalid Preset Name", str(err))
                return

        label: str = f"{alg_name} {normalize_version(ver)}" if enabled else "Disabled"

        done: List[str] = []
        failed: List[str] = []
        unchanged: List[str] = []

        self.tree.blockSignals(True)
        for item in checked_segs:
            data = item.data(0, Qt.UserRole)
            seg = data["name"]
            if per_segment and data["seg"] not in seg_params:
                unchanged.append(seg)
                continue
            try:
                doc = self.segment_config(seg)
                if enabled:
                    seg_pid: Optional[str] = pid
                    if per_segment:
                        seg_pid = self.alg_man.save_parameters(
                            alg_name,
                            ver,
                            seg_params[data["seg"]],
                            preset_name=seg_presets.get(data["seg"], ""),
                        )
                    assert isinstance(seg_pid, str)
                    self.alg_man.attach_alg_to_detector(
                        self.hutch, self.alias, doc, alg_name, ver, seg_pid
                    )
                else:
                    self.alg_man.detach_alg_from_detector(self.hutch, self.alias, doc)
            except Exception as err:
                failed.append(f"{seg}: {err}")
                continue

            data["ref"] = doc.get("drp_alg0")  # attach/detach mutate doc in place
            item.setData(0, Qt.UserRole, data)
            item.setText(1, label)
            done.append(seg)

        for parent in {id(i.parent()): i.parent() for i in checked_segs}.values():
            self._refresh_parent_label(parent)

        self.tree.blockSignals(False)

        if not failed:
            self.plugin_dirty = False  # Stored parameters now match the intent

        if preset and done:
            try:
                self.presets = self.alg_man.list_presets(alg_name, ver)
                self.refresh_presets()
            except Exception as err:
                logger.warning(f"Could not refresh presets: {err}")

        msg: str = f"Applied to: {', '.join(done) or 'none'}"
        if unchanged:
            msg += f"\n\nUnchanged (nothing to apply): {', '.join(unchanged)}"
        if failed:
            QMessageBox.critical(
                self, "Partial Failure", msg + "\n\nFailed:\n" + "\n".join(failed)
            )
        else:
            QMessageBox.information(self, "Applied", f"{label}\n\n{msg}")

    #############################################
    ### End Form Updates and Applying Changes ###

    ### Bridge Detector (Plugin) View and Per-Segment View ###
    ##########################################################

    def partition_plugin(
        self, params: Dict[str, Any]
    ) -> Optional[Dict[int, Dict[str, Any]]]:
        """Split the plugin's intent into complete, validated per-segment parameter sets.

        Shows blocking problems, and asks before continuing past warnings.

        Plugin's work in per-detector space (through intent system). But the rest of the
        DAQ works in per-DRP (per-segment, per-panel, per-process, etc.). This method
        bridges these two spaces, converting the detector-view into the individual
        segment/panel views.

        Args:
            params (Dict[str, Any]): Form parameters, shared by every segment.

        Returns:
            seg_params (Optional[Dict[int, Dict[str, Any]]]): Parameters per segment the
                intent applies to, or None if nothing should be applied.

        Raises:
            AlgValidationError: If a segment's parameters fail schema validation.
        """
        assert self.plugin is not None and self.adapter is not None

        problems: List[str] = self.check_plugin()
        errors: List[str] = [p for p in problems if p.startswith("Error:")]

        if errors:
            QMessageBox.critical(self, "Cannot Apply", "\n".join(errors))
            return None
        if problems:
            answer = QMessageBox.question(
                self, "Apply With Warnings?", "\n".join(problems) + "\n\nApply anyway?"
            )
            if answer != QMessageBox.Yes:
                return None

        geom: DetectorGeometry
        segs: List[int]
        geom, segs = self.detector_geometry()
        parts: Dict[int, Dict[str, Any]] = self.adapter.partition(
            self.plugin.intent(), geom, segs, params
        )
        if not parts:
            QMessageBox.warning(
                self,
                "Nothing To Apply",
                "The plugin settings cover no checked segment.",
            )
            return None

        seg_params: Dict[int, Dict[str, Any]] = {}
        for seg, part in parts.items():
            full: Dict[str, Any] = dict(params)
            full.update(part)
            try:
                validate_parameters_against_schema(full, self.current_schema)
            except AlgValidationError as err:
                raise AlgValidationError(f"Segment {seg}: {err}") from err
            seg_params[seg] = full
        return seg_params

    #################################################################
    ### End of Bridge Detector (Plugin) View and Per-Segment View ###

    ### Some ConfigDB Interactions ###
    ##################################

    def devices_from_configdb(self) -> Dict[str, List[int]]:
        tree: Dict[str, List[int]] = {}
        for dev in self.confdb.get_devices(self.alias, hutch=self.hutch):
            det, _, seg = dev.rpartition("_")
            if det and seg.isdigit():
                tree.setdefault(det, []).append(int(seg))
        return {det: sorted(segs) for det, segs in tree.items()}

    def segment_config(self, seg_id: str) -> Dict[str, Any]:
        """The segment's current configuration document (empty if unavailable)."""
        return self.confdb.get_configuration(self.alias, seg_id, hutch=self.hutch) or {}

    def current_alg_ref(self, seg_id: str) -> Optional[Dict[str, str]]:
        return self.segment_config(seg_id).get("drp_alg0")
