"""
Main window panel for configuring DRP algorithms and assigning config to segments.

Classes:
    CGWDrpAlgManagerSplitPanel: The split detector hierarchy/algorithm config panel.
"""

import json
from typing import Any, ClassVar, Dict, List, Optional

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

from psdaq.configdb.alg_parameter_manager import normalize_version, DrpAlgParamManager
from psdaq.configdb.alg_parameter_validator import (
    typerange,
    validate_parameters_against_schema,
    AlgValidationError,
)
from psdaq.control_gui.drp_alg_plugins import DrpAlgParamPlugin, PseudoGeometry, load_plugin
from psdaq.control_gui.CGJsonUtils import dict_platform

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


def make_param_widget(spec):
    """Returns (widget, getter) where getter() yields a correctly typed value."""
    t: str = spec.get("type", "string")
    default: str = spec.get("default")
    if t in ("boolEnum", "boolean"):  # before enum: boolEnum has enum
        w: QCheckBox = QCheckBox()
        w.setChecked(bool(default))
        return w, w.isChecked
    if "enum" in spec:
        opts = list(spec["enum"])
        w = QComboBox()
        w.addItems([str(o) for o in opts])
        if default in opts:
            w.setCurrentIndex(opts.index(default))
        return w, lambda: opts[w.currentIndex()]  # original typed value
    if t in _INT_T:
        lo, hi = typerange.get(t, _I32)
        lo, hi = spec.get("minimum", lo), spec.get("maximum", hi)
        if _I32[0] <= lo and hi <= _I32[1]:  # QSpinBox is int32
            w = QSpinBox()
            w.setRange(lo, hi)
            if default is not None:
                w.setValue(int(default))
            return w, w.value
        w = QLineEdit("" if default is None else str(default))
        return w, lambda: int(w.text())
    if t in _FLT_T:
        w = QDoubleSpinBox()
        w.setDecimals(6)
        w.setRange(spec.get("minimum", -1e12), spec.get("maximum", 1e12))
        if default is not None:
            w.setValue(float(default))
        return w, w.value
    if t == "array":
        w = QLineEdit(json.dumps(default) if default is not None else "[]")
        return w, lambda: json.loads(w.text())
    w = QLineEdit("" if default is None else str(default))
    return w, w.text


def set_param_widget(w, spec, value) -> None:
    if isinstance(w, QCheckBox):
        w.setChecked(bool(value))
    elif isinstance(w, QComboBox):
        opts = list(spec["enum"])
        if value in opts:
            w.setCurrentIndex(opts.index(value))
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        w.setValue(value)
    elif isinstance(w, QLineEdit):
        w.setText(json.dumps(value) if spec.get("type") == "array" else str(value))


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

    MIXED_DISP_ID: ClassVar[str] = (
        "<Mixed>"  # If different segments have different algs
    )
    CUSTOM: ClassVar[str] = "-- Custom --"  # For non-preset parameter sets

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

        self.current_schema = {}
        self.field_widgets = {}
        self.is_updating_checkboxes: bool = False

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

        # TODO: Use real intialization once configDB stuff is setup!
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
        self.refresh_tree() # resets self.detector_tree_data
        # self.detector_tree_data = self.devices_from_configdb()
        # self.populate_tree()

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

            for s in segs:
                child: QTreeWidgetItem = QTreeWidgetItem(parent)
                seg_id: str = f"{det_name}_{s}"
                child.setText(0, seg_id)
                try:
                    ref: Optional[Dict[str, Any]] = self.current_alg_ref(seg_id)
                    txt: str = f"{ref['alg_name']} {ref['version']}" if ref else "Disabled"
                    child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
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
                        "seg": s,
                        "ref": ref,
                    },
                )
            self._refresh_parent_label(parent)
            parent.setExpanded(True)

        self.tree.blockSignals(False)

    def pseudo_geometry(self) -> PseudoGeometry:
        """Pseudo multi-segment layout of the checked segments (no psana geometry)."""
        segs: List[int] = sorted(
            item.data(0, Qt.UserRole)["seg"] for item in self.get_checked_segment_items()
        ) or [0]
        in_shape: List[int] = (
            self.current_schema.get("properties", {}).get("in_shape", {}).get("default")
            or self.gui_plugin.get("in_shape", [6, 168, 192])
        )
        panel_grid: List[int] = self.gui_plugin.get("panel_grid", [2, 3])
        return PseudoGeometry.grid(segs, tuple(panel_grid), (in_shape[1], in_shape[2]))

    def refresh_plugin(self) -> None:
        """Update plugin geometry, and load existing values of newly checked segments."""
        if self.plugin is None:
            return
        self.plugin.set_geometry(self.pseudo_geometry())

        alg_name: str = self.combo_alg.currentText()
        version: str = self.combo_ver.currentText()
        for item in self.get_checked_segment_items():
            data: Dict[str, Any] = item.data(0, Qt.UserRole)
            ref: Optional[Dict[str, str]] = data.get("ref")
            if data["seg"] in self.plugin_loaded_segs or not ref:
                continue
            if ref["alg_name"] != alg_name or normalize_version(ref["version"]) != normalize_version(version):
                continue
            try:
                doc = self.alg_man.get_algorithm_parameters(alg_name, version, ref["params_id"])
                self.plugin.set_values(data["seg"], doc["parameters"])
                self.plugin_loaded_segs.add(data["seg"])
            except Exception as err:
                logger.warning(f"Could not load {data['name']} parameters into plugin: {err}")

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
        coll_name: str = f"alg/{alg_name}/{version}"

        self.current_schema = {}
        self.gui_plugin = {}
        self.presets = []
        if alg_name and version:
            try:
                self.current_schema = self.alg_man.get_algorithm_schema(
                    alg_name, version
                )
                metadata: Dict[str, Any] = self.alg_man.get_algorithm_metadata(alg_name, version)
                self.gui_plugin = metadata.get("gui_plugin") or {}
                self.presets = self.alg_man.list_presets(alg_name, version)
            except Exception as err:
                QMessageBox.critical(
                    self, "ConfigDB Error", f"{alg_name} {version}: {err}"
                )
        self.build_dynamic_form()
        self.refresh_presets()

    def build_dynamic_form(self) -> None:
        """Generates PyQt5 widgets from current_schema.properties."""
        # Clear existing form
        while self.param_form_layout.count() > 0:
            item = self.param_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.field_widgets.clear()
        self.plugin = None
        self.plugin_loaded_segs = set()

        properties = self.current_schema.get("properties", {})
        required = self.current_schema.get("required", [])

        plugin_cls = load_plugin(self.gui_plugin) if self.gui_plugin else None
        if self.gui_plugin and plugin_cls is None:
            self.param_form_layout.addRow(
                QLabel(
                    f"GUI plugin unavailable: {self.gui_plugin.get('module')}"
                    f".{self.gui_plugin.get('entry_point')}"
                )
            )

        # Choose appropriate widgets based on the type of each parameter
        plugin_specs: Dict[str, Dict[str, Any]] = {}
        for field_name, spec in properties.items():
            if plugin_cls is not None and field_name in plugin_cls.params:
                plugin_specs[field_name] = spec
                continue
            label = spec.get("title", field_name) + (
                " *" if field_name in required else ""
            )
            widget, getter = make_param_widget(spec)
            self.field_widgets[field_name] = (widget, getter, spec)
            self.param_form_layout.addRow(QLabel(label), widget)

        if plugin_cls is not None:
            self.plugin = plugin_cls(plugin_specs, self.gui_plugin)
            plugin_box: QGroupBox = QGroupBox(self.gui_plugin.get("label", plugin_cls.__name__))
            plugin_box_layout: QVBoxLayout = QVBoxLayout()
            plugin_box_layout.addWidget(self.plugin)
            plugin_box.setLayout(plugin_box_layout)
            self.param_form_layout.addRow(plugin_box)
            self.refresh_plugin()

    def extract_form_parameters(self) -> Dict[str, Any]:
        """Extracts parameter dictionary from form widgets."""
        return {name: getter() for name, (_, getter, _) in self.field_widgets.items()}

    def refresh_presets(self) -> None:
        self.combo_preset.blockSignals(True)
        self.combo_preset.clear()
        self.combo_preset.addItems(
            [p["preset_name"] for p in self.presets] + [self.CUSTOM]
        )
        self.combo_preset.setCurrentText(self.CUSTOM)
        self.combo_preset.blockSignals(False)

    def on_preset_changed(self, preset_name: str) -> None:
        """Handles preset selection changes."""
        preset = next(
            (p for p in self.presets if p["preset_name"] == preset_name), None
        )
        if preset is not None:
            self.fill_form(preset["parameters"])

    def fill_form(self, params: Dict[str, Any], segment: Optional[int] = None) -> None:
        for name, value in params.items():
            if name in self.field_widgets:
                widget, _, spec = self.field_widgets[name]
                set_param_widget(widget, spec, value)

        if self.plugin is not None and segment is not None:
            self.plugin.set_values(segment, params)
            self.plugin_loaded_segs.add(segment)

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
        """Traverses tree and returns list of all checked segment items."""
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
        """Validates parameters and updates checked segments in ConfigDB."""
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
        per_segment: bool = self.plugin is not None and self.plugin.segment_aware
        params: Dict[str, Any] = {}
        pid: Optional[str] = None

        if enabled:
            try:
                params = self.extract_form_parameters()
                if not per_segment:
                    if self.plugin is not None:
                        params.update(self.plugin.values_for_segment(-1))
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

        label = f"{alg_name} {normalize_version(ver)}" if enabled else "Disabled"
        done, failed = [], []
        self.tree.blockSignals(True)
        for item in checked_segs:
            data = item.data(0, Qt.UserRole)
            seg = data["name"]
            try:
                doc = self.confdb.get_configuration(self.alias, seg, hutch=self.hutch)
                if enabled:
                    seg_pid: Optional[str] = pid
                    if per_segment:
                        seg_params: Dict[str, Any] = dict(params)
                        seg_params.update(self.plugin.values_for_segment(data["seg"]))
                        validate_parameters_against_schema(seg_params, self.current_schema)
                        seg_pid = self.alg_man.save_parameters(
                            alg_name, ver, seg_params, preset_name=preset
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

        if failed:
            QMessageBox.critical(
                self,
                "Partial Failure",
                f"Applied to: {', '.join(done) or 'none'}\n\nFailed:\n"
                + "\n".join(failed),
            )
        else:
            QMessageBox.information(
                self, "Applied", f"{label} applied to: {', '.join(done)}"
            )

    def devices_from_configdb(self) -> Dict[str, List[int]]:
        tree: Dict[str, List[int]] = {}
        for dev in self.confdb.get_devices(self.alias, hutch=self.hutch):
            det, _, seg = dev.rpartition("_")
            if det and seg.isdigit():
                tree.setdefault(det, []).append(int(seg))
        return {det: sorted(segs) for det, segs in tree.items()}

    def current_alg_ref(self, seg_id: str) -> Optional[Dict[str, str]]:
        doc = self.confdb.get_configuration(self.alias, seg_id, hutch=self.hutch)
        return (doc or {}).get("drp_alg0")
