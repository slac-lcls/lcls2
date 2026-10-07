"""
Main window panel for configuring DRP algorithms and assigning config to segments.

Classes:
    CGWDrpAlgManagerSplitPanel: The split detector hierarchy/algorithm config panel.
"""

from typing import Any, Dict, Optional

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

# TODO: Replace placeholders for validation with configDB integration and real routines
class AlgValidationError(Exception):
    """Exception raised when parameter validation fails."""
    pass


def validate_parameters_against_schema(
    params: Dict[str, Any], json_schema: Dict[str, Any]
) -> None:
    ...


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

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        confdb: Optional[str] = None,
        hutch: str = "xpp",
        detector_tree_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.confdb = confdb
        self.hutch = hutch

        self.detector_tree_data = detector_tree_data or {
            "epixuhr3x2": 2,
            "jungfrau16m": 5,
            "wave8": 1,
        }

        self.current_schema = {}
        self.field_widgets = {}
        self.is_updating_checkboxes: bool = False

        self.setWindowTitle("DRP Algorithm Manager")
        self.resize(950, 550)
        self.init_ui()

    def init_ui(self) -> None:
        main_layout: QVBoxLayout = QVBoxLayout()
        self.setLayout(main_layout)

        # View split into detector/segment on left, and algorithm parameters on right
        splitter: QSplitter = QSplitter(Qt.Horizontal)

        # Detector/segment tree selector
        tree_group: QGroupBox = QGroupBox("Partition Detectors & Segments")
        tree_layout: QVBoxLayout = QVBoxLayout()

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

        inspector_group: QGroupBox = QGroupBox("Inspector & Algorithm Config")
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
        self.combo_alg.currentTextChanged.connect(self.on_alg_or_ver_changed)
        alg_select_layout.addRow("Algorithm:", self.combo_alg)

        self.combo_ver: QComboBox = QComboBox()
        self.combo_ver.addItems(["v1", "v2"])
        self.combo_ver.currentTextChanged.connect(self.on_alg_or_ver_changed)
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
        self.populate_tree()
        self.load_schema()

    def populate_tree(self):
        """Populates the QTreeWidget with detectors and their segment children."""
        self.tree.blockSignals(True)
        self.tree.clear()

        for det_name, seg_count in self.detector_tree_data.items():
            parent: QTreeWidget = QTreeWidgetItem(self.tree)
            parent.setText(0, f"{det_name} ({seg_count} segs)")
            parent.setText(1, "Binning v1")
            parent.setFlags(
                parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
            )
            parent.setCheckState(0, Qt.Checked)
            parent.setData(0, Qt.UserRole, {"type": "detector", "name": det_name})

            for s in range(seg_count):
                child: QTreeWidgetItem = QTreeWidgetItem(parent)
                seg_id: str = f"{det_name}_{s}"
                child.setText(0, seg_id)
                child.setText(1, "Binning v1")
                child.setFlags(
                    child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
                )
                child.setCheckState(0, Qt.Checked)
                child.setData(
                    0,
                    Qt.UserRole,
                    {"type": "segment", "name": seg_id, "det": det_name, "seg": s},
                )

            parent.setExpanded(True)

        self.tree.blockSignals(False)

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

    def on_alg_or_ver_changed(self) -> None:
        """Re-loads schema and re-builds parameters when Alg or Version changes."""
        self.load_schema()

    def load_schema(self) -> None:
        """Fetches schema from ConfigDB (or fallback dict) and builds dynamic form."""
        # TODO: This is still placeholder!!
        alg_name: str = self.combo_alg.currentText()
        version: int = int(self.combo_ver.currentText().replace("v", ""))
        coll_name: str = f"alg/{alg_name}/{version}"

        if self.confdb:
            try:
                schema_doc = self.confdb.get_configuration("_schema", coll_name)
                self.current_schema = schema_doc.get("json_schema", {})
            except Exception:
                self.current_schema = self._get_fallback_schema(alg_name)
        else:
            self.current_schema = self._get_fallback_schema(alg_name)

        self.build_dynamic_form()

    def build_dynamic_form(self) -> None:
        """Generates PyQt5 widgets from current_schema.properties."""
        # Clear existing form
        while self.param_form_layout.count() > 0:
            item = self.param_form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.field_widgets.clear()
        properties = self.current_schema.get("properties", {})
        required = self.current_schema.get("required", [])

        # Choose appropriate widgets based on the type of each parameter
        # TODO: Add in the hook here to allow handing off parameter selection
        #       to another widget/program (e.g. use an ROI selector program)
        for field_name, spec in properties.items():
            label_text = spec.get("title", field_name)
            if field_name in required:
                label_text += " *"

            field_type = spec.get("type", "string")
            default_val = spec.get("default")

            if "enum" in spec:
                widget = QComboBox()
                for opt in spec["enum"]:
                    widget.addItem(str(opt))
                if default_val in spec["enum"]:
                    widget.setCurrentText(str(default_val))

            elif field_type == "boolean":
                widget = QCheckBox()
                if default_val:
                    widget.setChecked(True)

            elif field_type == "integer":
                widget = QSpinBox()
                widget.setRange(
                    spec.get("minimum", -999999), spec.get("maximum", 999999)
                )
                if default_val is not None:
                    widget.setValue(int(default_val))

            elif field_type == "number":
                widget = QDoubleSpinBox()
                widget.setDecimals(4)
                widget.setRange(
                    spec.get("minimum", -999999.0), spec.get("maximum", 999999.0)
                )
                if default_val is not None:
                    widget.setValue(float(default_val))

            else:
                widget = QLineEdit()
                if default_val is not None:
                    widget.setText(str(default_val))

            self.field_widgets[field_name] = (widget, field_type, spec)
            self.param_form_layout.addRow(QLabel(label_text), widget)

    def extract_form_parameters(self) -> Dict[str, Any]:
        """Extracts parameter dictionary from form widgets."""
        params = {}
        for field_name, (widget, field_type, spec) in self.field_widgets.items():
            if "enum" in spec:
                params[field_name] = widget.currentText()
            elif field_type == "boolean":
                params[field_name] = widget.isChecked()
            elif field_type == "integer":
                params[field_name] = widget.value()
            elif field_type == "number":
                params[field_name] = widget.value()
            else:
                params[field_name] = widget.text()
        return params

    def on_preset_changed(self, preset_name: str) -> None:
        """Handles preset selection changes."""
        if preset_name == "-- Custom --":
            return
        # TODO: This is just a placeholder for development. Need real presets logic
        if (
            preset_name == "Standard Low Noise"
            and "threshold_value" in self.field_widgets
        ):
            widget = self.field_widgets["threshold_value"][0]
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(15.0)

    def get_checked_segment_names(self):
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

        MIXED_DISP_ID: str = "<Mixed>"
        self.tree.blockSignals(True)
        root = self.tree.invisibleRootItem()
        for i in range(root.childCount()):
            parent = root.child(i)
            parent_state: str = ""
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    child.setText(1, f"{alg_name} {ver}" if enabled else "Disabled")
                child_txt: str = child.text(1)
                if j == 0:
                    parent_state = child_txt
                elif child_txt != parent_state:
                    parent_state = MIXED_DISP_ID
            parent.setText(1, parent_state)

        self.tree.blockSignals(False)

    def on_apply_changes(self) -> None:
        """Validates parameters and updates checked segments in ConfigDB."""
        checked_segs = self.get_checked_segment_names()
        if not checked_segs:
            QMessageBox.warning(
                self,
                "No Segments Selected",
                "Please check at least one segment checkbox in the tree.",
            )
            return

        params = self.extract_form_parameters()
        try:
            validate_parameters_against_schema(params, self.current_schema)
        except AlgValidationError as e:
            QMessageBox.critical(self, "Validation Error", str(e))
            return

        alg_name: str = self.combo_alg.currentText()
        ver_str: str = self.combo_ver.currentText().replace("v", "")
        coll_name: str = f"alg/{alg_name}/{ver_str}"

        # Update labels in tree for checked items
        # TODO: How do we want to handle highlight vs checked in tree vs enable DRP box?
        self.update_det_tree_alg_col(alg_name=alg_name, ver=ver_str)

        msg: str = (
            f"Successfully applied {coll_name} to {len(checked_segs)} checked segment(s):\n"
            f"Segments: {', '.join(checked_segs)}\n"
            f"Parameters: {params}"
        )
        QMessageBox.information(self, "Assignment Successful", msg)

    def _get_fallback_schema(self, alg_name) -> Dict[str, Any]:
        """Fallback JSON schemas for testing."""
        if alg_name == "Binning":
            return {
                "title": "Binning Algorithm Parameters",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["PIXEL_VALUE", "DETECTOR_SUM"],
                        "default": "PIXEL_VALUE",
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["THRESHOLD", "REJECT_GREATER_THAN"],
                        "default": "THRESHOLD",
                    },
                    "threshold_value": {
                        "type": "number",
                        "minimum": 0.0,
                        "default": 10.0,
                    },
                },
                "required": ["field", "operation"],
            }
        else:
            return {
                "title": "Threshold Algorithm Parameters",
                "properties": {
                    "cutoff": {"type": "number", "minimum": 0.0, "default": 100.0},
                    "invert": {"type": "boolean", "default": False},
                },
                "required": ["cutoff"],
            }


if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication

    app: QApplication = QApplication(sys.argv)
    w: CGWDrpAlgManagerSplitPanel = CGWDrpAlgManagerSplitPanel()

    w.show()
    app.exec_()
