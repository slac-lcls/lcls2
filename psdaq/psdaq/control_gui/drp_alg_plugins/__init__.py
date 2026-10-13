"""Base classes and interfaces for DRP algorithm GUI plugins.

GUI plugins for editing DRP algorithm parameters that need more than a form field.
The core functionality is determined by an `intent` (described here).

-- Intents --
A plugin is an interface that works at the detector-level. It interacts with a
dictionary which is referred to as an "intent" - the intent describes a set of
intended parameters which need to be translated into the actual structures used by
the DRP.

An intent is returned by the plugin based on a DetectorGeometry it has been
given. The plugin itself does not have knowledge of an algorithm's parameters.
To bridge these two views of the system, a DRP algorithm must provide an
adapter - it translates the plugin's intent (as conceived for a DetectorGeometry)
into the actual set of parameters it must use.

-- Adapaters --
The adapters themselves live with the algorithms (since the algorithm understands
its parameters best). This code is in `DRP-algos` (in the Python package
`drpalgos.adapters`).

-- Intent and Adapter Types --
The GUI plugin will provide a class-level variable `intent_type`.
The adapter (DRP-algos) provides a class-level variable `plugin`.

The two are matched when these variables match: `intent_type == plugin`

Classes:
    DrpAlgParamPlugin: Base class for intent-editing plugins.

Functions:
    load_plugin: Resolve an algorithm's gui_plugin metadata to a plugin class.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, Type

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

if TYPE_CHECKING:
    from drpalgos.geometry import DetectorGeometry


class DrpAlgParamPlugin(QWidget):
    """Base class for GUI plugins editing a detector-level intent."""

    valuesChanged = pyqtSignal()  # Emitted on user edits only, not by set_intent
    intent_type: ClassVar[str] = ""
    """The intent_type MUST match the adapter `plugin` variable. This is how they are matched."""

    def __init__(self, options: Dict[str, Any], parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.options: Dict[str, Any] = options

    def set_geometry(self, geom: Optional[DetectorGeometry]) -> None:
        """Set the detector geometry to edit over (None if unavailable)."""
        raise NotImplementedError

    def set_intent(self, intent: Dict[str, Any]) -> None:
        """Replace the current intent (does not emit valuesChanged)."""
        raise NotImplementedError

    def intent(self) -> Dict[str, Any]:
        """Return the current intent."""
        raise NotImplementedError


_ALLOWED_PREFIX: str = __name__  # "psdaq.control_gui.drp_alg_plugins"


def load_plugin(gui_plugin: Dict[str, Any]) -> Optional[Type["DrpAlgParamPlugin"]]:
    """Resolve an algorithm's gui_plugin metadata to a plugin class."""

    module: str = gui_plugin.get("module", "")
    entry_point: str = gui_plugin.get("entry_point", "")

    if not (module == _ALLOWED_PREFIX or module.startswith(_ALLOWED_PREFIX + ".")):
        return None
    try:
        cls = getattr(importlib.import_module(module), entry_point)
    except (ImportError, AttributeError):
        return None

    return cls if isinstance(cls, type) and issubclass(cls, DrpAlgParamPlugin) else None
