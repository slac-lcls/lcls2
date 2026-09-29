#!/usr/bin/env python

"""configdb_GUI.py: browse the configuration database and change one parameter
across many detectors at once.

The window is four panes, left to right:

  1. hutch / configuration-alias tree   (what to look at)
  2. the detectors of that alias        (multi-select: what to change)
  3. the parameter tree of the first    (which parameter to change)
     selected detector
  4. a log of everything that happened

The hutches and their aliases are listed at start-up; only the alias you select
is expanded into detectors, and only the detector you select is read.  Nothing
scans the whole database.  Hutches, aliases, detectors and parameters all appear
in the order the database returns them until you click a column header to sort.

Design notes, because the previous version got these wrong:

  * All database access happens in a worker thread; widgets are only ever
    touched from the GUI thread, via signals.
  * Values are converted using the *schema* (the ':types:' tree that configdb
    ships inside every configuration), not by guessing from the type of the
    value that happens to be stored.  typed_json.updateValue does the path,
    type, range and enum checking for us, atomically.
  * The session starts read-only, and reads go to the unauthenticated URL, so
    browsing cannot write.  Writing is an explicit opt-in.

The production database is the default; pass --dev for development.  Browsing
production is safe -- a read-only session literally cannot write, because the
URL it uses has no authentication -- but note that once writes are enabled the
target is live DAQ configuration.
"""

__author__ = "Riccardo Melchiorri"

import argparse
import copy
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Qt binding.  PyQt5 is what the rest of psdaq uses; going to PyQt6 later
# should only mean deleting the first branch.  If neither is installed we
# still import cleanly, so that the schema/validation layer below can be
# unit-tested on a machine with no Qt and no display -- main() then exits
# with a clear message instead of a traceback.
try:
    from PyQt6 import QtCore, QtGui, QtWidgets                      # noqa: F401
    from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal, pyqtSlot
    HAVE_QT = True
except ImportError:
    try:
        from PyQt5 import QtCore, QtGui, QtWidgets                  # noqa: F401
        from PyQt5.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal, pyqtSlot
        HAVE_QT = True
    except ImportError:
        HAVE_QT = False

        class QObject(object):                                      # noqa: D401
            """Stand-in so the pure-logic layer imports without Qt."""

            def __init__(self, *args, **kwargs):
                pass

        QtCore = QtGui = QtWidgets = None
        QSettings = QThread = Qt = None

        def pyqtSignal(*args, **kwargs):
            return None

        def pyqtSlot(*args, **kwargs):
            return lambda fn: fn

from psdaq.configdb.configdb import configdb
from psdaq.configdb.typed_json import getType, getValue, typerange, updateValue

logger = logging.getLogger(__name__)

PSWWW = 'https://pswww.slac.stanford.edu/ws-auth/%s/ws/'
DEFAULT_ROOT = 'configDB'
MAX_WORKERS = 8
READ_TIMEOUT = 30.0                          # configdb's own default is 8.05 s
PREVIEW_ELEMENTS = 12                        # array elements shown before eliding
COLUMN_PADDING = 12                          # breathing room when fitting a column
COLUMN_MAX_WIDTH = 400                       # a long enum must not fill the pane

# updateValue() reports what went wrong as a small integer.
UPDATE_STATUS = {
    0: 'ok',
    1: 'no such parameter in this detector',
    2: 'value is not valid for this parameter type',
    3: 'malformed configuration',
}


# --------------------------------------------------------------------------
# Pure logic.  No Qt, no network -- all of this is testable headlessly.
# --------------------------------------------------------------------------

def resolve_url(url=None, dev=False):
    """Database URL.  An explicit --url wins, otherwise production unless --dev.

    Production is the default because that is what this tool is normally
    pointed at; browsing it is harmless, since reads use the unauthenticated
    endpoint and the session starts read-only.
    """
    if url:
        return url
    return PSWWW % ('devconfigdb' if dev else 'configdb')


def read_only_url(url):
    """Strip authentication from a URL; reads do not need it.

    The configdb CLI does exactly this for its read-only subcommands.  It also
    avoids renegotiating a Kerberos ticket on every single request.
    """
    return url.replace('ws-auth', 'ws').replace('ws-kerb', 'ws')


def is_prod(url):
    return 'devconfigdb' not in url


def is_hidden_key(key):
    """True for keys the editor should not show.

    ':types:' is the schema itself.  Keys carrying ':RO' are owned by the
    config-store scripts or written by the DRP at run time; 'help:RO' is the
    documented exception, as it is per-detector help text worth reading.
    """
    return key == ':types:' or (':RO' in key and key != 'help:RO')


class TypeInfo(object):
    """What getType() told us about one parameter, in a form the UI can use.

    kind is one of 'int', 'float', 'str', 'enum' or 'array'.
    """

    __slots__ = ('kind', 'base', 'labels', 'shape', 'limits', 'display')

    def __init__(self, kind, base=None, labels=None, shape=None, limits=None):
        self.kind = kind
        self.base = base
        self.labels = labels or {}
        self.shape = shape or ()
        self.limits = limits
        self.display = self._describe()

    def _describe(self):
        if self.kind == 'enum':
            return '|'.join(self.labels.keys())
        if self.kind == 'array':
            dims = 'x'.join(str(d) for d in self.shape)
            base = '|'.join(self.base.keys()) if isinstance(self.base, dict) else self.base
            return '%s[%s]' % (base, dims)
        if self.limits:
            return '%s (%d..%d)' % (self.base, self.limits[0], self.limits[1])
        return str(self.base)

    @property
    def numeric(self):
        """Can 'delta:N' be applied?  Only to a plain number."""
        return self.kind in ('int', 'float')

    def hint(self):
        if self.kind == 'enum':
            return 'one of: %s' % self.display
        if self.kind == 'array':
            return '%d values, space separated (%s)' % (
                _count(self.shape), self.display)
        return self.display


def _count(shape):
    total = 1
    for dim in shape:
        total *= dim
    return total


def describe_type(spec):
    """Turn a getType() result into a TypeInfo, or None if it is unusable.

    getType returns a type name for a scalar, an already-resolved {label: int}
    dict for an enum, or [base, *shape] for an array -- where base may itself
    be a resolved enum dict.
    """
    if spec is None:
        return None
    if isinstance(spec, dict):
        return TypeInfo('enum', labels=spec)
    if isinstance(spec, list):
        if not spec:
            return None
        return TypeInfo('array', base=spec[0], shape=tuple(spec[1:]))
    if spec == 'CHARSTR':
        return TypeInfo('str', base=spec)
    if spec in ('FLOAT', 'DOUBLE'):
        return TypeInfo('float', base=spec)
    if spec in typerange:
        return TypeInfo('int', base=spec, limits=typerange[spec])
    return None


def type_of(config, path):
    """describe_type(getType(...)), tolerating a schema we cannot make sense of."""
    try:
        return describe_type(getType(config, path))
    except (TypeError, KeyError, AttributeError, IndexError) as exc:
        logger.debug('no usable type for %s: %s', path, exc)
        return None


def sort_key(text):
    """Order text the way a person reads it.

    Digit runs compare as numbers, so hsd_2 comes before hsd_10 and the value
    9 before 100; a plain string comparison gets both of those backwards.
    Letters compare case-insensitively.
    """
    try:                                     # a bare number is the common case
        return [(0, float(text), '')]
    except (TypeError, ValueError):
        pass
    parts = []
    digits = ''
    for char in text or '':
        if char.isdigit():
            digits += char
        else:
            if digits:
                parts.append((0, float(digits), ''))
                digits = ''
            parts.append((1, 0.0, char.lower()))
    if digits:
        parts.append((0, float(digits), ''))
    return parts


def format_value(value, limit=None):
    """Render a stored value the way the user would type it back in.

    With a limit, long arrays are elided: a detector gain map is a six-figure
    list of numbers, and rendering all of it into a table cell is slow enough
    to be noticeable.  The full text is still what editing works on.
    """
    if isinstance(value, list):
        if limit is not None and len(value) > limit:
            head = ' '.join(format_value(v) for v in value[:limit])
            return '%s ... (%d values)' % (head, len(value))
        return ' '.join(format_value(v) for v in value)
    if value is None:
        return ''
    return str(value)


class ValueSpec(object):
    """What the user asked for: an absolute value, or a relative 'delta:N'."""

    __slots__ = ('text', 'delta')

    def __init__(self, text):
        self.text = text.strip()
        self.delta = None
        if self.text.startswith('delta:'):
            self.delta = self.text[len('delta:'):].strip()

    @property
    def relative(self):
        return self.delta is not None

    def __str__(self):
        return ('%s (relative)' % self.delta) if self.relative else self.text


class Change(object):
    """A planned or completed edit of one parameter on one detector."""

    __slots__ = ('detector', 'path', 'old', 'new', 'ok', 'error', 'config', 'key')

    def __init__(self, detector, path, old=None, new=None,
                 ok=False, error=None, config=None):
        self.detector = detector
        self.path = path
        self.old = old
        self.new = new
        self.ok = ok
        self.error = error
        self.config = config
        self.key = None

    @property
    def unchanged(self):
        return self.ok and self.new == self.old

    def describe(self):
        if not self.ok:
            return '%s: %s' % (self.detector, self.error)
        if self.unchanged:
            return '%s: %s already %s' % (
                self.detector, self.path,
                format_value(self.old, PREVIEW_ELEMENTS))
        return '%s: %s  %s -> %s' % (self.detector, self.path,
                                     format_value(self.old, PREVIEW_ELEMENTS),
                                     format_value(self.new, PREVIEW_ELEMENTS))


def plan_change(detector, config, path, spec):
    """Work out what writing `spec` to `path` would do, without touching `config`.

    Returns a Change.  Validation is done by actually performing the update on
    a copy: updateValue checks the path, the type, the integer range and the
    enum membership in one step, and leaves the dict alone when it fails.
    """
    if not path:                           # just carrying the configuration back
        return Change(detector, path, ok=True, config=config)

    info = type_of(config, path)
    if info is None:
        return Change(detector, path, error=UPDATE_STATUS[1])

    old = getValue(config, path)
    if spec is None:                       # read-only probe, no edit intended
        return Change(detector, path, old=old, new=old, ok=True)

    if spec.relative:
        if not info.numeric:
            return Change(detector, path, old=old,
                          error='delta: needs a numeric parameter, this is %s'
                                % info.kind)
        try:
            step = float(spec.delta) if info.kind == 'float' else int(spec.delta)
        except ValueError:
            return Change(detector, path, old=old,
                          error='delta: %r is not a number' % spec.delta)
        try:
            wanted = old + step
        except TypeError:
            return Change(detector, path, old=old,
                          error='cannot add %s to stored value %r' % (step, old))
        text = repr(wanted)
    else:
        text = spec.text

    work = copy.deepcopy(config)
    status = updateValue(work, path, text)
    if status != 0:
        return Change(detector, path, old=old,
                      error='%s (%r)' % (UPDATE_STATUS.get(status, status), text))
    return Change(detector, path, old=old, new=getValue(work, path),
                  ok=True, config=work)


# --------------------------------------------------------------------------
# Database access.  Every call here is a blocking HTTP round trip, so
# everything in this section runs in a worker thread.
# --------------------------------------------------------------------------

def open_db(url, hutch, root=DEFAULT_ROOT, user=None, password=None,
            timeout=READ_TIMEOUT):
    """A configdb client.  Each thread must build its own.

    The default 8.05 s is tight for a detector carrying pixel or gain maps, and
    the unauthenticated read path makes a single attempt with no retry, so a
    slow response would otherwise be a hard failure.
    """
    kwargs = {'create': False, 'root': root}
    if user:
        kwargs['user'] = user
    if password:
        kwargs['password'] = password
    db = configdb(url, hutch, **kwargs)
    db.timeout = timeout
    return db


class Worker(QObject):
    """Base for the background jobs: progress, failure, and a way to stop."""

    progress = pyqtSignal(int, int)          # done, total
    failed = pyqtSignal(int, str)            # generation, message
    finished = pyqtSignal()

    def __init__(self, generation, url, root, user=None, password=None):
        super(Worker, self).__init__()
        self.generation = generation
        self.url = url
        self.root = root
        self.user = user
        self.password = password
        self._running = True

    def stop(self):
        self._running = False

    def db(self, hutch):
        return open_db(self.url, hutch, self.root, self.user, self.password)

    @pyqtSlot()
    def run(self):
        try:
            self.work()
        except Exception as exc:
            logger.error('%s: %s', type(self).__name__, exc)
            self.failed.emit(self.generation, str(exc))
        finally:
            self.finished.emit()

    def work(self):
        raise NotImplementedError

    def _map(self, fn, items):
        """Run fn over items in parallel, yielding results in order.

        One item failing must never abort the rest, so every call is wrapped:
        pool.map re-raises on iteration, which would throw away the results
        that did succeed.
        """
        def safely(item):
            try:
                return fn(item)
            except Exception as exc:
                logger.warning('%s: %s', item, exc)
                return None

        total = len(items)
        done = 0
        self.progress.emit(done, total)
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for result in pool.map(safely, items):
                if not self._running:
                    return
                done += 1
                self.progress.emit(done, total)
                if result is not None:
                    yield result


class TreeWorker(Worker):
    """The hutches and their configuration aliases.

    Nothing per-device: one get_hutches plus one get_aliases per hutch.  Only
    runs when asked, so starting the window costs no database access at all.
    """

    loaded = pyqtSignal(int, object)         # generation, [(hutch, [alias, ...])]

    def work(self):
        hutches = self.db(None).get_hutches()
        tree = list(self._map(lambda h: (h, self._aliases(h)), hutches))
        if self._running:
            self.loaded.emit(self.generation, tree)

    def _aliases(self, hutch):
        try:
            return self.db(hutch).get_aliases(hutch=hutch)
        except Exception as exc:
            logger.warning('cannot list aliases of %s: %s', hutch, exc)
            return []


class DeviceWorker(Worker):
    """The detectors of one alias: a single get_devices call.

    Deliberately does not read each detector's configuration.  Doing so to
    learn its detType meant one full download per detector, which is the
    slowest thing the window could possibly do.
    """

    # generation, (hutch, alias), [device, ...] -- the key travels with the
    # payload so a late reply cannot be filed under a different alias.
    loaded = pyqtSignal(int, object, object)

    def __init__(self, generation, url, root, hutch, alias, **kwargs):
        super(DeviceWorker, self).__init__(generation, url, root, **kwargs)
        self.hutch = hutch
        self.alias = alias

    def work(self):
        devices = self.db(self.hutch).get_devices(self.alias, hutch=self.hutch)
        if self._running:
            self.loaded.emit(self.generation, (self.hutch, self.alias), devices)


class ScanWorker(Worker):
    """Read one parameter across several detectors, and optionally plan an edit.

    With spec=None this just reports the current value.  With a spec it
    reports what the edit would do, per detector, so the confirmation dialog
    can show a real diff before anything is written.
    """

    row = pyqtSignal(int, object)            # generation, Change
    scanned = pyqtSignal(int, object)        # generation, [Change, ...]

    def __init__(self, generation, url, root, hutch, alias, devices, path,
                 spec=None, keep_config=False, **kwargs):
        super(ScanWorker, self).__init__(generation, url, root, **kwargs)
        self.hutch = hutch
        self.alias = alias
        self.devices = devices
        self.path = path
        self.spec = spec
        self.keep_config = keep_config

    def work(self):
        changes = list(self._map(self._one, self.devices))
        if self._running:
            self.scanned.emit(self.generation, changes)

    def _one(self, device):
        try:
            config = self.db(self.hutch).get_configuration(
                self.alias, device, hutch=self.hutch)
        except Exception as exc:
            change = Change(device, self.path, error=str(exc))
        else:
            change = plan_change(device, config, self.path, self.spec)
            # Keep the body only when someone is going to use it: the parameter
            # tree needs it, a bulk scan across 20 detectors does not.
            if self.path and not self.keep_config:
                change.config = None
        self.row.emit(self.generation, change)
        return change


class WriteWorker(Worker):
    """Apply already-confirmed changes, one detector at a time.

    Each detector is re-read and re-planned immediately before writing: if its
    stored value no longer matches what the confirmation dialog showed, it is
    skipped rather than silently overwritten.  One failure never stops the
    others, and every successful modify_device mints a new configuration key.
    """

    result = pyqtSignal(int, object)         # generation, Change

    def __init__(self, generation, url, root, hutch, alias, changes, path, spec,
                 **kwargs):
        super(WriteWorker, self).__init__(generation, url, root, **kwargs)
        self.hutch = hutch
        self.alias = alias
        self.changes = changes
        self.path = path
        self.spec = spec

    def work(self):
        total = len(self.changes)
        db = self.db(self.hutch)
        for done, planned in enumerate(self.changes, start=1):
            if not self._running:
                return
            self.result.emit(self.generation, self._write(db, planned))
            self.progress.emit(done, total)

    def _write(self, db, planned):
        device = planned.detector
        try:
            config = db.get_configuration(self.alias, device, hutch=self.hutch)
        except Exception as exc:
            return Change(device, self.path, error='re-read failed: %s' % exc)

        fresh = plan_change(device, config, self.path, self.spec)
        if not fresh.ok:
            return fresh
        if fresh.old != planned.old:
            return Change(device, self.path, old=fresh.old,
                          error='skipped: value changed since preview (now %s)'
                                % format_value(fresh.old, PREVIEW_ELEMENTS))
        if fresh.unchanged:
            return fresh

        try:
            fresh.key = db.modify_device(self.alias, fresh.config, hutch=self.hutch)
        except Exception as exc:
            return Change(device, self.path, old=fresh.old, new=fresh.new,
                          error='write failed: %s' % exc)
        return fresh


# --------------------------------------------------------------------------
# Routing log records into the log pane.
# --------------------------------------------------------------------------

class LogBridge(QObject):
    """Carries log records to the GUI thread.

    A logging handler runs on whichever thread emitted the record, so it must
    not touch a widget.  It emits this signal instead; Qt queues it and the
    text lands in the pane on the GUI thread.
    """

    message = pyqtSignal(str, str)           # level name, formatted text


class SignalLogHandler(logging.Handler):
    def __init__(self, bridge):
        logging.Handler.__init__(self)
        self.bridge = bridge
        self.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))

    def emit(self, record):
        try:
            self.bridge.message.emit(record.levelname, self.format(record))
        except Exception:
            pass                             # never let logging break the GUI


# --------------------------------------------------------------------------
# Widgets and dialogs.
# --------------------------------------------------------------------------

def _exec(dialog):
    return dialog.exec() if hasattr(dialog, 'exec') else dialog.exec_()


def _qt_enum(owner, group, name):
    """PyQt5 puts enum members on the class, PyQt6 nests them in a scoped enum."""
    scoped = getattr(owner, group, None)
    if scoped is not None and hasattr(scoped, name):
        return getattr(scoped, name)
    return getattr(owner, name)


def _clear_sort_indicator(view):
    """Leave a sortable view unsorted until the user clicks a header.

    setSortingEnabled(True) immediately sorts by column 0, which would override
    the order the database gave us.  Setting the indicator to -1 keeps the
    headers clickable while leaving insertion order alone.
    """
    header = view.header() if hasattr(view, 'header') else view.horizontalHeader()
    header.setSortIndicator(-1, _qt_enum(Qt, 'SortOrder', 'AscendingOrder'))


class SortedTreeItem(QtWidgets.QTreeWidgetItem if HAVE_QT else object):
    """Tree row that sorts numerically where the text is numeric."""

    def __lt__(self, other):
        tree = self.treeWidget()
        column = tree.sortColumn() if tree is not None else 0
        if column < 0:
            column = 0
        try:
            return sort_key(self.text(column)) < sort_key(other.text(column))
        except TypeError:                     # mixed shapes: fall back to text
            return self.text(column) < other.text(column)


class ConfirmDialog(QtWidgets.QDialog if HAVE_QT else object):
    """Shows the diff that is about to be written, and asks."""

    def __init__(self, parent, target, path, spec, changes, skipped):
        super(ConfirmDialog, self).__init__(parent)
        self.setWindowTitle('Confirm configuration change')
        layout = QtWidgets.QVBoxLayout(self)

        head = QtWidgets.QLabel(
            'Setting <b>%s</b> to <b>%s</b> on <b>%d</b> detector(s)<br>'
            'target: <b>%s</b>' % (path, spec, len(changes), target))
        head.setTextFormat(_qt_enum(Qt, 'TextFormat', 'RichText'))
        layout.addWidget(head)

        table = QtWidgets.QTableWidget(len(changes), 3, self)
        table.setHorizontalHeaderLabels(['detector', 'current', 'new'])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(_qt_enum(QtWidgets.QAbstractItemView,
                                       'EditTrigger', 'NoEditTriggers'))
        for row, change in enumerate(changes):
            for col, text in enumerate((change.detector,
                                        format_value(change.old, PREVIEW_ELEMENTS),
                                        format_value(change.new, PREVIEW_ELEMENTS))):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        table.resizeColumnsToContents()
        layout.addWidget(table)

        if skipped:
            note = QtWidgets.QTextEdit(self)
            note.setReadOnly(True)
            note.setPlainText('\n'.join(c.describe() for c in skipped))
            note.setMaximumHeight(90)
            layout.addWidget(QtWidgets.QLabel('%d detector(s) will be skipped:'
                                              % len(skipped)))
            layout.addWidget(note)

        layout.addWidget(QtWidgets.QLabel(
            'Each detector written gets a new configuration key.'))

        buttons = QtWidgets.QDialogButtonBox(
            _qt_enum(QtWidgets.QDialogButtonBox, 'StandardButton', 'Ok')
            | _qt_enum(QtWidgets.QDialogButtonBox, 'StandardButton', 'Cancel'),
            parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(560, 420)


# --------------------------------------------------------------------------
# The window.
# --------------------------------------------------------------------------

class ConfigdbGUI(QtWidgets.QMainWindow if HAVE_QT else object):

    def __init__(self, url, root=DEFAULT_ROOT, user=None, password=None,
                 hutch=None, allow_write=False):
        super(ConfigdbGUI, self).__init__()
        self.auth_url = url
        self.root = root
        self.user = user
        self.password = password
        self.start_hutch = hutch

        self.tree = {}                       # hutch -> [alias, ...]
        self.devices = {}                    # (hutch, alias) -> [device, ...]
        self.hutch = None
        self.alias = None
        self.config = None                   # configuration behind the param tree
        self.path = None                     # dotted path of the selected leaf
        self.info = None                     # its TypeInfo
        self.spec = None                     # the edit being applied
        self._wanted_path = None             # selection to restore after a reload
        self.results = {'ok': 0, 'failed': 0, 'skipped': 0}

        self._generation = 0                 # discards replies for stale selections
        self._jobs = {}

        self._build_ui()
        self._install_log_handler()
        self._restore_settings()

        self.write_box.setChecked(not allow_write)
        self._refresh_target()
        # The hutch and alias lists are cheap -- one call each, nothing
        # per-device -- so they are fetched straight away, in the background.
        self.reload()

    # -- construction ------------------------------------------------------

    def _build_ui(self):
        self.setWindowTitle('configdb')
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)

        # top row: what we are pointed at, and the safety catch
        top = QtWidgets.QHBoxLayout()
        self.target_label = QtWidgets.QLabel()
        self.write_box = QtWidgets.QCheckBox('Read-only')
        self.write_box.setToolTip('uncheck to allow writing to the database')
        self.write_box.toggled.connect(self._read_only_toggled)
        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setVisible(False)

        top.addWidget(self.target_label, 1)
        top.addWidget(self.write_box)
        top.addWidget(self.pbar, 1)
        outer.addLayout(top)

        # the four panes.  Every header is clickable to sort, but nothing is
        # sorted until clicked: the database order is the default everywhere.
        self.splitter = QtWidgets.QSplitter(_qt_enum(Qt, 'Orientation', 'Horizontal'))
        self.hutch_tree = QtWidgets.QTreeWidget()
        self.hutch_tree.setHeaderLabel('hutch / alias')
        self.hutch_tree.itemSelectionChanged.connect(self._alias_selected)

        # A single-column tree rather than a list, so that it too has a header
        # to click.  Selection behaviour is unchanged.
        self.device_list = QtWidgets.QTreeWidget()
        self.device_list.setHeaderLabel('detector')
        self.device_list.setRootIsDecorated(False)
        self.device_list.setSelectionMode(
            _qt_enum(QtWidgets.QAbstractItemView, 'SelectionMode', 'ExtendedSelection'))
        self.device_list.itemSelectionChanged.connect(self._devices_selected)

        self.param_tree = QtWidgets.QTreeWidget()
        self.param_tree.setHeaderLabels(['parameter', 'type', 'value'])
        self.param_tree.itemSelectionChanged.connect(self._param_selected)

        for view in (self.hutch_tree, self.device_list, self.param_tree):
            view.setSortingEnabled(True)
            view.header().setSectionsClickable(True)
            _clear_sort_indicator(view)
            # Expanding a node indents its children, so the width that fitted
            # the collapsed tree no longer fits.  Re-fit whenever what is
            # visible changes, so a parameter name is never cut off.
            view.expanded.connect(lambda _index, v=view: self._fit_columns(v))
            view.collapsed.connect(lambda _index, v=view: self._fit_columns(v))
            view.itemSelectionChanged.connect(
                lambda v=view: self._fit_columns(v))

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(
            _qt_enum(QtWidgets.QTextEdit, 'LineWrapMode', 'NoWrap'))

        for widget in (self.hutch_tree, self.device_list, self.param_tree, self.log):
            self.splitter.addWidget(widget)
        self.splitter.setSizes([160, 160, 380, 300])
        outer.addWidget(self.splitter, 1)

        # bottom row: the edit
        bottom = QtWidgets.QHBoxLayout()
        self.path_label = QtWidgets.QLabel('select a parameter')
        self.value_edit = QtWidgets.QLineEdit()
        self.value_edit.setPlaceholderText('new value, or delta:N to add N')
        self.value_edit.textChanged.connect(self._validate)
        self.value_edit.returnPressed.connect(self._apply)
        self.value_combo = QtWidgets.QComboBox()
        self.value_combo.currentTextChanged.connect(self._validate)
        self.value_stack = QtWidgets.QStackedWidget()
        self.value_stack.addWidget(self.value_edit)
        self.value_stack.addWidget(self.value_combo)
        self.hint_label = QtWidgets.QLabel()
        self.apply_button = QtWidgets.QPushButton('Apply')
        self.apply_button.clicked.connect(self._apply)
        self.apply_button.setEnabled(False)

        bottom.addWidget(self.path_label, 2)
        bottom.addWidget(self.value_stack, 1)
        bottom.addWidget(self.hint_label, 1)
        bottom.addWidget(self.apply_button)
        outer.addLayout(bottom)

        self.setStatusBar(QtWidgets.QStatusBar(self))
        self.resize(1100, 560)

    def _install_log_handler(self):
        self.log_bridge = LogBridge()
        self.log_bridge.message.connect(self._append_log)
        handler = SignalLogHandler(self.log_bridge)
        handler.setLevel(logging.INFO)
        # The root logger, so that configdb.py's own error messages -- which it
        # sends straight to the root -- also show up in the pane.
        logging.getLogger().addHandler(handler)
        self._log_handler = handler

    # -- settings ----------------------------------------------------------

    def _restore_settings(self):
        settings = QSettings()
        geometry = settings.value('geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        columns = settings.value('columns')
        if columns:
            try:
                self.splitter.setSizes([int(c) for c in columns])
            except (TypeError, ValueError):
                pass
        if settings.value('log_expanded', 'true') in ('false', False):
            self.log.setVisible(False)
        for name, view in self._sortable_views():
            try:
                column = int(settings.value('sort_%s_column' % name, -1))
                descending = settings.value('sort_%s_desc' % name, 'false')
            except (TypeError, ValueError):
                continue
            if column < 0 or column >= view.columnCount():
                continue
            view.header().setSortIndicator(
                column,
                _qt_enum(Qt, 'SortOrder',
                         'DescendingOrder' if descending in ('true', True)
                         else 'AscendingOrder'))

    def _sortable_views(self):
        return (('hutches', self.hutch_tree),
                ('detectors', self.device_list),
                ('parameters', self.param_tree))

    def _save_settings(self):
        settings = QSettings()
        settings.setValue('geometry', self.saveGeometry())
        settings.setValue('columns', [str(s) for s in self.splitter.sizes()])
        settings.setValue('log_expanded', 'true' if self.log.isVisible() else 'false')
        descending = _qt_enum(Qt, 'SortOrder', 'DescendingOrder')
        for name, view in self._sortable_views():
            header = view.header()
            settings.setValue('sort_%s_column' % name,
                              str(header.sortIndicatorSection()))
            settings.setValue('sort_%s_desc' % name,
                              'true' if header.sortIndicatorOrder() == descending
                              else 'false')

    # -- jobs --------------------------------------------------------------

    def _start(self, name, worker, **connections):
        """Run a worker on its own thread, replacing any previous one."""
        if name == 'write' and self._busy('write'):
            logger.warning('a write is already running; ignoring this one')
            return
        self._stop(name)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.failed.connect(self._job_failed)
        for signal, slot in connections.items():
            getattr(worker, signal).connect(slot)
        worker.finished.connect(thread.quit)
        thread.finished.connect(lambda: self._job_done(name))
        self._jobs[name] = (thread, worker)
        thread.start()

    def _busy(self, name):
        job = self._jobs.get(name)
        return bool(job) and job[0].isRunning()

    def _stop(self, name):
        job = self._jobs.pop(name, None)
        if not job:
            return
        thread, worker = job
        worker.stop()
        thread.quit()
        if not thread.wait(3000):
            logger.warning('%s worker did not stop cleanly', name)

    def _job_done(self, name):
        self._jobs.pop(name, None)
        if not self._jobs:
            self.pbar.setVisible(False)

    def _progress(self, done, total):
        self.pbar.setVisible(total > 1 and done < total)
        self.pbar.setMaximum(max(total, 1))
        self.pbar.setValue(done)

    def _job_failed(self, generation, message):
        if generation == self._generation:
            self.statusBar().showMessage(message, 10000)

    def _fresh(self):
        """Invalidate outstanding replies and return the new generation."""
        self._generation += 1
        return self._generation

    def _current(self, generation):
        return generation == self._generation

    # -- reading -----------------------------------------------------------

    @property
    def read_url(self):
        return read_only_url(self.auth_url)

    def _worker_args(self, write=False):
        if write:
            # get_hutches() reports hutches in upper case but the operator
            # accounts are lower case, so 'TMO' has to become 'tmoopr'.
            user = self.user or (self.hutch.lower() + 'opr' if self.hutch else None)
            return dict(url=self.auth_url, root=self.root,
                        user=user, password=self.password)
        return dict(url=self.read_url, root=self.root)

    def reload(self):
        generation = self._fresh()
        self.hutch_tree.clear()
        self.device_list.clear()
        self.param_tree.clear()
        self.devices.clear()
        self.statusBar().showMessage('loading hutches...')
        self._start('tree',
                    TreeWorker(generation, **self._worker_args()),
                    loaded=self._tree_loaded)

    def _tree_loaded(self, generation, tree):
        if not self._current(generation):
            return
        self.tree = tree
        header = self.hutch_tree.header()
        column, order = header.sortIndicatorSection(), header.sortIndicatorOrder()
        sorting = column >= 0

        self.hutch_tree.setSortingEnabled(False)
        self.hutch_tree.clear()
        for hutch, aliases in tree:          # database order unless sorted
            node = SortedTreeItem(self.hutch_tree, [hutch])
            node.setFlags(node.flags() & ~_qt_enum(Qt, 'ItemFlag', 'ItemIsSelectable'))
            for alias in aliases:
                SortedTreeItem(node, [alias])
            if self.start_hutch and hutch.lower() == self.start_hutch.lower():
                node.setExpanded(True)
        self.hutch_tree.setSortingEnabled(True)
        if sorting:
            self.hutch_tree.sortItems(column, order)
        else:
            _clear_sort_indicator(self.hutch_tree)
        self._fit_columns(self.hutch_tree)
        self.statusBar().showMessage('%d hutches' % len(tree), 5000)

    def _alias_selected(self):
        items = self.hutch_tree.selectedItems()
        self.device_list.clear()
        self.param_tree.clear()
        self._clear_selection()
        if not items or items[0].parent() is None:
            self.hutch = self.alias = None
            return
        self.alias = items[0].text(0)
        self.hutch = items[0].parent().text(0)

        key = (self.hutch, self.alias)
        if key in self.devices:
            self._show_devices()
            return
        generation = self._fresh()
        self.statusBar().showMessage('loading %s/%s...' % key)
        self._start('devices',
                    DeviceWorker(generation, hutch=self.hutch, alias=self.alias,
                                 **self._worker_args()),
                    loaded=self._devices_loaded)

    def _devices_loaded(self, generation, key, devices):
        # Cache under the key the worker was asked about, even if the selection
        # has moved on -- the result is still valid for that alias.
        self.devices[key] = devices
        if not self._current(generation):
            return
        self._show_devices()
        self.statusBar().showMessage('%d detectors' % len(devices), 5000)

    def _show_devices(self):
        devices = self.devices.get((self.hutch, self.alias))
        if devices is None:
            return
        header = self.device_list.header()
        column, order = header.sortIndicatorSection(), header.sortIndicatorOrder()
        sorting = column >= 0                # keep the user's choice across aliases

        self.device_list.setSortingEnabled(False)
        self.device_list.clear()
        for device in devices:               # database order unless sorted
            SortedTreeItem(self.device_list, [device])
        self.device_list.setSortingEnabled(True)
        if sorting:
            self.device_list.sortItems(column, order)
        else:
            _clear_sort_indicator(self.device_list)
        self.device_list.resizeColumnToContents(0)
        self._fit_columns(self.device_list)

    def _selected_devices(self):
        return [item.text(0) for item in self.device_list.selectedItems()]

    def _devices_selected(self, keep_path=False):
        """Load the parameter tree for the first of the selected detectors.

        keep_path is set when we are only refreshing after a write: the user is
        still working on the same parameter, so the selection comes back.
        """
        wanted = self.path if keep_path else None
        self.param_tree.clear()
        self._clear_selection()
        self._wanted_path = wanted
        devices = self._selected_devices()
        if not devices:
            return
        generation = self._fresh()
        self.statusBar().showMessage('loading %s...' % devices[0])
        self._start('config',
                    ScanWorker(generation, hutch=self.hutch, alias=self.alias,
                               devices=devices[:1], path=None, keep_config=True,
                               **self._worker_args()),
                    scanned=self._config_loaded)

    def _config_loaded(self, generation, changes):
        """The first selected detector's configuration becomes the parameter tree."""
        if not self._current(generation) or not changes:
            return
        change = changes[0]
        device = change.detector
        if change.config is None:
            logger.error('cannot read %s: %s', device,
                         change.error or 'no configuration returned')
            return
        self.config = change.config
        wanted = self._wanted_path           # survives a rebuild after a write
        self._wanted_path = None
        # Rebuilding drops the header's sort choice, so put it back afterwards.
        header = self.param_tree.header()
        column, order = header.sortIndicatorSection(), header.sortIndicatorOrder()
        sorting = self.param_tree.isSortingEnabled() and column >= 0

        self.param_tree.setSortingEnabled(False)
        self.param_tree.clear()
        # A different detector needs different widths, so start from the
        # contents rather than from whatever the last one happened to need.
        for column in range(3):
            self.param_tree.resizeColumnToContents(column)
        self.param_tree.setHeaderLabels([device, 'type', 'value'])
        self._populate(self.param_tree, self.config, '')
        self.param_tree.setSortingEnabled(True)
        if sorting:
            self.param_tree.sortItems(column, order)
        else:
            _clear_sort_indicator(self.param_tree)
        self._fit_columns(self.param_tree)
        if wanted:
            self._reselect(wanted)

    def _fit_columns(self, view):
        """Widen columns so nothing visible is cut off.

        Called whenever the visible rows change: expanding a node indents its
        children past the width that fitted the collapsed tree, which is what
        used to leave parameter names truncated.  Columns only ever grow here,
        so a column the user has widened by hand is left alone; the last column
        is skipped because it takes the remaining space anyway.
        """
        if view is None:
            return
        header = view.header()
        for column in range(max(view.columnCount() - 1, 1)):
            needed = view.sizeHintForColumn(column)
            if needed <= 0:
                continue
            needed = min(needed + COLUMN_PADDING, COLUMN_MAX_WIDTH)
            if needed > header.sectionSize(column):
                view.setColumnWidth(column, needed)

    def _reselect(self, path):
        """Put the selection back on `path` after the tree has been rebuilt."""
        role = _qt_enum(Qt, 'ItemDataRole', 'UserRole')
        stack = [self.param_tree.topLevelItem(i)
                 for i in range(self.param_tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, role) == path:
                self.param_tree.setCurrentItem(item)
                parent = item.parent()
                while parent is not None:    # make sure it is actually visible
                    parent.setExpanded(True)
                    parent = parent.parent()
                return
            stack.extend(item.child(i) for i in range(item.childCount()))

    def _populate(self, parent, node, prefix):
        """Build the parameter tree, hiding the schema and read-only keys."""
        if isinstance(node, dict):
            items = list(node.items())
        else:                                # list of sub-structures
            items = [(str(i), v) for i, v in enumerate(node)]

        for key, value in items:
            if is_hidden_key(key):
                continue
            path = key if not prefix else '%s.%s' % (prefix, key)
            compound = isinstance(value, dict) or (
                isinstance(value, list) and value and
                all(isinstance(v, dict) for v in value))
            if compound:
                item = SortedTreeItem(parent, [key + ' *'])
                item.setFlags(item.flags()
                              & ~_qt_enum(Qt, 'ItemFlag', 'ItemIsSelectable'))
                self._populate(item, value, path)
                continue

            info = type_of(self.config, path)
            item = SortedTreeItem(
                parent, [key, info.display if info else '?',
                         format_value(value, PREVIEW_ELEMENTS)])
            item.setToolTip(0, 'path: %s' % path)
            if info is None or key == 'help:RO':
                item.setFlags(item.flags()
                              & ~_qt_enum(Qt, 'ItemFlag', 'ItemIsSelectable'))
            else:
                item.setData(0, _qt_enum(Qt, 'ItemDataRole', 'UserRole'), path)

    # -- the edit ----------------------------------------------------------

    def _clear_selection(self):
        self.path = None
        self.info = None
        self.path_label.setText('select a parameter')
        self.hint_label.setText('')
        self.apply_button.setEnabled(False)

    def _param_selected(self):
        items = self.param_tree.selectedItems()
        if not items:
            self._clear_selection()
            return
        path = items[0].data(0, _qt_enum(Qt, 'ItemDataRole', 'UserRole'))
        if not path:
            self._clear_selection()
            return

        self.path = path
        self.info = type_of(self.config, path)
        self.path_label.setText(path)
        self.hint_label.setText(self.info.hint() if self.info else '')

        current = getValue(self.config, path)

        # An enum has a known, short set of legal values; offer those instead
        # of free text.  Everything else is typed in, and validated live.
        if self.info is not None and self.info.kind == 'enum':
            self.value_combo.blockSignals(True)
            self.value_combo.clear()
            self.value_combo.addItems(list(self.info.labels.keys()))
            for label, number in self.info.labels.items():
                if number == current:
                    self.value_combo.setCurrentText(label)
                    break
            self.value_combo.blockSignals(False)
            self.value_stack.setCurrentWidget(self.value_combo)
        else:
            # Prefill, so that editing one element of an array does not mean
            # retyping the whole vector.
            self.value_edit.blockSignals(True)
            self.value_edit.setText(format_value(current))
            self.value_edit.blockSignals(False)
            self.value_stack.setCurrentWidget(self.value_edit)

        self._validate()
        self._scan_current()

    def _scan_current(self):
        """Show the current value of this parameter on every selected detector."""
        devices = self._selected_devices()
        if not devices or not self.path:
            return
        generation = self._fresh()
        self.log.append('--- %s' % self.path)
        self._start('scan',
                    ScanWorker(generation, hutch=self.hutch, alias=self.alias,
                               devices=devices, path=self.path,
                               **self._worker_args()),
                    row=self._scan_row)

    def _scan_row(self, generation, change):
        if self._current(generation):
            self._append_log('INFO', change.describe())

    def _value_text(self):
        if self.value_stack.currentWidget() is self.value_combo:
            return self.value_combo.currentText()
        return self.value_edit.text()

    def _validate(self):
        """Enable Apply only for an edit that would actually succeed."""
        text = self._value_text()
        if not self.path or not text:
            self.apply_button.setEnabled(False)
            return
        if self.config is None:
            return
        change = plan_change('preview', self.config, self.path, ValueSpec(text))
        writable = not self.write_box.isChecked()
        self.apply_button.setEnabled(change.ok and writable)
        if not change.ok:
            self.hint_label.setText(change.error)
        elif self.info is not None:
            self.hint_label.setText(self.info.hint())
        self.apply_button.setToolTip(
            'enable writing first' if change.ok and not writable else '')

    def _read_only_toggled(self, read_only):
        if not read_only and not self._writing_possible():
            self.write_box.setChecked(True)
            return
        self._refresh_target()
        self._validate()

    def _writing_possible(self):
        """Check we could actually authenticate before letting Apply light up."""
        if 'ws-kerb' in self.auth_url:
            if subprocess.call(['klist', '-s']) != 0:
                self._warn('No Kerberos ticket. Run kinit, then try again.')
                return False
        elif 'ws-auth' in self.auth_url:
            if not (self.password or os.getenv('CONFIGDB_AUTH')):
                self._warn('No password. Set CONFIGDB_AUTH or pass --password.')
                return False
        else:
            self._warn('This URL is unauthenticated and cannot be written to.\n'
                       'Restart with --prod or --url pointing at ws-auth.')
            return False
        return True

    def _refresh_target(self):
        writable = not self.write_box.isChecked()
        url = self.auth_url if writable else self.read_url
        prod = is_prod(url)
        self.target_label.setText('%s  %s' % ('PROD' if prod else 'dev', url))
        self.setWindowTitle('configdb - %s%s' % ('PROD' if prod else 'dev',
                                                 '' if writable else ' (read-only)'))
        # Writing to production is the one combination worth making loud.
        self.target_label.setStyleSheet(
            'background: #b00020; color: white; padding: 2px;'
            if writable and prod else '')

    # -- applying ----------------------------------------------------------

    def _apply(self):
        if self.write_box.isChecked() or not self.path:
            return
        devices = self._selected_devices()
        text = self._value_text()
        if not devices or not text:
            return
        self.spec = ValueSpec(text)
        self.apply_button.setEnabled(False)
        self.statusBar().showMessage('checking %d detectors...' % len(devices))
        generation = self._fresh()
        self._start('plan',
                    ScanWorker(generation, hutch=self.hutch, alias=self.alias,
                               devices=devices, path=self.path, spec=self.spec,
                               **self._worker_args()),
                    scanned=self._plan_ready)

    def _plan_ready(self, generation, changes):
        if not self._current(generation):
            return
        self._validate()
        writable = [c for c in changes if c.ok and not c.unchanged]
        skipped = [c for c in changes if not c.ok or c.unchanged]
        if not writable:
            self._warn('Nothing to do.\n\n%s'
                       % '\n'.join(c.describe() for c in skipped[:20]))
            return

        target = '%s  %s/%s' % (self.auth_url, self.hutch, self.alias)
        dialog = ConfirmDialog(self, target, self.path, self.spec, writable, skipped)
        if _exec(dialog) != _qt_enum(QtWidgets.QDialog, 'DialogCode', 'Accepted'):
            self.statusBar().showMessage('cancelled', 5000)
            return

        self.results = {'ok': 0, 'failed': 0, 'skipped': 0}
        generation = self._fresh()
        self.log.append('--- writing %s = %s' % (self.path, self.spec))
        self._start('write',
                    WriteWorker(generation, hutch=self.hutch, alias=self.alias,
                                changes=writable, path=self.path, spec=self.spec,
                                **self._worker_args(write=True)),
                    result=self._write_result,
                    finished=self._write_finished)

    def _write_result(self, generation, change):
        # Deliberately not filtered by generation: this change has already been
        # written to the database, so it gets reported whatever the user has
        # clicked on since.  Losing the record of a write would be worse than
        # showing it late.
        if not change.ok:
            self.results['failed'] += 1
            logger.error('%s', change.describe())
        elif change.unchanged:
            self.results['skipped'] += 1
            self._append_log('INFO', change.describe())
        else:
            self.results['ok'] += 1
            self._append_log('INFO', '%s  (key %s)' % (change.describe(), change.key))

    def _write_finished(self):
        summary = '%d written, %d failed, %d skipped' % (
            self.results['ok'], self.results['failed'], self.results['skipped'])
        self.statusBar().showMessage(summary, 15000)
        self._append_log('INFO', summary)
        # Re-read so the tree shows what is actually in the database now, while
        # keeping the user on the parameter they were working on.
        self._devices_selected(keep_path=True)

    # -- odds and ends -----------------------------------------------------

    def _append_log(self, level, text):
        if level in ('ERROR', 'CRITICAL'):
            self.log.append('<span style="color:#b00020">%s</span>' % text)
        elif level == 'WARNING':
            self.log.append('<span style="color:#9a6700">%s</span>' % text)
        else:
            self.log.append(text)

    def _warn(self, text):
        box = QtWidgets.QMessageBox(self)
        box.setIcon(_qt_enum(QtWidgets.QMessageBox, 'Icon', 'Warning'))
        box.setWindowTitle('configdb')
        box.setText(text)
        _exec(box)

    def closeEvent(self, event):
        if self._busy('write'):
            box = QtWidgets.QMessageBox(self)
            box.setIcon(_qt_enum(QtWidgets.QMessageBox, 'Icon', 'Warning'))
            box.setWindowTitle('configdb')
            box.setText('A write is still in progress. Quit anyway?')
            box.setStandardButtons(
                _qt_enum(QtWidgets.QMessageBox, 'StandardButton', 'Ok')
                | _qt_enum(QtWidgets.QMessageBox, 'StandardButton', 'Cancel'))
            if _exec(box) != _qt_enum(QtWidgets.QMessageBox,
                                      'StandardButton', 'Ok'):
                event.ignore()
                return
        # Detach the log handler before stopping the threads: a parting log
        # record must not emit into a widget that is being torn down.
        logging.getLogger().removeHandler(self._log_handler)
        for name in list(self._jobs):
            self._stop(name)
        self._save_settings()
        super(ConfigdbGUI, self).closeEvent(event)


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='browse the configuration database and change a parameter '
                    'across many detectors')
    parser.add_argument('--url', help='database URL (overrides --dev)')
    parser.add_argument('--dev', action='store_true',
                        help='use the development database (default: production)')
    parser.add_argument('--prod', action='store_true',
                        help=argparse.SUPPRESS)   # accepted, but now the default
    parser.add_argument('--root', default=DEFAULT_ROOT,
                        help='database root (default: %s)' % DEFAULT_ROOT)
    parser.add_argument('--user', help='user for ws-auth URLs')
    parser.add_argument('--password', default=os.getenv('CONFIGDB_AUTH'),
                        help='password for ws-auth URLs (default: $CONFIGDB_AUTH)')
    parser.add_argument('--hutch', help='hutch to expand at start-up')
    parser.add_argument('--allow-write', action='store_true',
                        help='start with writing enabled (default: read-only)')
    parser.add_argument('-v', '--verbose', action='store_true', help='be verbose')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        datefmt='%H:%M:%S',
        level=logging.DEBUG if args.verbose else logging.INFO)

    if not HAVE_QT:
        sys.exit('configdb_GUI needs PyQt5 (or PyQt6); neither could be imported.')

    url = resolve_url(args.url, args.dev)
    if 'ws-kerb' in url and 'devconfigdb' in url:
        logger.warning('there is no ws-kerb development endpoint; '
                       'writes to %s are likely to fail', url)
    # Left as None when not given: the window derives it from the hutch that is
    # actually selected, at the moment it needs to write.
    user = args.user

    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName('SLAC')
    app.setOrganizationDomain('slac.stanford.edu')
    app.setApplicationName('configdb_gui')

    window = ConfigdbGUI(url, root=args.root, user=user, password=args.password,
                         hutch=args.hutch, allow_write=args.allow_write)
    window.show()
    sys.exit(_exec(app))


if __name__ == '__main__':
    main()
