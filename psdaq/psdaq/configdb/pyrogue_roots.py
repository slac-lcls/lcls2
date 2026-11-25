import gc
import logging
import sys

# Py_Finalize joins non-daemon threads, so a pyrogue Root still running then hangs the exit
_roots = []

def register(root):
    _roots.append(root)

def _stop(root):
    try:
        root.stop()
    except Exception as e:
        logging.warning(f'Stopping pyrogue Root {type(root).__name__} failed: {e!r}')

def stop_all():
    for root in reversed(_roots):
        if root._running:
            _stop(root)

    pr = sys.modules.get('pyrogue')
    if pr is None:
        return
    for obj in gc.get_objects():
        if isinstance(obj, pr.Root) and obj._running:
            logging.warning(f'Stopping unregistered pyrogue Root {type(obj).__name__} {obj.name}: '
                            f'its config should stop it')
            _stop(obj)
