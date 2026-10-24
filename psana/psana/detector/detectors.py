# a detector interface class name must follow the
# naming convention: dettype_drpalg_major_minor_micro

"""Import all detector interface modules into one namespace.

`DgramManager` looks up interface classes here by name; the code comment says names follow
`dettype_drpalg_major_minor_micro`. The epixhremu import is commented out.
"""
from .test_detectors        import *
from .misc_detectors        import *
from .bld                   import *
from .envstore              import *
from psana.hsd              import *
from .opal                  import *
from .piranha4              import *
from .timetool              import *
from .ts                    import *
from .wave8                 import *
from .wave8he               import *
from .epix10ka              import *
from .epixhr2x2             import *
from .epix100               import *
from .epixm320              import *
from .epixuhr               import *
from .archon                import *
from .axis                  import *
from .jungfrau              import *
from .jungfrauemu           import *
#from .epixhremu             import *
from .generic               import *
from .epixuhr3x2            import *
