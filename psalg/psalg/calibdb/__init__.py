"""Read-only access to the calibration constants database (calib_ws).

The selection follows psana's psana.pscalib.calib.MDBWebUtils, which psdaq may not
import.  Reads need no Kerberos or JWT ticket.
"""

from .reader import calib_constants, short_name
