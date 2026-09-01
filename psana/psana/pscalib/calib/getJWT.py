#!/usr/bin/env python3
""" code from
    https://confluence.slac.stanford.edu/spaces/~mshankar/pages/695783226/JWT+s+for+the+LCLS2+calib+service

    WORKS ON INTERACTIVE NODES ONLY!!!!
    Get a JWT from KerberosTicket and print just the JWT out.
    Anything else; we exit with a non-zero exit status
"""
import sys
import requests
import json
import logging
from krtc import KerberosTicket

logger = logging.getLogger(__name__)

if len(sys.argv) != 2:
    logger.error("Please specify one and only one scope")
    sys.exit(1)

scope = sys.argv[1]

try:
    krbheaders = KerberosTicket("HTTP@pswww.slac.stanford.edu").getAuthHeaders()
    ws_url = f"https://psdm.slac.stanford.edu/pcdstkn/fromkerb"
    r = requests.get(ws_url, headers=krbheaders)
    r.raise_for_status()
    jwt = r.json()["value"]
    print(jwt)
    sys.exit(0)
except Exception as e:
    logger.exception("Exception getting a JWT from the server")
    sys.exit(1)

# EOF
