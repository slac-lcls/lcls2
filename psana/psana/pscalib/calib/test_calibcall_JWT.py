#!/usr/bin/env python3
""" example from
    https://confluence.slac.stanford.edu/spaces/~mshankar/pages/695783226/JWT+s+for+the+LCLS2+calib+service

    USAGE
    =====
    make env CALIB_JWT using:
      source get_JWT_from_s3df.sh
    or
      source get_JWT_from_kerberos.sh
    then run this test:
      ./test_calibcall_JWT.py
"""
import os
import json

from requests import Session

# Get the JWT from the environment
jwt = os.environ['CALIB_JWT']
if not jwt:
    raise Exception('Cannot determine the JWT')

session = Session()
session.headers.update({'Authorization': 'Bearer ' + jwt })

print('Getting some calib data using the JWT:')
ws_url = 'https://psdm.slac.stanford.edu/ws-jwt/calib_ws/cdb_xpptut15/cspad_detnum1234/'
r = session.get(ws_url)
r.raise_for_status()
print(json.dumps(r.json()))

print('\nMake sure we can edit calib information using the JWT:')
ws_url = 'https://psdm.slac.stanford.edu/ws-jwt/calib_ws/cdb_xpptut15/test_edit_privilege'
r = session.get(ws_url)
r.raise_for_status()
print(json.dumps(r.json()))

# EOF
