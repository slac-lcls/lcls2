#!/usr/bin/env python
 
"""
Sample for posting to the calibration service using a web service and kerberos authentication.
Make sure we have a kerberos ticket.

2019-07-27 Murali Shankar - version 1 
"""
 
import requests
import json
from krtc import KerberosTicket
from urllib.parse import urlparse
 
ws_url = "https://pswww.slac.stanford.edu/ws-kerb/calib_ws/"
krbheaders = KerberosTicket("HTTP@" + urlparse(ws_url).hostname).getAuthHeaders()

# Create a new document in the collection test_coll in the database test_db.
resp = requests.post(ws_url+"test_db/test_coll/", headers=krbheaders, json={"calib_count": 1})
print(resp.text)
new_id = resp.json()["_id"]
 
# Update an existing document
resp = requests.put(ws_url+"test_db/test_coll/"+new_id, headers=krbheaders, json={"calib_count": 2})
print(resp.text)
 
# Delete database and collection
#resp = requests.delete(ws_url+"test_db", headers=krbheaders)
#resp = requests.delete(ws_url+"test_db/test_coll", headers=krbheaders)

# Delete an existing document
resp = requests.delete(ws_url+"test_db/test_coll/"+new_id, headers=krbheaders)
print(resp.text)
 
# Create a new GridFS document, we upload an image called small_img.png
files = [("files",  ('small_img.png', open('small_img.png', 'rb'), 'image/png'))]
resp = requests.post(ws_url+"test_db/gridfs/", headers=krbheaders, files=files)
print(resp.text)
new_id = resp.json()["_id"]

#Uploading file as binarystring (2020-04-30)
with open("small_img.png", "rb") as f:
  data = f.read()
  resp = requests.post(ws_url+"test_db/gridfs/", data=data, headers={'Content-Type': 'application/octet-stream'})
  print(resp.json())

# Delete the GridFS document
resp = requests.delete(ws_url+"test_db/gridfs/"+new_id, headers=krbheaders)
print(resp.text)
