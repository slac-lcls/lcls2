# From Calient "How to collect Port power levels over the REST API"

import time
import platform
import sys
import os
import json
import urllib.parse

from datetime import datetime
from http.client import HTTPConnection
from base64 import b64encode



if len(sys.argv) < 4 :
  print('must have 3 arguments, the switch ip, the connection id, the interval in minutes.\r\n   e.g. python collect_power.py 192.168.1.1  1.1.1>1.2.1  10')
  exit()

ip = sys.argv[1]
conn = sys.argv[2]
interval_minutes = int(sys.argv[3])

if interval_minutes < 1 :
  print('interval must be 1 or more minutes')
  exit()

f = open('calient_power_data.csv', 'a+')
try:
  print('getting connection data for %s every %s minutes from %s to %s  (use ctrl-c to quit)'%(conn, interval_minutes, ip, f.name), flush=True)

  userAndPass = "YWRtaW46cHhjKioq" # b64encode(b'admin:******').decode('ascii')
  headers = { 'Authorization' : 'Basic %s'%(userAndPass), 'Accept' : 'application/json' }

  f.write('Time,ConnectionID,InputPower,OutputPower,Loss\r\n')
  while True:
    c1 = HTTPConnection(host=ip, port=80, timeout=60)
    try :
      c1.request('GET', 'http://%s/rest/crossconnects/?id=detail&conn=%s'%(ip,urllib.parse.quote(conn)), headers=headers)
      response = c1.getresponse()
      now = datetime.now()
      cookie = response.getheader('Set-Cookie')
      if cookie :
        cookieval = cookie[0 : cookie.index(';')]
        headers['Cookie'] = cookieval
        headers.pop('Authorization')

      data  = response.read()
      jsondata = json.loads(data)
    finally :
      c1.close()

    half1 = jsondata[0].get('half1')
    f.write('%s,%s,%s,%s,%s\r\n'%(now.strftime('%Y-%m-%d %H:%M:%S'),half1.get('conn'),half1.get('inp'),half1.get('outp'),half1.get('loss')))

    if 'half2' in jsondata[0]:
      half2 = jsondata[0].get('half2')
      f.write('%s,%s,%s,%s,%s\r\n'%(now.strftime('%Y-%m-%d %H:%M:%S'),half2.get('conn'),half2.get('inp'),half2.get('outp'),half2.get('loss')))

    print('%s\t%s\tin %s\tout %s\tloss %s'%(now.strftime('%Y-%m-%d %H:%M:%S'),half1.get('conn'),half1.get('inp'),half1.get('outp'),half1.get('loss')), flush=True)
    time.sleep(interval_minutes * 60)

finally:
  if f:
    f.close()

