#!/usr/bin/env python
'''
Get/start/stop the current run for the specified experiment.
'''
import os
import sys
import json
import requests
from requests.auth import HTTPBasicAuth
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", help="experiment name")
    parser.add_argument("--user", default="xppopr")
    parser.add_argument("--password", default=os.getenv("CONFIGDB_AUTH"))
    parser.add_argument("--start", action="store_true", help="start run")
    parser.add_argument("--end", action="store_true", help="end run")
    defaultURL = "https://pswww.slac.stanford.edu/ws-auth/devlgbk/"
    parser.add_argument("--rundb", metavar='URL', help="URL to post to; this is only the prefix. Defaults to " + defaultURL, default=defaultURL)
    parser.add_argument("--verbose", "-v", action="count", default=0, help="verbose (may be repeated)")
    args = parser.parse_args()

    experiment_name = args.experiment

    serverURLPrefix = "{0}run_control/{1}/ws/".format(args.rundb + "/" if not args.rundb.endswith("/") else args.rundb, experiment_name)

    if args.verbose:
        print('serverURLPrefix = %s' % serverURLPrefix)

    if args.start:
        # start run
        if args.verbose:
            print("Start a run (start_run)")
        try:
            resp = requests.post(serverURLPrefix + "start_run", auth=HTTPBasicAuth(args.user, args.password)  )
        except Exception as ex:
            print('start_run error')
            if args.verbose >= 2:
                print('HTTP request: %s' % ex)
        else:
            if args.verbose >= 2:
                print("Response: %s" % resp.text)
            if resp.json().get("success", None):
                print("start_run success")
            else:
                print("start_run failure")

    # get current run
    if args.verbose:
        print("Get the current run (current_run)")
    try:
        resp = requests.get(serverURLPrefix + "current_run?skipClosedRuns=true", auth=HTTPBasicAuth(args.user, args.password)  )
    except Exception as ex:
        print('current_run error')
        if args.verbose >= 2:
            print('HTTP request: %s' % ex)
    else:
        if args.verbose >= 2:
            print("Response: %s" % resp.text)
        if resp.status_code == requests.codes.ok:
            if resp.json().get("success", None):
                num = resp.json().get("value", {}).get("num", None)
                print("run number: %s" % num)
            else:
                print("current_run failure")
        else:
            exit("Error: status code %d" % resp.status_code)

    if args.end:
        # end run
        if args.verbose:
            print("End a run (end_run)")
        try:
            resp = requests.post(serverURLPrefix + "end_run", auth=HTTPBasicAuth(args.user, args.password)  )
        except Exception as ex:
            print('end_run error')
            if args.verbose >= 2:
                print('HTTP request: %s' % ex)
        else:
            if args.verbose >= 2:
                print("Response: %s" % resp.text)
            if resp.json().get("success", None):
                print("end_run success")
            else:
                print("end_run failure")

if __name__ == '__main__':
    main()
