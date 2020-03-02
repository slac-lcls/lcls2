#!/usr/bin/env python
'''
Determine the active experiment for the specified instrument/station.
'''
import os
import sys
import json
import requests
import argparse

def main():

    # process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("instrument", help="instrument_name[:station_number]")
    # unauthenticated URL is OK for read-only access
    defaultURL = "https://pswww.slac.stanford.edu/ws/devlgbk/"
    parser.add_argument("--url", help="URL to post to; this is only the prefix. Defaults to " + defaultURL, default=defaultURL)
    parser.add_argument("--verbose", "-v", action="count", default=0, help="verbose (may be repeated)")
    args = parser.parse_args()

    # parse instrument_name[:station_number]
    if ':' in args.instrument:
        instrument_name, station_number = args.instrument.split(':', maxsplit=1)
        try:
            station = int(station_number)
        except ValueError:
            exit("Error: invalid station number '%s'" % station_number)
    else:
        instrument_name = args.instrument
        station = 0

    try:
        resp = requests.get((args.url + "/" if not args.url.endswith("/") else args.url) + "/lgbk/ws/activeexperiment_for_instrument_station",
                            params={"instrument_name": instrument_name, "station": station}, timeout=10)
    except requests.exceptions.RequestException as ex:
        exit("Error: request exception: %s" % ex)

    if args.verbose >= 2:
        print("response: %s" % resp.text)
    experiment_name = None
    if resp.status_code == requests.codes.ok:
        if args.verbose >= 2:
            print("headers: %s" % resp.headers)
        if 'application/json' in resp.headers['Content-Type']:
            try:
                experiment_name = resp.json().get("value", {}).get("name", None)
            except json.decoder.JSONDecodeError:
                exit("Error: failed to decode JSON")
        else:
            exit("Error: failed to receive JSON")
    else:
        exit("Error: status code %d" % resp.status_code)

    if not experiment_name:
        exit("No experiment found for instrument %s:%d" % (instrument_name, station))

    if args.verbose > 0:
        print("Instrument name: %s" % instrument_name)
        print("Station number: %d" % station)
        print("Experiment name: %s" % experiment_name)
    else:
        print("%s:%d %s" % (instrument_name, station, experiment_name))

if __name__ == '__main__':
    main()
