import os

import requests


class Bos(object):

    def __init__(self, urlBase, user="admin", password=os.getenv("BOS_AUTH"), verbose=False):
        self._session = requests.Session()
        self._urlBase = urlBase
        self._auth    = (user, password)
        self._verbose = verbose

    def _handle_response(self, response, result=False):
        if not result:
            result = self._verbose
        if response.status_code == 200:
            return response.json() if result else None
        print(f"BOS returned error {response.status_code}")
        if response.status_code == 401:
            print("Unauthorized")
            return
        elif response.status_code == 404:
            print("Not found")
            return
        return response.json()

    def add(self, inPort, outPort):
        url = self._urlBase + 'crossconnects/'
        params = [('id',   'add')]
        data = {'in':   inPort,
                'out':  outPort,
                'conn': inPort+'-'+outPort,
                'dir':  'bi',
                'band': 'O'}
        response = self._session.post(url, auth=self._auth, params=params, json=data)
        return self._handle_response(response)

    def delete(self, conn):
        url = self._urlBase + 'crossconnects/'
        params = [('conn', conn)]
        response = self._session.delete(url, auth=self._auth, params=params)
        return self._handle_response(response)

    def list(self):
        url = self._urlBase + 'crossconnects/'
        params = [('id', 'list')]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response, True)

    def detail(self, conn):
        url = self._urlBase + 'crossconnects/'
        params = [('id',   'detail'),
                  ('conn', conn)]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response, True)

    def activate(self, conn):
        url = self._urlBase + 'crossconnects/'
        params = [('id',   'activate'),
                  ('conn', conn)]
        response = self._session.post(url, auth=self._auth, params=params)
        return self._handle_response(response)

    def deactivate(self, conn):
        url = self._urlBase + 'crossconnects/'
        params = [('id',   'deactivate'),
                  ('conn', conn)]
        response = self._session.post(url, auth=self._auth, params=params)
        return self._handle_response(response)

    def portSummary(self):
        url = self._urlBase + 'ports/'
        params = [('id', 'summary')]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response, True)

    def portDetail(self, port):
        url = self._urlBase + 'ports/'
        params = [('id',   'detail'),
                  ('port', port)]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response, True)

    def alarms(self, type_, class_):
        url = self._urlBase + 'alarms/'
        params = [('id',    'all'),]
        if type_ is not None:
            params += [('type',  type_),]
        if class_ is not None:
            params += [('class', class_),]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response, True)

    def backup(self):
        url = self._urlBase + 'node/'
        params = [('id',           'backup'),
                  ('type',         'Remote'),
                  ('host',         'my.backup.host'),
                  ('backupfolder', '/tmp'),
                  ('username',     'backup'),
                  ('password',     'mypassword')]
        response = self._session.get(url, auth=self._auth, params=params)
        return self._handle_response(response)


import argparse
import pprint

def main():

    def _add(bos, args):
        response = bos.add(args.inPort, args.outPort)
        if response is not None:
            return response
        if args.activate:
            response = bos.activate(args.inPort+'-'+args.outPort)
        return response

    def _delete(bos, args):
        if args.deactivate:
            response = bos.deactivate(args.conn)
            if response is not None:
                return response
        return bos.delete(args.conn)

    def _list(bos, args):
        return bos.list()

    def _detail(bos, args):
        return bos.detail(args.conn)

    def _activate(bos, args):
        return bos.activate(args.conn)

    def _deactivate(bos, args):
        return bos.deactivate(args.conn)

    def _portSummary(bos, args):
        return bos.portSummary()

    def _portDetail(bos, args):
        return bos.portDetail(args.port)

    def _alarms(bos, args):
        return bos.alarms(args.type, args.klass)

    # create the top-level parser
    parser = argparse.ArgumentParser(description='Big Optical Switch CLI')
    parser.add_argument('--url', default='http://osw-daq-calients320.pcdsn/rest/',
                        help='Big Optical Switch connection')
    parser.add_argument('--user', help='user for BOS authentication', type=str, default='admin')
    parser.add_argument('--password', help='password for BOS authentication', type=str, default=os.getenv('BOS_AUTH'))
    parser.add_argument('--verbose', help='Be verbose', action='store_true')
    subparsers = parser.add_subparsers()

    # create the parser for the "add" command
    parser_add = subparsers.add_parser('add', help='Add Cross Connection')
    parser_add.add_argument('inPort',  help='in port')
    parser_add.add_argument('outPort', help='out port')
    parser_add.add_argument('--activate', help='Activate after add', action='store_true')
    parser_add.set_defaults(func=_add)

    # create the parser for the "delete" command
    parser_delete = subparsers.add_parser('delete', help='Delete Cross Connection')
    parser_delete.add_argument('conn', help='connection ID')
    parser_delete.add_argument('--deactivate', help='Deactivate before delete', action='store_true')
    parser_delete.set_defaults(func=_delete)

    # create the parser for the "list" command
    parser_list = subparsers.add_parser('list', help='Cross Connection List')
    parser_list.set_defaults(func=_list)

    # create the parser for the "detail" command
    parser_detail = subparsers.add_parser('detail', help='Cross Connection Detail')
    parser_detail.add_argument('conn', help='connection ID')
    parser_detail.set_defaults(func=_detail)

    # create the parser for the "activate" command
    parser_activate = subparsers.add_parser('activate', help='Cross Connection Activate')
    parser_activate.add_argument('conn', help='connection ID')
    parser_activate.set_defaults(func=_activate)

    # create the parser for the "deactivate" command
    parser_deactivate = subparsers.add_parser('deactivate', help='Cross Connection Deactivate')
    parser_deactivate.add_argument('conn', help='connection ID')
    parser_deactivate.set_defaults(func=_deactivate)

    # create the parser for the "portSummary" command
    parser_portSummary = subparsers.add_parser('ports', help='Port Summary')
    parser_portSummary.set_defaults(func=_portSummary)

    # create the parser for the "portDetail" command
    parser_portDetail = subparsers.add_parser('port', help='Port Detail')
    parser_portDetail.add_argument('port', help='port number')
    parser_portDetail.set_defaults(func=_portDetail)

    # create the parser for the "alarms" command
    parser_alarms = subparsers.add_parser('alarms', help='Alarms/Events')
    parser_alarms.add_argument('--type',  default=None, help='CRITICAL or MAJOR or MINOR')
    parser_alarms.add_argument('--klass', default=None, help='COM,CRS,ENV,EQPT, SECU')
    parser_alarms.set_defaults(func=_alarms)

    # parse the args and call whatever function was selected
    args = parser.parse_args()
    try:
        subcommand = args.func
    except Exception:
        parser.print_help(sys.stderr)
        sys.exit(1)

    bos = Bos(args.url, user=args.user, password=args.password, verbose=args.verbose)
    response = subcommand(bos, args)
    if response is not None:
        pprint.pprint(response)


if __name__ == '__main__':
    main()
