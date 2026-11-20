"""Report cluster routing; never submit, connect remotely, or change profiles."""
import argparse
import json
import os
from native_contract import detect_site, require, write


def select(target='auto', environment=None, hostname=None):
    detected = detect_site(environment, hostname)
    require(target in ('auto', 'sdf', 'perlmutter'), 'unsupported target')
    require(detected is not None or target != 'auto', 'unknown cluster; specify execution cluster explicitly')
    chosen = detected if target == 'auto' else target
    return dict(detected_site=detected, execution_site=chosen, explicit_override=target != 'auto',
                remote_required=detected != chosen,
                adapter='perlmutter_adapter.py',
                next_action='prepare on this cluster' if detected == chosen else
                'connect to the requested cluster and recheck identity before preparation; do not run locally')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', choices=('auto', 'sdf', 'perlmutter'), default='auto')
    parser.add_argument('--output')
    args = parser.parse_args()
    result = select(args.site)
    result['host'] = os.uname().nodename
    if args.output:
        write(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
