import argparse
import datetime
import glob
import sys
import subprocess
import logging

def dump_errors(a, limit=5):
    cmd = f"grep '<E>' {a}"
    result = subprocess.run(cmd, capture_output=True, shell=True, encoding='utf-8')
#            logging.info(f'return code {result.returncode}')
#            logging.info(f'stdout: {result.stdout}')
#            logging.info(f'stderr: {result.stderr}')
    if result.returncode == 0:
        logging.info(f'\t{a}')
        for l in result.stdout.split('\n')[:limit]:
            logging.info(f'\t\t{l}')

    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', metavar='HUTCH', default='tmo',
                        help='hutch logfiles to search')
    parser.add_argument('--path', metavar='PATH', default=None,
                        help='path to search for logfiles (overrides -H)')
    parser.add_argument('-D', metavar='DATE', default=None,
                        help='Can also specify YYYY/MM/DD_HH:MM:SS or some leading part')
    parser.add_argument('-v', action='store_true',
                        help='be verbose')

    args = parser.parse_args()

    level = logging.DEBUG if args.v else logging.INFO
    logging.basicConfig(level=level)

    if args.path is None:
        args.path = f'/cds/home/opr/{args.H}opr/'

    if args.D is None:
        today = datetime.date.today()
        args.path += f'/{today.year}/{today.month}/{today.day}_' 
    else:
        args.path += args.D

    logging.debug(f'Path is {args.path}')

    #  assume all logfiles share the same time in the filename

    #  collect the first few errors from each file

    logging.debug(f'Searching {args.path}*control.log')

    files = glob.glob(f'{args.path}*control.log')
    files.sort()
#    logging.debug(files)

    ncontrol_logs = 0
    ncontrol_errors = 0
    last_day = ''
    for f in files:
        day = '/'.join(f.split('_')[0].rsplit('/',3)[1:])
        if day != last_day:
            if ncontrol_logs:
                logging.info(f'{last_day} control logfiles with errors: {ncontrol_errors}/{ncontrol_logs}')
                ncontrol_logs = 0
                ncontrol_errors = 0
            last_day = day

        hdr = '_'.join(f.split('_')[:2])
        logging.info(hdr)
        
        all_files = glob.glob(f'{hdr}*.log')

        # Ignore control_gui.log
        for a in all_files:
            if 'control_gui.log' in a:
                all_files.remove(a)
                break
        # control.log
        for a in all_files:
            if 'control.log' in a:
                ncontrol_logs += 1
                if dump_errors(a):
                    ncontrol_errors += 1
                all_files.remove(a)
                break
        # teb0.log
        for a in all_files:
            if 'teb0.log' in a:
                dump_errors(a)
                all_files.remove(a)
                break

        other_files = []
        # meb
        for a in all_files:
            if 'meb' in a:
                dump_errors(a)
            else:
                other_files.append(a)

        # all others
        for a in other_files:
            dump_errors(a)

    if ncontrol_logs:
        logging.info(f'{last_day} control logfiles with errors: {ncontrol_errors}/{ncontrol_logs}')

if __name__ == '__main__':
    main()

