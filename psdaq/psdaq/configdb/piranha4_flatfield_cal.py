#!/usr/bin/env python3
"""

example command from cpo: (sep 15, 2026)
python piranha4_flatfield_cal.py --dark-only --lane 0 --user-set 1 --exptime 8000  --pixel-format 12 --linerate 1000

Standalone dark (FPN) + flat field (PRNU) calibration for a Teledyne DALSA
Piranha4 line camera attached to an SLAC cameralink-gateway FEB.

This is piranha4_dark_cal.py plus the bright half of the calibration, and it
imports that script's serial plumbing (Piranha4Rx / Piranha4 / gcp parsing), so
keep the two files together in this directory.

Both calibrations are done by the camera itself (see the "Commands" appendix of
the Piranha4 2K/4K user manual, 03-032-20176):

  ccf <2048|4096>                   Calibrate User FPN - average N *dark* lines
                                    and compute the per-pixel offset
  cpa <alg> <2048|4096> <target>    Calibrate Flatfield - average N *bright*
                                    lines and compute the per-pixel gain such
                                    that the output reaches <target> DN
  uss <1-8>                         save both coefficient sets, plus everything
                                    gcp reports, to non-volatile memory

The two phases need opposite illumination, so the script stops between them and
waits for you:

    phase 1  DARK   - lens capped / shutter closed / beam off
    phase 2  BRIGHT - uniform white reference at the operating light level

Per the manual's calibration section: set the system gain so the peak is at the
level you want *before* calibrating, use a clean white plastic or ceramic
reference rather than paper (any dust or scratch on it ends up in the
coefficients), keep the reference moving if you can, and pick a target DN
*above* the peak you saw while setting up. The camera calibrates at the
conditions you give it, so --linerate/--exptime/--gain should match operating
conditions as closely as possible - both halves depend on them.

Order of operations, and why:

    ffm 0 ; ccf N        FPN is measured with correction off, so it sees the
                         raw dark image
    ffm 1 ; cpa a N T    PRNU is measured with the FPN correction *on*, since
                         the pipeline is (raw - FPN) * PRNU and the gain has to
                         be computed from offset-corrected data
    restore timing       ssf/set before sem/stm - the camera only accepts a
                         line rate with stm 0 and an exposure with sem 0
    uss <n>              persist coefficients + settings

Usage (the DAQ must not be running on this PCIe device):

    # both phases, prompting for the illumination change in between
    python piranha4_flatfield_cal.py --lane 0 --user-set 1 --target 3000

    # two passes, if changing the light means walking into the hutch
    python piranha4_flatfield_cal.py --lane 0 --dark-only --no-save
    python piranha4_flatfield_cal.py --lane 0 --flat-only --user-set 1 --target 3000

    # see the plan without touching the camera
    python piranha4_flatfield_cal.py --lane 0 --no-save --dry-run

Same LCLS-II DAQ caveats as the dark-only script: piranha4_config_store.py sets
FFM=0 and USD=0, so a Configure turns the correction back off and makes the
factory set the power-up default. Set the configdb FFM to 1 (and USD to the
user set used here) if the DAQ should use these coefficients.
"""

import argparse
import os
import sys
import time

# Import the shared plumbing from the dark-cal script next to this one; that
# module also sets up the cameralink-gateway/surf python path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from piranha4_dark_cal import (Piranha4, Piranha4Error, Piranha4Rx,
                                   check_timing, gcp_int, parse_gcp, print_gcp)
except ImportError:
    from psdaq.configdb.piranha4_dark_cal import (
        Piranha4, Piranha4Error, Piranha4Rx,
        check_timing, gcp_int, parse_gcp, print_gcp)

import pyrogue as pr
import cameralink_gateway

# cpa algorithm argument
ALGORITHMS = {'basic': 0, 'filter': 1}


def build_parser():
    parser = argparse.ArgumentParser(
        description='Piranha4 dark (ccf) + flat field (cpa) calibration, '
                    'saved with uss',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--dev', type=str, default='/dev/datadev_0',
                        help='path to the PCIe device')
    parser.add_argument('--lane', type=int, default=0, choices=range(4),
                        help='PGP lane the FEB is on')
    parser.add_argument('--chan', type=int, default=0, choices=range(2),
                        help='camera link channel on the FEB')
    parser.add_argument('--pgp4', action='store_true',
                        help='FEB link is PGPv4 (default PGP2b)')

    phase = parser.add_mutually_exclusive_group()
    phase.add_argument('--dark-only', action='store_true',
                       help='only the dark FPN phase ("ccf")')
    phase.add_argument('--flat-only', action='store_true',
                       help='only the bright PRNU phase ("cpa"), using the '
                            'FPN coefficients already in the camera')

    parser.add_argument('--dark-lines', type=int, default=4096,
                        choices=(2048, 4096),
                        help='dark lines averaged by ccf')
    parser.add_argument('--flat-lines', type=int, default=4096,
                        choices=(2048, 4096),
                        help='bright lines averaged by cpa')
    parser.add_argument('--algorithm', type=str, default='basic',
                        choices=sorted(ALGORITHMS),
                        help='cpa algorithm: "basic", or "filter" to smooth '
                             'the average line and interpolate outliers when '
                             'the white reference is not featureless')
    parser.add_argument('--target', type=int, default=None,
                        help='cpa target level in DN (default: --target-frac '
                             'of full scale for the pixel format in use)')
    parser.add_argument('--target-frac', type=float, default=0.75,
                        help='target as a fraction of full scale, when '
                             '--target is not given')

    parser.add_argument('--user-set', type=int, default=None,
                        choices=range(1, 9), metavar='{1..8}',
                        help='non-volatile user set to save into with "uss". '
                             'Required unless --no-save is given')
    parser.add_argument('--no-save', action='store_true',
                        help='calibrate but do not "uss": the coefficients are '
                             'used until the next power cycle or "rc"')
    parser.add_argument('--default-set', action='store_true',
                        help='also "usd <user-set>": load that set at power-up')
    parser.add_argument('--verify', action='store_true',
                        help='after saving, "lpc <user-set>" to read the '
                             'coefficients back out of non-volatile ram')

    parser.add_argument('--linerate', type=int, default=10000,
                        help='internal line rate [Hz] used for both phases')
    parser.add_argument('--exptime', type=int, default=None,
                        help='internal exposure time [ns] used for both phases '
                             '(default: the camera\'s current value, or 4000 '
                             'if that does not fit)')
    parser.add_argument('--pixel-format', type=int, default=None,
                        choices=(8, 10, 12),
                        help='calibrate in this pixel format ("spf"). '
                             'ClinkDevRoot.start() leaves the camera in 8 bit')
    parser.add_argument('--gain', type=float, default=None,
                        help='set the system gain ("ssg 0 f<gain>") first')
    parser.add_argument('--roi', type=int, nargs=2, default=None,
                        metavar=('FIRST', 'LAST'),
                        help='only calibrate pixels FIRST..LAST ("roi"); '
                             'applies to both ccf and cpa')
    parser.add_argument('--reset-coefficients', action='store_true',
                        help='issue "rpc" first, clearing both the FPN and the '
                             'PRNU coefficients. Reasonable here, since both '
                             'are about to be recalibrated')
    parser.add_argument('--load-coefficients', type=int, default=None,
                        choices=range(0, 9), metavar='{0..8}',
                        help='issue "lpc SET" first, loading the FPN/PRNU '
                             'coefficients from that user set (0 = factory). '
                             'Use with --flat-only so the new PRNU is paired '
                             'with a known FPN')
    parser.add_argument('--leave-correction-off', action='store_true',
                        help='do not "ffm 1" at the end')

    parser.add_argument('--settle', type=float, default=2.0,
                        help='seconds to let the camera run at the '
                             'calibration timing before each calibration')
    parser.add_argument('--baud', type=int, default=9600,
                        help='FEB uart baud rate (camera default is 9600)')
    parser.add_argument('--throttle', type=int, default=10000,
                        help='FEB uart tx byte throttle [us]')

    parser.add_argument('--dry-run', action='store_true',
                        help='read the camera configuration, print the '
                             'command sequence, change nothing')
    parser.add_argument('--no-pause', action='store_true',
                        help='do not wait for confirmation between the dark '
                             'and bright phases. Only sensible if the '
                             'illumination change is automated')
    parser.add_argument('-y', '--yes', action='store_true',
                        help='skip the initial confirmation prompt')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='echo every line the camera sends')

    return parser


def pixel_bits(args, cfg):
    """Bit depth the calibration will run at, and the resulting full scale"""
    if args.pixel_format is not None:
        bits = args.pixel_format
    else:
        bits = gcp_int(cfg, 'Pixel Fmt', 8)     # 'Pixel Fmt   8 bits'
    if bits not in (8, 10, 12):
        raise Piranha4Error('cannot make sense of pixel format %r' % bits)
    return bits, (1 << bits) - 1


def make_plan(args, cfg):
    """Build the (command, timeout) lists from the current camera state"""

    exptime = args.exptime
    if exptime is None:
        exptime = gcp_int(cfg, 'Exp. Time[0]', 4000)
        if exptime + 1500.0 >= 1.0e9 / args.linerate:
            exptime = 4000
    check_timing(args, exptime)

    maxRate = gcp_int(cfg, 'Max L.R.')
    if maxRate is not None and args.linerate > maxRate:
        raise Piranha4Error('--linerate %d Hz exceeds the camera\'s maximum '
                            'line rate of %d Hz in this configuration'
                            % (args.linerate, maxRate))

    bits, fullScale = pixel_bits(args, cfg)
    target = args.target
    if target is None:
        target = int(round(args.target_frac * fullScale))
    if not args.dark_only and not 0 <= target <= fullScale:
        raise Piranha4Error('--target %d is outside 0..%d for %d bit pixels'
                            % (target, fullScale, bits))

    # Original timing configuration, to be put back afterwards
    extTrig = 1 if cfg.get('Ext Trig', 'Off').lower().startswith('on') else 0
    expMode = 1 if 'pulse' in cfg.get('Exp. Mode', 'Timed').lower() else 0
    lineRate = gcp_int(cfg, 'Line Rate', args.linerate)
    expTime = gcp_int(cfg, 'Exp. Time[0]', exptime)
    testPat = cfg.get('Test Pat.', '0:Off')

    pre = []
    if args.load_coefficients is not None:
        pre.append(('lpc %d' % args.load_coefficients, 60.0))
    if not testPat.startswith('0'):
        print('  *** WARNING: test pattern %s is on, turning it off' % testPat)
        pre.append(('svm 0', 5.0))
    if args.pixel_format is not None:
        pre.append(('spf %d' % {8: 0, 10: 1, 12: 2}[args.pixel_format], 5.0))
    if args.gain is not None:
        pre.append(('ssg 0 f%.3f' % args.gain, 5.0))
    if args.roi is not None:
        pre.append(('roi %d %d' % tuple(args.roi), 5.0))
    if args.reset_coefficients:
        pre.append(('rpc', 30.0))
    # ccf and cpa both average lines the camera has to generate itself
    pre += [('stm 0', 5.0),
            ('sem 0', 5.0),
            ('ssf %d' % args.linerate, 5.0),
            ('set %d' % exptime, 5.0)]

    def tmo(lines, extra):
        return max(extra, 10.0 * lines / args.linerate + extra)

    # phase 1: dark.  Correction off so ccf measures the raw dark image.
    dark = []
    if not args.flat_only:
        dark += [('ffm 0', 5.0),
                 ('ccf %d' % args.dark_lines, tmo(args.dark_lines, 60.0))]

    # phase 2: bright.  Correction on so cpa works on FPN-corrected data.
    flat = []
    if not args.dark_only:
        flat += [('ffm 1', 10.0),
                 ('cpa %d %d %d' % (ALGORITHMS[args.algorithm],
                                    args.flat_lines, target),
                  tmo(args.flat_lines, 90.0))]

    # Restore in an order the camera accepts: ssf needs stm 0, set needs sem 0
    post = []
    if not args.leave_correction_off:
        post.append(('ffm 1', 10.0))
    post += [('ssf %d' % lineRate, 5.0),
             ('set %d' % expTime, 5.0),
             ('sem %d' % expMode, 15.0),
             ('stm %d' % extTrig, 15.0)]

    save = []
    if args.user_set is not None:
        save.append(('uss %d' % args.user_set, 60.0))
        if args.default_set:
            save.append(('usd %d' % args.user_set, 30.0))
        if args.verify:
            save.append(('lpc %d' % args.user_set, 60.0))

    return pre, dark, flat, post, save, dict(target=target, bits=bits,
                                             fullScale=fullScale,
                                             exptime=exptime)


def confirm(prompt, args):
    """Stop and wait for the illumination to be set up"""
    print()
    print('*' * 72)
    for line in prompt.split('\n'):
        print('* %s' % line)
    print('*' * 72)
    if args.dry_run or args.no_pause:
        print('(not waiting)')
        return
    try:
        answer = input('Ready? [yes/no] ').strip().lower()
    except EOFError:
        answer = ''
    if answer not in ('y', 'yes'):
        raise Piranha4Error('aborted at the illumination prompt')


DARK_PROMPT = ("Phase 1 of 2: DARK.  The sensor must see NO LIGHT.\n"
               "Cap the lens / close the shutter / turn the illumination off.")

FLAT_PROMPT = ("Phase 2 of 2: BRIGHT.  The sensor must now see a UNIFORM\n"
               "WHITE REFERENCE at the operating light level.\n"
               "Uncap the lens, turn the illumination on, and if you can,\n"
               "keep the reference moving.  Clean plastic or ceramic beats\n"
               "paper: dust and scratches end up in the coefficients.")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.user_set is None and not (args.no_save or args.dry_run):
        parser.error('pick a non-volatile user set to save into with '
                     '--user-set {1..8}, or pass --no-save')
    if args.no_save:
        args.user_set = None
    if args.reset_coefficients and args.flat_only:
        parser.error('--reset-coefficients with --flat-only would throw away '
                     'the dark calibration you are about to rely on')
    if args.reset_coefficients and args.load_coefficients is not None:
        parser.error('--reset-coefficients and --load-coefficients contradict '
                     'each other: rpc clears what lpc just loaded')

    if not (args.yes or args.dry_run):
        print('This will recalibrate the Piranha4 on lane %d and %s'
              % (args.lane, 'save to user set %d' % args.user_set
                 if args.user_set else 'not save the result'))
        try:
            if input('Continue? [yes/no] ').strip().lower() not in ('y', 'yes'):
                return 1
        except EOFError:
            return 1

    # Register access (VC0) and the camera serial port (VC2) only; no YAML
    # configuration is loaded and the data path is left alone.
    root = cameralink_gateway.ClinkDevRoot(
        dev=args.dev,
        pollEn=False,
        initRead=True,
        laneConfig={args.lane: 'Piranha4'},
        dataDebug=False,
        enLclsII=True,
        startupMode=True,
        pgp4=args.pgp4,
        enableConfig=False,
        enVcMask=0x5,
        zmqSrvEn=False,
    )

    # Swap in the multi-line receiver before start(), which already talks to
    # the camera (Esc, spf, gcp).
    uartDev = getattr(getattr(root, 'ClinkFeb[%d]' % args.lane).ClinkTop,
                      'Ch[%d]' % args.chan).UartPiranha4
    uartDev._rx = Piranha4Rx(uartDev._rx._path, verbose=args.verbose)
    pr.streamConnect(root.dmaStreams[args.lane][2], uartDev._rx)

    root.start()
    try:
        if root.RemRxLinkReady[args.lane].get() != 1:
            raise Piranha4Error('PGP link on lane %d is down' % args.lane)

        cam = Piranha4(root, args.lane, args.chan, verbose=args.verbose)
        cam.channel.BaudRate.set(args.baud)
        cam.channel.SerThrottle.set(args.throttle)
        time.sleep(0.1)

        cam.sync()

        cfg = parse_gcp(cam.cmd('gcp', tmo=15.0))
        print_gcp(cfg, 'camera configuration before calibration')
        bist = cfg.get('BiST', '')
        if bist and bist != 'Good':
            print('  *** WARNING: BiST = %s (see the manual for the meaning)'
                  % bist)
        print('    Temperature    %s' % cam.cmd('vt')[-1])
        print('    Voltage        %s' % cam.cmd('vv')[-1])

        pre, dark, flat, post, save, info = make_plan(args, cfg)

        print('--- plan')
        print('    pixel format   %d bit, full scale %d DN'
              % (info['bits'], info['fullScale']))
        print('    timing         %d Hz, %d ns exposure'
              % (args.linerate, info['exptime']))
        if flat:
            print('    cpa target     %d DN (%.0f%% of full scale), algorithm %s'
                  % (info['target'], 100.0 * info['target'] / info['fullScale'],
                     args.algorithm))
            if cfg.get('System Gain') is not None:
                print('    system gain    %s   (the target is divided by gain '
                      'and binning, then the offset subtracted)'
                      % cfg.get('System Gain'))

        if args.dry_run:
            print('--- commands that would be sent (dry run)')
            for label, steps in (('prepare', pre), ('phase 1 dark', dark),
                                 ('phase 2 bright', flat),
                                 ('restore', post), ('save', save)):
                if not steps:
                    continue
                print('    [%s]' % label)
                for text, t in steps:
                    print('      %-18s (timeout %.0f s)' % (text, t))
            if not save:
                print('    *** no --user-set given: nothing would be saved to '
                      'non-volatile ram')
            print('--- dry run: the camera was not changed')
            return 0

        # The camera free-runs during both calibrations; drop those frames at
        # the FEB rather than pushing them at an idle data path.
        blowoff = cam.channel.Blowoff.get()
        cam.channel.Blowoff.set(True)

        calibrated = False
        try:
            print('--- preparing the camera')
            for text, t in pre:
                cam.cmd(text, tmo=t)

            if dark:
                confirm(DARK_PROMPT, args)
                print('--- phase 1: dark FPN calibration, averaging %d lines '
                      'at %d Hz (~%.1f s of data)'
                      % (args.dark_lines, args.linerate,
                         args.dark_lines / args.linerate))
                time.sleep(args.settle)
                for text, t in dark:
                    cam.cmd(text, tmo=t, progress=True)

            if flat:
                confirm(FLAT_PROMPT, args)
                print('--- phase 2: flat field PRNU calibration, averaging %d '
                      'lines at %d Hz to target %d DN'
                      % (args.flat_lines, args.linerate, info['target']))
                time.sleep(args.settle)
                for text, t in flat:
                    cam.cmd(text, tmo=t, progress=True)

            calibrated = True
        finally:
            print('--- restoring the original camera timing')
            for text, t in post:
                try:
                    cam.cmd(text, tmo=t)
                except Piranha4Error as e:
                    print('  *** WARNING: restore step failed: %s' % e)
            cam.channel.Blowoff.set(blowoff)

        if calibrated and save:
            print('--- saving to camera non-volatile ram')
            for text, t in save:
                cam.cmd(text, tmo=t)
        elif calibrated:
            print('--- no --user-set given: the new coefficients are in use '
                  'but will be lost on the next power cycle or "rc"')

        cfg = parse_gcp(cam.cmd('gcp', tmo=15.0))
        print_gcp(cfg, 'camera configuration after calibration')

        print('--- done')
        if dark:
            print('    dark FPN:      ccf %d' % args.dark_lines)
        if flat:
            print('    flat field:    cpa %d %d %d'
                  % (ALGORITHMS[args.algorithm], args.flat_lines,
                     info['target']))
        print('    correction:    %s' % cfg.get('Flat Field', 'unknown'))
        if args.user_set is not None:
            print('    saved to set:  %d (uss)' % args.user_set)
            print('    power-up set:  %s'
                  % (str(args.user_set) + ' (usd)' if args.default_set
                     else cfg.get('DefaultSet', 'unknown') + ' (unchanged)'))
            if not args.default_set:
                print('    *** the coefficients are only reloaded at power-up '
                      'if the default user set is %d: rerun with '
                      '--default-set, or set the configdb USD' % args.user_set)
        print('    *** a DAQ Configure applies the configdb values: FFM must '
              'be 1 there, or the correction gets turned back off')
        if flat:
            print('    *** check the result on an image: a flat field cal is '
                  'only as good as the uniformity of the reference')

    finally:
        root.stop()

    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Piranha4Error as e:
        print('*** Piranha4 calibration failed: %s' % e)
        sys.exit(1)
    except KeyboardInterrupt:
        print('*** interrupted')
        sys.exit(1)
