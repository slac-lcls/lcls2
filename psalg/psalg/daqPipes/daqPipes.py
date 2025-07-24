#!/usr/bin/env python3
#

import os
import time
from datetime import datetime, timedelta
import requests
import argparse
import curses

# Name the colors
_white   = 1
_black   = 2
_cyan    = 3
_red     = 4
_yellow  = 5
_green   = 6
_magenta = 7

_maxDetNameSz  = 16
_maxInstanceSz = 21
_defColWidth   = 12

class Debug:
    def __init__(self, args, filename):
        self._args = args
        self._file = None
        self._filename = filename

    def write(self, line):
        if self._args.debug:
            if self._file is None:
                self._file = open(self._filename, "w")
            self._file.write(line)

    def done(self):
        if self._args.debug:
            if self._file is not None:
                self._file.close()

class PromMetric:
    def __init__(self, srvurl, query, column, width=_defColWidth):
        self._srvurl = srvurl
        self._query  = query[0]
        self.dpyFmt  = query[1] # A callable function, so no '_'
        self._descr  = query[2]
        self._width  = width if len(query) < 4 else query[3]+1 # +1 for a space
        self._column = column

    def descr(self):
        return self._descr

    def column(self):
        return self._column

    def width(self):
        return self._width

    def query(self, query, time):
        srvurl = self._srvurl
        payload = {"query": query}
        if time is not None:
            payload["time"] = time
        url = f"{srvurl}/api/v1/query"
        r = requests.get(url, params=payload)
        #print(r.url)

        data = r.json()
        #print("Stats:", len(data["data"]["result"]))
        return data

    def query_range(self, query, start, stop, step="5s"):
        if stop is None:
            return self.query(query, start)

        srvurl = self._srvurl
        payload = {"query": query, "start": int(start), "end": int(stop), "step": step}
        url = f"{srvurl}/api/v1/query_range"
        r = requests.get(url, params=payload)
        #print(r.url)

        data = r.json()
        #print("Stats:", len(data["data"]["result"]))
        return data

    def get(self, time):
        #print("query:", self._query)
        #data = self.query(self._query, time)
        #print("query data: ", data)

        start = time
        stop  = start + 15 if time is not None else None
        data = self.query_range(self._query, start, stop)
        #print("query_range data: ", data)

        self._status = data['status']
        self._type   = data['data']['resultType']

        return data['data']['result']

        #data = [{'metric': {'__name__': 'drp_dma_in_use', 'detname': 'epics', 'detseg': '0', 'instance': 'drp-tst-dev004:9201', 'instrument': 'tst', 'job': 'drpmon', 'norm': '1048576', 'partition': '3'}, 'value': [1598915557.572, '63']}, {'metric': {'__name__': 'drp_dma_in_use', 'detname': 'tmoandor', 'detseg': '0', 'instance': 'drp-tst-dev004:9202', 'instrument': 'tst', 'job': 'drpmon', 'norm': '1048576', 'partition': '3'}, 'value': [1598915557.572, '17']}, {'metric': {'__name__': 'drp_dma_in_use', 'detname': 'tmocam0', 'detseg': '0', 'instance': 'drp-tst-dev004:9200', 'instrument': 'tst', 'job': 'drpmon', 'norm': '1048576', 'partition': '3'}, 'value': [1598915557.572, '93']}, {'metric': {'__name__': 'drp_dma_in_use', 'detname': 'tmots', 'detseg': '0', 'instance': 'drp-tst-dev009:9200', 'instrument': 'tst', 'job': 'drpmon', 'norm': '131072', 'partition': '3'}, 'value': [1598915557.572, '105']}]
        #return data #['data']['result']

def update(metrics, time):

    samples = {}
    for column, metric in metrics.items():
        data = metric.get(time)
        #print('column:', column, ', data:', data)
        for result in data:
            #print('result:', result)
            labels, values = result.values()
            instance = labels['instance']
            detName = ''
            if 'detname' in labels.keys():
                detName = labels['detname']
                if 'detseg' in labels.keys():
                    detName += '_' + labels['detseg']
            if instance not in samples.keys():
                samples[instance] = [detName, {}]
            if detName and not samples[instance][0]:
                samples[instance][0] = detName
            samples[instance][1][column] = values if time is None else values[0]
            #print('instance:', instance, ', column:', column, ', values:', values)

    return samples

def showHelp(stdscr, args, metrics):

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Attempt to turn on the cursor
    curses.curs_set(True)

    # Update the screen periodically, independent of input
    stdscr.timeout(-1)

    # Loop where k is the last character pressed
    k = -1

    start_row = 0

    while (True):

        # Initialization
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        y = 0;  r = 0
        x = 0

        line = f'DAQ blockages monitor for partition {args.part}, instrument "{args.inst}"'
        if r >= start_row:
            stdscr.addstr(y, x, line, curses.color_pair(_white))
            y += 1
        r += 1

        if r >= start_row:
            y += 1              # Blank line
        r += 1

        line = 'Available keystrokes:'
        keys = [('Arrow keys', 'Scroll by column or row'),
                ('Page up/down', 'Scroll rows by page'),
                ('i', 'Toggle display of the process "instance" name'),
                ('n/p', 'Advance/retreat time by one step'),
                ('<space>', 'Advance time by one step'),
                ('+/-', 'Increase/decrease time step size by 1 second'),
                ('t', 'Toggle use of current vs "start" parameter time'),
                ('h/?', 'Help'),
                ('q', 'Quit'),]

        if r >= start_row:
            stdscr.addstr(y, x, line, curses.color_pair(_white))
            y += 1
        r += 1
        for key in keys:
            if r >= start_row:
                stdscr.addstr(y, 2+x,    key[0], curses.color_pair(_white))
                stdscr.addstr(y, 2+x+14, key[1], curses.color_pair(_white))
                y += 1
            r += 1

        if r >= start_row:
            y += 1              # Blank line
        r += 1
        line = 'Column header descriptions:'
        if r >= start_row:
            stdscr.addstr(y, x, line, curses.color_pair(_white))
            y += 1
        r += 1
        last_row = r
        for column, metric in metrics.items():
            if r >= start_row:
                stdscr.addstr(y, 2+x,    column,         curses.color_pair(_white))
                stdscr.addstr(y, 2+x+14, metric.descr(), curses.color_pair(_white))
                y += 1
            r += 1
            if y > height - 2:  break

        last_row += len(metrics.items())
        last_row -= height - 1

        y = height - 1
        stdscr.addstr(y, x, "Scroll or hit any character to continue", curses.color_pair(_white))

        # Wait for next input
        k = stdscr.getch()

        # Handle operator key presses
        if   k == curses.KEY_DOWN:
            if start_row < last_row:  start_row += 1
            if start_row > last_row:  start_row  = last_row
        elif k == curses.KEY_UP:
            if start_row > 0:         start_row -= 1
            if start_row < 0:         start_row  = 0
        elif k == curses.KEY_NPAGE:
            if start_row < last_row:  start_row += height
            if start_row > last_row:  start_row  = last_row
        elif k == curses.KEY_PPAGE:
            if start_row > 0:         start_row -= height
            if start_row < 0:         start_row  = 0
        elif k != -1:
            break

    # Attempt to turn off the cursor
    curses.curs_set(False)

    # Update the screen periodically, independent of input
    stdscr.timeout(1000)

    stdscr.erase()


def draw(stdscr, args, metrics, size_x):

    # Clear and refresh the screen for a blank canvas
    stdscr.clear()
    stdscr.refresh()

    # Start colors in curses
    curses.start_color()
    curses.init_pair(_white,   curses.COLOR_WHITE,   curses.COLOR_BLACK)
    curses.init_pair(_black,   curses.COLOR_BLACK,   curses.COLOR_WHITE)
    curses.init_pair(_cyan,    curses.COLOR_CYAN,    curses.COLOR_BLACK)
    curses.init_pair(_red,     curses.COLOR_RED,     curses.COLOR_BLACK)
    curses.init_pair(_yellow,  curses.COLOR_YELLOW,  curses.COLOR_BLACK)
    curses.init_pair(_green,   curses.COLOR_GREEN,   curses.COLOR_BLACK)
    curses.init_pair(_magenta, curses.COLOR_MAGENTA, curses.COLOR_BLACK)

    # Attempt to turn off the cursor
    curses.curs_set(False)

    # Update the screen periodically, independent of input
    stdscr.timeout(1000)

    # Send debug output to a file, if requested
    dbg = Debug(args, "./daqPipes.dbg")

    # Loop where k is the last character pressed
    k = 0
    start_y = 0
    start_x = 0
    start_row = 0               # In units of rows    of some height
    start_col = 0               # In units of columns of some width
    size_y = 0
    size_x += _maxDetNameSz + 1 # Add space for detName column
    new_y_size = size_y
    new_x_size = size_x
    showInstance = False
    if args.start is not None:
        time = datetime.fromisoformat(args.start).timestamp()
    else:
        time = None
    step = 5                    # Seconds

    try:
        while (k != ord('q')):

            # Initialization
            if k != -1:
                stdscr.erase()
            height, width = stdscr.getmaxyx()
            dbg.write('height %d, width %d\n' % (height, width))
            ht = height - 1 if time is not None else height

            # Sample the metrics
            samples = update(metrics, time)
            new_y_size = 1 + len(samples)
            if time is not None:  new_y_size += 2

            # Set up a sub-window that fits the whole thing
            if new_y_size > size_y or new_x_size != size_x:
                size_y = new_y_size
                size_x = new_x_size

                dbg.write('size_y, size_x: %d, %d\n' % (size_y, size_x))

                pad = curses.newpad(max(height, size_y), max(width, size_x))

            # Clear the sub-window
            pad.erase()

            dbg.write('len(samples): %d, size_y %d\n' % (len(samples), size_y))
            dbg.write('len(metrics): %d, size_x %d\n' % (len(metrics), size_x))

            # Establish bounds
            tot_rows = 1 + len(samples) # 1 for the header bar
            rows = min(ht, tot_rows)

            start_row = max(0,               start_row)
            start_row = min(tot_rows - rows, start_row)
            start_row = max(0,               start_row)

            tw = _maxDetNameSz  # DetName width
            cols = 1            # DetName
            if showInstance:
                tw += _maxInstanceSz
                cols += 1       # Instance
            tot_cols = cols + len(metrics)
            for metric in metrics.values():
                mw = metric.width()
                if tw + mw <= width:
                    tw += mw
                    cols += 1

            start_col = max(0,               start_col)
            start_col = min(tot_cols - cols, start_col)
            start_col = max(0,               start_col)

            dbg.write('start_row, max: %d, %d\n' % (start_row,
                                                    max(0, tot_rows - rows)))
            dbg.write('start_col, max: %d, %d\n' % (start_col,
                                                    max(0, tot_cols - cols)))

            # Render header bar
            pad.attron(curses.color_pair(_black))
            sc = 0
            start_x = 0
            y = 0
            x = 0
            cw = 0
            if showInstance:
                header = 'Instance'
                cw = _maxInstanceSz
                pad.addstr(y, x, header)
                pad.addstr(y, x + len(header), " " * (cw - len(header)))
                if sc < start_col:
                    start_x += cw
                    sc += 1
                x += cw
            header = 'DetName'
            cw = _maxDetNameSz  # DetName width
            pad.addstr(y, x, header)
            pad.addstr(y, x + len(header), " " * (cw - len(header)))
            if sc < start_col:
                start_x += cw
                sc += 1
            for header, metric in metrics.items():
                x += cw
                cw = metric.width()
                dbg.write('cw %d, x %d, len %d, %d %d, header "%s"\n' %
                          (cw, x, len(header), x+len(header), cw - len(header), header))
                if x - start_x + cw <= width:
                    pad.addstr(y, x, header)
                    pad.addstr(y, x + len(header), " " * (cw - len(header)))
                    if sc < start_col:
                        start_x += cw
                        sc += 1
            dbg.write('sc %d, start_x %d\n' % (sc, start_x))
            pad.attroff(curses.color_pair(_black))

            # Render the columns
            rh = 1                  # Revisit: For now row height is 1 line
            sr = 0
            start_y = 0
            cw = 0
            for nInstance, instance in enumerate(samples): # Rows
                dbg.write('nInstance: %d, instance %s\n' % (nInstance, instance))
                y = 1 + nInstance
                x = 0
                if showInstance:
                    cw = _maxInstanceSz
                    pad.addstr(y, x, instance, curses.color_pair(_black))
                    if sr < start_row:
                        start_y += rh
                        sr += 1
                    x += cw

                sample = samples[instance]
                cw = _maxDetNameSz # DetName width
                pad.addstr(y, x, sample[0], curses.color_pair(_white))
                if sr < start_row:
                    start_y += rh
                    sr += 1
                sx = x + cw
                for item, values in sample[1].items():      # Columns
                    dbg.write('item %s, value %s\n' % (item, values))
                    x = sx + metrics[item].column()
                    entry, color = metrics[item].dpyFmt(values[1])
                    if x - start_x + len(entry) <= width:
                        pad.addstr(y, x, entry, curses.color_pair(color))
                        if sr < start_row:
                            start_y += rh
                            sr += 1
            dbg.write('sr %d, start_y %d\n' % (sr, start_y))

            dbg.write('start_y %d, height %d, size_y %d\n' % (start_y, ht,    size_y))
            dbg.write('start_x %d, width  %d, size_x %d\n' % (start_x, width, size_x))

            if tot_cols > cols and start_col < tot_cols - cols:
                pad.addch(start_y, min(size_x - 1, start_x + width - 1), curses.ACS_RARROW, curses.A_STANDOUT)
            if start_col > 0:
                pad.addch(start_y, start_x, curses.ACS_LARROW, curses.A_STANDOUT)
            if start_row > 0:
                pad.addch(start_y, min(size_x - 1, start_x + width - 2), curses.ACS_UARROW, curses.A_STANDOUT)
            if tot_rows > rows and start_row < tot_rows - rows:
                pad.addch(min(size_y - 1, start_y + height - 1), min(size_x - 1, start_x + width - 1), curses.ACS_DARROW, curses.A_STANDOUT)

            if time is not None:
                entry = str(datetime.fromtimestamp(time))
                entry = entry + ',  step = ' + str(step) + ' s'
                pad.addstr(start_y + height - 1, 0, entry, curses.color_pair(_black))

            # Refresh the screen
            stdscr.refresh()
            pad.refresh( start_y,start_x, 0,0, height-1,width-1 )

            # Wait for next input
            k = stdscr.getch()

            # Handle operator key presses
            if   k == curses.KEY_DOWN:
                if start_row < size_y - height:  start_row += 1
                if start_row > size_y - height:  start_row  = size_y - height
            elif k == curses.KEY_UP:
                if start_row > 0:                start_row -= 1
                if start_row < 0:                start_row  = 0
            elif k == curses.KEY_RIGHT:
                if start_col < size_x - width:   start_col += 1
                if start_col > size_x - width:   start_col  = size_x - width
            elif k == curses.KEY_LEFT:
                if start_col > 0:                start_col -= 1
                if start_col < 0:                start_col  = 0
            elif k == curses.KEY_NPAGE:
                if start_row < size_y - height:  start_row += size_y - height
                if start_row > size_y - height:  start_row  = size_y - height
            elif k == curses.KEY_PPAGE:
                if start_row > 0:                start_row -= size_y - height
                if start_row < 0:                start_row  = 0
            elif (k == ord('n') or k == ord(' ')) and time is not None:
                time += step
            elif (k == ord('p')) and time is not None:
                time -= step
            elif k == ord('+') and time is not None:
                step += 1
            elif k == ord('-') and time is not None:
                if step > 1:  step -= 1
            elif k == ord('t') and time is not None:
                time = None
            elif k == ord('t') and time is None:
                if args.start is not None:
                    time = datetime.fromisoformat(args.start).timestamp()
            elif k == ord('i'):
                showInstance = not showInstance
                new_x_size += _maxInstanceSz if showInstance else -_maxInstanceSz
            elif k == ord('h') or k == ord('?'):
                showHelp(stdscr, args, metrics)
                k = 0
    except:
        dbg.done()
        raise
    else:
        dbg.done()


def test(args, metrics):

    if args.start is not None:
        time = datetime.fromisoformat(args.start).timestamp()
    else:
        time = None

    samples = update(metrics, time)
    print('samples:', samples)

    print(0, 0, 'DetName')
    w = _maxDetNameSz           # DetName width
    for header, metric in metrics.items():
        print(0, w, header)
        w += metric.width()

    for instance in samples:
        print(instance, ':', samples[instance])

    for nInstance, instance in enumerate(samples):
        sample = samples[instance]
        print('instance:', nInstance, instance, sample[0])
        print('sample:', sample)
        for item, values in sample[1].items():
            column = metrics[item].column()

            print('item:', item, values[1], column)


def daqPipes(srvurl, args):

    def _q(a, m, eb=None):
        if eb is None:
            return f'{m}{{instrument="{a.inst}",partition="{a.part}"}}'
        else:
            return f'{m}{{instrument="{a.inst}",partition="{a.part}",eb="{eb}"}}'

    # Format the metrics into queries with the scope limiting labels
    DRP_PgpCtMax  = _q(args, 'drp_num_pgp_bufs')
    DRP_PgpInUsr  = _q(args, 'drp_num_pgp_in_user')
    DRP_PgpInHw   = _q(args, 'drp_num_pgp_in_hw')
    DRP_PgpInPre  = _q(args, 'drp_num_pgp_in_prehw')
    DRP_PgpInRx   = _q(args, 'drp_num_pgp_in_rx')
    DRP_DmaCtMax  = _q(args, 'drp_dma_in_use_max')
    DRP_DmaInUse  = _q(args, 'drp_dma_in_use')
    DRP_WrkQueDp  = _q(args, 'drp_worker_queue_depth')
    DRP_WrkInQue  = _q(args, 'drp_worker_input_queue')
    DRP_WrkOutQue = _q(args, 'drp_worker_output_queue')
    TCtb_BEMax    = _q(args, 'TCtb_BEMax')
    TCtbO_BtEnt   = _q(args, 'TCtbO_BtEnt')
    TCtb_IUMax    = _q(args, 'TCtb_IUMax')
    TCtbO_IFMax   = _q(args, 'TCtbO_IFMax')
    TCtbO_InFlt   = _q(args, 'TCtbO_InFlt')
    TCtbO_BatCt   = _q(args, 'TCtbO_BatCt')
    TCtbI_BatCt   = _q(args, 'TCtbI_BatCt')
    TEB_BEMax     = _q(args, 'TEB_BEMax')
    TEB_BtEnt     = _q(args, 'TEB_BtEnt')
    TEB_BfInCt    = _q(args, 'EB_BfInCt', 'TEB')
    TEB_EvFrCt    = _q(args, 'EB_EvFrCt', 'TEB')
    TEB_EvAlCt    = _q(args, 'EB_EvAlCt', 'TEB')
    TEB_EvPlDp    = _q(args, 'EB_EvPlDp', 'TEB')
    MEB_EvFrCt    = _q(args, 'EB_EvFrCt', 'MEB')
    MEB_EvAlCt    = _q(args, 'EB_EvAlCt', 'MEB')
    MEB_EvPlDp    = _q(args, 'EB_EvPlDp', 'MEB')
    DRP_RecDpMax  = _q(args, 'DRP_RecordDepthMax')
    DRP_RecDp     = _q(args, 'DRP_RecordDepth')
    MRQ_BufCt     = _q(args, 'MRQ_BufCt')
    MRQ_BufCtMax  = _q(args, 'MRQ_BufCtMax')

    # Set up a set of formatting functions to turn a query result value into a
    # colored table cell entry
    def _fmtPct(value):
        number = float(value)
        color  = _cyan if number < 95.0 else _yellow if number < 99.0 else _red
        #entry  = ('% 11.6f' if '.' in value else '% 11.0f') % (number)
        entry  = '% 6.2f' % (number) # Ignores space needed for a sign
        return entry, color

    def _fmtPctIC(value):       # Inverted color
        number = float(value)
        color  = _red if number < 1.0 else _yellow if number < 5.0 else _cyan
        #entry  = ('% 11.6f' if '.' in value else '% 11.0f') % (number)
        entry  = '% 6.2f' % (number) # Ignores space needed for a sign
        return entry, color

    def _fmtPctNC(value):       # No Color
        number = float(value)
        color  = _cyan
        #entry  = ('% 11.6f' if '.' in value else '% 11.0f') % (number)
        entry  = '% 6.2f' % (number) # Ignores space needed for a sign
        return entry, color

    def _fmtBool(value):
        number = int(value)
        color  = _green if number == 0 else _red
        #entry  = '   ok' if number == 0 else '  blk' if number < 2 else ('  blk%d' % number)
        if   number == 0:   entry = '   ok'
        elif number == 1:   entry = '  blk'
        elif number < 10:   entry = ' blk%1d' % number
        elif number < 100:  entry = 'blk%2d' % number
        else:               entry = value #hex(number)
        return entry, color

    def _fmtBoolNC(value):      # No Color
        number = int(value)
        color  = _cyan
        #entry  = '   ok' if number == 0 else '  blk' if number < 2 else ('  blk%d' % number)
        if   number == 0:   entry = '   ok'
        elif number == 1:   entry = '  blk'
        elif number < 10:   entry = ' blk%1d' % number
        elif number < 100:  entry = 'blk%2d' % number
        else:               entry = value #hex(number)
        return entry, color

    def _fmtHex(value):
        number = int(value)
        color  = _green if number == 0 else _red
        entry  = '   ok' if number == 0 else '%016x' % (number)
        return entry, color

    # Set up the list of prometheus queries in the order they will appear on the screen (l to r)
    queries = {
        # key         : (query, format func., help string, cell width)
        '%InUsr'      : (f'100.0*{DRP_PgpInUsr}/{DRP_PgpCtMax}',
                         _fmtPct, 'Percentage of PGP buffers in user', 6),
        '%InHw'       : (f'100.0*{DRP_PgpInHw}/{DRP_PgpCtMax}',
                         _fmtPct, 'Percentage of PGP buffers in hw', 6),
        '%InPre'      : (f'100.0*{DRP_PgpInPre}/{DRP_PgpCtMax}',
                         _fmtPct, 'Percentage of PGP buffers in pre-hw queue', 6),
        '%InRx'       : (f'100.0*{DRP_PgpInRx}/{DRP_PgpCtMax}',
                         _fmtPct, 'Percentage of PGP buffers in Rx queue', 6),
        '%DMA'        : (f'100.0*{DRP_DmaInUse}/{DRP_DmaCtMax}',
                         _fmtPct, 'Percentage of occupied DRP DMA buffers', 6),
        '%WkrI'       : (f'100.0*{DRP_WrkInQue}/{DRP_WrkQueDp}',
                         _fmtPct, 'Percentage occupancy of all Input work queues on a DRP', 6),
        '%WkrO'       : (f'100.0*{DRP_WrkOutQue}/{DRP_WrkQueDp}',
                         _fmtPct, 'Percentage occupancy of all Output work queues on a DRP', 6),
        '%IBat'       : (f'100.0*{TCtbO_BtEnt}/{TCtb_BEMax}',
                         _fmtPctNC, 'Entry occupancy of DRP-to-TEB (Input) batches', 6),
        '%BatInFlt'   : (f'100.0*{TCtbO_InFlt}/{TCtbO_IFMax}',
                         _fmtPct, 'Percentage of DRP Input batches queued to await a Result', 9),
        'DRP->TEB'    : (_q(args, 'TCtbO_TxPdg'),
                         _fmtBool, 'Indicator of when traffic from DRP to TEB is stalled', 8),
        '%TEB'        : (f'100.0*({TEB_EvAlCt}-{TEB_EvFrCt})/{TEB_EvPlDp}',
                         _fmtPct, 'Percentage of allocated TEB event buffers', 6),
        '%RBat'       : (f'100.0*{TEB_BtEnt}/{TEB_BEMax}',
                         _fmtPctNC, 'Entry occupancy of TEB-to-DRP (Result) batches', 6),
        'TEB->DRP'    : (_q(args, 'TEB_TxPdg'),
                         _fmtHex, 'Indicator of when traffic from TEB to DRP is stalled', 8),
        '%FileW'      : (f'100.0*(1.0 - {DRP_RecDp}/{DRP_RecDpMax})',
                         _fmtPct, 'Percentage occupancy of the recording queue', 6),
        'BfPndB'      : (_q(args, 'DRP_bufPendBlk'),
                         _fmtBoolNC, 'Indicates when FileWriter is pending for a buffer', 6),
        'FlWrB'       : (_q(args, 'DRP_fileWriting'),
                         _fmtBool, 'Indicates when file writing is stalled', 5),
        'BfFrB'       : (_q(args, 'DRP_bufFreeBlk'),
                         _fmtBool, 'Indicates when FileWriter is blocked freeing a buffer', 5),
        'SmdWrB'      : (_q(args, 'DRP_smdWriting'),
                         _fmtBool, 'Indicates when SMD file writing is stalled', 6),
        'DRP->MEB'    : (_q(args, 'MCtbO_TxPdg'),
                         _fmtBool, 'Indicator of when traffic from DRP to MEB is blocked', 8),
        '%MEB'        : (f'100.0*({MEB_EvAlCt}-{MEB_EvFrCt})/{MEB_EvPlDp}',
                         _fmtPct, 'Percentage of allocated MEB event buffers', 6),
        '%MonReq'     : (f'100.0*({MRQ_BufCt}/{MRQ_BufCtMax})',
                         _fmtPctIC, 'Percentage of occupied shmem buffers', 6),
    }

    width   = 0
    metrics = {}
    for metric, query in queries.items():
        metrics[metric] = PromMetric(srvurl, query, width)
        width += metrics[metric].width()

    #if not args.debug:
    curses.wrapper(draw, args, metrics, width)
    #else:
    #test(args, metrics)


def main():

    promserver = os.environ.get("DM_PROM_SERVER", "http://psmetric03:9090")

    partition = '0'
    hutch     = 'tst'
    start     = None            # Format is 'YYYY-MM-DD hh:mm:ss'

    parser = argparse.ArgumentParser(description='DAQ data flow display')
    parser.add_argument('-p', '--part', help='partition ['+partition+']', type=str, default=partition)
    parser.add_argument('--inst', help='hutch ['+hutch+']', type=str, default=hutch)
    parser.add_argument('--start', help='start time (\'YYYY-MM-DD hh:mm:ss\') [now]', type=str, default=start)
    parser.add_argument('--debug', help='debug flag', action='store_true')

    daqPipes(promserver, parser.parse_args())


if __name__ == "__main__":

    main()
