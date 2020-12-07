#!/usr/bin/env python3
#

import os
import time
import jmespath
from datetime import datetime, timedelta
import requests
import argparse
import curses


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
    def __init__(self, srvurl, query, column, width=12):
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

    while (k == -1):

        # Initialization
        stdscr.erase()
        height, width = stdscr.getmaxyx()

        y = 0
        x = 0

        line = f'DAQ blockages monitor for partition {args.part}, instrument "{args.inst}"'
        stdscr.addstr(y, x, line, curses.color_pair(1))
        y += 2

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

        stdscr.addstr(y, x, line, curses.color_pair(1))
        y += 1
        for key in keys:
            stdscr.addstr(y, 2+x,    key[0], curses.color_pair(1))
            stdscr.addstr(y, 2+x+14, key[1], curses.color_pair(1))
            y += 1

        y += 1
        line = 'Column header descriptions:'
        stdscr.addstr(y, x, line, curses.color_pair(1))
        y += 1
        for column, metric in metrics.items():
            stdscr.addstr(y, 2+x,    column,         curses.color_pair(1))
            stdscr.addstr(y, 2+x+14, metric.descr(), curses.color_pair(1))
            y += 1
            if y > height - 1:  break

        y = height - 1
        stdscr.addstr(y, x, "Hit any character to continue", curses.color_pair(1))

        # Wait for next input
        k = stdscr.getch()

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
    curses.init_pair(1, curses.COLOR_WHITE,  curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_BLACK,  curses.COLOR_WHITE)
    curses.init_pair(3, curses.COLOR_CYAN,   curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED,    curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    # Attempt to turn off the cursor
    curses.curs_set(False)

    # Update the screen periodically, independent of input
    stdscr.timeout(1000)

    dbg = Debug(args, "./daqPipes.dbg")

    # Loop where k is the last character pressed
    k = 0
    start_y = 0
    start_x = 0
    start_row = 0               # In units of rows    of some height
    start_col = 0               # In units of columns of some width
    size_y = 0
    size_x = 12 + size_x + 1    # Add space for detName column
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

            if   k == curses.KEY_DOWN:
                start_row += 1
            elif k == curses.KEY_UP:
                start_row -= 1
            elif k == curses.KEY_RIGHT:
                start_col += 1
            elif k == curses.KEY_LEFT:
                start_col -= 1
            elif k == curses.KEY_NPAGE:
                start_col += height
            elif k == curses.KEY_PPAGE:
                start_col -= height
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
                new_x_size += 20 if showInstance else -20
            elif k == ord('h') or k == ord('?'):
                showHelp(stdscr, args, metrics)
                k = 0

            # Sample the metrics
            samples = update(metrics, time)
            new_y_size = 1 + len(samples)
            if time is not None:  new_y_size += 2

            # Set up a sub-window that fits the whole thing
            if new_y_size > size_y or new_x_size != size_x:
                size_y = new_y_size
                size_x = new_x_size

                dbg.write('size_y, size_x: %d, %d\n' % (size_y, size_x))

                pad = curses.newpad(size_y, size_x)

            pad.erase()

            dbg.write('len(samples): %d, size_y %d\n' % (len(samples), size_y))
            dbg.write('len(metrics): %d, size_x %d\n' % (len(metrics), size_x))

            # Establish bounds
            tot_rows = 1 + len(samples) # 1 for the header bar
            rows = min(height, tot_rows)

            start_row = max(0, start_row)
            start_row = min(tot_rows - rows, start_row)
            start_row = max(0, start_row)

            tw = 12
            cols = 1            # DetName
            if showInstance:
                tw += 20
                cols += 1       # Instance
            tot_cols = cols + len(metrics)
            for metric in metrics.values():
                mw = metric.width()
                if tw + mw <= width:
                    tw += mw
                    cols += 1

            start_col = max(0, start_col)
            start_col = min(tot_cols - cols, start_col)
            start_col = max(0, start_col)

            dbg.write('start_row, max: %d, %d\n' % (start_row,
                                                    max(0, tot_rows - rows)))
            dbg.write('start_col, max: %d, %d\n' % (start_col,
                                                    max(0, tot_cols - cols)))

            # Render header bar
            pad.attron(curses.color_pair(2))
            sc = 0
            start_x = 0
            y = 0
            x = 0
            cw = 0
            if showInstance:
                header = 'Instance'
                cw = 20
                pad.addstr(y, x, header)
                pad.addstr(y, x + len(header), " " * (cw - len(header)))
                if sc < start_col:
                    start_x += cw
                    sc += 1
                x += cw
            header = 'DetName'
            cw = 12
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
            pad.attroff(curses.color_pair(2))

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
                    cw = 20
                    pad.addstr(y, x, instance, curses.color_pair(2))
                    if sr < start_row:
                        start_y += rh
                        sr += 1
                    x += cw

                sample = samples[instance]
                cw = 12
                pad.addstr(y, x, sample[0], curses.color_pair(1))
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

            dbg.write('start_y %d, height %d, size_y %d\n' % (start_y, height, size_y))
            dbg.write('start_x %d, width  %d, size_x %d\n' % (start_x, width,  size_x))

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
                pad.addstr(size_y - 1, 0, entry, curses.color_pair(2))

            # Refresh the screen
            stdscr.refresh()
            pad.refresh( start_y,start_x, 0,0, height-1,width-1 )

            # Wait for next input
            k = stdscr.getch()
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
    w = 12
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

    DRP_DmaCtMax  = _q(args, 'drp_dma_in_use_max')
    DRP_DmaInUse  = _q(args, 'drp_dma_in_use')
    DRP_WrkQueDp  = _q(args, 'drp_worker_queue_depth')
    DRP_WrkInQue  = _q(args, 'drp_worker_input_queue')
    DRP_WrkOutQue = _q(args, 'drp_worker_output_queue')
    TCtb_IUMax    = _q(args, 'TCtb_IUMax')
    TCtb_IUBats   = _q(args, 'TCtb_IUBats')
    TCtbO_IFMax   = _q(args, 'TCtbO_IFMax')
    TCtbO_InFlt   = _q(args, 'TCtbO_InFlt')
    TCtbO_BatCt   = _q(args, 'TCtbO_BatCt')
    TCtbI_BatCt   = _q(args, 'TCtbI_BatCt')
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

    def _fmtPct(value):
        number = float(value)
        color  = 3 if number < 95.0 else 5 if number < 99.0 else 4
        #entry  = ('% 11.6f' if '.' in value else '% 11.0f') % (number)
        entry  = '% 11.6f' % (number)
        return entry, color

    def _fmtBool(value):
        number = int(value)
        color  = 3 if number == 0 else 4
        #entry  = '   ok' if number == 0 else '  blk' if number < 2 else ('  blk%d' % number)
        if number == 0:
            entry = '   ok'
        elif number == 1:
            entry = '  blk'
        elif number < 10:
            entry = ' blk%1d' % number
        elif number < 100:
            entry = 'blk%2d' % number
        else:
            entry = value #hex(number)
        return entry, color

    def _fmtHex(value):
        number = int(value)
        color  = 3 if number == 0 else 4
        entry  = '   ok' if number == 0 else '%016x' % (number)
        return entry, color

    queries = {
        '%_DMA_occ'   : (f'200.0*{DRP_DmaInUse}/{DRP_DmaCtMax}', # Compensate for the nextPowerOf2() in DrpBase
                         _fmtPct, 'Percentage of occupied DRP DMA buffers'),
        '%_WkrI_occ'  : (f'100.0*{DRP_WrkInQue}/{DRP_WrkQueDp}',
                         _fmtPct, 'Percentage occupancy of all Input work queues on a DRP'),
        '%_WkrO_occ'  : (f'100.0*{DRP_WrkOutQue}/{DRP_WrkQueDp}',
                         _fmtPct, 'Percentage occupancy of all Output work queues on a DRP'),
        '%_Bat_InUse' : (f'100.0*{TCtb_IUBats}/{TCtb_IUMax}',
                         _fmtPct, 'Percentage of DRP Input batches allocated'),
        'Bat_Wtg'     : (_q(args, 'TCtbO_BtWtg'),
                         _fmtBool, 'Indicator of the DRP Input batch pool being exhausted', 7),
        '%_Bat_InFlt' : (f'100.0*{TCtbO_InFlt}/{TCtbO_IFMax}',
                         _fmtPct, 'Percentage of DRP Input batches queued to await a Result'),
        'DRP->TEB'    : (_q(args, 'TCtbO_TxPdg'),
                         _fmtBool, 'Indicator of when traffic from DRP to TEB is stalled', 8),
        '%_TEB_Full'  : (f'100.0*({TEB_EvAlCt}-{TEB_EvFrCt})/{TEB_EvPlDp}',
                         _fmtPct, 'Percentage of allocated TEB event buffers'),
        'TEB->DRP'    : (_q(args, 'TEB_TxPdg'),
                         _fmtHex, 'Indicator of when traffic from TEB to DRP is stalled', 16),
        '%_FileW_occ' : (f'100.0*(1.0 - {DRP_RecDp}/{DRP_RecDpMax})',
                         _fmtPct, 'Percentage occupancy of the recording queue'),
        'FlWrB'       : (_q(args, 'DRP_fileWriting'),
                         _fmtBool, 'Indicates when file writing is stalled', 6),
        'BfFrB'       : (_q(args, 'DRP_bufFreeBlk'),
                         _fmtBool, 'Indicates when FileWriter is blocked for a free buffer', 6),
        'BfPndB'      : (_q(args, 'DRP_bufPendBlk'),
                         _fmtBool, 'Indicates when FileWriter is blocked on a pending buffer', 6),
        'SmdWrB'      : (_q(args, 'DRP_smdWriting'),
                         _fmtBool, 'Indicates when SMD file writing is stalled', 6),
        'DRP->MEB'    : (_q(args, 'MCtbO_TxPdg'),
                         _fmtBool, 'Indicator of when traffic from DRP to MEB is blocked', 8),
        '%_MEB_Full'  : (f'100.0*({MEB_EvAlCt}-{MEB_EvFrCt})/{MEB_EvPlDp}',
                         _fmtPct, 'Percentage of allocated MEB event buffers'),
        '%_MonReqOcc' : (f'100.0*({MRQ_BufCt}/{MRQ_BufCtMax})',
                         _fmtPct, 'Percentage of occupied shmem buffers'),
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
    parser.add_argument('--start', help='start time [now]', type=str, default=start)
    parser.add_argument('--debug', help='debug flag', action='store_true')

    daqPipes(promserver, parser.parse_args())


if __name__ == "__main__":

    main()
