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

def showHelp(stdscr, args, listOfQueries, dbg):

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
        for queries in listOfQueries:
            for metric, query in queries.items():
                dbg.write('y, x: %d, %d, metric \'%s\', descr \'%s\'\n' % (y, x, metric, query[2]))
                stdscr.addstr(y, 2+x,    metric,   curses.color_pair(1))
                stdscr.addstr(y, 2+x+14, query[2], curses.color_pair(1))
                y += 1
                if y > height - 1:  break
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

class Table:
    def __init__(self, srvurl, queries, dbg):
        self._dbg    = dbg
        self._pad    = None
        self._size_y = 0
        self._size_x = 0
        self.metrics = {}
        for metric, query in queries.items():
            self.metrics[metric] = PromMetric(srvurl, query, self._size_x)
            self._size_x += self.metrics[metric].width()
        self._showInstance = False
        self._rarrow = False
        self._darrow = False

    def update(self, rows, start_row, start_col, showInstance, scrWidth):
        width  = 12 + self._size_x + 1 # Add space for detName column
        height = 1 + rows              # One row for the header

        if showInstance != self._showInstance:
            self._showInstance = showInstance
            width += 20 if showInstance else -20

        # Set up a sub-window that fits the whole Table
        if height > self._size_y or width != self._size_x:
            self._size_y = height
            self._size_x  = width

            self._dbg.write('pad height, width: %d, %d\n' % (height, width))

            self._pad = curses.newpad(height, width)
            self._pad.erase()

        self._dbg.write('rows         %d, height %d\n' % (rows, height))
        self._dbg.write('len(metrics) %d, width  %d\n' % (len(self.metrics), width))

        # Establish bounds
        tot_rows = height       # 1 for the header bar
        rows = min(height, tot_rows)

        start_row = max(0,               start_row)
        start_row = min(tot_rows - rows, start_row)
        start_row = max(0,               start_row)
        self._start_row = start_row

        tw = 12
        cols = 1            # DetName
        if showInstance:
            tw += 20
            cols += 1       # Instance
        tot_cols = cols + len(self.metrics)
        for metric in self.metrics.values():
            mw = metric.width()
            if tw + mw <= scrWidth:
                tw += mw
                cols += 1

        start_col = max(0,               start_col)
        start_col = min(tot_cols - cols, start_col)
        start_col = max(0,               start_col)
        self._start_col = start_col

        self._dbg.write('start_row, max: %d, %d\n' % (start_row,
                                                      max(0, tot_rows - rows)))
        self._dbg.write('start_col, max: %d, %d\n' % (start_col,
                                                      max(0, tot_cols - cols)))
        self._rarrow = tot_cols > cols and start_col < tot_cols - cols
        self._darrow = tot_rows > rows and start_row < tot_rows - rows

    def _header(self, scrWidth):
        self._pad.attron(curses.color_pair(2))
        sc = 0
        start_y = 0
        start_x = 0
        y = 0
        x = 0
        cw = 0
        if self._showInstance:
            header = 'Instance'
            cw = 20
            self._pad.addstr(y, x, header)
            self._pad.addstr(y, x + len(header), " " * (cw - len(header)))
            if sc < self._start_col:
                start_x += cw
                sc += 1
            x += cw
        header = 'DetName'
        cw = 12
        self._pad.addstr(y, x, header)
        self._pad.addstr(y, x + len(header), " " * (cw - len(header)))
        if sc < self._start_col:
            start_x += cw
            sc += 1
        for header, metric in self.metrics.items():
            x += cw
            cw = metric.width()
            self._dbg.write('cw %d, x %d, len %d, %d %d, header "%s"\n' %
                      (cw, x, len(header), x+len(header), cw - len(header), header))
            if x - start_x + cw <= scrWidth:
                self._dbg.write('y %d, x %d, header \'%s\'\n' % (y, x, header))
                self._pad.addstr(y, x, header)
                self._pad.addstr(y, x + len(header), " " * (cw - len(header)))
                if sc < self._start_col:
                    start_x += cw
                    sc += 1
        self._dbg.write('sc %d, start_x %d\n' % (sc, start_x))
        self._pad.attroff(curses.color_pair(2))

        return start_x

    def _rows(self, samples, start_x, scrWidth):
        rh = 1                  # Revisit: For now row height is 1 line
        sr = 0
        start_y = 0
        cw = 0
        for nInstance, instance in enumerate(samples): # Rows
            self._dbg.write('nInstance: %d, instance %s\n' % (nInstance, instance))
            y = 1 + nInstance
            x = 0
            if self._showInstance:
                cw = 20
                self._pad.addstr(y, x, instance, curses.color_pair(2))
                if sr < self._start_row:
                    start_y += rh
                    sr += 1
                x += cw

            sample = samples[instance]
            cw = 12
            self._pad.addstr(y, x, sample[0], curses.color_pair(1))
            if sr < self._start_row:
                start_y += rh
                sr += 1
            sx = x + cw
            for item, values in sample[1].items():      # Columns
                x = sx + self.metrics[item].column()
                entry, color = self.metrics[item].dpyFmt(values[1])
                if x - start_x + len(entry) <= scrWidth:
                    self._pad.addstr(y, x, entry, curses.color_pair(color))
                    if sr < self._start_row:
                        start_y += rh
                        sr += 1
        self._dbg.write('sr %d, start_y %d\n' % (sr, start_y))
        return start_y

    def draw(self, samples, height, width, y):
        # Render header bar
        start_x = self._header(width)

        # Render the columns
        start_y = self._rows(samples, start_x, width)

        self._dbg.write('start_y %d, height %d, size_y %d\n' % (start_y, height, self._size_y))
        self._dbg.write('start_x %d, width  %d, size_x %d\n' % (start_x, width,  self._size_x))

        if self._rarrow:
            self._pad.addch(start_y, min(self._size_x - 1, start_x + width - 1), curses.ACS_RARROW, curses.A_STANDOUT)
        if self._start_col > 0:
            self._pad.addch(start_y, start_x, curses.ACS_LARROW, curses.A_STANDOUT)
        if self._start_row > 0:
            self._pad.addch(start_y, min(self._size_x - 1, start_x + width - 2), curses.ACS_UARROW, curses.A_STANDOUT)
        if self._darrow:
            self._pad.addch(min(self._size_y - 1, start_y + height - 1), min(self._size_x - 1, start_x + width - 1), curses.ACS_DARROW, curses.A_STANDOUT)

        self._pad.noutrefresh( start_y,start_x, y,0, height-1,width-1 )
        return self._size_y

def draw(stdscr, srvurl, args, listOfQueries, dbg):

    start_row = 0               # In units of rows    of some height
    start_col = 0               # In units of columns of some width
    showInstance = False
    step = 5                    # Seconds
    time = None
    if args.start is not None:
        time = datetime.fromisoformat(args.start).timestamp()

    tables = []
    for queries in listOfQueries:
        tables.append(Table(srvurl, queries, dbg))

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

    try:
        # Loop where k is the last character pressed
        k = 0
        while (k != ord('q')):

            # Initialization
            #stdscr.erase()
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
            elif k == ord('h') or k == ord('?'):
                showHelp(stdscr, args, listOfQueries, dbg)
                k = 0
            elif k == ord('\f'): # ^l
                stdscr.clear()

            y = 0
            for table in tables:
                samples = update(table.metrics, time)
                table.update(len(samples), start_row, start_col, showInstance, width)
                rows = table.draw(samples, height, width, y)
                y += rows + 1
                stdscr.addstr(y-1, 0, ' ' * width) # Clear line between tables

            if time is not None:
                entry = str(datetime.fromtimestamp(time))
                entry = entry + ',  step = ' + str(step) + ' s'
                stdscr.addstr(height - 1, 0, entry, curses.color_pair(2))

            # Refresh the screen
            curses.doupdate() #refresh()

            # Wait for next input
            k = stdscr.getch()
    except:
        dbg.done()
        raise
    else:
        dbg.done()


def test(srvurl, args, listOfQueries, dbg):

    tables = []
    for queries in listOfQueries:
        tables.append(Table(srvurl, queries, dbg))

    if args.start is not None:
        time = datetime.fromisoformat(args.start).timestamp()
    else:
        time = None

    for table in tables:
        samples = update(table.metrics, time)
        print('samples:', samples)

        print(0, 0, 'DetName')
        w = 12
        for header, metric in table.metrics.items():
            print(0, w, header)
            w += metric.width()

        for instance in samples:
            print(instance, ':', samples[instance])

        for nInstance, instance in enumerate(samples):
            sample = samples[instance]
            print('instance:', nInstance, instance, sample[0])
            print('sample:', sample)
            for item, values in sample[1].items():
                column = metric.column()

                print('item:', item, values[1], column)


def daqStats(srvurl, args):

    def _q(a, m, eb=None):
        if eb is None:
            return f'{m}{{instrument="{a.inst}",partition="{a.part}"}}'
        else:
            return f'{m}{{instrument="{a.inst}",partition="{a.part}",eb="{eb}"}}'

    def _fmtN(value):
        color  = 3
        return value, color

    def _fmtF(value):
        color  = 3
        entry  = '%.1f' % (float(value))
        return entry, color

    def _fmtBool(value):
        color  = 3 if int(value) == 0 else 4
        return value, color

    def _fmtHex(value):
        color  = 3
        entry  = '%016x' % (int(value))
        return entry, color

    drpQueries = {
        'EvtRt'     : (_q(args, 'TCtbO_EvtRt'),             _fmtF,    'Event rate',                                               8),
        'DmaInUse'  : (_q(args, 'drp_dma_in_use'),          _fmtN,    '# of DMA buffers in use',                                  8),
        'WrkInQ'    : (_q(args, 'drp_worker_input_queue'),  _fmtN,    '# of Events on the worker Input  Queue',                   6),
        'WrkOutQ'   : (_q(args, 'drp_worker_output_queue'), _fmtN,    '# of Events on the worker Output Queue',                   7),
        'TO_EvtCt'  : (_q(args, 'TCtbO_EvtCt'),             _fmtN,    '# of Input  events posted   to   TEB',                     8),
        'TI_EvtCt'  : (_q(args, 'TCtbI_EvtCt'),             _fmtN,    '# of Result events received from TEB',                     8),
        'TO_BatCt'  : (_q(args, 'TCtbO_BatCt'),             _fmtN,    '# of Input  batches',                                      8),
        'TI_BatCt'  : (_q(args, 'TCtbI_BatCt'),             _fmtN,    '# of Result batches',                                      8),
        'TO_InFlt'  : (_q(args, 'TCtbO_InFlt'),             _fmtN,    '# of Events in flight from DRP Input side to EbReceiver',  8),
        'BtAlCt'    : (_q(args, 'TCtbO_BtAlCt'),            _fmtN,    '# of Input batches allocated',                             6),
        'BtFrCt'    : (_q(args, 'TCtbO_BtFrCt'),            _fmtN,    '# of Input batches freed',                                 6),
        'BtInUse'   : (_q(args, 'TCtb_IUBats'),             _fmtN,    '# of Input batches in use',                                7),
        'BtWtg'     : (_q(args, 'TCtbO_BtWtg'),             _fmtBool, 'Input batch pool exhaustion flag',                         5),
        'InpTxPdg'  : (_q(args, 'TCtbO_TxPdg'),             _fmtHex,  'Input batch transmit-to-TEB pending list',                16),
        'ResRxPdg'  : (_q(args, 'TCtbI_RxPdg'),             _fmtN,    'Result batch receive pending flag',                        8),
        'BypCt'     : (_q(args, 'TCtbI_BypCt'),             _fmtN,    '# of Events bypassing the TEB',                            5),
        'InpMisCt'  : (_q(args, 'TCtbI_MisCt'),             _fmtN,    '# of Results missing an Input event',                      8),
        'RecDp'     : (_q(args, 'DRP_RecordDepth'),         _fmtN,    '# of Free slots on the Record Queue',                      5),
        'FlWrB'     : (_q(args, 'DRP_fileWriting'),         _fmtBool, 'Indicates when file writing is stalled',                   6),
        'BfFrB'     : (_q(args, 'DRP_bufFreeBlk'),          _fmtBool, 'Indicates when FileWriter is blocked for a free buffer',   6),
        'BfPndB'    : (_q(args, 'DRP_bufPendBlk'),          _fmtBool, 'Indicates when FileWriter is blocked on a pending buffer', 6),
        'SmdWrB'    : (_q(args, 'DRP_smdWriting'),          _fmtBool, 'Indicates when SMD file writing is stalled',               6),
        'MO_EvCt'   : (_q(args, 'MCtbO_EvCt'),              _fmtN,    '# of Events posted to the MEB',                            7),
        'MO_TxPdg'  : (_q(args, 'MCtbO_TxPdg'),             _fmtHex,  'Event Transmit-to-MEB pending list',                      16),
    }

    tebQueries = {
	      'EvtRt'        : (_q(args, 'TEB_EvtRt'),        _fmtF,   'Event rate',             8),
	      'EvtCt'        : (_q(args, 'TEB_EvtCt'),        _fmtN,   '# of Events handled',    8),
	      'RxPdg'        : (_q(args, 'EB_RxPdg',  'TEB'), _fmtN,   'Receive pending flag',   5),
	      'BfInCt'       : (_q(args, 'EB_BfInCt', 'TEB'), _fmtN,   '# of Input Batches',     8),
	      'EvAlCt'       : (_q(args, 'EB_EvAlCt', 'TEB'), _fmtN,   '# of Allocated events',  8),
	      'EvFrCt'       : (_q(args, 'EB_EvFrCt', 'TEB'), _fmtN,   '# of Freed events',      8),
	      'FxUpCt'       : (_q(args, 'EB_FxUpCt', 'TEB'), _fmtN,   '# of Swept out events',  8),
	      'ToEvCt'       : (_q(args, 'EB_ToEvCt', 'TEB'), _fmtN,   '# of Timed out events',  8),
	      'CtrbMissing'  : (_q(args, 'EB_CbMsMk', 'TEB'), _fmtHex, 'Missing contributors',   16),
# Unused: 'BtWtg'        : (_q(args, 'TEB_BtWtg'),       _fmtBool, 'Result batch pool exhaustion flag', 5),
	      'BatCt'        : (_q(args, 'TEB_BatCt'),        _fmtN,   '# of Result Batches',    8),
	      'BtAlCt'       : (_q(args, 'TEB_BtAlCt'),       _fmtN,   '# of Batches Allocated', 8),
	      'BtFrCt'       : (_q(args, 'TEB_BtFrCt'),       _fmtN,   '# of Batched freed',     8),
	      'TxPdg'        : (_q(args, 'TEB_TxPdg'),        _fmtHex, 'Transmit pending list',  16),
	      'WrtCt'        : (_q(args, 'TEB_WrtCt'),        _fmtN,   '# of Record  triggers',  8),
	      'MonCt'        : (_q(args, 'TEB_MonCt'),        _fmtN,   '# of Monitor triggers',  8),
	      'PsclCt'       : (_q(args, 'TEB_PsclCt'),       _fmtN,   'Prescale count',         8),
	      'SpltCt'       : (_q(args, 'TEB_SpltCt'),       _fmtN,   'Split event count',      8),
    }

    mebQueries = {
	      'EvtRt'        : (_q(args, 'MEB_EvtRt'),        _fmtF,   'Event rate',                     8),
	      'EvtCt'        : (_q(args, 'MEB_EvtCt'),        _fmtN,   '# of Events handled',            8),
	      'RxPdg'        : (_q(args, 'EB_RxPdg',  'MEB'), _fmtN,   'Receive pending flag',           5),
	      'BfInCt'       : (_q(args, 'EB_BfInCt', 'MEB'), _fmtN,   '# of Input Batches',             8),
	      'EvAlCt'       : (_q(args, 'EB_EvAlCt', 'MEB'), _fmtN,   '# of Allocated events',          8),
	      'EvFrCt'       : (_q(args, 'EB_EvFrCt', 'MEB'), _fmtN,   '# of Freed events',              8),
	      'FxUpCt'       : (_q(args, 'EB_FxUpCt', 'MEB'), _fmtN,   '# of Swept out events',          8),
	      'ToEvCt'       : (_q(args, 'EB_ToEvCt', 'MEB'), _fmtN,   '# of Timed out events',          8),
	      'CtrbMissing'  : (_q(args, 'EB_CbMsMk', 'MEB'), _fmtHex, 'Missing contributors',   16),
        'RqBufCt'      : (_q(args, 'MRQ_BufCt'),        _fmtN,   '# of Available Request buffers', 8),
	      'ReqRt'        : (_q(args, 'MEB_ReqRt'),        _fmtF,   'Monitor request rate ',          8),
	      'ReqCt'        : (_q(args, 'MEB_ReqCt'),        _fmtN,   '# of Monitor requests',          8),
    }

    #width  = 0
    #metrics  = [{}, {}, {}]
    #for metric, query in drpQueries.items():
    #    metrics[0][metric] = PromMetric(srvurl, query, width)
    #    width += metrics[0][metric].width()
    #totWidth = width
    #
    #width    = 0
    #for metric, query in tebQueries.items():
    #    metrics[1][metric] = PromMetric(srvurl, query, width)
    #    width += metrics[1][metric].width()
    #if width > totWidth:  totWidth = width
    #
    #width  = 0
    #for metric, query in mebQueries.items():
    #    metrics[2][metric] = PromMetric(srvurl, query, width)
    #    width += metrics[2][metric].width()
    #if width > totWidth:  totWidth = width

    queries = [drpQueries, tebQueries, mebQueries]
    debug   = Debug(args, "./daqStats.dbg")
    if not args.test:
        curses.wrapper(draw, srvurl, args, queries, debug)
    else:
        test(srvurl, args, queries, debug)


def main():

    promserver = os.environ.get("DM_PROM_SERVER", "http://psmetric03:9090")

    partition = '0'
    hutch     = 'tst'
    start     = None            # Format is 'YYYY-MM-DD hh:mm:ss'

    parser = argparse.ArgumentParser(description='DAQ statistics display')
    parser.add_argument('-p', '--part', help='partition ['+partition+']', type=str, default=partition)
    parser.add_argument('--inst', help='hutch ['+hutch+']', type=str, default=hutch)
    parser.add_argument('--start', help='start time [now]', type=str, default=start)
    parser.add_argument('--debug', help='debug flag', action='store_true')
    parser.add_argument('--test', help='test flag', action='store_true')

    daqStats(promserver, parser.parse_args())


if __name__ == "__main__":

    main()
