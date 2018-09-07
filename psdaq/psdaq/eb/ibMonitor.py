import zmq
import time
import itertools
from functools import partial
from bokeh.plotting import figure
from bokeh.layouts import gridplot
from bokeh.palettes import Category10
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.models.ranges import DataRange1d
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, Legend, LegendItem
from bokeh.application.handlers.function import FunctionHandler
from bokeh.application.handlers.handler import Handler

'''
class MyHandler(Handler):
    def __init__():
        self.context = zmq.Context()

    def on_session_created(self, session_context):

        setattr(session_context, 'socket', dfDict)
'''

counters = [ "excessive_buffer_overrun_errors", #  0
             "link_downed",                     #  1
             "link_error_recovery",             #  2
             "local_link_integrity_errors",     #  3
             "multicast_rcv_packets",           #  4
             "multicast_xmit_packets",          #  5
             "port_rcv_constraint_errors",      #  6
             "port_rcv_data",                   #  7
             "port_rcv_errors",                 #  8
             "port_rcv_packets",                #  9
             "port_rcv_remote_physical_errors", # 10
             "port_rcv_switch_relay_errors",    # 11
             "port_xmit_constraint_errors",     # 12
             "port_xmit_data",                  # 13
             "port_xmit_discards",              # 14
             "port_xmit_packets",               # 15
             "port_xmit_wait",                  # 16
             "symbol_error",                    # 17
             "unicast_rcv_packets",             # 18
             "unicast_xmit_packets",            # 19
             "VL15_dropped" ]                   # 20


def make_document(context, doc):
    print('make document')

    socket = context.socket(zmq.SUB)
    socket.connect('tcp://psdev7b:55560')
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    sources = {}
    figures = []
    columns = []
    color_cycle = itertools.cycle(Category10[10])

    formatter = DatetimeTickFormatter(
        seconds = ["%H:%M:%S"],
        minsec  = ["%H:%M:%S"],
        minutes = ["%H:%M:%S"],
        hourmin = ["%H:%M:%S"],
        hours   = ["%H:%M:%S"],
        days    = ["%H:%M:%S"],
        months  = ["%H:%M:%S"],
        years   = ["%H:%M:%S"],
    )

    x_range = DataRange1d(follow='end', follow_interval=5*60*1000, range_padding=0)
    legend  = Legend(border_line_color=None, padding=0, location=(0, 0))
    figleg  = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Legend')
    figleg.xaxis.formatter = formatter

    for counter in counters:
        fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title=counter)
        fig.xaxis.formatter  = formatter
        fig.yaxis.axis_label = 'Counts/Second'
        figures.append(fig)
        columns.append(counter)

    #figures[7].add_layout(legend, 'above')

    layout = gridplot([[figleg,      None,        None       ],
                       [figures[ 7], figures[ 9], figures[18]],
                       [figures[13], figures[15], figures[19]],
                       [figures[ 4], figures[ 5], figures[16]],
                       [figures[ 6], figures[ 8], figures[10]],
                       [figures[12], figures[14], figures[11]],
                       [figures[ 2], figures[ 3], figures[ 1]],
                       [figures[ 0], figures[17], figures[20]]])#, sizing_mode='scale_both')
    doc.add_root(layout)
    doc.title = 'Infiniband monitor'

    def update():
        while True:
            socks = dict(poller.poll(timeout=0))
            if not socks:
                break
            hostname, metrics = socket.recv_json()
            #print(hostname, metrics)
            if hostname not in sources:
                data = {'time': [],}
                for counter in counters:
                    data[counter] = []
                source = ColumnDataSource(data=data)
                color  = next(color_cycle)
                for i in range(len(figures)):
                    line = figures[i].line(x='time', y=columns[i], source=source,
                                           line_width=1, color=color)
                    #if i == 0:
                    #    legend.items.append(LegendItem(label=hostname, renderers=[line]))
                figleg.line(x=0, y=0, line_width=2, color=color, legend=hostname)
                sources[hostname] = source
                print('new host', hostname)
            # shift timestamp from UTC to current timezone and convert to milliseconds
            metrics['time'] = [(t - time.altzone)*1000 for t in metrics['time']]
            sources[hostname].stream(metrics)

    doc.add_periodic_callback(update, 1000)


context = zmq.Context()
apps    = {'/': Application(FunctionHandler(partial(make_document, context)))}

server  = Server(apps, port=50006, allow_websocket_origin=['pslogin7c:50006'])
server.start()
server.io_loop.add_callback(server.show, '/')
server.io_loop.start()
