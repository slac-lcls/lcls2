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


def make_document(context, doc):
    print('make document')

    socket = context.socket(zmq.SUB)
    socket.connect('tcp://psdev7b:55562')
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    sources = {}
    figures = []
    columns = []
    color_cycle = itertools.cycle(Category10[10])

    formatter = DatetimeTickFormatter(
        seconds=["%H:%M:%S"],
        minsec=["%H:%M:%S"],
        minutes=["%H:%M:%S"],
        hourmin=["%H:%M:%S"],
        hours=["%H:%M:%S"],
        days=["%H:%M:%S"],
        months=["%H:%M:%S"],
        years=["%H:%M:%S"],
    )

    x_range = DataRange1d(follow='end', follow_interval=5*60*1000, range_padding=0)
    legend  = Legend(border_line_color=None, padding=0, location=(0, 0))
    figleg  = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Legend')
    figleg.xaxis.formatter = formatter

    fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Event rate')
    fig.xaxis.formatter  = formatter
    fig.yaxis.axis_label = 'KHz'
    figures.append(fig)
    columns.append('EventCount')

    fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Batch rate')
    fig.xaxis.formatter  = formatter
    fig.yaxis.axis_label = 'KHz'
    figures.append(fig)
    columns.append('BatchCount')

    fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Free batch count')
    fig.xaxis.formatter  = formatter
    fig.yaxis.axis_label = 'Count'
    figures.append(fig)
    columns.append('FreeBatchCnt')

    fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Free epoch count')
    fig.xaxis.formatter  = formatter
    fig.yaxis.axis_label = 'Count'
    figures.append(fig)
    columns.append('FreeEpochCnt')

    fig = figure(x_axis_type='datetime', x_range=x_range, plot_width=400, plot_height=233, title='Free event count')
    fig.xaxis.formatter  = formatter
    fig.yaxis.axis_label = 'Count'
    figures.append(fig)
    columns.append('FreeEventCnt')

    layout = gridplot([[figleg,     None,       None      ],
                       [figures[0], figures[1], None      ],
                       [figures[2], figures[3], figures[4]]])#, sizing_mode='scale_both')
    doc.add_root(layout)
    doc.title = 'Event Builder monitor'

    def update():
        while True:
            socks = dict(poller.poll(timeout=0))
            if not socks:
                break
            hostname, metrics = socket.recv_json()
            #print(hostname, metrics)
            if hostname not in sources:
                data = {'time': [], 'EventCount': [], 'BatchCount': [],
                                    'FreeBatchCnt': [], 'FreeEpochCnt': [], 'FreeEventCnt': []}
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

server  = Server(apps, port=50007, allow_websocket_origin=['pslogin7c:50007'])
server.start()
server.io_loop.add_callback(server.show, '/')
server.io_loop.start()
