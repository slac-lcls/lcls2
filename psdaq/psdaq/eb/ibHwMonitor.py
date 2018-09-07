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

counters = [ "lifespan",                    #  0
             "num_cqovf",                   #  1
             "rq_num_dup",                  #  2
             "rq_num_lle",                  #  3
             "rq_num_lpe",                  #  4
             "rq_num_lqpoe",                #  5
             "rq_num_oos",                  #  6
             "rq_num_rae",                  #  7
             "rq_num_rire",                 #  8
             "rq_num_rnr",                  #  9
             "rq_num_udsdprd",              # 10
             "rq_num_wrfe",                 # 11
             "sq_num_bre",                  # 12
             "sq_num_lle",                  # 13
             "sq_num_lpe",                  # 14
             "sq_num_lqpoe",                # 15
             "sq_num_mwbe",                 # 16
             "sq_num_oos",                  # 17
             "sq_num_rae",                  # 18
             "sq_num_rire",                 # 19
             "sq_num_rnr",                  # 20
             "sq_num_roe",                  # 21
             "sq_num_rree",                 # 22
             "sq_num_to",                   # 23
             "sq_num_tree",                 # 24
             "sq_num_wrfe" ]                # 25

def make_document(context, doc):
    print('make document')

    socket = context.socket(zmq.SUB)
    socket.connect('tcp://psdev7b:55566')
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
                       [figures[ 0], figures[ 1], figures[ 2]],
                       [figures[ 3], figures[ 4], figures[ 5]],
                       [figures[ 6], figures[ 7], figures[ 8]],
                       [figures[ 9], figures[10], figures[11]],
                       [figures[12], figures[13], figures[14]],
                       [figures[15], figures[16], figures[17]],
                       [figures[18], figures[19], figures[20]],
                       [figures[21], figures[22], figures[23]],
                       [figures[24], figures[25]             ]])#, sizing_mode='scale_both')
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

server  = Server(apps, port=50008, allow_websocket_origin=['pslogin7c:50008'])
server.start()
server.io_loop.add_callback(server.show, '/')
server.io_loop.start()
