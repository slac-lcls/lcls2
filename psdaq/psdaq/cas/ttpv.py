#!/usr/bin/env python3
from textwrap import dedent

from caproto.server import PVGroup, ioc_arg_parser, pvproperty, run


class SimpleIOC(PVGroup):
    """
    An IOC with one read/writable array PV.

    Array PVs
    ---------
    C (array of int)
    """
    TTALL = pvproperty(
        value=[1, 2, 3, 4, 5, 6],
        doc='An array of integers (max length 6)'
    )


if __name__ == '__main__':
    ioc_options, run_options = ioc_arg_parser(
        default_prefix='simple:',
        desc=dedent(SimpleIOC.__doc__))
    ioc = SimpleIOC(**ioc_options)
    run(ioc.pvdb, **run_options)
