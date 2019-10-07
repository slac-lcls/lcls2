from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import sys


def write_scratch_pad(prescaling):

    create = True
    dbname = 'configDB'
    instrument = 'TMO'
    mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)
    my_dict = mycdb.get_configuration("BEAM","tmotimetool")

    top = cdict()
    top.setInfo('timetool', 'tmotimetool', 'serial1234', 'No comment')
    top.setAlg('timetoolConfig', [0,0,1])

    #There are many rogue fields, but only a handful need to be configured.  what are they?
    #1)  the FIR coefficients.  after accounting for parabolic fitting detection
    #2)  start and stop?
    #3)  main by pass. (still isn't working with FEX in hardware, but does in simulation.)
    #4)  all prescalers.  This has potential to crash linux kernel during hi rate and low prescaling.  How to add protection?  Slow ramp?
    #5)  low pass on background
    #6)  op code (now called readout group). there's no rogue counter part of this yet.
    #7)  the load coefficients bit needs to set to one for the FIR coefficients to be written.
    #8)  camera rate and soft or hard trigger. The soft trigger is for offline testing.  For users or just for me?
    #9)  the batcher bypass so soft trigger doesn't halt.

    ######################################################################
    ####### Keeping it simple.  Just what Giaccomo will need #############
    ######################################################################


    top.set("cl.Application.AppLane1.Prescale.ScratchPad",int(prescaling),'UINT32')
    mycdb.modify_device('BEAM', top)


if __name__ == "__main__":
    write_scratch_pad(sys.argv[1])
