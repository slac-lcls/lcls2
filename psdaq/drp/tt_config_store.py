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

    top.set("cl.Application.AppLane1.Prescale.ScratchPad",int(prescaling),'UINT32')
    mycdb.modify_device('BEAM', top)


if __name__ == "__main__":
    write_scratch_pad(sys.argv[1])
