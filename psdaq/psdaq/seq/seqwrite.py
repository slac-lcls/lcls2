import json
from psdaq.seq.seq import *

#  The output is only usable by XPM tools
def seq_write_py(instr, output):
    #  Write the python file for direct programming
    fname = output+'.py'
    f = open(fname,'w')
    #    f.write('from tools.seq import *\n')
    f.write('\n')

    for i in instr:
        f.write('{}\n'.format(i))

    f.close()

#  The output is usable by TPG tools
def seq_write_json(name, instr, output):

    seqstr = 'from psdaq.seq.seq import *\n'
    for i in instr:
        seqstr += f'{i}\n'

    config = {'title':'TITLE', 'descset':None, 'instrset':None, 'seqcodes':None}
    exec(compile(seqstr, name, 'exec'), {}, config)

    instrset = preproc(config['instrset'], isTPG=True)

    encoding = [len(instrset)]
    for i in instrset:
        encoding = encoding + i.encoding()

    #  Populate a new dictionary with only the fields we want
    cc = {'title'   :name,
          'descset' :None,
          'encoding':encoding}

    open(output+'.json','w').write(json.dumps(cc))

