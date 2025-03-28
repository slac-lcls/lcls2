
"""
Usage ::

    # Import
    import psana.pscalib.calib.MDBUtils as mu

    ts    = mu._timestamp(time_sec:int,int,float)
    ts    = mu.timestamp_id(id)
    ts    = mu.timestamp_doc(doc)
    t, ts = mu.time_and_timestamp(**kwa)
    etc.

    --- 2023-11-02: grep def psana/psana/pscalib/calib/MDBUtils.py

    is_valid_type(pname, o, otype):
    is_valid_dbname(dbname):
    is_valid_cname(cname):
    is_valid_objectid(id):
    is_valid_time_sec(time_sec):
    db_prefixed_name(name, prefix=cc.DBNAME_PREFIX):
    get_dbname(**kwa):
    get_colname(**kwa):
    _timestamp(time_sec):
    timestamp_id(id): # e.g. id=5b6cde201ead14514d1301f1 or ObjectId
    timestamp_doc(doc):
    time_and_timestamp(**kwa):
    docdic(data, dataid, **kwa):
    doc_add_id_ts(doc):
    doc_info(doc, fmt='\n  %16s: %s'):
    doc_keys_info(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars'), fmt='  %s: %s'):
    print_doc(doc):
    print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars')):
    encode_data(data):
    _error_msg(msg):
    valid_experiment(experiment):
    valid_detector(detector):
    valid_ctype(ctype):
    valid_run(run):
    valid_version(version):
    valid_comment(comment):
    valid_data(data, detector, ctype):
    exec_command(cmd):
    dict_from_data_string(s):
    object_from_data_string(s, doc):
    get_data_for_doc(fs, doc):
    dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dtype=None):
    document_keys(doc):
    document_info(doc, keys=('time_sec','time_stamp','experiment',\
    request_confirmation(msg=''):
    out_fname_prefix(fmt='clb-%s-%s-r%04d-%s', **kwa):
    save_doc_and_data_in_file(doc, data, prefix, control={'data': True, 'meta': True}):
    data_from_file(fname, ctype, dtype, verb=False):
    _doc_detector_name(detname, dettype, detnum):
    _short_for_partial_name(detname, ldocs):
    _pro_detector_name(detname, maxsize=cc.MAX_DETNAME_SIZE, add_shortname=False)

    ...
    2023-11-02 remove all methods directly depending on pymongo
"""

import logging
logger = logging.getLogger(__name__)

import pickle

import os
import sys
from time import time
import json

import numpy as np
import psana.pyalgos.generic.Utils as gu
from   psana.pyalgos.generic.NDArrUtils import print_ndarr, info_ndarr
import psana.pscalib.calib.CalibConstants as cc

from bson.objectid import ObjectId
from psana.pscalib.calib.Time import Time

ASCENDING  = 1  # pymongo.ASCENDING
DESCENDING =-1
TSFORMAT = cc.TSFORMAT #'%Y-%m-%dT%H:%M:%S%z' # e.g. 2018-02-07T09:11:09-0800

def is_valid_type(pname, o, otype):
    if isinstance(o, otype): return True
    logger.warning('parameter "%s" type "%s" IS NOT %s' % (pname, str(o), str(otype)))
    return False

def is_valid_dbname(dbname):
    return is_valid_type('dbname', dbname, str)

def is_valid_cname(cname):
    return is_valid_type('cname', cname, str)

def is_valid_objectid(id):
    return is_valid_type('id', id, ObjectId)

def is_valid_time_sec(time_sec):
    return is_valid_type('time_sec', time_sec, int)

def db_prefixed_name(name, prefix=cc.DBNAME_PREFIX):
    """Returns database name with prefix, e.g. name='exp12345' -> 'cdb_exp12345'."""
    if name is None: return None
    assert isinstance(name,str), 'db_prefixed_name parameter should be str'
    nchars = len(name)
    assert nchars < 128, 'name length should be <128 characters'
    dbname = '%s%s' % (prefix, name)
    logger.debug('db_prefixed_name: %s' % dbname)
    return dbname

def get_dbname(**kwa):
    """Returns (str) dbname or None.
       Implements logics for dbname selection:
       -- dbname is used if defined else
       -- prefixed experiment else
       -- prefixed detector else None
    """
    exp    = kwa.get('experiment', None)
    det    = kwa.get('detector', None)
    dbname = kwa.get('dbname', None)
    mode   = kwa.get('cli_mode', None)
    if dbname is None:
        name = exp if not (exp is None) else det
        if name is None:
            if mode != 'print':
                logger.warning('dbname, experiment, and detector name are NOT SPECIFIED')
            return None
        dbname = db_prefixed_name(name)
    return dbname

def get_colname(**kwa):
    """Returns (str) collection name or None.
       Implements logics for collection selection:
       -- dbname is not defined returns None
       -- colname is defined returns colname
       -- returns detector name, it might be None
    """
    dbname = get_dbname(**kwa)
    if dbname is None: return None
    colname = kwa.get('colname', None)
    if colname is not None: return colname
    return kwa.get('detector', None)

def _timestamp(time_sec):
    """Converts time_sec in timestamp of adopted format TSFORMAT."""
    if not is_valid_time_sec(time_sec): return None
    return gu.str_tstamp(TSFORMAT, int(time_sec))

def timestamp_id(id): # e.g. id=5b6cde201ead14514d1301f1 or ObjectId
    """Converts MongoDB (str) id to (str) timestamp of adopted format."""
    oid = id
    if isinstance(id, str):
        if len(id) != 24: return str(id) # protection aginst non-valid id
        oid = ObjectId(id)
    if isinstance(oid, ObjectId):
        str_ts = str(oid.generation_time) # '2018-03-14 21:59:37+00:00'
        tobj = Time.parse(str_ts)         # Time object from parsed string
        tsec = int(tobj.sec())            # 1521064777
        str_tsf = _timestamp(tsec)        # re-formatted time stamp
        return str_tsf
    return str(id) # protection aginst non-valid id

def timestamp_doc(doc):
    """Returns document creation (str) timestamp from its id."""
    return timestamp_id(doc['_id'])

def time_and_timestamp(**kwa):
    """Returns "time_sec" and "time_stamp" from **kwa.
       If one of these parameters is missing, another is reconstructed from available one.
       If both missing - current time is used.
    """
    time_sec   = kwa.get('time_sec', None)
    time_stamp = kwa.get('time_stamp', None)
    if time_sec is not None:
        time_sec = int(float(time_sec))
        assert isinstance(time_sec, int) , 'time_and_timestamp - parameter time_sec should be int'
        assert 0 < time_sec < 5000000000,  'time_and_timestamp - parameter time_sec should be in allowed range'
        if time_stamp is None:
            time_stamp = gu.str_tstamp(TSFORMAT, time_sec)
    else:
        if time_stamp is None:
            time_sec_str, time_stamp = gu.time_and_stamp(TSFORMAT)
        else:
            time_sec_str = gu.time_sec_from_stamp(TSFORMAT, time_stamp)
        time_sec = int(time_sec_str)
    return time_sec, time_stamp

def docdic(data, dataid, **kwa):
    """Returns dictionary for db document in style of JSON object."""
    dic_extpars = kwa.get('extpars', {})
    if isinstance(dic_extpars, str): dic_extpars = json.loads(dic_extpars)
    dic_extpars.setdefault('content', 'extended parameters dict->json->str')
    dic_extpars.setdefault('command', ' '.join(sys.argv))
    str_extpars = str(json.dumps(dic_extpars))
    vers = kwa.get('version', None)
    undef = 'undefined'
    doc = {
          'experiment': kwa.get('experiment', undef),
          'run'       : kwa.get('run', 0),
          'run_end'   : kwa.get('run_end', 'end'),
          'detector'  : kwa.get('detector', undef),
          'ctype'     : kwa.get('ctype', undef),
          'dtype'     : kwa.get('dtype', undef),
          'time_sec'  : int(kwa.get('time_sec', 1000000000)),
          'time_stamp': kwa.get('time_stamp', '2001-09-09T01:46:40-0000'),
          'version'   : vers if vers is not None else 'V2021-10-08',
          'comment'   : kwa.get('comment', 'no comment'),
          'uid'       : gu.get_login(),
          'host'      : gu.get_hostname(),
          'cwd'       : gu.get_cwd(),
          'id_data'   : dataid,
          'id_doc_exp': kwa.get('id_doc_exp', 0),
          'id_data_exp':kwa.get('id_data_exp', 0),
          'detname'   : kwa.get('detname', ''),
          'longname'  : kwa.get('longname', ''),
          'shortname' : kwa.get('shortname', ''),
          'dettype'   : kwa.get('dettype', ''),
          'run_orig'  : kwa.get('run_orig', 0),
          'tstamp_orig':kwa.get('tstamp_orig', ''),
          'iofname'   : kwa.get('iofname', ''),
          'extpars'   : str_extpars,
         }
    if isinstance(data, np.ndarray):
        doc['data_type']  = 'ndarray'
        doc['data_dtype'] = str(data.dtype)
        doc['data_size']  = '%d' % data.size
        doc['data_ndim']  = '%d' % data.ndim
        doc['data_shape'] = str(data.shape)
    elif isinstance(data, str):
        doc['data_type']  = 'str'
        doc['data_size']  = '%d' % len(data)
    else:
        doc['data_type']  = 'any'
    logger.debug('doc data type: %s' % doc['data_type'])
    return doc

def doc_add_id_ts(doc):
    """add items with timestamp for id-s as '_id_ts', 'id_data_ts', 'id_exp_ts'"""
    for k in ('_id', 'id_data', 'id_exp'):
        v = doc.get(k, None)
        if v is not None: doc['%s_ts'%k] = timestamp_id(v)

def doc_info(doc, fmt='\n  %16s: %s'):
    s = 'Data document attributes'
    if doc is None: return '%s\n   doc_info: Data document is None...' % s
    for k,v in doc.items(): s += fmt % (k,v)
    return s

def doc_keys_info(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars'), fmt='  %s: %s'):
    s = ''
    for k in keys: s += fmt % (k, doc.get(k,'N/A'))
    return s

def print_doc(doc):
    print(doc_info(doc))

def print_doc_keys(doc, keys=('run', 'time_stamp', 'data_size', 'id_data', 'extpars')):
    print(doc_keys_info(doc, keys))

def encode_data(data):
    """Converts any data type into octal string to save in gridfs."""
    s = None
    if   isinstance(data, np.ndarray): s = data.tobytes()
    elif isinstance(data, str):        s = str.encode(data)
    else:
        logger.warning('DATA TYPE "%s" IS NOT "str" OR "numpy.ndarray" CONVERTED BY pickle.dumps ...'%\
                       type(data).__name__)
        s = pickle.dumps(data)
    return s

def _error_msg(msg):
    return 'wrong parameter %s' % msg

def valid_experiment(experiment):
    assert isinstance(experiment,str), _error_msg('type')
    assert 7 < len(experiment) < 10, _error_msg('length')

def valid_detector(detector):
    assert isinstance(detector,str), _error_msg('type')
    assert 1 < len(detector) < 65, _error_msg('length')

def valid_ctype(ctype):
    assert isinstance(ctype,str), _error_msg('type')
    assert 4 < len(ctype) < 32, _error_msg('length')

def valid_run(run):
    assert isinstance(run,int), _error_msg('type')
    assert -1 < run < 10000, _error_msg('value')

def valid_version(version):
    assert isinstance(version,str), _error_msg('type')
    assert len(version) < 128, _error_msg('length')

def valid_comment(comment):
    assert isinstance(comment,str), _error_msg('type')
    assert len(comment) < 1000000, _error_msg('length')

def valid_data(data, detector, ctype):
    pass

def exec_command(cmd):
    from psana.pscalib.proc.SubprocUtils import subproc
    logger.debug('Execute shell command: %s' % cmd)
    if not gu.shell_command_is_available(cmd.split()[0], verb=True): return
    out,err = subproc(cmd, env=None, shell=False, do_wait=True)
    if out or err:
        logger.warning('err: %s\nout: %s' % (err,out))

def dict_from_data_string(s):
    import ast
    try:
        d = ast.literal_eval(s) # retreive dict from str
    except Exception as err:
        d = None
        logger.error('ast.literal_eval("%s") err:\n%s' % (s,err))
    if not isinstance(d, dict):
        logger.debug('dict_from_data_string: literal_eval returns type: %s which is not "dict"' % type(d))
        return None
    from psana.pscalib.calib.MDBConvertUtils import deserialize_dict
    deserialize_dict(d)     # deserialize dict values
    return d

def object_from_data_string(s, doc):
    """Returns str, ndarray, or dict"""
    data_type = doc.get('data_type', None)
    if data_type is None:
        logger.warning('object_from_data_string: data_type is None in the doc: %s' % str(doc))
        return None
    logger.debug('object_from_data_string: %s' % data_type)
    if data_type == 'str':
        data = s.decode()
        if doc.get('ctype', None) in ('xtcav_lasingoff', 'xtcav_pedestals', 'lasingoffreference'): # 'pedestals'):
            return dict_from_data_string(data)
        return data
    elif data_type == 'ndarray':
        str_dtype = doc.get('data_dtype', None)
        nda = np.frombuffer(s, dtype=str_dtype)
        nda.shape = eval(doc.get('data_shape', None)) # eval converts string shape to tuple
        return nda
    elif data_type == 'any':
        import pickle
        return pickle.loads(s)
    else:
        logger.warning('get_data_for_doc: UNEXPECTED data_type: %s' % data_type)
        return None

def get_data_for_doc(fs, doc):
    """Returns data referred by the document."""
    if not is_valid_fs(fs):
        logger.warning('Document %s\n  associated DATA IS NOT AVAILABLE in fs %s' % (str(doc), str(fs)))
        return None
    if doc is None:
        logger.warning('get_data_for_doc: Data document is None...')
        return None
    idd = doc.get('id_data', None)
    if idd is None:
        logger.debug("get_data_for_doc: key 'id_data' is missing in selected document...")
        return None
    idd = ObjectId(idd)
    if not fs.exists(idd):
        logger.debug("get_data_for_doc: NON EXISTENT fs data for data_id %s" % str(idd))
        return None
    out = fs.get(idd)
    s = out.read()
    return object_from_data_string(s, doc)

def dbnames_collection_query(det, exp=None, ctype='pedestals', run=None, time_sec=None, vers=None, dtype=None):
    """Returns dbnames for detector, experiment, collection name, and query."""
    cond = (run is not None) or (time_sec is not None) or (vers is not None)
    assert cond, 'Not sufficeint info for query: run, time_sec, and vers are None'
    _det = _pro_detector_name(det)
    query={'detector':_det,} # 'ctype':ctype}
    if ctype is not None: query['ctype'] = ctype
    if dtype is not None: query['dtype'] = dtype
    runq = run if not(run in (0,None)) else 9999 # by cpo request on 2020-01-16
    query['run'] = {'$lte': runq} #query['run_end'] = {'$gte': runq}
    if time_sec is not None: query['time_sec'] = {'$lte': int(time_sec)}
    if vers is not None: query['version'] = vers
    logger.debug('query: %s' % str(query))
    db_det, db_exp = db_prefixed_name(_det), db_prefixed_name(str(exp))
    if None in (db_det, db_exp):
        logger.debug('WARNING: dbnames_collection_query: db_det:%s db_exp:%s' % (db_det, db_exp))
        return None,None,None,None
    if 'None' in db_det: db_det = None
    if 'None' in db_exp: db_exp = None
    return db_det, db_exp, _det, query

def document_keys(doc):
    """Returns formatted strings of document keys."""
    keys = sorted(doc.keys())
    s = '%d document keys:' % len(keys)
    for i,k in enumerate(keys):
        if not(i%5): s += '\n      '
        s += ' %s' % k.ljust(16)
    return s

def document_info(doc, keys=('time_sec','time_stamp','experiment',\
                  'detector','ctype','run','id_data','id_data_ts', 'data_type','data_dtype','version'),\
                  fmt='%10s %24s %11s %16s %12s %4s %24s %24s %10s %10s %7s'):
    """Returns (str, str) for formatted document values and title made of keys."""
    id_data = str(doc.get('id_data',None))
    doc['id_data_ts'] = timestamp_id(id_data)
    doc_keys = sorted(doc.keys())
    if 'experiment' in doc_keys: # CDDB type of document
        vals = tuple([str(doc.get(k,None)) for k in keys])
        return fmt % vals, fmt % keys
    else: # OTHER type of document
        title = '  '.join(doc_keys)
        vals = tuple([str(doc.get(k,None) if k != 'data' else '<some data>') for k in doc_keys])
        info = '  '.join(vals)
        return info, title

def request_confirmation(msg=''):
    """Dumps request for confirmation of specified (delete) action."""
    logger.warning(msg+'Use confirm "-C" option to proceed with request.')

def out_fname_prefix(fmt='clb-%s-%s-r%04d-%s', **kwa):
    """Returns output file name prefix like doc-cxid9114-cspad_0001-r0116-pixel_rms"""
    exp = kwa.get('experiment', 'exp')
    det = kwa.get('detector', 'det')
    _det = _pro_detector_name(det)
    run = int(kwa.get('run', 0))
    ctype = kwa.get('ctype', 'ctype')
    return fmt % (exp, _det, run, ctype)

def save_doc_and_data_in_file(doc, data, prefix, control={'data': True, 'meta': True}):
    """Saves document and associated data in files."""
    msg = '\n'.join(['%12s: %s' % (k,doc[k]) for k in sorted(doc.keys())])
    data_type = doc.get('data_type', None)
    ctype     = doc.get('ctype', None)
    dtype     = doc.get('dtype', None)
    verb      = doc.get('vebous', False)
    logger.debug('Save in file(s) "%s" data and document metadata:\n%s' % (prefix, msg))
    #logger.debug(info_ndarr(data, 'data', first=0, last=100))
    logger.debug('save_doc data_type:%s ctype:%s type(data):%s' % (data_type, ctype, type(data).__name__))
    if control['data']:
        fname = '%s.data' % prefix
        if data_type=='ndarray':
            from psana.pscalib.calib.NDArrIO import save_txt # load_txt
            save_txt(fname, data, cmts=(), fmt='%.3f')
            logger.info('saved file: %s' % fname)
            fname = '%s.npy' % prefix
            np.save(fname, data, allow_pickle=False)
        elif ctype == 'geometry':
            gu.save_textfile(data, fname, mode='w', verb=verb)
        elif data_type=='str' and (ctype in ('lasingoffreference', 'pedestals')):
            logger.info('save_doc XTCAV IS RECOGNIZED ctype "%s"' % ctype)
            from psana.pscalib.calib.MDBConvertUtils import serialize_dict
            s = dict(data)
            serialize_dict(s)
            gu.save_textfile(str(s), fname, mode='w', verb=verb)
        elif dtype in ('pkl', 'pickle'):
            gu.save_pickle(data, fname, mode='wb')
        elif dtype == 'json':
            gu.save_json(data, fname, mode='w')
        elif data_type == 'any':
            gu.save_textfile(str(data), fname, mode='w', verb=verb)
        else:
            gu.save_textfile(str(data), fname, mode='w', verb=verb)
        logger.info('saved file: %s' % fname)
    if control['meta']:
        fname = '%s.meta' % prefix
        gu.save_textfile(msg, fname, mode='w', verb=verb)
        logger.info('saved file: %s' % fname)

def data_from_file(fname, ctype, dtype, verb=False):
    """Returns data object loaded from file."""
    from psana.pscalib.calib.NDArrIO import load_txt
    assert os.path.exists(fname), 'File "%s" DOES NOT EXIST' % fname
    if dtype == 'xtcav':
        from psana.pscalib.calib.XtcavUtils import load_xtcav_calib_file
    ext = os.path.splitext(fname)[-1]
    data = gu.load_textfile(fname, verb=verb) if ctype == 'geometry' or dtype in ('str', 'txt', 'text') else\
           load_xtcav_calib_file(fname)       if dtype == 'xtcav' else\
           np.load(fname)                     if ext == '.npy' else\
           gu.load_json(fname)                if ext == '.json' or dtype == 'json' else\
           gu.load_pickle(fname)              if ext == '.pkl' or dtype in ('pkl', 'pickle') else\
           load_txt(fname) # input NDArrIO
    logger.debug('fname:%s ctype:%s dtype:%s verb:%s data:\n%s\n...'%\
                 (fname, ctype, dtype, verb, str(data)[:1000]))
    return data

# 2020-05-11

def _doc_detector_name(detname, dettype, detnum):
    """Returns (dict) document for Detector Name Database (for long <detname> to short <dettype-detnum>)."""
    t0_sec = time()
    return {'long'      : detname,\
            'short'     : '%s_%06d'%(dettype, detnum),\
            'seqnumber' : detnum,\
            'uid'       : gu.get_login(),
            'host'      : gu.get_hostname(),
            'cwd'       : gu.get_cwd(),
            'time_sec'  : t0_sec,
            'time_stamp': _timestamp(int(t0_sec))
           }

def _short_for_partial_name(detname, ldocs):
    """Returns full from partial detector name or None if not found.
       Parameters:
       - detname (str) - partial or full detector name, e.g. <dettype>_<panalN-id>_<panalM-id>_...
       - ldocs (list of dict) - list of documents returning after submitting quiery to cdb_detnames DB
    """
    logger.debug('detname: %s\nlist of docs: %s'%(detname, str(ldocs)))
    name_fields = detname.split('_')
    if len(name_fields)<2:
        logger.warning('Partial detname %s does not cantain enough fields to find long name.' %  detname)
        return None
    #dettype = name_fields[0]
    pnames = name_fields[1:]
    for doc in ldocs:
        longname = doc.get('long', None)
        if longname is None: continue
        if all([name in longname for name in pnames]):
            shortname = doc.get('short', None)
            logger.debug('found associated detector names\n  long: %s\n  short: %s'%(longname, shortname))
            return shortname # longname
    return None

#def pro_detector_name(detname, maxsize=cc.MAX_DETNAME_SIZE, add_shortname=False):
#    """ Returns short detector name if its length exceeds cc.MAX_DETNAME_SIZE chars."""
#    if detname is None:
#        logger.debug('WARNING: pro_detector_name: input detname is None')
#        return None
#    return detname if len(detname)<maxsize else _short_detector_name(detname, add_shortname=add_shortname)

def _pro_detector_name(detname, maxsize=cc.MAX_DETNAME_SIZE, add_shortname=False):
    #import psana.pscalib.calib.MDBWebUtils as mwu
    from psana.pscalib.calib.MDBWebUtils import pro_detector_name
    return pro_detector_name(detname, maxsize, add_shortname)


if __name__ == "__main__":
    sys.exit('\nFor test use ./ex_%s <test-number> <mode> <...>' % sys.argv[0].rsplit('/')[-1])

# EOF
