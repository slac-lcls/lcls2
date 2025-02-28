#!/usr/bin/env python

"""Test of MDBUtils
"""

import logging
logger = logging.getLogger(__name__)
from psana.pscalib.calib.MDBUtils import *


def get_test_nda():
    """Returns random standard nupmpy array for test purpose."""
    import psana.pyalgos.generic.NDArrGenerators as ag
    return ag.random_standard(shape=(32,185,388), mu=20, sigma=5, dtype=np.float32)

def get_test_dic():
    """Returns dict for test purpose."""
    arr = np.array(range(12))
    arr.shape = (3,4)
    return {'1':1, '5':'super', 'a':arr, 'd':{'c':'name'}}

def get_test_txt():
    """Returns text for test purpose."""
    return '%s\nThis is a string\n to test\ncalibration storage' % gu.str_tstamp()


if __name__ == "__main__":

  TEST_FNAME_PNG = '/reg/g/psdm/detector/data2_test/misc/small_img.png'
  TEST_EXPNAME = 'testexper'
  TEST_DETNAME = 'testdet_1234'

  def test_connect(tname):
    """Connect to host, port get db handls."""
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment='cxid9114', detector='cspad_0001')
        #connect(host=cc.HOST, port=cc.PORT, detector='cspad_0001')


  def test_insert_one(tname):
    """Insert one calibration data in data base."""
    data = None
    if   tname == '1': data, ctype = get_test_txt(), 'testtext'; logger.debug('txt: %s' % str(data))
    elif tname == '2': data, ctype = get_test_nda(), 'testnda';  logger.debug(info_ndarr(data, 'nda'))
    elif tname == '3': data, ctype = get_test_dic(), 'testdict'; logger.debug('dict: %s' % str(data))

    kwa = {'user': gu.get_login(), 'upwd':cc.USERPW}
    t0_sec = int(time())
    insert_constants(data, TEST_EXPNAME, TEST_DETNAME, ctype, 20+int(tname), t0_sec,\
                     time_stamp=_timestamp(t0_sec), **kwa)


  def test_insert_many(tname):
    """Insert many documents in loop"""
    user = gu.get_login()
    kwa = {'user': user}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
        connect(host=cc.HOST, port=cc.PORT, experiment=TEST_EXPNAME, detector=TEST_DETNAME, **kwa)

    t_data = 0
    nloops = 3
    kwa = {'user'     : user,
           'experiment': expname,
           'detector' : detname,
           'ctype'    : 'testnda'}

    for i in range(nloops):
        logger.info('%s\nEntry: %4d' % (50*'_', i))
        data = get_test_nda()
        print_ndarr(data, 'data nda')
        t0_sec = time()
        t0_int = int(t0_sec)
        kwa['run'] = 10 + i
        kwa['time_sec'] = t0_sec
        kwa['time_stamp'] = _timestamp(t0_int)
        id_data_exp, id_data_det, id_exp, id_det = insert_data_and_two_docs(data, fs_exp, fs_det, col_exp, col_det, **kwa)

        dt_sec = time() - t0_sec
        t_data += dt_sec
        logger.info('Insert data in fs of dbs: %s and %s, time %.6f sec '%\
                     (fs_exp._GridFS__database.name, fs_det._GridFS__database.name, dt_sec))

    logger.info('Average time to insert data and two docs: %.6f sec' % (t_data/nloops))


  def test_get_data(tname):
    """Get doc and data"""
    kwa = {'detector': TEST_DETNAME}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwa)

    t0_sec = time()
    data_type='any'
    if tname == '11': data_type='str'
    if tname == '12': data_type='ndarray'

    doc = find_doc(col_det, query={'data_type': data_type})
    logger.info('Find doc time %.6f sec' % (time()-t0_sec))
    logger.info('doc:\n%s' % str(doc))
    print_doc(doc)

    t0_sec = time()
    data = get_data_for_doc(fs_det, doc)
    logger.info('get data time %.6f sec' % (time()-t0_sec))
    s = info_ndarr(data, '', first=0, last=100) if isinstance(data, np.ndarray) else str(data)
    logger.info('data:\n%s' % s)


  #def test_get_data_for_id(tname, det='cspad_0001', data_id='5bbbc6de41ce5546e8959bcf'):
  def test_get_data_for_id(tname, det='cspad_0001', data_id='5bca02bbd1cc55246a67f263'):
    """Get data from GridFS using its id"""
    kwa = {'detector': det}
    client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det = connect(**kwa)
    print('XXX expname', expname)
    print('XXX data_id', data_id)

    out = fs_det.get(ObjectId(data_id))
    print('XXX out: ', out)

    s = out.read()

    print('XXX type(out.read()): ', type(s))
    print('XXX out.read(): ', s[:200])

    nda = np.fromstring(s, dtype=np.float32)
    print_ndarr(nda, 'XXX: nda: ', first=0, last=10)



  def test_database_content(tname, level=3):
    """Print in loop database content"""
    #client, expname, detname, db_exp, db_det, fs_exp, fs_det, col_exp, col_det =\
    #    connect(host=cc.HOST, port=cc.PORT)

    client = connect_to_server(host=cc.HOST, port=cc.PORT, user=cc.USERNAME, upwd=cc.USERPW)

    print('type(client):', type(client))
    print('dir(client):', dir(client))
    logger.info('host:%s\nport:%d' % (client_host(client), client_port(client)))
    dbnames = database_names(client)
    prefix = db_prefixed_name('') # = "cdb_"
    logger.info('databases: %s' % str(dbnames))
    for idb, dbname in enumerate(dbnames):
        db = database(client, dbname) # client[dbname]
        cnames = collection_names_pro(db)
        logger.info('== DB %2d: %12s # cols:%2d' % (idb, dbname, len(cnames)))
        if dbname[:4] != prefix:
            logger.info('     skip non-calib dbname: %s' % dbname)
            continue
        if level==1: continue
        for icol, cname in enumerate(cnames):
            col = collection(db, cname) # or db[cname]
            docs = col.find()
            logger.info('     COL %2d: %12s #docs: %d' % (icol, cname.ljust(12), docs.count()))
            if level==2: continue
            #for idoc, doc in enumerate(docs):
            if docs.count() > 0:
                #logger.info('%s %4d  %s %s' % (10*' ', idoc, doc['time_stamp'], doc['ctype']))
                doc = docs[0]
                logger.info('%s doc[0] %s' % (10*' ', str(doc.keys())))


  def collection_names_pro(db):
     try:
       cnames = collection_names(db)
     except Exception as err:
       cnames = []
       print('  Failed to access collections... err: %s' % str(err))
     return cnames


  def test_dbnames_colnames():
    """Prints the list of DBs and in loop list of collections for each DB."""
    client = connect_to_server(host=cc.HOST, port=cc.PORT, user=cc.USERNAME, upwd=cc.USERPW)
    dbnames = database_names(client)
    print('== client DBs: %s...' % str(dbnames))

    for dbname in dbnames:
        db = database(client, dbname)
        cnames = collection_names_pro(db)
        print('== collections of %s: %s' % (dbname.ljust(20),cnames))
        #if dbname=='config': break


  def test_calib_constants_nda():
    det = 'cspad_0001'
    data, doc = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None)
    print('== doc: %s' % str(doc))
    print_ndarr(data, '== test_calib_constants_nda data', first=0, last=5)


  def test_calib_constants_text():
    #det = 'cspad_0001'
    #data, doc = calib_constants(det, exp='cxic0415', ctype='geometry', run=50, time_sec=None, vers=None)
    #print('==== test_calib_constants_text data:', data)
    #print('==== doc: %s' % str(doc))

    det = 'tmo_quadanode'
    data, doc = calib_constants(det, exp='amox27716', ctype='calibcfg', run=100)
    print('==== test_calib_constants_text data:\n', data)
    print('==== doc: %s' % str(doc))


  def test_print_dict(d, offset='  '):
    """ prints dict content
        re-defined from psana.pscalib.calib.MDBConvertUtils.print_dict
    """
    print('%sprint_dict' % offset)
    for k,v in d.items():
        if isinstance(v, dict): test_print_dict(v, offset = offset+'  ')
        if isinstance(v, np.ndarray): print_ndarr(v, '%sk:%s nda' % (offset,k), first=0, last=5)
        else: print('%sk:%s t:%s v:%s' % (offset, str(k).ljust(10), type(v).__name__, str(v)[:120]))


  def test_calib_constants_dict():
    det = 'opal1000_0059'
    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    #print('==== test_calib_constants_dict data:', data)
    print('==== test_calib_constants_dict type(data):', type(data))
    print('==== doc: %s' % str(doc))
    test_print_dict(data)


  def test_pro_detector_name(shortname='testdet_1234'):
    longname = shortname + '_this_is_insane_long_detector_name_exceeding_55_characters_in_length_or_longer'
    tmode = sys.argv[2] if len(sys.argv) > 2 else '0'
    dname = shortname if tmode=='0' else\
            longname  if tmode=='1' else\
            longname + '_' + tmode # mu._timestamp(int(time()))
    print('==== test_pro_detector_name for detname:', dname)
    name = pro_detector_name(dname)
    print('Returned protected detector name:', name)


  def dict_usage(tname=None):
      d = {'0': 'test_connect <do not forget server password after each command>',
           '1': 'test_insert_one txt',
           '2': 'test_insert_one nda',
           '3': 'test_insert_one dic',
           '4': 'test_insert_many',
           '5': 'test_dbnames_colnames',
           '6': 'test_database_content',
           '7': 'test_calib_constants_nda',
           '8': 'test_calib_constants_text',
           '9': 'test_calib_constants_dict',
           '10': 'test_get_data_for_id',
           '11': 'test_get_data txt',
           '12': 'test_get_data nda',
           '13': 'test_get_data dict',
           '20': 'test_pro_detector_name [test-number=0-short name, 1-fixed long name, n-long name +"_n"]'\
          }
      if tname is None: return d
      return d.get(tname, 'NON IMPEMENTED TEST')


  def usage(tname=None):
    s = '%s\nUsage:' % (50*'_')
    for k,v in dict_usage().items(): s += '\n  %2s: %s' % (k,v)
    print('%s\n%s'%(s,50*'_'))


if __name__ == "__main__":
    #fmt='%(asctime)s %(name)s %(lineno)d %(levelname)s: %(message)s'
    #logging.basicConfig(format=fmt', datefmt='%Y-%m-%dT%H:%M:%S', level=logging.INFO)
    #logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d: %(message)s', level=logging.DEBUG) # logging.INFO
    cc.USERPW = sys.argv[2] if len(sys.argv) > 2 else ''
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    if len(sys.argv) < 2: usage()
    logger.info('%s Test %s %s: %s' % (25*'=', tname, 25*'=', dict_usage(tname)))
    if   tname == '0': test_connect(tname)
    elif tname in ('1','2','3'): test_insert_one(tname)
    elif tname == '4': test_insert_many(tname)
    elif tname == '5': test_dbnames_colnames()
    elif tname == '6': test_database_content(tname)
    elif tname == '7': test_calib_constants_nda()
    elif tname == '8': test_calib_constants_text()
    elif tname == '9': test_calib_constants_dict()
    elif tname =='10': test_get_data_for_id(tname)
    elif tname in ('11','12','13'): test_get_data(tname)
    elif tname =='20': test_pro_detector_name()
    else: logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

# EOF
