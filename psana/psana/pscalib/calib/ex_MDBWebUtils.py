#!/usr/bin/env python

"""Test of MDBWebUtils"""

import inspect
import logging
logger = logging.getLogger(__name__)
from psana.pscalib.calib.MDBWebUtils import *

if __name__ == "__main__":

  TEST_FNAME_PNG = '/reg/g/psdm/detector/data2_test/misc/small_img.png'
  TEST_EXPNAME = 'testexper'
  TEST_DETNAME = 'testdet_1234'

  def test_database_names():
    print('test_database_names:', database_names())

  def test_collection_names():
    dbname = sys.argv[2] if len(sys.argv) > 2 else 'cdb_cspad_0001'
    print('test_collection_names:', collection_names(dbname))

  def test_find_docs():
    docs = find_docs('cdb_cspad_0001', 'cspad_0001')
    print('find_docs: number of docs found: %d' % len(docs))
    print('test_find_docs returns:', type(docs))
    for i,d in enumerate(docs):
        print('%04d %12s %10s run:%04d time_sec:%10s %s' % (i, d['ctype'], d['experiment'], d['run'], str(d['time_sec']), d['detector']))

    if len(docs)==0: return
    doc0 = docs[0]
    print('doc0 type:', type(doc0))
    print('doc0:', doc0)
    print('doc0.keys():', doc0.keys())

  def test_get_random_doc_and_data_ids(det='cspad_0001'):
    dbname = mu.db_prefixed_name(det)
    colname = det
    doc = find_doc(dbname, colname, query={'ctype':'pedestals'})
    print('Pick up any doc for dbname:%s colname:%s pedestals: ' % (dbname,colname))
    print('Document: %s' % str(doc))
    id_doc  = doc.get('_id', None)
    id_data = doc.get('id_data', None)
    print('_id : %s   id_data : %s' % (id_doc, id_data))
    return id_doc, id_data, dbname, colname

  def test_find_doc():
    #doc = find_doc('cdb_cxic0415', 'cspad_0001', query={'ctype':'pedestals', 'run':{'$lte':40}})
    #print('====> test_find_doc for run: %s' % str(doc))

    #doc = find_doc('cdb_cxid9114', 'cspad_0001', query={'ctype':'pedestals', 'time_sec':{'$lte':1402851400}})
    #print('====> test_find_doc for time_sec: %s' % str(doc))

    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0001')
    _,_,_,_ = test_get_random_doc_and_data_ids(det='cspad_0002')

  def test_get_data_for_id():
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_id(dbname, id_data)
    print('test_get_data_for_id: r.content raw data: %s ...' % str(o[:500]))

  def test_get_data_for_docid():
    id_doc, id_data, dbname, colname = test_get_random_doc_and_data_ids(det='cspad_0001')
    o = get_data_for_docid(dbname, colname, id_doc)
    #o = get_data_for_docid('cdb_cxid9114', 'cspad_0001', '5b6cdde71ead144f115319be')
    print_ndarr(o, 'test_get_data_for_docid o:', first=0, last=10)

  def test_dbnames_collection_query():
    det='cspad_0001'
    db_det, db_exp, colname, query = dbnames_collection_query(det, exp=None, ctype='pedestals', run=50, time_sec=None, vers=None)
    print('test_dbnames_collection_query:', db_det, db_exp, colname, query)

  def test_calib_constants():
    det = 'cspad_0001'
    data, doc = calib_constants('cspad_0001', exp='cxic0415', ctype='pedestals', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print_ndarr(data, '==== test_calib_constants', first=0, last=5)
    print('==== doc: %s' % str(doc))

  def test_calib_constants_text():
    det = 'cspad_0001'
    data, doc = calib_constants(det, exp='cxic0415', ctype='geometry', run=50, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

    det = 'tmo_quadanode'
    data, doc = calib_constants(det, exp='amox27716', ctype='calibcfg', run=100, time_sec=None, vers=None) #, url=cc.URL)
    print('==== test_calib_constants_text data:', data)
    print('==== doc: %s' % str(doc))

  def test_calib_constants_dict():
    det = 'opal1000_0059'
    #data, doc = calib_constants(det, exp='amox23616', ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    data, doc = calib_constants(det, exp=None, ctype='lasingoffreference', run=60, time_sec=None, vers=None)
    print('==== test_calib_constants_dict data:', data)
    print('==== type(data)', type(data))
    print('==== type(doc) ', type(doc))
    print('==== doc: %s' % doc)

  def test_calib_constants_all_types():
    #resp = calib_constants_all_types('tmo_quadanode', exp='amox27716', run=100, time_sec=None, vers=None) #, url=cc.URL)
    #resp = calib_constants_all_types('pnccd_0001', exp='amo86615', run=200, time_sec=None, vers=None) #, url=cc.URL)
    resp = calib_constants_all_types('epixhr2x2_000001', exp='rixx45619', run=200, time_sec=None, vers=None, dbsuffix='mytestdb')
    print('==== test_calib_constants_text data:') #, resp)

    for k,v in resp.items():
        print('ctype:%16s    data and meta:' % k, type(v[0]), type(v[1]))

    import pickle
    s = pickle.dumps(resp)
    print('IF YOU SEE THIS, dict FOR ctypes SHOULD BE pickle-d')

  def test_insert_constants(tname='11', expname=TEST_EXPNAME, detname=TEST_DETNAME, ctype='test_ctype', runnum=10, data='test text sampele'):
    """ Inserts constants using direct MongoDB interface from MDBUtils.
    """
    import psana.pyalgos.generic.Utils as gu

    print('test_delete_database 1:', database_names())
    #txt = '%s\nThis is a string\n to test\ncalibration storage' % gu.str_tstamp()
    #data, ctype = txt, 'testtext'; logger.debug('txt: %s' % str(data))
    #data, ctype = get_test_nda(), 'testnda';  logger.debug(info_ndarr(data, 'nda'))
    #data, ctype = get_test_dic(), 'testdict'; logger.debug('dict: %s' % str(data))

    kwa = {'user': gu.get_login()}
    t0_sec = time()
    ts = gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z', time_sec=t0_sec)
    mu.insert_constants('%s - saved at %s'%(data,ts), expname, detname, ctype, runnum+int(tname), int(t0_sec),\
                        time_stamp=ts, **kwa)
    print('test_delete_database 2:', database_names())

  def test_delete_database(dbname='cdb_testexper'):
    print('test_delete_database %s' % dbname)
    print('test_delete_database BEFORE:', database_names())
    resp = delete_database(dbname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_database AFTER :', database_names())

  def test_delete_collection(dbname='cdb_testexper', colname=TEST_DETNAME):
    print('test_delete_collection %s collection: %s' % (dbname, colname))
    print('test_delete_collection BEFORE:', collection_names(dbname, url=cc.URL))
    resp = delete_collection(dbname, colname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_collection AFTER :', collection_names(dbname, url=cc.URL))

  def test_delete_document(dbname='cdb_testexper', colname=TEST_DETNAME, query={'ctype':'test_ctype'}):
    doc = find_doc(dbname, colname, query=query, url=cc.URL)
    print('find_doc:', doc)
    if doc is None:
        logger.warning('test_delete_document: Non-found document in db:%s col:%s query:%s' % (dbname,colname,str(query)))
        return
    id = doc.get('_id', None)
    print('test_delete_document for doc _id:', id)
    resp = delete_document(dbname, colname, id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_document resp:', resp)

  def test_delete_document_and_data(dbname='cdb_testexper', colname=TEST_DETNAME):
    ldocs = find_docs(dbname, colname, query={}, url=cc.URL)
    if not ldocs:
        print('test_delete_document_and_data db/collection: %s/%s does not have any document' % (dbname, colname))
        return
    doc = ldocs[0]
    print('==== test_delete_document_and_data db/collection: %s/%s contains %d documents\n==== try to delete doc: %s'%\
          (dbname, colname, len(ldocs), str(doc)))
    doc_id = doc.get('_id', None)
    resp = delete_document_and_data(dbname, colname, doc_id, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_delete_document_and_data resp:', resp)

  def test_add_data_from_file(dbname='cdb_testexper', fname=TEST_FNAME_PNG):
    resp = add_data_from_file(dbname, fname, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data_from_file resp: %s of type: %s' % (resp, type(resp)))

  def test_add_data(dbname='cdb_testexper'):
    #data = 'some text is here'
    data = mu.get_test_nda() # np.array(range(12))
    resp = add_data(dbname, data, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('test_add_data: %s\n  to: %s/gridfs/\n  resp: %s' % (str(data), dbname, resp))

  def test_add_document(dbname='cdb_testexper', colname=TEST_DETNAME, doc={'ctype':'test_ctype'}):
    from psana.pyalgos.generic.Utils import str_tstamp
    doc['time_stamp'] = str_tstamp(fmt='%Y-%m-%dT%H:%M:%S%z')
    resp = add_document(dbname, colname, doc, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS)
    print('\ntest_add_document: %s\n  to: %s/%s\n  resp: %s' % (str(doc), dbname, colname, resp))

  def test_add_data_and_two_docs(exp=TEST_EXPNAME, det=TEST_DETNAME):
    from psana.pyalgos.generic.Utils import get_login
    t0_sec = time()
    kwa = {'user'      : get_login(),
           'experiment': exp,
           'detector'  : det,
           'ctype'     : 'testnda',
           'run'       : 123,
           'time_sec'  : t0_sec,
           'time_stamp': mu._timestamp(int(t0_sec)),
          }
    data = mu.get_test_nda()
    res = add_data_and_two_docs(data, exp, det, url=cc.URL_KRB, krbheaders=cc.KRBHEADERS, **kwa)
    if res is None: print('\n!!!! add_data_and_two_docs responce is None')

    print('time to insert data and two docs: %.6f sec' % (time()-t0_sec))

  def test_pro_detector_name():
    shortname = TEST_DETNAME
    longname = shortname + '_this_is_insane_long_detector_name_exceeding_55_characters_in_length_or_longer'
    detname1 = 'epixhremu_00cafe0003-0000000000-0000000000-0000000000-0000000000-0000000000-0000000000'
    tmode = sys.argv[2] if len(sys.argv) > 2 else '0'
    dname = shortname if tmode=='0' else\
            longname  if tmode=='1' else\
            detname1  if tmode=='2' else\
            longname + '_' + tmode # mu._timestamp(int(time()))
    print('==== test_pro_detector_name for detname:', dname)
    name = pro_detector_name(dname)
    print('Long detector name: %s' % dname)
    print('associated in %s with short name: %s' % (cc.DETNAMESDB, name))

  def test_valid_post_privilege():
      for dbname in ('cdb_xpptut15', 'cdb_epix_000001', 'cdb_ueddaq02'):
          print('\n=== test_test_post_privilege for DB: %s' % dbname)
          r = valid_post_privilege(dbname)
          print('     responce: %s' % r)

  def test_collection_info():
    s = collection_info('cdb_cspad_0001', 'cspad_0001')
    print('test_collection_info:\n%s' % str(s))

  def test_tmp():

    from requests import get

    url = 'https://pswww.slac.stanford.edu/calib_ws/cdb_testexper/testdet_1234'

    q1 = {'query_string': '{}'}
    r = get(url, q1).json()
    print('====\n  url  :%s\n  query:%s\n  resp :%s' % (url, str(q1), str(r)))

    doc_id = r[0]['_id']
    print('Selected doc _id:', doc_id, type(doc_id))

    #q2 = {'query_string': u'{"_id":"5eb49463851779a9b1c40966"}'} DOES NOT WORK
    q2 = {'query_string': '{"_id":"ObjectId(%s)"}'%doc_id}
    r = get(url, q2).json()
    print('====\n  url  :%s\n  query:%s\n  resp :%s' % (url, str(q2), str(r)))

    #ldocs = find_docs(dbname, colname, query={})
    #print('==== query={} ldocs:\n', ldocs)

    #doc_id = ldocs[0]['_id']
    #print('==== selected doc _id:', doc_id)

    #ldocs = find_docs(dbname, colname, query={'_id':doc_id})
    #print('==== query={"_id":doc_id}} ldocs:\n', ldocs)

if __name__ == "__main__":

  def test_MDBWebUtils():
    import os
    from psana.pyalgos.generic.NDArrUtils import print_ndarr # info_ndarr, print_ndarr
    global print_ndarr
    logging.basicConfig(format='[%(levelname).1s] L%(lineno)04d : %(message)s', level=logging.DEBUG) # logging.INFO

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.info('%s Test %s %s' % (25*'_',tname, 25*'_'))
    if   tname == '0': test_database_names()
    elif tname == '1': test_collection_names() # [dbname]
    elif tname == '2': test_find_docs()
    elif tname == '3': test_find_doc()
    elif tname == '4': test_get_data_for_id()
    elif tname == '5': test_get_data_for_docid()
    elif tname == '6': test_dbnames_collection_query()
    elif tname == '7': test_calib_constants()
    elif tname == '8': test_calib_constants_text()
    elif tname == '9': test_calib_constants_dict()
    elif tname =='10': test_calib_constants_all_types()
    elif tname =='11': test_insert_constants(tname=tname) # [using direct access methods of MDBUtils]
    elif tname =='12': test_delete_database()
    elif tname =='13': test_delete_collection()
    elif tname =='14': test_delete_document()
    elif tname =='15': test_delete_document_and_data()
    elif tname =='16': test_add_data_from_file()
    elif tname =='17': test_add_data()
    elif tname =='18': test_add_document()
    elif tname =='19': test_add_data_and_two_docs()
    elif tname =='20': test_pro_detector_name() # [test-mode=0-short name, 1-fixed long name, n-long name +"_n"]
    elif tname =='21': test_valid_post_privilege()
    elif tname =='22': test_collection_info()
    elif tname =='00': test_tmp()
    else: logger.info('Not-recognized test name: %s' % tname)
    sys.exit('End of test %s' % tname)

  USAGE = '\nUsage: %s <tname>\n' % sys.argv[0].split('/')[-1]\
      + '\n'.join([s for s in inspect.getsource(test_MDBWebUtils).split('\n') \
                   if "tname ==" in s or 'GWViewImage' in s])  # s[9:]

if __name__ == "__main__":
    print('\n%s\n' % USAGE) # usage())
    test_MDBWebUtils()
# EOF
