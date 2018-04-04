"""
Comparison of original xtcav calibration constants 
with the same constants saved and retrieved in CDB.
"""
import sys
from psana.pscalib.calib.XtcavConstants import Load, dict_from_xtcav_calib_object, compare_dicts
import psana.pscalib.calib.MDBUtils as dbu
from psana.pscalib.calib.CalibUtils import parse_calib_file_name #history_dict_for_file, history_list_of_dicts
from psana.pscalib.calib.MDBConvertLCLS1 import detname_conversion
from psana.pscalib.calib.CalibConstants import HOST, PORT

#------------------------------

def usage(fname) :
    msg = 'Usage: python lcls2/psana/psana/pscalib/examples/ex07-cdb-xtcav.py [<full-path-to-calib-file>]'\
          '\n       by default calib file is %s' % fname
    print(msg)

#------------------------------

def test_xtcav_calib_constants(fname=
    '/reg/d/psdm/XPP/xpptut15/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/101-102.data') :

    _, exp, _, cvers, detname, ctype, cfname = fname.rsplit('/',6) 
    resp = parse_calib_file_name(cfname)
    begin, end, ext = resp if resp is not None else (None, None, None)
    det = detname_conversion(detname)
    run = begin
    dbname_exp = dbu.db_prefixed_name(exp)
    dbname_det = dbu.db_prefixed_name(det)

    print('LCLS1 Xtcav calibration file: %s' % fname)
    print('Parameters form path: exp:%s det:%s ctype:%s run:%s dbname_exp:%s dbname_det:%s'%\
          (exp, det, ctype, run, dbname_exp, dbname_det))

    #Save(ct,fname)
    o1 = Load(fname)
    d1 = dict_from_xtcav_calib_object(o1)
    print('Xtcav calibration constants as dict:\n', d1)


    #---- Delete databases for experiment and detector

    client = dbu.connect_to_server(HOST, PORT)
    print('Open client on host:%s port:%s' % (HOST, PORT))

    print('Delete database %s'% dbname_exp)
    dbu.delete_database(client, dbname_exp)

    print('Delete database %s'% dbname_det)
    dbu.delete_database(client, dbname_det)

    #---- Add data to experiment and detector dbs
    print('Add Xtcav constants') 

    kwargs = {'host' : HOST,\
              'port' : PORT,\
              'version' : 'V01',\
              'comment' : 'test of add-retrieve xtcav constants'
             }
    #insert_calib_data(data, *kwargs)
    dbu.insert_constants(o1, exp, det, ctype, run, time_sec='1000000000', **kwargs)

    #msg = dbu.database_info(client, dbname_exp, level=10)
    #print(msg)

    print('Xtcav constants inserted, now retrieve them from db:%s collection:%s' % (dbname_exp, det))

    db, fs = dbu.db_and_fs(client, dbname_exp)
    col = dbu.collection(db, det)

    #for doc in col.find() :
    #    print(doc)

    doc = dbu.find_doc(col, query={'ctype':ctype, 'run':run})
    print('Found doc:\n', doc)

    o2 = dbu.get_data_for_doc(fs, doc)
    d2 = dict_from_xtcav_calib_object(o2)
    print('Xtcav calib object converted to dict:\n', d2)

    #print('cmp(d1,d2) :', str(d1==d2))

    print('\nCompare dictionaries for Xtcav calib objects loaded directly from calib file and passed through the CDB')
    compare_dicts(d1,d2)

    client.close()
    return


#------------------------------

if __name__ == "__main__":
    path='/reg/d/psdm/XPP/xpptut15/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/101-102.data'
    #path='/reg/d/psdm/XCS/xcsm9816/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/lasingoffreference/31-end.data'
    #path='/reg/d/psdm/XCS/xcsm9816/calib/Xtcav::CalibV1/XrayTransportDiagnostic.0:Opal1000.0/pedestals/30-end.data'

    fname = sys.argv[1] if len(sys.argv) > 1 else path

    test_xtcav_calib_constants(fname)
    usage(path)

#------------------------------
