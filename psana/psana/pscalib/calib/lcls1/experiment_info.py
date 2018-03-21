#--------------------------------------------------------------------------
# File and Version Information:
#  $Id:$
#
# Description:
#  The API for inquering various information about instruments and
#  experiments registered at PCDS.
#
#------------------------------------------------------------------------

"""
The API for inquering various information about instruments and
experiments registered at PCDS.

This software was developed for the LCLS project.  If you use all or
part of it, please give an appropriate acknowledgment.

@version $Id:$

@author Igor Gaponenko
"""

#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision:$"
# $Source:$

#--------------------------------
#  Imports of standard modules --
#--------------------------------

import sys
import time

#-----------------------------
# Imports for other modules --
#-----------------------------

import MySQLdb as db

__host   = 'psdb.slac.stanford.edu'
__user   = 'regdb_reader'
__passwd = ''
__db     = 'regdb'

# ------------------------------------------------------------------------
# Connect to MySQL server and execute the specified SELECT statement which
# is supposed to return a single row (if it return more then simply ignore
# anything before the first one). Return result as a dictionary. Otherwise return None.
#
#   NOTE: this method won't catch MySQL exceptions. It's up to
#         the caller to do so. See the example below:
#
#           try:
#               result = __do_select('SELECT...')
#           except db.Error, e:
#               print 'MySQL connection failed: '.str(e)
#               ...
#
# ------------------------------------------------------------------------------

__connection = None

def __get_connection():
    global __connection
    if __connection is None: __connection = db.connect(host=__host, user=__user, passwd=__passwd, db=__db)
    return __connection

def __escape_string(str):

    return __get_connection().escape_string(str)

def __do_select_many(statement):

    cursor = __get_connection().cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    return cursor.fetchall()

def __do_select(statement):

    rows = __do_select_many(statement)
    if not rows : return None

    return rows[0]

# ------------------------------------------------------------------------------------------
# Execute any SQL statement which doesn't return a result set
#
# Notes:
# - exceptions are thrown exactly as explained for the previously defined method __do_select
# - the statement will be surrounded by BEGIN and COMMIT transaction statements
# ------------------------------------------------------------------------------------------

def __do_sql(statement):

    cursor = __get_connection().cursor(db.cursors.SSDictCursor)
    cursor.execute("SET SESSION SQL_MODE='ANSI'")
    cursor.execute("BEGIN")
    cursor.execute(statement)
    cursor.execute("COMMIT")

# ------------------------------------------------------------
# Return the current time expressed in nanoseconds. The result
# will be packed into a 64-bit number.
# ------------------------------------------------------------

def __now_64():
    t = time.time()
    sec = int(t)
    nsec = int(( t - sec ) * 1e9 )
    return sec*1000000000L + nsec

# ---------------------------------------------------------------------
# Look for an experiment with specified identifier and obtain its name.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def id2name(id):

    row = __do_select("SELECT name FROM experiment WHERE id=%s" % id)
    if not row : return None
    return row['name']

# ---------------------------------------------------------------------
# Look for an experiment with specified identifier.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

def getexp(id):

    row = __do_select("SELECT * FROM experiment WHERE id=%s" % id)
    return row

# -----------------------------------------------------------------------------
# Look for an experiment with specified name and obtain its numeric identifier.
# Return None if no such experiment exists in the database.
# ------------------------------------------------------------------------------

__name2id = dict()

def name2id(name):

    global __name2id

    if name not in __name2id:
        row = __do_select("SELECT id FROM experiment WHERE name='%s'" % name)
        if not row : return None
        __name2id[name] = int(row['id'])

    return __name2id[name]


# ---------------------------------------------------------------
# Return a numeric identifier of the experiment's run. If none is
# found the function will return None
# ---------------------------------------------------------------

__run2id = dict()

def run2id(exper_name, runnum):

    exper_id = name2id(exper_name)
    if exper_id is None: return None

    global __run2id

    if exper_id not in __run2id:
        __run2id[exper_id] = dict()

    if runnum not in __run2id[exper_id]:
        row = __do_select("SELECT id from logbook.run WHERE exper_id=%d AND num=%d" % (exper_id,runnum,))
        if not row : return None
        __run2id[exper_id][runnum] = int(row['id'])

    return __run2id[exper_id][runnum]


# --------------------------------------------------------------------
# Get data path for an experiment. Use a numeric identifier to specify
# the experiment.
# Return None if no data path is configured for the experiment.
# --------------------------------------------------------------------

def getexp_datapath(id):
    row = __do_select("SELECT val FROM experiment_param WHERE exper_id=%s AND param='DATA_PATH'" % id)
    if not row : return None
    return row['val']


def active_experiment(instr, station=0):
    """
    Get a record for the latest experiment activated for the given instrument.
    The function will return a tuple of:

      (instrument, experiment, id, activated, activated_local_str, user)

    Where:

      instrument - the name of the instrument

      experiment - the name of the experiment

      id - a numeric identifier of the experiment

      activated - a 64-bit integer timestamp representing a time when the experiment
                  was activated as the 'current' experiment of the instrument. The value
                  of the timestamp is calculated as: nsec + sec * 10^9

      activated_local_str - a human-readable repreentation of the activation time
                            in the local timezone

      user - a UID of a user who's requested the activation

    The function wil return None if no record database was found for the requested
    instrument name. This may also be an indication that the instrument name is not valid.
    """

    row = __do_select(
        """
        SELECT e.name AS `name`, e.id AS `id`, sw.switch_time AS `switch_time`, sw.requestor_uid AS `requestor_uid` FROM expswitch sw, experiment e, instrument i
        WHERE sw.exper_id = e.id AND e.instr_id = i.id AND i.name='%s' AND sw.station=%d ORDER BY sw.switch_time DESC LIMIT 1
        """ % (instr,station,))

    if not row : return None

    timestamp = int(row['switch_time'])
    return (instr, row['name'], int(row['id']), timestamp, time.ctime(timestamp / 1e9), row['requestor_uid'])

# -----------------------------------------------------------------
# Return a list of files open/created in a context of the specified
# experiment and a run. If the run number is omitted then the most
# recent run will be assumed.
# -----------------------------------------------------------------

def get_open_files(exper_id,runnum=None):

    """
    Return a list of files created (by the DAQ system) in a context of
    the specified experiment and a run. If the run number is omitted
    then the most recent run will be assumed.

      @param exper_id  - numeric identifier of an experiemnt
      @param runnum    - optional run number

    The function will return a list of entries, where each entry will
    represent one file described by a dictionary of:

      'open'      - a floating point number corresponding to a time when
                    the file was created. The number will have the same semantics
                    as the one of Python Library function time.time().

      'exper_id'  - a numeric identifier of teh experiment

      'runnum'    - a run number

      'stream'    - a stream number

      'chunk'     - a chunk number

      'host'      - a DSS host name where the file was created

      'dirpath'   - an absolute path name of a base directory where the file is
                    located

    Note that the list won't be sorted. It's up to a caller's code
    to do a proper sorting and an interpretation of the returned file
    description entries. Here is a reminder how to build file names from
    the above presented fields:

      "e%d_r%04d_s%02d_c%02d", (exper_id,runnum,stream,chunk)

    This will yield the output which would look like this:

      e167_r0002_s01_c00

    The function will return an empty list if no files were found for the
    specified experiment/run.
    """

    if runnum is None:
        row = __do_select("SELECT MAX(run) AS 'run' FROM file WHERE exper_id=%d" % (exper_id,))
        if not row or not row['run']: return []
        runnum = int(row['run'])

    files = []
    for row in __do_select_many("SELECT * FROM file WHERE exper_id=%d and run=%d ORDER BY stream, chunk" % (exper_id,runnum)):
        row['open'] = row['open'] / 1e9
        files.append(row)

    return files

def experiment_runs(instr, exper=None, station=0):

    """
    Return a list of runs taken in a context of the specified experiment.

    Each run will be represented with a dictionary with the following keys:

      'exper_id'        : a numeric identifier of the experiment
      'id'              : a numeric identifier of a run
      'num'             : a run number
      'begin_time_unix' : a UNIX timestamp (32-bits since Epoch) for the start of the run
      'end_time_unix'   : a UNIX timestamp (32-bits since Epoch) for the start of the run.

    NOTES:

      1. if no experiment name provided to the function then  the current
      experiment for the specified station will be assumed. The station parameter
      will be ignored if the experiment name is provided.

      2. if the run is still going on then its 'end_time_unix' will be set to None

    PARAMETERS:

      @param instr: the name of the instrument
      @param exper: the optional name of the experiment (default is the current experiment of the instrument)
      @param station: the optional station number (default is 0)
      @return: the list of run descriptors as explained above

    """

    if exper is None:
        e = active_experiment(instr,station)
        if e is None: return []
        exper_id = e[2]
    else:
        exper_id = name2id(exper)
    runs = []
    for row in __do_select_many("SELECT * FROM logbook.run WHERE exper_id=%d ORDER BY begin_time" % (exper_id,)):
        row['begin_time_unix'] =  int(row['begin_time']/1000000000L)
        row['end_time_unix'] = None
        if row['end_time']:  row['end_time_unix'] = int(row['end_time']/1000000000L)
        runs.append(row)

    return runs

def unique_detector_names():

    """
    Return a list of all known detector names configured in the DAQ system
    accross all known experiments and runs
    """

    query = "SELECT DISTINCT name FROM logbook.run_attr WHERE class IN ('DAQ_Detectors','DAQ Detectors') ORDER BY name"
    return [row['name'] for row in __do_select_many(query)]


def detectors(instr, exper, run):

    """
    Return a list of detector names configured in the DAQ system for a particular
    experiment and a run.
 
    PARAMETERS:

      @param instr: the name of the instrument
      @param exper: the name of the experiment
      @param run: the run number
      @return: the list of detector names

    """

    run_id = run2id(exper,run)
    if run_id is None: raise ValueError('wrong experiment name or run number')

    query = "SELECT name FROM logbook.run_attr WHERE run_id=%d AND class IN ('DAQ_Detectors','DAQ Detectors') ORDER BY name" % run_id
    return [row['name'] for row in __do_select_many(query)]


def run_attributes(instr, exper, run, attr_class=None):

    """
    Return a list of attrubutes of the specified run of an experiment.
    The result set may be (optionally) narrowed to a class.
    Each entry in the result list will be represented by a dictionary of
    the following keys:

      'class' : the class of the attribute
      'name'  : the name of the attribute within a scope of its class
      'descr' : the descrption of the attribute (can be empty)
      'type'  : the type of the attribute's value ('INT','DOUBLE' or 'TEXT')
      'val'   : the value of the attribute (None if no value was set)

    PARAMETERS:

      @param instr: the name of the instrument
      @param exper: the name of the experiment
      @param run: the run number
      @param attr_class: the name of the attribute's class (optional)
      @return: the list of dictinaries representing attributes

    """

    run_id = run2id(exper,run)
    if run_id is None: raise ValueError('wrong experiment name or run number')

    attr_class_opt = ''
    if attr_class is not None: attr_class_opt = " AND class='%s' " % __escape_string(attr_class)

    sql = "SELECT * FROM logbook.run_attr WHERE run_id=%d %s ORDER BY class,name" % (run_id,attr_class_opt,)

    result = []
    for attr in __do_select_many(sql):

        attr_type =     attr['type']
        attr_id   = int(attr['id'])

        sql4val = "SELECT val FROM logbook.run_attr_%s WHERE attr_id=%d" % (attr_type,attr_id,)
        row4val = __do_select(sql4val)

        attr_val = None
        if row4val is not None:
            if   attr_type == 'INT'   : attr_val = int  (row4val['val'])
            elif attr_type == 'DOUBLE': attr_val = float(row4val['val'])
            elif attr_type == 'TEXT'  : attr_val =       row4val['val']

        result.append({'class': attr['class'],'name':attr['name'],'descr':attr['descr'],'type':attr_type,'val':attr_val})

    return result

def run_attribute_classes(instr, exper, run):

    """
    Return a list with names of attrubute classes for the specified run of
    an experiment.

    PARAMETERS:

      @param instr: the name of the instrument
      @param exper: the name of the experiment
      @param run: the run number
      @return: the list of class names

    """

    run_id = run2id(exper,run)
    if run_id is None: raise ValueError('wrong experiment name or run number')

    sql = "SELECT DISTINCT class FROM logbook.run_attr WHERE run_id=%d ORDER BY class" % (run_id,)

    return [attr['class'] for attr in __do_select_many(sql)]

def calibration_runs(instr, exper, runnum=None):

    """
    Return the information about calibrations associated with the specified run
    (or all runs of the experiment if no specific run number is provided).

    The result will be packaged into a dictionary of the following type:

      <runnum> : { 'calibrations' : [<calibtype1>, <calibtype2>, ... ] ,
                   'comment'      :  <text>
                 }

    Where:

      <runnum>     : the run number
      <calibtype*> : the name of the calibration ('dark', 'flat', 'geometry', etc.)
      <text>       : an optional comment for the run

    PARAMETERS:

      @param instr: the name of the instrument
      @param exper: the name of the experiment
      @param run: the run number (optional)

    """

    run_numbers = []
    if runnum is None:
        run_numbers = [run['num'] for run in experiment_runs(instr, exper)]
    else:
        run_numbers = [runnum]


    result = {}

    for runnum in run_numbers:
        run_info = {'calibrations': [], 'comment':''}
        for attr in run_attributes(instr, exper, runnum, 'Calibrations'):
            if   attr['name'] == 'comment': run_info['comment'] = attr['val']
            elif attr['val']              : run_info['calibrations'].append(attr['name'])
        result[runnum] = run_info

    return result

# -------------------------------
# Here folow a couple of examples
# -------------------------------

if __name__ == "__main__" :

    import datetime

    try:

        print 'experiment id 47 translates into %s' % id2name(47)
        print 'experiment sxrcom10 translates into id %d' % name2id('sxrcom10')
        print 'data path for experiment id 116 set to %s' % getexp_datapath(116)
        print 'current time is %d nanoseconds' % __now_64()





        print """
 -------+------------+------+------------------------------------------------+----------
        |            |      |            activation time                     |
  instr | experiment |   id +---------------------+--------------------------+ by user
        |            |      |         nanoseconds | local timezone           |
 -------+------------+------+---------------------+--------------------------+----------"""

        for instr in ('AMO','SXR','XPP','XCS','CXI','MEC','XYZ'):
            exp_info = active_experiment(instr)
            if exp_info is None:
                print "  %3s   | *** no experiment found in the database for this instrument ***" % instr
            else:
                print "  %3s   | %-10s | %4d | %19d | %20s | %-8s" % exp_info

        print ""

        print 'open files of the last run of experiment id 161:'
        for file in get_open_files(161):
            print file

        print 'open files of run 1332 of experiment id 55:'
        for file in get_open_files(55,1332):
            print file

        print 'open files of a  non-valid experiment id 9999999:'
        for file in get_open_files(9999999):
            print file






        print """

 Runs for the current experiment at XPP:

 --------+-------+---------------------+---------------------
  exp_id |  run  |     begin  time     |      end  time
 --------+-------+---------------------+---------------------"""

        for run in experiment_runs('XPP'):
            begin_time = datetime.datetime.fromtimestamp(run['begin_time_unix']).strftime('%Y-%m-%d %H:%M:%S')
            end_time = ''
            if run['end_time_unix']:
                end_time = datetime.datetime.fromtimestamp(run['end_time_unix']).strftime('%Y-%m-%d %H:%M:%S')
            print "  %6d | %5d | %19s | %19s" % (run['exper_id'],run['num'],begin_time,end_time,)


        instr_name = 'SXR'
        exper_name = 'sxr39612'







        print """

 Detector names for all runs for experiment %s/%s :

 --------+---------------------------------------------------------------------------------------------------------
    run  |  detectors
 --------+---------------------------------------------------------------------------------------------------------""" % (instr_name, exper_name,)

        for run in experiment_runs(instr_name, exper_name):
            runnum = run['num']
            print "   %4d  |  %s" % (runnum, '  '.join(detectors(instr_name, exper_name, runnum,)),)






        instr_name = 'CXI'
        exper_name = 'cxic0213'
        runnum     = 215

        print """

 Attributes for run %d of experiment %s/%s :

 -------------------+--------------------------------+------------+-------------+------------------------------------------------------------------------
              class |                           name |       type |       descr | value
 -------------------+--------------------------------+------------+-------------+------------------------------------------------------------------------""" % (runnum,instr_name,exper_name,)


        for attr in run_attributes(instr_name, exper_name, runnum):
            attr_val = attr['val']
            if attr_val is None: attr_val = ''
            print "  %17s | %30s | %10s | %11s | %s" % (attr['class'],attr['name'],attr['type'],attr['descr'][:11],str(attr_val),)






        print """

 Calibration runs of experiment %s/%s :

 --------+--------------------------------------------+-----------------------------------------
    run  |  Calibrations                              |  Comment
 --------+--------------------------------------------+-----------------------------------------""" % (instr_name, exper_name,)


        entries = calibration_runs(instr_name, exper_name)
        for run in sorted(entries.keys()):

            info = entries[run]
            comment = info['comment']
            calibtypes =  ' '.join([calibtype for calibtype in info['calibrations']])

            # report runs which have at least one calibratin type
            if calibtypes:
                print "   %4d  |  %-40s  |  %s"  % (run, calibtypes, comment,)


        print """

 Unique detector names which have ever been used for any experiments and runs
 ----------------------------------------------------------------------------"""

        names = unique_detector_names()
        for name in names:
            print "   %s" % name

        print """ -----------------------------------------
   Total: %d
 """ % len(names)


    except db.Error, e:
         print 'MySQL operation failed because of:', e
         sys.exit(1)

    sys.exit(0)

