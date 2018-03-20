#--------------------------------------------------------------------------
# File and Version Information:
#  $Id: $
#
# Description:
#  Module RegDb...
#
#------------------------------------------------------------------------

""" Interface class for RegDb.

This software was developed for the LUSI project.  If you use all or
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id: $

@author Igor Gaponenko
"""


#------------------------------
#  Module's version from SVN --
#------------------------------
__version__ = "$Revision: 702 $"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys
import os
import logging

#---------------------------------
#  Imports of base class module --
#---------------------------------

#-----------------------------
# Imports for other modules --
#-----------------------------

from psana.pscalib.calib.Time import Time

#----------------------------------
# Local non-exported definitions --
#----------------------------------

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------

class RegDb ( object ) :

    #--------------------
    #  Class variables --
    #--------------------

    #----------------
    #  Constructor --
    #----------------
    def __init__ ( self, conn, log=None ) :
        """Constructor.

        @param conn      database connection object
        """

        self._conn = conn
        self._log = log or logging.getLogger()

    def begin(self):
        self._conn.cursor().execute( 'BEGIN' )

    def commit(self):
        self._conn.cursor().execute( 'COMMIT' )


    # ===================
    # Get instrument list
    # ===================
    def get_instruments(self):
        """
        Get a list of currently defined instruments. Returns a list of tuples
        (instr_id, instr_name, descr). 
        """
        
        cursor = self._conn.cursor()
        cursor.execute("SELECT i.id, i.name, i.descr FROM instrument i ORDER BY i.id")
        
        return map(tuple, cursor.fetchall())


    # ===================
    # Get experiment list
    # ===================
    def get_experiments(self, instr=None):
        """
        Get a list of currently defined experiments for given instrument name. Returns
        the list of dictionaries, elements will have the same structure as dictionary
        return by find_experiment_by_id(). If instr is None then experiments for all 
        instruments are returned.
        """
        
        cursor = self._conn.cursor( True )
        q = """SELECT i.name AS instr_name, i.descr AS instr_descr, e.* FROM experiment e, instrument i WHERE e.instr_id=i.id"""
        if instr is not None:
            q += """ AND i.name=%s"""
            cursor.execute(q,  (instr,))
        else:
            cursor.execute(q)
        
        return map(dict, cursor.fetchall())


    # =================================
    # Find experiment by its identifier
    # =================================
    def find_experiment_by_id(self, id):
        """
        Find experiment by its identifier. Return none or False if no such experiment
        or a dictionary with parameters of the experiment. The following keys will
        be found in the dictionary:
            - instr_id
            - instr_name
            - instr_descr
            - id
            - name
            - descr
            - leader_account
            - posix_gid
            - contact_info
            - registration_time
            - begin_time
            - end_time
        """

        # find the experiment
        cursor = self._conn.cursor( True )
        cursor.execute( """SELECT i.name AS instr_name, i.descr AS instr_descr, e.* FROM experiment e, instrument i WHERE e.id=%s AND e.instr_id=i.id""",  (id, ))
        rows = cursor.fetchall()
        if not rows : return None
        if len(rows) > 1:
            raise Exception("Too many rows for experiment id %s" % id)

        row = rows[0]

        row['registration_time'] = Time.from64( row['registration_time'] )
        row['begin_time']        = Time.from64( row['begin_time'] )
        row['end_time']          = Time.from64( row['end_time'] )

        return row


    # ===========================
    # Find experiment by its name
    # ===========================
    def find_experiment_by_name(self, instrName, expName):
        """
        Find experiment by its name (plus instrument name). Returns
        the same result as find_experiment_by_id().
        """

        # find the experiment
        cursor = self._conn.cursor( True )
        q = """SELECT i.name AS instr_name, i.descr AS instr_descr, e.* 
            FROM experiment e, instrument i 
            WHERE i.name=%s AND e.name=%s AND e.instr_id=i.id"""
        cursor.execute(q, (instrName, expName))
        rows = cursor.fetchall()
        if not rows : return None
        if len(rows) > 1:
            raise Exception("Too many rows for experiment %s:%s" % (instrName, expName))

        row = rows[0]

        row['registration_time'] = Time.from64( row['registration_time'] )
        row['begin_time']        = Time.from64( row['begin_time'] )
        row['end_time']          = Time.from64( row['end_time'] )

        return row


    # ======================
    # Get latest experiments
    # ======================
    def last_experiment_switch(self, instr, limit=1):
        """Get one or few latest experiments for the given instrument name,
        returns list of tuples (expName, time, user), time is LusiTime.Time object,
        user is user name who requested switch. Returns empty list if cannot find
        information for given instrument name. Returned list is ordered by time, 
        latest entries come first."""

        cursor = self._conn.cursor()
        q = """SELECT e.name, sw.switch_time, sw.requestor_uid 
            FROM expswitch sw, experiment e, instrument i
            WHERE sw.exper_id = e.id AND e.instr_id = i.id AND i.name=%s
            ORDER BY sw.switch_time DESC LIMIT %s"""
        cursor.execute(q, (instr, limit))

        return map(lambda x: (x[0], Time.from64(x[1]), x[2]), cursor.fetchall())


    # =========================
    # Get experiment parameters
    # =========================
    def get_experiment_param(self, instr, exper, param=None):
        """
        Returns experiment parameters defined in database. It takes instrument name,
        experiment name, and optional parameter name. If parameter name is not 
        specified or is None the it returns the list of all parameters, returned value 
        in this case list of tuples (param_name, param_value, description). If parameter 
        name is not None then it returns a value of that parameter or None if parameter
        does not exist. Parameter value in all cases is string (possibly empty).
        """
        
        cursor = self._conn.cursor()
        if param:

            # parameter name is specified, find that param only
            q = """SELECT p.val FROM experiment_param p, experiment e, instrument i
                WHERE p.exper_id = e.id AND e.instr_id = i.id 
                AND i.name=%s AND e.name=%s AND p.param=%s"""
            cursor.execute(q, (instr, exper, param))

            rows = cursor.fetchall()
            if not rows: return None
            return rows[0][0]
        
        else:
            
            # get all parameters
            q = """SELECT p.param, p.val, p.descr 
                FROM experiment_param p, experiment e, instrument i
                WHERE p.exper_id = e.id AND e.instr_id = i.id AND i.name=%s AND e.name=%s"""
            cursor.execute(q, (instr, exper))
            
            return map(tuple, cursor.fetchall())

    # =========================
    # Set experiment parameters
    # =========================
    def set_experiment_param(self, instr, exper, param, value):
        """
        Change experiment parameters defined in database. It takes instrument name,
        experiment name, parameter name and new parameter value which must be a string.
        """
        
        cursor = self._conn.cursor()
        
        # find experiment id
        q = """SELECT e.id FROM experiment e, instrument i WHERE e.instr_id = i.id AND i.name=%s AND e.name=%s"""
        cursor.execute(q, (instr, exper))
        rows = cursor.fetchall()
        if not rows: 
            raise ValueError("unknown instrument or experiment name: %s/%s" % (instr, exper))
        exper_id = rows[0][0]

        # try to get its value and lock it if it's there
        q = """SELECT p.val FROM experiment_param p WHERE p.exper_id = %s AND p.param=%s FOR UPDATE"""
        cursor.execute(q, (exper_id, param))

        rows = cursor.fetchall()
        if not rows: 

            # need to make new parameter
            q = """INSERT INTO experiment_param (exper_id, param, val, descr) VALUES (%s, %s, %s, '')"""
            cursor.execute(q, (exper_id, param, value))

        else:
            
            # update existing parameter            
            q = """UPDATE experiment_param p SET val=%s WHERE p.exper_id=%s AND p.param=%s"""
            cursor.execute(q, (value, exper_id, param))

        
    # ===========================
    # Delete experiment parameter
    # ===========================
    def delete_experiment_param(self, instr, exper, param):
        """
        Delete specified experiment parameter
        """

        cursor = self._conn.cursor()

        # remove matching row
        q = """DELETE FROM experiment_param
            WHERE exper_id=(SELECT e.id FROM experiment e, instrument i WHERE e.instr_id = i.id AND i.name=%s AND e.name=%s)
            AND param=%s"""
        cursor.execute(q, (instr, exper, param))
        

    # 

    def get_all_datapath(self, instr=None, datapath=None):
        """ Get (datapath, experiment-name, instrument-name) for selected instrument and/or
        particular datapath. 
        """

        cursor = self._conn.cursor()
        
        q = """ SELECT p.val, e.name, i.name FROM experiment e, instrument i, experiment_param p WHERE 
                p.param = 'DATA_PATH' AND p.exper_id = e.id AND e.instr_id = i.id"""
        param = []
        if instr:
            q += " AND i.name=%s"
            param.append(instr.upper())
        if datapath:
            q += " AND p.val like %s"
            param.append(datapath + '%')

        if param:
            cursor.execute(q, param)
        else:
            cursor.execute(q)

        return cursor.fetchall()

    def get_datapath(self, instr, exper):        
        return self.get_experiment_param(instr, exper, param="DATA_PATH")
