#!/bin/env python
# ProcMgr.py - configure (start, stop, status) the DAQ processes
# $Id$

import os, sys, string, telnetlib, subprocess
import stat, errno, time
import re
from time import sleep, strftime
import socket
import io
from platform import node
from getpass import getuser

uniqueid_maxlen = 30;

#
# getConfigFileNames
#
# This function returns the paths of the p<partition>.conf.running and
# p<partition>.conf.last files, based on the partition# and config path.
#
# The paths are returned irregardless of whether the files are present.
#
# RETURNS: Two values: running config filename, last config filename
#
def getConfigFileNames(argconfig, partition):

    run_name = 'p%d.cnf.running' % partition
    last_name = 'p%d.cnf.last' % partition

    # prepend directory name if necessary
#    if ('/' in argconfig):
#        run_name = os.path.dirname(argconfig) + '/' + run_name
#        last_name = os.path.dirname(argconfig) + '/' + last_name

    # return filenames
    return (run_name, last_name)

#
# getCurrentExperiment
#
# This function returns the current experiment ID from the
# offline database based on the instrument (AMO, SXR, CXI:0, CXI:1, etc.)
#
# RETURNS:  Two values:  experiment number, experiment name
#

def getCurrentExperiment(exp, cmd, station):

    exp_id = int(-1)
    exp_name = ''

    if (cmd):
      returnCode = 0
      fullCommand = '%s %s:%u' % (cmd, exp.upper(), station)
      p = subprocess.Popen([fullCommand],
                           shell = True,
                           stdin = subprocess.PIPE,
                           stdout = subprocess.PIPE,
                           stderr = subprocess.PIPE,
                           close_fds = True)
      out, err = subprocess.Popen.communicate(p)
      if (p.returncode):
        returnCode = p.returncode
     
      if len(err) == 0 and len(out) != 0:
        out = out.strip()
        try:
          exp_name = out.split()[1]
          exp_id = int(out.split()[2])
        except:
          exp_id = int(-1)
          exp_name = ''
          err = 'failed to parse \"%s\"' % out

      if returnCode != 0:
        print("Unable to get current experiment ID")
        if len(err) != 0:
          print("Error from '%s': %s" % (fullCommand, err))
        exp_id = int(-1)
        exp_name = ''

    return (exp_id, exp_name)

#
# getUser
#
# RETURNS: Two values: username, hostname
#
def getUser():
    return getuser(), node()

#
# name2uniqueid - translate procServ name to uniqueid
#
# For example:
#   '/reg/lab2/home/caf/2012/03/29_16:27:22_localhost:helloX.log' -> 'helloX'
#
def name2uniqueid(name):
    rv = name
    try:
      if name.endswith(b".log"):
        rv = name[0:-4].split(b":")[-1]
    except:
      rv = name

    return (rv)

#
# progressMessage
#
def progressMessage(msg):
    print('%-60s ...' % msg, end=' ')
    sys.stdout.flush()
    return

#
# The ProcMgr class maintains a dictionary with keys of
# the form "<host>:<uniqueid>".  The following helper functions
# are used to convert back and forth among <key>, <host>, and <uniqueid>.
#

#
# makekey - given <host> and <uniqueid>, generate a dictionary key
#
def makekey(host, uniqueid):
    return (host + ':' + uniqueid)
    
#
# key2host - given a dictionary key, get <host>
#
def key2host(key):
    return (key.split(':')[0])
    
#
# key2uniqueid - given a dictionary key, get <uniqueid>
#
def key2uniqueid(key):
    return (key.split(':')[1])

#
# mkdir_p - emulate mkdir -p
#
def mkdir_p(path):
    rv = 1
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else: raise
    else:
        rv = 0
    return rv

#
# idFoundInList - look for match between ID and list of substrings
#
# Substring '.' is a special case that matches all IDs.
#
def idFoundInList(id, substrings):
    found = False
    for item in substrings:
      if (id.find(item) != -1) or (item.strip() == '.'):
        found = True
        break
    return found

#
# deduce_platform - deduce platform (-p) from contents of config file
#
# Returns: non-negative platform number, or -1 on error.
#
def deduce_platform(configfilename):
    rv = -1   # return -1 on error
    cc = {'platform': None, 'procmgr_config': None,
          'id':'id', 'cmd':'cmd', 'flags':'flags', 'port':'port', 'host':'host',
          'rtprio':'rtprio', 'env':'env', 'evr':'evr', 'procmgr_macro': {}}
    try:
      exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, cc)
      if type(cc['platform']) == type('') and cc['platform'].isdigit():
        rv = int(cc['platform'])
    except:
      print('deduce_platform Error:', sys.exc_info()[1])

    return rv

#
# deduce_platform2 - deduce platform (-p) and macros
#
# RETURNS: Two values: platform number (or -1 on error), and macros
#
def deduce_platform2(configfilename):
    platform_rv = -1   # return -1 on error
    macro_rv = {}
    cc = {'platform': None, 'procmgr_config': None,
          'id':'id', 'cmd':'cmd', 'flags':'flags', 'port':'port', 'host':'host',
          'rtprio':'rtprio', 'env':'env', 'evr':'evr', 'procmgr_macro': {}}
    try:
      exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, cc)
      macro_rv = cc['procmgr_macro']
      if type(cc['platform']) == type('') and cc['platform'].isdigit():
        platform_rv = int(cc['platform'])
    except:
      print('deduce_platform2 Error:', sys.exc_info()[1])

    return platform_rv, macro_rv


#
# deduce_instrument - deduce instrument (AMO, SXR, etc) from contents of config file
#
# An optional station number is supported.  For example, CXI:1 translates to CXI station 1.
#
# The default instrument name is ''.  The default station number is 0.
# The default currentexp command path is ''.
#
# RETURNS: Three values: instrument name, station number, and currentexp command path.
#
def deduce_instrument(configfilename):
    instr_name = ''
    currentexpcmd = ''
    station_number = 0
    cc = {'instrument': None, 'platform': None, 'procmgr_config': None,
          'id':'id', 'cmd':'cmd', 'flags':'flags', 'port':'port', 'host':'host',
          'rtprio':'rtprio', 'env':'env', 'evr':'evr', 'procmgr_macro': {}, 'currentexpcmd': None}

    try:
      exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, cc)
      if cc['instrument'] != None:
        tmplist = cc['instrument'].split(b":")
        instr_name = tmplist[0].upper()
        if len(tmplist) > 1:
          station_number = int(tmplist[1])
        if cc['currentexpcmd'] != None:
          currentexpcmd = cc['currentexpcmd']
    except:
      print('deduce_instrument Error:', sys.exc_info()[1])        
      instr_name = ''
      currentexpcmd = ''
      station_number = 0

    return instr_name, station_number, currentexpcmd

#
# parse_cmd
#
# Parse the cmd string looking for -E, -e, or -f and replacing
# 'expname' and 'expnum' with current experiment name and number
# from database
#
# Caution:  If the string 'expname' or 'expnum' appears anywhere
# else in the command, it will also be replaced
#
def parse_cmd(cmd, expnum, expname):
    # if -E (experiment name) is passed in, replace 'expname' with current
    if cmd.find('-E') != -1 and cmd.find('expname') != -1:
        cmd = cmd.replace('expname',expname)

    # if -e (experiment number) is passed in, replace 'expnum' with current
    if cmd.find('-e') != -1 and cmd.find('expnum') != -1:
        cmd = cmd.replace('expnum', str(expnum))

    # if -f (filename) and 'expname' are passed in, replace 'expname' with current
    if cmd.find('-f') != -1 and cmd.find('expname') != -1:
        fname = cmd.split(b'-f')[1].strip()
        newfname = fname.replace('expname', expname)
        cmd = cmd.replace(fname, newfname)
    return cmd

#
# add_macro_config
#
def add_macro_config(procmgr_macro, oldfilename, newfilename):

  #
  # read old file into memory
  #
  try:
    oldfile = open(oldfilename, 'r')
    oldfilecontents = oldfile.read()
    oldfile.close()
  except IOError:
    print('%s: i/o error occurred while reading from \'%s\'' % (sys.argv[0], oldfilename))
  except:
    print('%s: error occurred while reading from \'%s\': %r' % (sys.argv[0], oldfilename, sys.exc_info()[1]))
  else:
  #
  # create temporary file (in memory)
  #
    try:
      tmpfile = io.StringIO()
      tmpfile.write('# --- automatically generated file - DO NOT EDIT -----------------------------\n')
      tmpfile.write('# COMMAND:')
      for aa in sys.argv:
        tmpfile.write(' %s' % aa)
      tmpfile.write('\n# DATE: %s\n' % strftime('%c'))
      for key in sorted(procmgr_macro.keys()):
        tmpfile.write('procmgr_macro[\'%s\'] = \'%s\'\n' % (key, procmgr_macro[key]))
      tmpfile.write('# ----------------------------------------------------------------------------\n')
      tmpfile.write(oldfilecontents)
    except IOError:
      print('%s: i/o error occurred while creating temporary file' % (sys.argv[0]))
    except:
      print('%s: error occurred while creating temporary file: %r' % (sys.argv[0], sys.exc_info()[1]))
      raise
    else:
  #
  # copy temporary file to new file
  #
      try:
        tmpfilecontents = tmpfile.getvalue()
        tmpfile.close()
        newfile = open(newfilename, 'w')
        newfile.write(tmpfilecontents)
        newfile.close()
      except IOError:
        print('%s: i/o error occurred while updating \'%s\'' % (sys.argv[0], newfilename))
      except:
        print('%s: error occurred while updating \'%s\': %r' % (sys.argv[0], newfilename, sys.exc_info()[1]))

  return

#
# ConfigFileError - this exception is raised to report configuration file errors
# 
class ConfigFileError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

#
# ProcMgr
#
class ProcMgr:

    # index into arrays managed by this class
    DICT_STATUS = 0
    DICT_PID = 1
    DICT_CMD = 2
    DICT_CTRL = 3
    DICT_PPID = 4
    DICT_FLAGS = 5
    DICT_GETID = 6

    # a managed executable can be in the following states
    STATUS_NOCONNECT = "NOCONNECT"
    STATUS_RUNNING = "RUNNING"
    STATUS_SHUTDOWN = "SHUTDOWN"
    STATUS_ERROR = "ERROR"

    # placeholder when PID is not valid
    STRING_NOPID = "-"

    # paths
    PATH_XTERM = "/usr/bin/xterm"
    PATH_TELNET = "/usr/bin/telnet"
    PATH_LESS = "/usr/bin/less"
    PATH_CAT = "/bin/cat"

    # messages expected from procServ
    MSG_BANNER_END = b"server started at"
    MSG_ISSHUTDOWN = b"is SHUT DOWN"
    MSG_ISSHUTTING = b"is shutting down"
    MSG_KILLED     = b"process was killed"
    MSG_RESTART    = b"new child"
    MSG_PROMPT     = b"\x0d\x0a> "
    MSG_SPAWN      = b"procServ: spawning daemon"

    # procmgr control port initialized in __init__
    EXECMGRCTRL = -1

    # platform initialized in __init__
    PLATFORM = -1

    # instrument and station initialized in __init__
    INSTRUMENT = ''
    STATION = 0
    CURRENTEXPCMD = ''
    
    valid_flag_list = ['X', 'x', 'k', 's', 'u', 'p'] 
    valid_instruments = ['AMO','SXR','XPP','XCS','CXI','MEC','MFX','DET']

    def __init__(self, configfilename, platform, Xterm_list=[], xterm_list=[], procmgr_macro={}, baseport=29000):
        self.pid = self.STRING_NOPID
        self.ppid = self.STRING_NOPID
        self.getid = None
        self.telnet = telnetlib.Telnet()
        self.Xterm_list = Xterm_list
        self.xterm_list = xterm_list
        self.procmgr_macro = procmgr_macro

        # configure the default socket timeout in seconds
        socket.setdefaulttimeout(2.5)

        # configure the port numbers
        self.EXECMGRCTRL = baseport

        # create new empty 'self' dictionary
        self.d = dict()

        # create dictionaries for port assignments
        nextCtrlPort = dict()
        staticPorts = dict()

        # create list for detecting duplicate ids
        dup_list = list()

        if (platform == -1):
            print('ERR: platform not specified')
            return
        else:
            self.PLATFORM = platform

        if (self.PLATFORM > 0):
            self.EXECMGRCTRL += (self.PLATFORM * 100)

        # initialize the experiment
        # (only used by online_ami to get current experiment)
        self.INSTRUMENT, self.STATION, self.CURRENTEXPCMD = deduce_instrument(configfilename)
        if self.INSTRUMENT not in self.valid_instruments:
            if self.INSTRUMENT != '':
              print('ERR: Invalid instrument ', self.INSTRUMENT)
            (expnum, expname) = (-1, '')
        elif self.STATION < 0:
            print('ERR: Invalid station ', self.STATION)
            (expnum, expname) = (-1, '')
        elif self.CURRENTEXPCMD != '':
            (expnum, expname) = getCurrentExperiment(self.INSTRUMENT, self.CURRENTEXPCMD, self.STATION)

        # set HOST and USER macros
        try:
            user, host = getUser()
            if host:
                procmgr_macro['HOST'] = host
            if user:
                procmgr_macro['USER'] = user
        except:
            print('Error determining HOST and USER')

        # The static port allocations must be processed first.
        # Read the config file and make a list with statically assigned
        # entries at the front, dynamic entries at the back.

        # In case a host appears as both 'localhost' and another name,
        # ensure that 'localhost' ports are not reused on other hosts.
        localPorts = set()
        remotePorts = set()

        configlist = []         # start out with empty list

        config = {'platform': repr(self.PLATFORM), 'procmgr_config': None,
                  'id':'id', 'cmd':'cmd', 'flags':'flags', 'port':'port', 'host':'host',
                  'rtprio':'rtprio', 'env':'env', 'evr':'evr', 'procmgr_macro': procmgr_macro}
        try:
          exec(compile(open(configfilename).read(), configfilename, 'exec'), {}, config)
        except:
          print('Error parsing configuration file:', sys.exc_info()[1])

        if type(config['procmgr_config']) == type([]):
          for dd in config['procmgr_config']:
            if type(dd) == type({}):
              if 'port' in dd:
                # static port assignments at the beginning of the list
                configlist.insert(0, dd)
              else:
                # dynamic port assignments at the end of the list
                configlist.append(dd)
            else:
              print('Error: procmgr_config entry not key:value:', dd)
        else:
          print('Error: procmgr_config not a list', config['procmgr_config'])

        # for each entry in the list...
        for entry in configlist:
          # ...process the fields

          # --- real-time priority (optional) ---
          self.rtprio = None
          tmpsum = 0
          if 'rtprio' in entry:
            try:
              tmpsum = int(entry['rtprio'])
            except:
              raise ConfigFileError('malformed rtprio value: %s' % entry)
            if tmpsum:
              # check if rtprio is in valid range
              if (tmpsum < 1) or (tmpsum > 99):
                raise ConfigFileError('rtprio not in range 1-99: %s' % entry)
              else:
                self.rtprio = tmpsum

          # --- environment (optional) ---
          self.env = None
          if 'env' in entry:
            if '=' in entry['env']:
              self.env = entry['env']
            else:
              raise ConfigFileError("env value is missing '=': %s" % entry)

          # --- evr (optional) ---
          self.evr = None
          if 'evr' in entry:
            match = re.match('^(\d)\,(\d)(\d)?$', entry['evr'])
            if match:
              self.evr = match.group()
            else:
              raise ConfigFileError("evr value does not match '<digit>,<digit>[<digit>]': %s" % entry)

          # --- cmd (required) ---
          if 'cmd' in entry:

            # use os.path.realpath() to resolve any symbolic links
            cmdSplit = entry['cmd'].split(None, 1)
            cmdZero = os.path.expanduser(cmdSplit[0])
            if (os.path.isfile(cmdZero)):
              if (len(cmdSplit) > 1):
                entry['cmd'] = os.path.realpath(cmdZero) + ' ' + cmdSplit[1]
              else:
                entry['cmd'] = os.path.realpath(cmdZero)
            else:
              print('ERR: %s not found' % cmdZero)
              entry['cmd'] = '/bin/echo \"File not found: ' + cmdZero + '"'

            # if rtprio is set, prefix with /usr/bin/chrt
            if (self.rtprio):
              entry['cmd'] = '/usr/bin/chrt -f %d %s' % (self.rtprio, entry['cmd'])

            # if env is specified, prefix the command with /bin/env
            if (self.env):
              entry['cmd'] = '/bin/env %s %s' % (self.env, entry['cmd'])

            if self.CURRENTEXPCMD != '':
              # Do something special if -E, -e, or -f appear in cmd string
              self.cmd = parse_cmd(entry['cmd'], expnum, expname)
            else:
              self.cmd = entry['cmd']
          else:
            raise ConfigFileError("procmgr_config entry %s missing cmd" % entry)
            self.cmd = 'error'

          # --- id (required) ---
          if 'id' in entry:
            tmpid = entry['id']
            if len(tmpid) > uniqueid_maxlen:
              raise ConfigFileError("ID '%s' exceeds %d characters" % (tmpid, uniqueid_maxlen))
            if tmpid in dup_list:
              raise ConfigFileError("ID '%s' appears in multiple procmgr_config entries" % tmpid)
            else:
              dup_list.append(tmpid)
              self.uniqueid = tmpid
          else:
            raise ConfigFileError("procmgr_config entry %s missing id" % entry)
            self.uniqueid = 'error'

          # --- host (optional) ---
          if 'host' in entry:
            self.host = entry['host']
          else:
            self.host = 'localhost'

          # --- flags (optional) ---
          if 'flags' in entry:
            self.flags = entry['flags']
            # evr keyword forces p flag
            if self.evr:
              self.flags += 'p'
            for nextflag in self.flags:
              if (nextflag not in self.valid_flag_list):
                print('ERR: invalid flag:', nextflag)
          else:
            self.flags = '-'

          # append '-u <UniqueId>' to command if 'u' flag is set
          if 'u' in self.flags:
            self.cmd += (' -u ' + self.uniqueid)

          # append '-p <platform>[<mod>,<chan>]' to command if 'p' flag is set
          if 'p' in self.flags:
            if self.evr:
              self.cmd += (' -p ' + repr(self.PLATFORM) + ',' + self.evr)
            else:
              self.cmd += (' -p ' + repr(self.PLATFORM))

          # update flags to reflect -x or -X on command line
          # ...order matters: X flag takes priority over x flag
          if idFoundInList(self.uniqueid, Xterm_list):
            self.flags += 'X'
          elif idFoundInList(self.uniqueid, xterm_list):
            self.flags += 'x'

          # initialize dictionaries used for port assignments
          if not self.host in nextCtrlPort:
              # on each host, two ports are reserved for a master server: ctrl and log
              nextCtrlPort[self.host] = self.EXECMGRCTRL + 2
              staticPorts[self.host] = set()

          # --- port (optional) ---
          tmpsum = 0
          if 'port' in entry:
            try:
              tmpsum = int(entry['port'])
            except:
              print('Error: malformed port value:', entry)

          if tmpsum:
            # assign the port statically
            if tmpsum in staticPorts[self.host]:
                print('ERR: port #%d duplicated in the config file' % tmpsum)
            else:
                # avoid dup: update the set of statically assigned ports
                staticPorts[self.host].add(tmpsum)
            self.ctrlport = str(tmpsum)                             # string
            self.flags += 'k'
          else:
              # assign port dynamically
              tmpport = nextCtrlPort[self.host]
              # avoid dup: check the set of statically assigned ports
              if (self.host == 'localhost'):
                  while (tmpport in staticPorts[self.host]) or (tmpport in remotePorts):
                      tmpport += 1
              else:
                  while (tmpport in staticPorts[self.host]) or (tmpport in localPorts):
                      tmpport += 1

              self.ctrlport = str(tmpport)                            # string
              nextCtrlPort[self.host] = tmpport + 1                   # integer

              # update set of local or remote ports to avoid conflict
              if (self.host == 'localhost'):
                  localPorts.add(tmpport)
              else:
                  remotePorts.add(tmpport)

          self.pid = b"-"
          self.ppid = b"-"
          self.getid = b"-"
          # open a connection to the control port (procServ)
          telnethost = self.host
          if telnethost == 'localhost':
              telnethost = self.procmgr_macro.get('HOST', 'localhost')
          try:
              self.telnet.open(telnethost, self.ctrlport)
          except:
              # telnet failed
              self.tmpstatus = self.STATUS_NOCONNECT
              # TODO ping each host first, as telnet could fail due to an error
          else:
              # telnet succeeded: gather status from procServ banner
              try: 
                ok = self.readLogPortBanner()
              except EOFError:
                print('EOFError in readLogPortBanner') 
                ok = False
              except:
                ok = False
              if not ok:
                  # reading procServ banner failed
                  print("ERR: failed to read procServ banner for \'%s\' on host %s" \
                          % (self.uniqueid, self.host))
              # close connection to the logging port (procServ)
              self.telnet.close()

          if self.getid.endswith(b".log"):
            # '/reg/lab2/home/caf/2012/03/29_16:27:22_localhost:helloX.log' -> 'helloX'
            gotid = self.getid[0:-4].split(b":")[-1]
          else:
            gotid = self.getid

          if ((self.tmpstatus != self.STATUS_NOCONNECT) and \
              (self.tmpstatus != self.STATUS_ERROR) and \
              (gotid != bytes(self.uniqueid, 'utf-8')) and \
              (not gotid.endswith(bytes(self.uniqueid+".log", 'utf-8')))):
              print("ERR: found %r, expected %r on host %s port %s" % \
                  (gotid, self.uniqueid, self.host, self.ctrlport))
          else:
              # add an entry to the dictionary
              key = makekey(self.host, self.uniqueid)
              self.d[key] = \
                [ self.tmpstatus, self.pid, self.cmd, self.ctrlport, self.ppid, self.flags, self.getid]
                # DICT_STATUS  DICT_PID  DICT_CMD  DICT_CTRL      DICT_PPID  DICT_FLAGS  DICT_GETID

    def spawnXterm(self, name, host, port, large=False):
        if large:
            args = [self.PATH_XTERM, "-bg", "midnightblue", "-fg", "white", "-fa", "18", "-T", name, \
                    "-e", self.PATH_TELNET, host, port]
        else:
            args = [self.PATH_XTERM, "-T", name, "-fn", "fixed", "-e", self.PATH_TELNET, host, port]
        subprocess.Popen(args)
        return

    def spawnConsole(self, uniqueid, large=False):
        rv = 1      # return value (0=OK, 1=ERR)
        found = False
        for key in self.d.keys():
            if key2uniqueid(key) == uniqueid:
                found = True
                break
        if not found:
            print('spawnConsole: process \'%s\' not found' % uniqueid)
        elif ((self.d[key][self.DICT_STATUS] == self.STATUS_RUNNING) or
              (self.d[key][self.DICT_STATUS] == self.STATUS_SHUTDOWN)):
            try:
                name = uniqueid
                host = key2host(key)
                port = self.d[key][self.DICT_CTRL]
                logfile = self.d[key][self.DICT_GETID]
                cmd  = self.PATH_TELNET + " " + host + " " + port
                if os.path.exists(logfile):
                    cmd = self.PATH_CAT + " " + logfile + ";" + cmd
                if large:
                    args = [self.PATH_XTERM, "-bg", "midnightblue", "-fg", "white", "-fa", "18", "-T", name, \
                            "-e", cmd]
                else:
                    args = [self.PATH_XTERM, "-T", name, "-e", cmd]
                subprocess.Popen(args)
            except:
                print('spawnConsole failed for process \'%s\'' % uniqueid)
            else:
                rv = 0
        else:
            print('spawnConsole: process \'%s\' neither RUNNING nor SHUTDOWN' % uniqueid)
        return rv


    def spawnLogfile(self, uniqueid, large=False):
        rv = 1      # return value (0=OK, 1=ERR)
        logfile = ''
        found = False
        for key in self.d.keys():
            if key2uniqueid(key) == uniqueid:
                found = True
                logfile = self.procmgr_macro.get('LOGPATH', '.') + '/' + self.d[key][self.DICT_GETID].decode()
                break
        if not found:
            print('spawnLogfile: process \'%s\' not found' % uniqueid)
        elif not os.path.exists(logfile):
            print('spawnLogfile: process \'%s\' logfile not found' % uniqueid)
        elif ((self.d[key][self.DICT_STATUS] == self.STATUS_RUNNING) or
              (self.d[key][self.DICT_STATUS] == self.STATUS_SHUTDOWN)):
            try:
                name = uniqueid
                if large:
                    args = [self.PATH_XTERM, "-bg", "midnightblue", "-fg", "white", "-fa", "18", "-T", name, \
                           "-e", self.PATH_LESS, "+F", logfile]
                else:
                    args = [self.PATH_XTERM, "-T", name, "-e", self.PATH_LESS, "+F", logfile]
                subprocess.Popen(args)
            except:
                print('spawnLogfile failed for process \'%s\'' % uniqueid)
            else:
                rv = 0
        else:
            print('spawnLogfile: process \'%s\' neither RUNNING nor SHUTDOWN' % uniqueid)
        return rv

    def readLogPortBanner(self):
        response = self.telnet.read_until(self.MSG_BANNER_END, 1)
        if not response.count(self.MSG_BANNER_END):
            print('readLogPortBanner: banner not found in response: '+response)
            self.tmpstatus = self.STATUS_ERROR
            # when reading banner fails, set the ID so the error output includes name instead of '-'
            self.getid = self.uniqueid
            return 0
        if re.search(b'SHUT DOWN', response):
            self.tmpstatus = self.STATUS_SHUTDOWN
            self.ppid = re.search(b'@@@ procServ server PID: ([0-9]*)', response).group(1)
            self.getid = re.search(b'@@@ Child \"(.*)\" start', response).group(1)
        else:
            self.tmpstatus = self.STATUS_RUNNING
            self.pid = re.search(b'@@@ Child \"(.*)\" PID: ([0-9]*)', response).group(2)
            self.getid = re.search(b'@@@ Child \"(.*)\" PID: ([0-9]*)', response).group(1)
            self.ppid = re.search(b'@@@ procServ server PID: ([0-9]*)', response).group(1)
        return 1

    #
    # show - call status() with an empty id_list
    #
    def show(self, verbose=0):
        return self.status([], verbose)

    #
    # status
    #
    def status(self, id_list, verbose=0, only_static=0):

        nonePrinted = 1

        if self.isEmpty():
            if verbose:
                print("(configuration is empty)")
            return 1

        # print contents of dictionary (sorted by key)
        for key in sorted(self.d.keys()):

            if len(id_list) > 0:
                # if id_list is nonempty and UniqueID is not in it,
                # skip this entry
                if key2uniqueid(key) not in id_list:
                    continue

            if only_static and ('k' not in self.d[key][self.DICT_FLAGS]):
                # only_static flag was passed in and this entry does not 
                # have the 'k' flag set: skip this entry
                continue

            if (nonePrinted == 1):
              # print heading, once
              print("Host          UniqueID     Status     PID    PORT   Command+Args")
              nonePrinted = 0

            if (self.d[key][self.DICT_STATUS] == self.STATUS_NOCONNECT):
                showId = key2uniqueid(key)  # string
            else:
                showId = name2uniqueid(self.d[key][self.DICT_GETID]).decode()

            showhost = key2host(key)
            if showhost == 'localhost':
                showhost = self.procmgr_macro.get('HOST', 'localhost')

            print("%-13s %-12s %-10s %-5s  %-5s  %s" % \
                    (showhost, showId, \
                    self.d[key][self.DICT_STATUS], \
                    self.d[key][self.DICT_PID].decode(), \
                    self.d[key][self.DICT_CTRL], \
                    self.d[key][self.DICT_CMD]))

            if (self.d[key][self.DICT_STATUS] == self.STATUS_RUNNING):
                if idFoundInList(showId, self.Xterm_list):
                    # spawn large xterm with console
                    self.spawnConsole(key2uniqueid(key), True)
                elif idFoundInList(showId, self.xterm_list):
                    # spawn small xterm with console
                    self.spawnConsole(key2uniqueid(key), False)
            elif (self.d[key][self.DICT_STATUS] == self.STATUS_SHUTDOWN):
                if idFoundInList(showId, self.Xterm_list):
                    # spawn large xterm with logfile
                    self.spawnLogfile(key2uniqueid(key), True)
                elif idFoundInList(showId, self.xterm_list):
                    # spawn small xterm witih logfile
                    self.spawnLogfile(key2uniqueid(key), False)

            if verbose:
                if self.d[key][self.DICT_GETID].endswith(b".log"):
                    print("  Logfile:", self.procmgr_macro.get('LOGPATH', '.') + \
                                        '/' + self.d[key][self.DICT_GETID].decode())
        if (nonePrinted == 1):
          print("(none found)")

        # done
        return 1

    #
    # getStatus - machine readable status
    #
    def getStatus(self, id_list=[], verbose=0, only_static=0):

        resultlist = list()

        if self.isEmpty():
          if verbose:
            print("(configuration is empty)")
        else:
          # get contents of dictionary (sorted by key)
          for key in sorted(self.d.keys()):
            # start with empty dictionary
            statusdict = dict()

            if len(id_list) > 0:
              # if id_list is nonempty and UniqueID is not in it,
              # skip this entry
              if key2uniqueid(key) not in id_list:
                continue

            if only_static and ('k' not in self.d[key][self.DICT_FLAGS]):
              # only_static flag was passed in and this entry does not 
              # have the 'k' flag set: skip this entry
              continue
                
            if (self.d[key][self.DICT_STATUS] == self.STATUS_NOCONNECT):
              statusdict['showId'] = key2uniqueid(key)
            else:
              statusdict['showId'] = name2uniqueid(self.d[key][self.DICT_GETID])

            statusdict['status'] = self.d[key][self.DICT_STATUS]
            statusdict['host'] = key2host(key)
            # add dictionary to list
            resultlist.append(statusdict)
              
        # done
        return resultlist

    #
    # checkConnection
    #
    def checkConnection(self, key, value, verbose=0):
        connected = False
        # open a connection to the procServ control port
        started = False
        connected = False
        telnetCount = 0
        host = key2host(key)
        while (not connected) and (telnetCount < 2):
            telnetCount += 1
            try:
                self.telnet.open(host, value[self.DICT_CTRL])
            except:
                sleep(.25)
            else:
                connected = True

        if connected:
            # close telnet connection
            self.telnet.close()

        if verbose:
            print(' --- checkConnection(key=%s) returning %s ---' % (key, connected))

        return connected

    #
    # restart
    #
    def restart(self, key, value, verbose=0):

        # open a connection to the procServ control port
        started = False
        connected = False
        telnetCount = 0
        host = key2host(key)
        while (not connected) and (telnetCount < 2):
            telnetCount += 1
            try:
                self.telnet.open(host, value[self.DICT_CTRL])
            except:
                sleep(.25)
            else:
                connected = True

        if connected:
            # wait for SHUT DOWN message
            response = self.telnet.read_until(self.MSG_ISSHUTDOWN, 1)
            if not response.count(self.MSG_ISSHUTDOWN):
                print('ERR: no SHUT DOWN message in ', end=' ')
                print('response: <<%s>>' % response)

            # send ^R to restart child process
            self.telnet.write(b"\x12");

            # wait for restart message
            response = self.telnet.read_until(self.MSG_RESTART, 1)
            if not response.count(self.MSG_RESTART):
                print('ERR: no restart message... ')
            else:
                started = True

            # close telnet connection
            self.telnet.close()
        else:
            print('ERR: restart() telnet to %s port %s failed' % \
                (host, value[self.DICT_CTRL]))

        return started

    #
    # startAll - call start() with an empty id_list
    #
    def startAll(self, verbose=0, logpathbase=None, coresize=0):
        return self.start([], verbose, logpathbase, coresize)

    #
    # start
    #
    # RETURNS: 0 if any processes were started, otherwise 1.
    #
    def start(self, id_list, verbose=0, logpathbase=None, coresize=0):

        rv = 1                  # return value
        started_count = 0       # count successful start commands

        # create sets of entries with X or x flag enabled (empty for now)
        xlist = list()
        Xlist = list()

        if self.isEmpty():
            # configuration is empty -- nothing to start
            if verbose:
                print('startAll: empty configuration')
            return 1

        if (self.PLATFORM < 0) or (self.PLATFORM > 9):
            print('platform %d not in range 0-9' % self.PLATFORM)
            return 1

        # for redirecting to /dev/null
        nullOut = open(os.devnull, 'w')

        # create a dictionary mapping hosts to a set of start commands
        startdict = dict()
        for key, value in self.d.items():
            if len(id_list) > 0:
                # if id_list is nonempty and UniqueID is not in it,
                # skip this entry
                if key2uniqueid(key) not in id_list:
                    continue

            if value[self.DICT_STATUS] == self.STATUS_SHUTDOWN:
                # double check to see if process is actually NOCONNECT
                if not self.checkConnection(key, value, verbose):
                    value[self.DICT_STATUS] = self.STATUS_NOCONNECT

            if value[self.DICT_STATUS] == self.STATUS_NOCONNECT:
                starthost = key2host(key)
                # order matters: X flag takes priority over x flag
                if 'X' in value[self.DICT_FLAGS]:
                    Xlist.append([key, value])
                    waitflag = '--wait'
                elif 'x' in value[self.DICT_FLAGS]:
                    xlist.append([key, value])
                    waitflag = '--wait'
                else:
                    waitflag = ''

                if ('>' in value[self.DICT_CMD]) or (logpathbase == None) or (logpathbase == "/dev/null"):
                    redirect_string = ''
                else:
                    #
                    # Construct path similar to:
                    #
                    #  <logpath>/2009/08/21_10:35_atca01:opal1k.log
                    #
                    logpath = '%s/%s' % (logpathbase, time.strftime('%Y/%m'))
                    try:
                      mkdir_p(logpath)
                    except:
                      # mkdir
                      print('ERR: mkdir <%s> failed' % logpath)
                      redirect_string = ''
                    else:
                      time_string = time.strftime('%d_%H:%M:%S')
                      loghost = key2host(key)
                      localFlag = False
                      if loghost == 'localhost':
                          localFlag = True
                          loghost = self.procmgr_macro.get('HOST', 'localhost')
                      logkey = loghost+':'+key2uniqueid(key)
                      logfile = '%s/%s_%s.log' % (logpath, time_string, logkey)
                      if verbose:
                          print('log file: <%s>' % logfile)
                      if localFlag:
                          # local: bash shell
                          redirect_string = '>> \"%s\" 2>&1' % logfile
                      else:
                          # remote: tcsh shell
                          redirect_string = '>>& \"%s\"' % logfile

                    pbits = (stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                    try:
                        statmode = os.stat(logpath).st_mode
                    except:
                        print('ERR: stat %s failed' % logpath)
                        redirect_string = ''
                    else:
                        if (statmode & pbits) != pbits:
                          try:
                            # make log path readable/writable/searchable by all
                            os.chmod(logpath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                          except:
                            print('ERR: chmod %s failed' % logpath)
                            redirect_string = ''

                # encode logfile path as part of procServ name
                if (len(redirect_string) > 1):
                  name = logfile.replace(logpathbase+'/', '', 1)
                  if not os.path.exists(logfile):
                    try:
                      outfile = open(logfile, 'w')
                      outfile.write("# ID:      %s\n" % key2uniqueid(key))
                      outfile.write("# PLATFORM:%s\n" % self.PLATFORM)
                      outfile.write("# HOST:    %s\n" % loghost)
                      outfile.write("# CMDLINE: %s\n" % value[self.DICT_CMD])
                      outfile.close()
                    except:
                      print("ERR: writing log file '%s' failed" % logfile)
                    else:
                      pbits = (stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
                      if (os.stat(logfile).st_mode & pbits) != pbits:
                        try:
                          # make log file readable/writable by all
                          os.chmod(logfile, pbits)
                        except:
                          print('ERR: chmod %s failed' % logfile)
                else:
                  name = key2uniqueid(key)

                startcmd = \
                        '/reg/common/package/procServ/2.6.0-SLAC/x86_64-rhel6-gcc44-opt/bin/procServ --noautorestart --name %s %s --allow --coresize %d %s %s %s' % \
                       (name, \
                        waitflag, \
                        coresize, \
                        value[self.DICT_CTRL], \
                        value[self.DICT_CMD],
                        redirect_string)
                # is this host already in the dictionary?
                if starthost in startdict:
                    # yes: add to set of start commands
                    startdict[starthost].append([startcmd, key])
                else:
                    # no: initialize set of start commands
                    startdict[starthost] = [[startcmd, key]]
            elif value[self.DICT_STATUS] == self.STATUS_SHUTDOWN:
                # restart
                if self.restart(key, value, verbose):
                    started_count += 1

        # now use the newly created dictionary to run start command(s)
        # on each host

        for host, value in startdict.items():

            if (host == 'localhost'):
                # process list of commands
                while len(value) > 0:
                    # send command
                    args, key = value.pop()
                    if verbose:
                        print('Run locally: %s' % args)

                    yy = subprocess.Popen(args, stdout=nullOut, stderr=nullOut, shell=True)
                    yy.wait()
                    if (yy.returncode != 0):
                        print("ERR: failed to run '%s' (procServ returned %d)" % \
                            (args, yy.returncode))
                    else:
                        self.setStatus([key], self.STATUS_RUNNING)
                        started_count += 1
            else:
                # open a connection to the procmgr control port (procServ)
                try:
                    self.telnet.open(host, self.EXECMGRCTRL)
                except:
                    # telnet failed
                    print('ERR: telnet to procmgr (%s port %d) failed' % \
                            (host, self.EXECMGRCTRL))
                    print('>>> Please start the procServ process on host %s!' % host)
                else:
                    # telnet succeeded

                    # send ^U followed by carriage return to safely reach the prompt
                    self.telnet.write(b"\x15\x0d");

                    # wait for prompt (procServ)
                    response = self.telnet.read_until(self.MSG_PROMPT, 2)
                    if not response.count(self.MSG_PROMPT):
                        print('ERR: no prompt at %s port %s' % \
                            (key2host(key), self.EXECMGRCTRL))

                    # process list of commands
                    while len(value) > 0:

                        nextcmd, nextkey = value.pop()

                        if verbose:
                            print('Run on %s: %s' % (host, nextcmd))

                        # send command
                        self.telnet.write(bytes('%s\n' % nextcmd, 'utf-8'))
                        # wait for prompt
                        response = self.telnet.read_until(self.MSG_PROMPT, 2)
                        if not response.count(self.MSG_PROMPT):
                            print('ERR: no prompt at %s port %s' % \
                                (host, self.EXECMGRCTRL))
                        else:
                            #
                            # If X flag is set, procServ --wait is used so
                            # the next state is actually STATUS_SHUTDOWN.
                            # It will be STATUS_RUNNING after restart, below.
                            #
                            self.setStatus([nextkey], self.STATUS_RUNNING)
                            started_count += 1

                    # close telnet connection
                    self.telnet.close()

        if len(xlist) > 0 or len(Xlist) > 0:
          # is xterm available?
          if not os.path.exists(self.PATH_XTERM):
            print('ERR: %s not available' % self.PATH_XTERM)
          else:
            # order matters: start large xterms last so they will be on top

            # small xterm support
            for item in xlist:
              # spawn small xterm
              self.spawnXterm(item[0], key2host(item[0]), item[1][self.DICT_CTRL])
            for item in xlist:
              if self.restart(item[0], item[1], verbose):
                started_count += 1
                            
            # large xterm support
            for item in Xlist:
              # spawn large xterm
              self.spawnXterm(item[0], key2host(item[0]), item[1][self.DICT_CTRL], True)
            for item in Xlist:
              if self.restart(item[0], item[1], verbose):
                started_count += 1
                        
        # done
        # cleanup
        nullOut.close()

        if started_count > 0:
            rv = 0

        return rv

    #
    # isEmpty
    #
    def isEmpty(self):
        return (len(self.d) < 1)

    #
    # stopDictionary
    #
    def stopDictionary(self, stopdict, verbose, sigdelay):
        rv = 0      # return value
        stopcount = 0

        telnetdict = dict()

        # open telnet connections
        for key, value in stopdict.items():
            connected = False
            telnetCount = 0
            host = key2host(key)
            if host == 'localhost':
                host = self.procmgr_macro.get('HOST', 'localhost')

            connection = telnetlib.Telnet()
            while (not connected) and (telnetCount < 2):
                telnetCount = telnetCount + 1
                try:
                    connection.open(host, value[self.DICT_CTRL])
                except:
                    sleep(.25)
                else:
                    connected = True

            if connected:
                telnetdict[key] = connection
            else:
                print('ERR: telnet to %s port %r failed' % (host, value[self.DICT_CTRL]), end=' ')

        # send ^C to selected connections
        for key, connection in telnetdict.items():
            if (not 's' in stopdict[key][self.DICT_FLAGS]):
                # no 's' flag: skip sending ^C
                continue
            stopcount += 1
            if verbose:
                progressMessage('sending ^C to %r (%s port %s)' % (key, key2host(key), stopdict[key][self.DICT_CTRL]))
            try:
                # 0x03 = ^C
                telnetdict[key].write(b"\x03");
            except:
                rv = 1
                if verbose:
                    print('FAILED')
                print('ERR: Exception while shutting down %r client: %r' % (key, sys.exc_info()[1]))
            else:
                if verbose:
                    print('done')

        # wait
        if (sigdelay > 0) and (stopcount > 0):
            if verbose:
                progressMessage('waiting %d seconds' % sigdelay)
            sleep(sigdelay)
            if verbose:
                print('done')

        # check for SHUTDOWN connections
        for key, connection in telnetdict.items():
            if (not 's' in stopdict[key][self.DICT_FLAGS]):
                # no 's' flag: skip checking
                continue
            try:
                # wait for SHUT DOWN message
                response = telnetdict[key].read_very_eager()
            except:
                rv = 1
                if verbose:
                    print('FAILED')
                print('ERR: Exception while reading %r client: %r' % (key, sys.exc_info()[1]))
            else:
                if response.count(self.MSG_ISSHUTTING)  or response.count(self.MSG_ISSHUTDOWN):
                    if verbose:
                        print('%r is SHUTDOWN' % key)
                    # change status to SHUTDOWN
                    self.setStatus([key], self.STATUS_SHUTDOWN)

        # send ^X to connections where status is not SHUTDOWN
        retrylist = []
        for key, connection in telnetdict.items():
            if (self.d[key][self.DICT_STATUS] == self.STATUS_SHUTDOWN):
                continue    # skip
            if verbose:
                progressMessage('sending ^X to %r (%s port %s)' % (key, key2host(key), stopdict[key][self.DICT_CTRL]))
            try:
                # 0x18 = ^X
                telnetdict[key].write(b"\x18");
                # wait for KILLED message
                response = telnetdict[key].read_until(self.MSG_KILLED, 1)
            except:
                response = '(exception)'
                rv = 1
                if verbose:
                    print('FAILED')
                retrylist.append(key)
                print('ERR: Exception while killing %r client: %r' % (key, sys.exc_info()[1]))
            else:
                if response.count(b"Restarting"):
                    retrylist.append(key)
                    if verbose:
                        print('FAILED')
                elif verbose:
                    print('done')

        # Retry: send ^X to connections for which first attempt failed
        for key in retrylist:
            if verbose:
                progressMessage('retry sending ^X to %r (%s port %s)' % (key, key2host(key), stopdict[key][self.DICT_CTRL]))
            try:
                # 0x18 = ^X
                telnetdict[key].write(b"\x18");
                # wait for KILLED message
                response = telnetdict[key].read_until(self.MSG_KILLED, 1)
            except:
                response = '(exception)'
                rv = 1
                if verbose:
                    print('FAILED')
                print('ERR: Exception while killing %r client: %r' % (key, sys.exc_info()[1]))
            else:
                if verbose:
                    if response.count(b"Restarting"):
                        print('FAILED')
                    else:
                        print('done')

        # send ^Q to all connections
        for key, connection in telnetdict.items():
            if verbose:
                progressMessage('sending ^Q to %r (%s port %s)' % (key, key2host(key), stopdict[key][self.DICT_CTRL]))
            try:
                # 0x11 = ^Q
                telnetdict[key].write(b"\x11");
            except:
                rv = 1
                print('ERR: Exception while quitting %r procServ: %r' % (key, sys.exc_info()[1]))
            else:
                # change status to NOCONNECT
                self.setStatus([key], self.STATUS_NOCONNECT)
                if verbose:
                    print('done')

        # close all connections
        for key, connection in telnetdict.items():
            connection.close()

        return rv

    #
    # stopAll - call stop() with an empty id_list
    #
    def stopAll(self, verbose=0, sigdelay=1):
        return self.stop([], verbose, sigdelay)

    #
    # stop
    #
    def stop(self, id_list, verbose=0, sigdelay=1, only_static=0):
        rv = 0      # return value

        if self.isEmpty():
            # configuration is empty -- nothing to disconnect
            if verbose:
                print('nothing to disconnect')
        else:
            stopdict = dict()
            for key, value in self.d.items():

                if len(id_list) > 0:
                    # if id_list is nonempty and UniqueID is not in it,
                    # skip this entry
                    if key2uniqueid(key) not in id_list:
                        continue
                else:
                    # if id_list is empty and 'k' flag is set,
                    # skip this entry
                    if 'k' in value[self.DICT_FLAGS]:
                        if verbose and value[self.DICT_STATUS] != self.STATUS_NOCONNECT:
                            print('\'%s\' not stopped: this is a static task' % \
                                key2uniqueid(key))
                        continue

                if only_static and ('k' not in self.d[key][self.DICT_FLAGS]):
                    # only_static flag was passed in and this entry does not 
                    # have the 'k' flag set: skip this entry
                    continue
                    
                # if process is not NOCONNECT, add it to dictionary
                if value[self.DICT_STATUS] != self.STATUS_NOCONNECT:
                    stopdict[key] = value
            try:
                rv = self.stopDictionary(stopdict, verbose, sigdelay)
            except:
                print('stopDictionary() Error:', sys.exc_info()[1])

        # done
        return rv

    #
    # getProcessCounts
    #
    # RETURNS: Two values: static process count, dynamic process count
    #
    def getProcessCounts(self):

        # count the processes that are not NOCONNECT
        staticProcessCount = 0
        dynamicProcessCount = 0
        for key, value in self.d.items():
            if (value[self.DICT_STATUS] != self.STATUS_NOCONNECT):
                if ('k' in self.d[key][self.DICT_FLAGS]):
                    staticProcessCount += 1
                else:
                    dynamicProcessCount += 1

        return staticProcessCount, dynamicProcessCount

    #
    # getStartUser
    #
    # RETURNS: Two values: username, hostname
    #
    def getStartUser(self):
        return self.procmgr_macro.get('USER', '(unknown)'), self.procmgr_macro.get('HOST', '(unknown)')

    #
    # setStatus
    #
    # This method sets the status for each process in a specified list.
    # 
    # RETURNS: 0 on success, 1 on error.
    #
    def setStatus(self, key_list, newStatus):

        for key in key_list:
            if key in self.d:
                self.d[key][self.DICT_STATUS] = newStatus
            else:
                print("ERR: setStatus: key '%s' not found" % key)
                return 1

        return 0

#
# main
#
if __name__ == '__main__':
    basename = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    usage = "Usage: %s configfile [platform]" % basename
    default_platform = 1
    if len(sys.argv) == 2:
        platform = default_platform
    elif len(sys.argv) != 3:
        print(usage)
        sys.exit(1)
    else:
        try:
            platform = int(sys.argv[2])
        except ValueError:
            platform = default_platform
            print("%s: invalid platform (%s), using default (%d)" % (sys.argv[0], sys.argv[2], platform))

    # collect the status, reading from the config file
    print('-------- calling ProcMgr(%s, %d)' % (sys.argv[1], platform))
    try:
        procMgr = ProcMgr(sys.argv[1], platform)
    except IOError:
        print("%s: error while accessing %s %d" % (sys.argv[0], sys.argv[1], platform))
        sys.exit(1)

    # error check
    if procMgr.isEmpty():
        print('configuration is empty.  exiting now')
        sys.exit(1)

    # show the status
    print('-------- calling show(0)')
    procMgr.show(0)

    # stop all
    print('-------- calling stopAll()')
    procMgr.stopAll()

    # delete the previous status (is this right?)
    del procMgr

    # collect the status again
    print('-------- calling ProcMgr(%s, %d)' % (sys.argv[1], platform))
    try:
        procMgr = ProcMgr(sys.argv[1], platform)
    except IOError:
        print("%s: error while accessing %d" % (sys.argv[0], sys.argv[1], platform))
        sys.exit(1)

    # error check
    if procMgr.isEmpty():
        print('configuration is empty.  exiting now')
        sys.exit(1)

    # show the status again
    print('-------- calling show(0)')
    procMgr.show(0)

    # start all
    print('-------- calling startAll(1)')
    procMgr.startAll(1)

    # delete the previous status (is this right?)
    del procMgr

    # collect the status again
    print('-------- calling ProcMgr(%s, %d)' % (sys.argv[1], platform))
    try:
        procMgr = ProcMgr(sys.argv[1], platform)
    except IOError:
        print("%s: error while accessing %s %d" % (sys.argv[0], sys.argv[1], platform))
        sys.exit(1)

    # show the status again
    print('-------- calling show(0)')
    procMgr.show(0)

    print('-------- done')
