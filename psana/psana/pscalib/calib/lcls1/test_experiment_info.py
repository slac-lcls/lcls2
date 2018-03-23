#!/bin/env /usr/bin/python

"""
Test modules found in the local directory. Note that this test is designed
not to have side effects such as modifying any database or file system content.

"""

print """
__________________________________
testing module psdm.file_status...
"""

import psana.pscalib.calib.experiment_info as expinfo

print "  experiment id=47    translates  as %s"    % expinfo.id2name(47)
print "  experiment sxrcom10 translates  as id=%d" % expinfo.name2id('sxrcom10')
print "  experiment data path for id=116 is %s"    % expinfo.getexp_datapath(116)

print """
 -------+------------+------+------------------------------------------------+----------
        |            |      |            activation time                     |
  instr | experiment |   id +---------------------+--------------------------+ by user
        |            |      |         nanoseconds | local timezone           |
 -------+------------+------+---------------------+--------------------------+----------"""

for instr in ('AMO','SXR','XPP','XCS','CXI','MEC','XYZ'):
    exper = expinfo.active_experiment(instr)
    if exper is None:
        print "  %3s   | *** no experiment found in the database for this instrument ***" % instr
    else:
        print "  %3s   | %-10s | %4d | %19d | %20s | %-8s" % exper


print ''
print 'open files of the last run of experiment id 161:'
for file in expinfo.get_open_files(161):
    print file

print ''
print 'open files of run 1332 of experiment id 55:'
for file in expinfo.get_open_files(55,1332):
    print file


