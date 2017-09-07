'''
This script is the second of the primitive offline event builder trilogy.
This is a single core script that matches the indices of all identical 
timestamps and stores them into an array which is dumped into a pickle
file for future analysis.
'''

import pickle, h5py
import numpy as np

#this is how many h5 files the script has to run through
#CURRENTLY ONLY 10 H5 FILES ARE SUPPORTED
runs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
times = []
files = []
#opens all the h5 files.
for  g in runs:
    filename = 'file'+str(g)+'.h5'
    f = h5py.File(filename)
    files.append(f)

all_ts_list = []
#look for interesting events => truncated small data
truncSD = [i for i, x in enumerate(files[0]['smalldata']) if 'red' in x]
#pull timestamps of interesting events => truncated timestamps
truncTS = [files[0]['timestamp1'][i] for i in truncSD]
#Create a list of all the timestamps
all_ts_list.append(np.array(truncTS))

#opening the timestamp arrays to crate a master list of timestamps
for i in range(1, len(files)):
    all_ts_list.append(np.array(files[i]['timestamp%s' %(str(i+1))]))
number_of_files = len(all_ts_list)

#parameters used to prepare for the loop 
matched_indices = []
currindex = number_of_files*[0]
master_index = 0
indices = []
singleMatch = []
#looping through h5 files to find matching indices and appending them to an array
while master_index < len(truncTS):
  filenumber = 0
  while filenumber < number_of_files:  
    compared_index = 0  
    value_to_compare = truncTS[master_index]
    while compared_index < len(truncTS):
        value_being_compared = all_ts_list[filenumber][compared_index]
        if value_to_compare == value_being_compared:
            singlematch.append([filenumber, compared_index])
        compared_index +=1
    filenumber+=1
  indices.append(singleMatch)
  singleMatch = []
  master_index +=1
print 'ALL MATCHES:', indices

#dumping the nested arrays of mathced indices into a pickle.
file_Name = "eventpickle"
fileObject = open(file_Name, 'wb')
pickle.dump(indices, fileObject)
fileObject.close()
