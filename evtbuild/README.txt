The (Primitive) Offline Event Builder

---------Quick Start Guide:----------
The Primitive Offline event builder consists of three scripts that should be run in order:
1) The h5 file creation script should be run first in order to create the mock data.
The file creation script is usually ran by submitting a batch job. Examples of this are:
  bsub -q psnehq -n 1 -o dataCreationLog.txt python filecreate.py

2) The pickle creation script is the second script in the process and is a single core script
that gathers the indices of all matching timestamps across all of the h5 files. 
Pickle is created via batch job as well:
  bsub -q psnehq -n 1 -o pickleCreationLog.txt python picklecreate.py

3) The analysis script is the last script to be executed. In this script the client cores fetch
event data from the indices retrieved from the pickle file created in the previous step.
This script uses MPI and so the number of cores for analysis will need to be specified.
The event indices are distributed in user-defined batches that are specified at the end of
the execution line. A batch size of 5 means that each core has to gather data from 5 
matched events. Examples of analysis exection are as follows:
  bsub -q psnehq -n 34 -o 16coreAnalysis.txt mpirun python mpiscript.py 100

----------Overview----------
The primitive offline event builder is used to create a mock up of LCLS-II event data
to test a data reduction pipeline intended to filter through "interesting" events. The 
file creation script simply makes h5 files to simulate the amount of diffraction data
expected to be produced by LCLS. In the peamble of the file creation script you may
specify the number of h5 files desired though the number_of_files variable. An 
approximate filesize varible called file1Size is also intended to be specified but is 
CURRENTLY NOT WORKING. To test the script using kilobyte size files, a file1Size of 0
should be used. For larger files any number can be specified instead. However, since
the script was created by an intern it is off by about 50%. A file1Size of ~5 will
constitute a 10GB h5 file. This is intended to be fixed at a later date.
Standard tests for the summer of 2017 composed of 10 h5 files with a file1Size of 10GB.

The first h5 file created is the principle h5 file. It contains a complete set of
timestamps and an array of event data that correspond to the timestamps. Also included
is an array of "small data" used to filter out "interesting" events. This small data 
array consists of randomly generated colors; red, blue, and green. The analysis script
takes all indices labeled "red" as being interesting for continued analysis and only
pairs indices from timestamps with the red label. All of the secondary h5 files contain 
timestamp arrays and event arrays. Random timestamps and their corresponding events have
been removed in order to create variable length timestamp data. These secondary h5 files
are therefore about half the size of the principle h5 file. The analysis script is tested
based on how fast it can match the remaining events over the rest of the h5 files. 

Our second script is the pickle creation script. Since our event builder relies on a single
master core to find the indices of matching timestamps, analysis times can be agonizingly 
slow. In order to combat this issue, a seperate script has been made for just the master
core to pair the matching events and dump the reults into a pickle file. This pickle file
contains nested lists consisting of tuples. These tuples reveal the file number and array
index of matching timestamps.

The final script in the offline event builder is the analysis algorithm. In this script,
the master core takes the indices stored in the pickle file and distrubtes them to client
cores by batch. This script is timed and each core will print out the time it takes to
complete their analysis. The clock starts when the master core starts distributing data and
each client core prints out their finishing time. Since all client cores return their analysis
time, only the last print statement containing the longest time should be considered for the
time it takes for complete analysis of all the event data.

For analysis, core numbers as multiples of 16 are usually used such as 32, 64, and 128. Batch
numbers typically range from 1 to 10,000. Batch numbers less than 50 are seen to run for a
very long time. Batch numbers greater than 2000 usually flood client cores with data and tend
to impede analysis time. Because of this, a batch size of 1000 on 16, 32, or 64, cores is
suggested.

----------Trivial Details----------
In the file creation script, event arrays have variable length but every single event has 
250000 integers, making each event about a megabyte in size. Using traditional print statements
to check data of this size usually results in memory errors. Thus HDFview is recommended
to look through and check the h5 files. 

HDFview can be found here:
  https://support.hdfgroup.org/products/java/release/download.html

On NERSC, h5ls can be used instead:
   module load cray-hdf5
   h5ls file1.h5

Because of the large amount of data being created, arrays are written in "chunks" that are 
determined in the script. The current chunk size is optimized for our megabyte size event data
and should only be changed in the most dire of circumstances. The event data is written to a
data set called "bigData."

The anslysis file loops through the event data and grabs matching events based on the indices
distrbuted by the master core. This script may need to be changed to loop through the data 
faster. Currently, the loops made by Ian are rudimentary and more sophisticated tactics may
make the script run smoother such as numpy's searchsorted.

----------Future Developments----------
This offline event builder, after being optimized and formalized should be translated into C
(probably by the use of cython). The C code will be closer to the computer, allowing the 
script to generally run faster. This code will also need to be tested with a chunk size
of 4 to test for memory errors as a chunk size of 5 is too large while chunks of 3 make 
incomplete datasets. (Chunking for this script needs to be in even numbers) This code is
also intended to be tested on NERSC's burst buffer using 100GB h5 files. The burst buffer
file system is closer to the filesystem LCLS-II will use.

----------Known Issues----------
The analysis only can accomodate 10 h5 files so the pickle creation script will need to be
changed in order to run through any number of scripts.

The pickle creation is VERY slow. This will need to be changed to make the analysis times
more reasonable. Time is wasted mainly due to the persistence of the script. Each timestamp
will have at most one match in any given file. When the pickle file finds a match, it records
it in the indices array but continues to check the rest of the timestamps in the h5 file though
there will be no more matches to the index being analysed. An "if" statement will need to be
included so that the script will move on to the next index when it finds a match rather than
persisting to look through indices after a match has been found.

The analysis script uses definitions to determine what the master and client cores should do.
Because of this, the variables used in the script need to be defined globally and referred to
globally. It is suggested that a class be created rather than a definition to allow the use
of variables without having to call them globally.
