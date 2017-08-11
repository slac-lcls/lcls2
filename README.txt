dataSource.cc iterates over datagrams in a file, but does not properly handle reference counts
---Contains the c extension for a dataSource object. This object opens a data file on initialization, and parses through it with a function called "nextDgram". This function takes a single datagram from the data file, assigns it to a Dgram object, then returns that object.
------(INTENDED USE: open a file as a datasource object, parse through that whole file in a python loop with multiple calls to        "nextDgram", returning a navigatable Dgram object with each pass of the loop)
---I have been linking this file to the dgram.so shared library in the CMake text file, to then be accessed in python via the "dgram" module. This file also contains the Dgram python type object, so when dataSource.cc is linked to the dgram.so shared library, both of these objects are accessible from the "dgram" module.
---dSourceTest.py creates an instance of a dataSource object and then calls its function "nextDgram" twice to create two new Dgram instances.
 
 
Current best example of reference count handeling is in testprog.cc.
---testProg.cc is compiled to shared library testType.so
---testType module contains an object called "Noddy" that has a function called "create" that calls "SimpleNewFromData" on some test data. Calling this function increases the reference count of "Noddy" instance, and returns a new array. if this array is used, its reference count is increased accordingly; if this array is deleted, the "Noddy" instance's reference count is decreased. This behavior is displayed in the testref.py routine.
---testref.py tests reference counting behavior for testType. This program tests how the testType "Noddy" behaves while going out of scope in multiple functions.


clemens.cc is the same as testprog.cc, except instead of using a "create" function, it uses the "[]" operator
---This program displays the same reference counting behavior that we saw in "testProg.cc". The only difference is it accomplishes this using the "[]" operator instead of a "create" function


getItemTest.cc is my current attempt to use the __getItem__ "[]" operator inside the "Dgram" object
---This code is having issues with storing the datagram information. It cannot loop over the fields in the descriptor. I think there are issues with how the data is being accessed.
---This is old code that I haven't kept up to date with the changes we've been making, so the solution to this problem might be in some of the other programs.


Data structure hierarchy idea:
---Have an upper-level python type object called "Dgram" that contains a dictionary and the datagram information.
---This dictionary would be named "__dict__" so that this object could have tab completion. This dictinoary would be populated with lower-level python type objects called "Xtc".
---The "Xtc" type would correspond to a single sensor (a single xtc) inside the datagram and would contain either:

      A) Another dictionary called "__dict__" with all the information from that xtc (for tab complete purposes)
      B) A list of keys for the data in that xtc and a specified __getitem__ function for using the "[]" operator
      
--- Option A) requires the tp_getattro function to call SimpleNewFromData in order to get the reference counting correct. This could pose problems later on in development.
--- Option B) requires the "[]" operator to call SimpleNewFromData (this code is already written in the "clemens.cc" routine), and would NOT have the tab completing utility.
