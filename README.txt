dataSource.cc iterates over datagrams in a file, but does not properly handle reference counts
---create a python script that demonstrates the properties of this module
---creates the dgram.so file, and contains the dgram python type 

Current best example of reference count handeling is in testprog.cc.
---testProg.cc creates shared library testType.so
---testref.py tests reference counting behavior for testType
---create another python script that tests multiple deallocs

clemens.cc uses the tp_getitem "[]" operator and calls simplnewfromdata on arrays in dgram
---at the moment, it is unclear whether reference counting works in this script
