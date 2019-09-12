#!/usr/bin/env python3
#import pyrogue.gui
import TimeToolDev
import sys
import time
import gc

#tool for iterating through pyrogue objects to find their offsets.


def recursive_memory_validator(this_object,this_offset,running_dict,this_name,node_path,this_node):
	#check if this object has the _nodes member
	if("_nodes" in dir(this_object)):
		pass
	else:
		return -1	#-1 means not a node

	#now get the offset of this node
	this_offset = this_object.offset + this_offset
	this_name = this_name+"/"+this_object.name
	node_path = node_path+"/"+this_node

	for i in this_object._nodes:
		if("offset" in dir(this_object._nodes[i])):
			recursive_memory_validator(this_object._nodes[i],this_offset,running_dict,this_name,node_path,i)
		else:
			pass
	#print(this_object.name+", "+str(format(this_offset,'02x')))
	#running_dict[this_object.name] = str(format(this_offset,'02x'))
	try:
		#print(this_object.bitOffset)
		#running_dict[this_name] = {"total":str(format(this_offset+this_object.bitOffset[0],'02x')),"base":str(format(this_offset,'02x'))}
		running_dict[this_name+"/total"] = str(format(this_offset+this_object.bitOffset[0],'02x'))
		running_dict[this_name+"/base"]  = str(format(this_offset,'02x'))
		running_dict[this_name+"/node_path"]  = node_path
	except:
		pass
