from psalg.configdb.typed_json import cdict
import psalg.configdb.configdb as cdb
import sys
import IPython


def write_scratch_pad(prescaling):

    #database contains collections which are sets of documents (aka json objects).
    #each type of device has a collection.  The elements of that collection are configurations of that type of device.
    #e.g. there will be OPAL, EVR, and YUNGFRAU will be collections.  How they are configured will be a document contained within that collection
    #Each hutch is also a collection.  Documents contained within these collection have an index, alias, and list of devices with configuration IDs
    #How is the configuration of a state is described by the hutch and the alias found.  E.g. TMO and BEAM.  TMO is a collection.  
    #BEAM is an alias of some of the documents in that collection. The document with the matching alias and largest index is the current 
    #configuration for that hutch and alias.  
    #When a device is configured, the device has a unique name OPAL7.  Need to search through document for one that has an NAME called OPAL7.  This will have
    #have two fields "collection" and ID field (note how collection here is a field. ID points to a unique document).  This collection field and 
    #ID point to the actuall Mongo DB collection and document

    create = True
    dbname = 'sioanDB'     #this is the name of the database running on the server.  Only client care about this name.
    instrument = 'TST'      #

    mycdb = cdb.configdb('mcbrowne:psana@psdb-dev:9306', instrument, create, dbname)    #mycdb.client.drop_database('configDB_szTest') will drop the configDB_szTest database
    #mycdb.client.drop_database('sioanDB')
    mycdb.add_alias("BEAM")
    mycdb.add_device_config('timetool')
    
    top = cdict()
    top.setInfo('timetool', 'tmotimetool', 'serial1234', 'No comment')
    top.setAlg('timetoolConfig', [0,0,1])

    #There are many rogue fields, but only a handful need to be configured.  what are they?
    #1)  the FIR coefficients.  after accounting for parabolic fitting detection
    #2)  start and stop?
    #3)  main by pass. (still isn't working with FEX in hardware, but does in simulation.)
    #4)  all prescalers.  This has potential to crash linux kernel during hi rate and low prescaling.  How to add protection?  Slow ramp?
    #5)  low pass on background
    #6)  op code (now called readout group). there's no rogue counter part of this yet.
    #7)  the load coefficients bit needs to set to one for the FIR coefficients to be written.
    #8)  camera rate and soft or hard trigger. The soft trigger is for offline testing.  For users or just for me?
    #9)  the batcher bypass so soft trigger doesn't halt.

    ######################################################################
    ####### Keeping it simple.  Just what Giaccomo will need #############
    ######################################################################

    ########################################
    #####        prescaling          #######
    ########################################
    top.set("cl.Application.AppLane[0].Prescale.ScratchPad",int(prescaling),'UINT32')                     # testing ability to write to database
    top.set("cl.Application.AppLane[0].Prescale.DialInPreScaling",12,'UINT32')                            # prescaled raw data 
    top.set("cl.Application.AppLane[0].Fex.background_prescaler.DialInPreScaling",11,'UINT32')            # prescaled raw backgrounds (may consider accumulated backgrounds instead)
    ########################################
    #####      initial fir filter    #######
    ########################################
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet0","7f7f7f7f",'CHARSTR')                      #high part of step 
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet1","7f7f7f7f",'CHARSTR')                      #high part of step 
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet2","7f7f7f7f",'CHARSTR')                      #high part of step 
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet3","7f7f7f7f",'CHARSTR')                      #high part of step 
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet4","81818181",'CHARSTR')                      #low  part of step
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet5","81818181",'CHARSTR')                      #low  part of step
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet6","81818181",'CHARSTR')                      #low  part of step
    top.set("cl.Application.AppLane[0].Fex.FIR.CoefficientSet7","81818181",'CHARSTR')                      #low  part of step

    #top.set("cl.Application.AppLane[0].Fex.FIR.LoadCoefficients",0,'CHARSTR')                             #low  part of step

    ########################################
    #####      time constants        #######
    ########################################
    top.set("cl.Application.AppLane[0].Fex.FrameIIR.TimeConstant",8,'UINT32')


    ########################################
    #####      subtraction enabled     #####
    ########################################
    top.set("cl.Application.AppLane[0].Fex.FrameSubtractor.SubtractionActive",0,'UINT32')                   #turn background subtract on



    mycdb.modify_device('BEAM', top)

    #IPython.embed()


if __name__ == "__main__":
    write_scratch_pad(sys.argv[1])
