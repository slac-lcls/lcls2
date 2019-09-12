#!/usr/bin/env python3
import pyrogue as pr

import rogue.interfaces.stream

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import scipy.signal

import IPython

import random

import TimeToolDev.eventBuilderParser as eventBuilderParser

matplotlib.use("Qt5agg")
plt.ion()

# import h5py





#####################################################################        
#####################################################################        
##################################################################### 

# This class wraps matplotlib for pyrogue integration
class dsp_plotting():


    def __init__(self):

        #constants        
        self.EDGE_IDX        = 96
        self.LOW_RATE        = 120
        self.IMAGE_START     = 144
        self.IMAGE_SIZE      = 2048
        self.EDGE_LIST_SIZE  = 30

        #non constants
        self.my_kernel               = np.append(np.ones(16),-np.ones(16))

        self.edge_position           = np.array([])
        self.edge_position_low_rate  = np.array([])
        self.py_calc_edge_position   = np.array([])
        
        self.counter = 0



        self.fig, self.axes = plt.subplots(4)

        self.fig.set_figwidth(5)
        self.fig.set_figheight(12)

        markers = [None,None,None,'o']
        linewidths = [1,1,1,0]
        
        self.lines = [self.axes[i].plot([],[],marker=markers[i],linewidth=linewidths[i]) for i in range(len(self.axes))]
        for i in range(len(self.axes)):
            self.axes[i].set_xlim(0,4500)
            self.axes[i].set_ylim(0,256)
            self.axes[i].set_title("camera image")
            self.axes[i].set_xlabel("pixel")
            self.axes[i].set_ylabel("intensity")
            

        self.axes[2].set_xlim(0,1000)
        self.axes[2].set_ylim(0,2500)
        self.axes[2].set_title("edge")
        self.axes[2].set_ylabel("position")
        self.axes[2].set_xlabel("event #")
        
        self.axes[3].set_ylim(0,2500)
        self.axes[3].set_xlim(0,2500)
        self.axes[3].set_title("edge position")
        self.axes[3].set_ylabel("firmware")
        self.axes[3].set_xlabel("software")

        self.thismanager = plt.get_current_fig_manager()
        
        #self.thismanager.resize(500,1250)

        plt.tight_layout()
        plt.show()

    def high_rate_processing(self, *args,**kwargs):
    
        pframe= args[0]

        self.edge_position = np.append(self.edge_position,pframe.edge_position)[-1000:]
        self.counter += 1 


        return

    def low_rate_processing(self,*args,**kwargs):

        pframe = args[0]

        #if (self.counter % self.LOW_RATE ) == 0:
        if (pframe.prescaled_frame is not None):


            #print(self.edge_position[-10:])
            #print(p_array[:144])
            #print("____________________________________________________")


            if pframe.prescaled_frame is not None:
                self.edge_position_low_rate  = np.append(self.edge_position_low_rate,  pframe.edge_position)[-self.EDGE_LIST_SIZE:]            
                self.py_calc_edge_position   = np.append(self.py_calc_edge_position ,  np.argmax(scipy.signal.convolve(pframe.prescaled_frame,self.my_kernel)))[-self.EDGE_LIST_SIZE:]


                self.plot(*args,**kwargs)


            self.counter = 1



        return
    
    def plot(self,*args,**kwargs):
        pframe  = args[0]
        if(pframe.prescaled_frame is not None):
            for i in range(2):
                self.lines[i][0].set_data(np.arange(len(pframe.prescaled_frame)),pframe.prescaled_frame)        
                #plt.pause(0.05)
        
        #self.edge_position = np.append(self.edge_position,args[96]+args[97]*256)[:100]
        self.lines[2][0].set_data(np.arange(len(self.edge_position)),self.edge_position)

        self.lines[3][0].set_data(self.py_calc_edge_position,self.edge_position_low_rate)
        
        return        



#####################################################################        
#####################################################################        
##################################################################### 

# This class emulates the Piranha4 Test Pattern
class TimeToolTxEmulation(rogue.interfaces.stream.Master):
    # Init method must call the parent class init
    def __init__(self):
        super().__init__()
        self._maxSize = 2048

    # Method for generating a frame
    def myFrameGen(self,*args):
        # First request an empty from from the primary slave
        # The first arg is the size, the second arg is a boolean
        # indicating if we can allow zero copy buffers, usually set to true
        frame = self._reqFrame(self._maxSize, True) # Here we request a frame capable of holding 2048 bytes

        # Create a 2048 byte array with an incrementing value
        #ba = bytearray([(i&0xFF) for i in range(self._maxSize)])
        #IPython.embed()
        if(0==len(args)):
              ba = self.make_byte_array()
        else:
              ba=args[0]
        #print(self.make_byte_array())

        # Write the data to the frame at offset 0
        frame.write(ba,0)
        
        # Send the frame to the currently attached slaves
        self._sendFrame(frame)

    def make_byte_array(self):
        return bytearray([(i&0xFF) for i in range(self._maxSize)])

#####################################################################        
#####################################################################        
#####################################################################        
        
#One of the goals this code satisfies is to to facilitate communication between registers in Xilinx's kcu1500 FPGA and a host linux machine.
#See comment in TimeTool.py for how to make rogue aware of a FPGA register to communicate with.
class TimeToolRx(pr.Device,rogue.interfaces.stream.Slave):

    def __init__(self, name='TimeToolRx', **kwargs):
        print("Initializing TimeToolRx")
        rogue.interfaces.stream.Slave.__init__(self)
        pr.Device.__init__(self,name=name,**kwargs)

        self.add(pr.LocalVariable(
            name        = 'frameCount',   
            value       = 0, 
            mode        = 'RO', 
            pollInterval= 1,
        ))
        
        self.add(pr.LocalVariable(
            name        = 'lengthErrors', 
            value       = 0, 
            mode        = 'RO', 
            pollInterval= 1,
        ))
        
        self.add(pr.LocalVariable( 
            name        = 'dataErrors',   
            value       = 0, 
            mode        = 'RO', 
            pollInterval= 1,
        ))
        
        self.add(pr.LocalVariable( 
            name        = 'frameLength',   
            description = 'frameLength = 2052 # sn : medium mode, 8 bit, frameLength = 4100 # sn : medium mode, 12 bit',
            value       = 2052, 
            mode        = 'RW', 
        )) 

        self.dsp_plotting = dsp_plotting()

        self.to_save_to_h5 = []

        for i in range(8):
            self.add(pr.LocalVariable( name='byteError{}'.format(i), disp='{:#x}', value=0, mode='RO', pollInterval=1))


    def _acceptFrame(self,frame):
        self.frame = frame
        p = bytearray(frame.getPayload())
        frame.read(p,0)

        my_mask = np.arange(36)
        if(len(p)>100):
              my_mask = np.append(my_mask,np.arange(int(len(p)/2),int(len(p)/2)+36))
              my_mask = np.append(my_mask,np.arange(len(p)-36,len(p)))

        to_print = np.array(p)[-1:] 
        #print(np.array(p)[:96],to_print) #comment out for long term test
        if(len(p)==0):
            return         
        
        ###################################################################
        ###### DSP processing and display updating ########################
        ###################################################################

        #parse the output before displaying

        #p_array = np.array(p)

        pframe = eventBuilderParser.timeToolParser()
        pframe.parseData(p)
        #print(pframe.edge_position)

        
        self.dsp_plotting.high_rate_processing(pframe)
        self.dsp_plotting.low_rate_processing(pframe)

        #full_frame.print_info()
 
         
    
        ###################################################################
        ###################################################################
        ###################################################################

        self.frameCount.set(self.frameCount.value() + 1,False)
     
        berr = [0,0,0,0,0,0,0,0]
        frameLength = self.frameLength.get()
        if len(p) != frameLength:
            #print('length:',len(p))
            self.lengthErrors.set(self.lengthErrors.value() + 1,False)
        else:
            for i in range(frameLength-4):
                exp = i & 0xFF
                if p[i] != exp:
                    #print("Error at pos {}. Got={:2x}, Exp={:2x}".format(i,p[i],exp))
                    d = p[i] ^ exp
                    c = i % 8
                    berr[c] = berr[c] | d
                    self.dataErrors.set(self.dataErrors.value() + 1,False)
        #print(len(p))
        # to_print = np.array(p)[-16:]
        # print(np.array(p)[:24],to_print) #comment out for long term test
        #self.to_save_to_h5.append(np.array(p))
        for i in range(8):
            self.node('byteError{}'.format(i)).set(berr[i],False)


    def close_h5_file(self):
        print("the thing that is not a destructor is working")
        self.my_h5_file['my_data'] = self.to_save_to_h5
        self.my_h5_file.close()
        print(self.to_save_to_h5)
        
    def countReset(self):
        self.frameCount.set(0,False)
        self.lengthErrors.set(0,False)
        self.dataErrors.set(0,False)
        
#####################################################################        
#####################################################################        
#####################################################################         

# sub-classing the TimeToolRx class
class TimeToolRxVcs(TimeToolRx):

    def __init__(self, name='TimeToolRx', **kwargs):
        print("Initializing TimeToolRxVcs")
        super().__init__(name=name,**kwargs)
       

    def _acceptFrame(self,frame):
        print("TimeToolRxVcs accepting frame ")
        p = bytearray(frame.getPayload())
        frame.read(p,0)
        self.unparsed_data = p
        print(len(p))
        my_mask = np.arange(36)
        if(len(p)>100):
              my_mask = np.append(my_mask,np.arange(int(len(p)/2),int(len(p)/2)+36))
              my_mask = np.append(my_mask,np.arange(len(p)-36,len(p)))

        to_print = np.array(p)[-1:] 
        #print(np.array(p)[:96],to_print) #comment out for long term test
        #print(np.array(p)[my_mask])
        self.parsed_data = np.array(p)[my_mask]
        print("____________________________________________________")
        self.frameCount.set(self.frameCount.value() + 1,False)
     
        '''berr = [0,0,0,0,0,0,0,0]
        #frameLength = 4100 # sn : medium mode, 12 bit
        frameLength = 2052 # sn : medium mode, 8 bit
        #if len(p) != 2048: 
        if len(p) != frameLength:
            #print('length:',len(p))
            self.lengthErrors.set(self.lengthErrors.value() + 1,False)
        else:
            for i in range(frameLength-4):
                exp = i & 0xFF
                if p[i] != exp:
                    #print("Error at pos {}. Got={:2x}, Exp={:2x}".format(i,p[i],exp))
                    d = p[i] ^ exp
                    c = i % 8
                    berr[c] = berr[c] | d
                    self.dataErrors.set(self.dataErrors.value() + 1,False)
        #print(len(p))
        to_print = np.array(p)[-16:]
        print(np.array(p)[:24],to_print) #comment out for long term test
        #self.to_save_to_h5.append(np.array(p))
        for i in range(8):
            self.node('byteError{}'.format(i)).set(berr[i],False)'''
        
#####################################################################        
#####################################################################        
#####################################################################          
        
