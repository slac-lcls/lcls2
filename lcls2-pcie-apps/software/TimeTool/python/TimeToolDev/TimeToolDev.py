#!/usr/bin/env python3
import pyrogue as pr

import TimeToolDev.TimeToolStreams as streams
import rogue.protocols
import rogue.interfaces.stream

import XilinxKcu1500Pgp       as kcu1500
import ClinkFeb               as feb
import TimeTool               as timeTool
import surf.axi               as axi
import surf.protocols.batcher as batcher
import LclsTimingCore         as timingCore

class TimeToolDev(kcu1500.Core):

    def __init__(self,
            name        = 'TimeToolDev',
            description = 'Container for TimeTool Dev',
            dataDebug   = False,
            dev         = '/dev/datadev_0',# path to PCIe device
            version3    = False,           # true = PGPv3, false = PGP2b
            pollEn      = True,            # Enable automatic polling registers
            initRead    = True,            # Read all registers at start of the system
            numLane     = 1,               # Number of PGP lanes
            enVcMask    = 0xff,             # channel to ignore for pgp read to work
            **kwargs
        ):
        super().__init__(
            name        = name, 
            description = description, 
            dev         = dev, 
            version3    = version3, 
            pollEn      = pollEn, 
            initRead    = initRead, 
            numLane     = numLane, 
            enVcMask    = enVcMask,
            **kwargs
        )
        self.dev    = dev,
        
                        
        # Check if not doing simulation
        if (dev!='sim'):            
            
            # Create arrays to be filled
            self._srp = [None for lane in range(numLane)]
            
            # Create the stream interface
            for lane in range(numLane):
                    
                # SRP
                self._srp[lane] = rogue.protocols.srp.SrpV3()
                pr.streamConnectBiDir(self._dma[lane][0],self._srp[lane])
                         
                # CameraLink Feb Board
                self.add(feb.ClinkFeb(      
                    name        = (f'ClinkFeb[{lane}]'), 
                    memBase     = self._srp[lane], 
                    serial      = [self._dma[lane][2],None],
                    camType     = ['Piranha4',        None],
                    version3    = version3,
                    enableDeps  = [self.Hardware.PgpMon[lane].RxRemLinkReady], # Only allow access if the PGP link is established
                    expand      = False,
                ))

        # Else doing Rogue VCS simulation
        else:
        
            # Create arrays to be filled
            self._frameGen = [None for lane in range(numLane)]
            
            # Create the stream interface
            for lane in range(numLane):  
            
                # Create the frame generator
                self._frameGen[lane] = streams.TimeToolTxEmulation()
                
                # Connect the frame generator
                pr.streamConnect(self._frameGen[lane],self._pgp[lane][1]) 
                    
                # Create a command to execute the frame generator
                self.add(pr.BaseCommand(   
                    name         = f'GenFrame[{lane}]',
                    function     = lambda cmd, lane=lane: self._frameGen[lane].myFrameGen(),
                ))  
                # Create a command to execute the frame generator. Accepts user data argument
                self.add(pr.BaseCommand(   
                    name         = f'GenUserFrame[{lane}]',
                    function     = lambda cmd, lane=lane: self._frameGen[lane].myFrameGen,
                ))               
                
        # Create arrays to be filled
        self._dbg = [None for lane in range(numLane)]        
        
        # Create the stream interface
        for lane in range(numLane):        
            # Debug slave
            if dataDebug:
                
                # Check if VCS or not
                if (dev!='sim'): 
                    #print("using TimeToolRx")
                    self._dbg[lane] = streams.TimeToolRx(expand=True)
                else:
                    #print("using TimeToolRxVcs")
                    self._dbg[lane] = streams.TimeToolRxVcs(expand=True)
                
                # Connect the streams
                pr.streamTap(self._dma[lane][1],self._dbg[lane])
                
                # Add stream device to root class
                self.add(self._dbg)
                
        # Time tool application
        self.add(timeTool.Application(
            memBase = self._memMap,
            offset  = 0x00C00000,
            numLane = numLane,
        ))
        
        self.add(pr.LocalVariable(
            name        = 'RunState', 
            description = 'Run state status, which is controlled by the StopRun() and StartRun() commands',
            mode        = 'RO', 
            value       = False,
        ))        
        
        @self.command(description  = 'Stops the triggers and blows off data in the pipeline')        
        def StopRun():
            print ('TimeToolDev.StopRun() executed')
            
            # Get devices
            trigChDev = self.find(typ=timingCore.EvrV2ChannelReg)
            eventDev  = self.find(typ=batcher.AxiStreamBatcherEventBuilder)
            
            # Turn off the triggering
            for devPtr in trigChDev:
                devPtr.EnableReg.set(False)
                
            # Turn on the blowoff to clear out the pipeline
            for devPtr in eventDev:
                devPtr.Blowoff.set(True) 

            # Update the run state status variable
            self.RunState.set(False)
                
        @self.command(description  = 'starts the triggers and allow steams to flow to DMA engine')        
        def StartRun():
            print ('TimeToolDev.StartRun() executed')
            
            # Get devices
            trigChDev = self.find(typ=timingCore.EvrV2ChannelReg)
            eventDev  = self.find(typ=batcher.AxiStreamBatcherEventBuilder)    
            
            # Turn off the blowoff to allow steams to flow to DMA engine
            for devPtr in eventDev:
                devPtr.Blowoff.set(False)                  
            
            # Turn on the triggering
            for devPtr in trigChDev:
                devPtr.EnableReg.set(True)  
                
            # Reset all counters
            self.CountReset()
                
            # Update the run state status variable
            self.RunState.set(True)        

        # Start the system
        self.start(
            pollEn   = self._pollEn,
            initRead = self._initRead,
            timeout  = self._timeout,
        )
        
        # Check if not simulation
        if (dev!='sim'):           
            # Read all the variables
            self.ReadAll()
            # Some initialization after starting root
            for lane in range(numLane):
                self.ClinkFeb[lane].ClinkTop.Ch[0].BaudRate.set(9600)
                self.ClinkFeb[lane].ClinkTop.Ch[0].SerThrottle.set(10000)
                #self.ClinkFeb[lane].ClinkTop.Ch[0].UartPiranha4.GCP()
                self.ClinkFeb[lane].ClinkTop.Ch[0].LinkMode.setDisp('Full')
                self.ClinkFeb[lane].ClinkTop.Ch[0].DataMode.setDisp('8Bit')
                self.ClinkFeb[lane].ClinkTop.Ch[0].FrameMode.setDisp('Line')
                self.ClinkFeb[lane].ClinkTop.Ch[0].TapCount.set(8)                    
                self.ClinkFeb[lane].ClinkTop.Ch[0].UartPiranha4.SendEscape()
                self.ClinkFeb[lane].ClinkTop.Ch[0].UartPiranha4.SPF.setDisp('0')
                #self.ClinkFeb[lane].ClinkTop.Ch[0].UartPiranha4.GCP()
        else:
            # Disable the PGP PHY device (speed up the simulation)
            self.Hardware.enable.set(False)
            self.Hardware.hidden = True
            # Bypass the time AXIS channel
            eventDev = self.find(typ=batcher.AxiStreamBatcherEventBuilder)
            for dev in eventDev:
                dev.Bypass.set(0x1)            
                
    def initialize(self):
        self.StopRun()
        
