import pyrogue as pr

# Modules from surf
import pyrogue as pr
import surf.axi as axi
import surf.devices.micron as micron
import surf.devices.microchip as microchip
import surf.devices.ti as ti
import surf.ethernet.udp as udp
import surf.misc as misc
import surf.protocols.rssi as rssi
import surf.xilinx as xilinx

# Modules from amc-carrier-core
import AmcCarrierCore as amcc
import AppMps as mps

class AmcCarrierCore(pr.Device):
    def __init__(   self, 
            name                = "AmcCarrierCore", 
            description         = "AmcCarrierCore", 
            rssiNotInterlaved   = True,
            rssiInterlaved      = False,            
            enableBsa           = True,
            enableMps           = True,
            expand	            = False,
            **kwargs):
        super().__init__(name=name, description=description, expand=expand, **kwargs)  

        ##############################
        # Variables
        ##############################                        
        self.add(axi.AxiVersion(            
            offset       =  0x00000000, 
            expand       =  False
        ))

        self.add(xilinx.AxiSysMonUltraScale(   
            offset       =  0x01000000, 
            expand       =  False
        ))
        
        self.add(micron.AxiMicronN25Q(
            name         = "MicronN25Q",
            offset       = 0x2000000,
            addrMode     = True,                                    
            expand       = False,                                    
            hidden       = True,                                    
        ))        

        self.add(microchip.AxiSy56040(    
            offset       =  0x03000000, 
            expand       =  False,
            description  = "\n\
                Timing Crossbar:  https://confluence.slac.stanford.edu/x/m4H7D   \n\
                -----------------------------------------------------------------\n\
                OutputConfig[0] = 0x0: Connects RTM_TIMING_OUT0 to RTM_TIMING_IN0\n\
                OutputConfig[0] = 0x1: Connects RTM_TIMING_OUT0 to FPGA_TIMING_IN\n\
                OutputConfig[0] = 0x2: Connects RTM_TIMING_OUT0 to BP_TIMING_IN\n\
                OutputConfig[0] = 0x3: Connects RTM_TIMING_OUT0 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n\
                OutputConfig[1] = 0x0: Connects FPGA_TIMING_OUT to RTM_TIMING_IN0\n\
                OutputConfig[1] = 0x1: Connects FPGA_TIMING_OUT to FPGA_TIMING_IN\n\
                OutputConfig[1] = 0x2: Connects FPGA_TIMING_OUT to BP_TIMING_IN\n\
                OutputConfig[1] = 0x3: Connects FPGA_TIMING_OUT to RTM_TIMING_IN1 \n\
                -----------------------------------------------------------------\n\
                OutputConfig[2] = 0x0: Connects Backplane DIST0 to RTM_TIMING_IN0\n\
                OutputConfig[2] = 0x1: Connects Backplane DIST0 to FPGA_TIMING_IN\n\
                OutputConfig[2] = 0x2: Connects Backplane DIST0 to BP_TIMING_IN\n\
                OutputConfig[2] = 0x3: Connects Backplane DIST0 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n\
                OutputConfig[3] = 0x0: Connects Backplane DIST1 to RTM_TIMING_IN0\n\
                OutputConfig[3] = 0x1: Connects Backplane DIST1 to FPGA_TIMING_IN\n\
                OutputConfig[3] = 0x2: Connects Backplane DIST1 to BP_TIMING_IN\n\
                OutputConfig[3] = 0x3: Connects Backplane DIST1 to RTM_TIMING_IN1\n\
                -----------------------------------------------------------------\n"\
            ))
                            
        self.add(ti.AxiCdcm6208(     
            offset       =  0x05000000, 
            expand       =  False,
        ))

        self.add(amcc.AmcCarrierBsi(   
            offset       =  0x07000000, 
            expand       =  False,
        ))

        self.add(amcc.AmcCarrierTiming(
            offset       =  0x08000000, 
            expand       =  False,
        ))

        self.add(amcc.AmcCarrierBsa(   
            offset       =  0x09000000, 
            enableBsa    =  enableBsa,
            expand       =  False,
        ))
                            
        self.add(udp.UdpEngineClient(
            name         = "BpUdpCltApp",
            offset       =  0x0A000000,
            description  = "Backplane UDP Client for Application ASYNC Messaging",
            expand       =  False,
        ))

        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvXvc",
            offset       =  0x0A000800,
            description  = "Backplane UDP Server: Xilinx XVC",
            expand       =  False,
        ))
        
        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvFsbl",
            offset       =  0x0A000808,
            description  = "Backplane UDP Server: FSBL Legacy SRPv0 register access",
            expand       =  False,
        )) 

        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvRssi[0]",
            offset       =  0x0A000810,
            description  = "Backplane UDP Server: Legacy Non-interleaved RSSI for Register access and ASYNC messages",
            expand       =  False,
        )) 

        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvRssi[1]",
            offset       =  0x0A000818,
            description  = "Backplane UDP Server: Legacy Non-interleaved RSSI for bulk data transfer",
            expand       =  False,
        ))         
        
        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvRssi[2]",
            offset       =  0x0A000830,
            description  = "Backplane UDP Server: Interleaved RSSI",
            expand       =  False,
        ))             
        
        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvApp",
            offset       =  0x0A000820,
            description  = "Backplane UDP Server for Application ASYNC Messaging",
            expand       =  False,
        ))  

        self.add(udp.UdpEngineServer(
            name         = "BpUdpSrvTiming",
            offset       =  0x0A000828,
            description  = "Backplane UDP Server for Timing ASYNC Messaging",
            expand       =  False,
        ))          
        
        for i in range(2):
            self.add(rssi.RssiCore(
                name         = f'SwRssiServer[{i}]',
                offset       =  0x0A010000 + (i * 0x1000),
                description  = "SwRssiServer Server: %i" % (i),                                
                expand       =  False,                                    
            ))       
            
        self.add(rssi.RssiCore(
            name         = "SwRssiServer[2]",
            offset       =  0x0A020000,
            description  = "SwRssiServer Server",                                
            expand       =  False,                                    
        ))            

        if (enableMps):
            self.add(mps.AppMps(      
                offset =  0x0C000000, 
                expand =  False
            ))            

    def writeBlocks(self, force=False, recurse=True, variable=None, checkEach=False):
        """
        Write all of the blocks held by this Device to memory
        """
        if not self.enable.get(): return

        # Process local blocks.
        if variable is not None:
            #variable._block.startTransaction(rogue.interfaces.memory.Write, check=checkEach)  # > 2.4.0
            variable._block.backgroundTransaction(rogue.interfaces.memory.Write)
        else:
            for block in self._blocks:
                if force or block.stale:
                    if block.bulkEn:
                        #block.startTransaction(rogue.interfaces.memory.Write, check=checkEach)  # > 2.4.0
                        block.backgroundTransaction(rogue.interfaces.memory.Write)

        # Process rest of tree
        if recurse:
            for key,value in self.devices.items():
                #value.writeBlocks(force=force, recurse=True, checkEach=checkEach)  # > 2.4.0
                value.writeBlocks(force=force, recurse=True)
                        
        # Retire any in-flight transactions before starting
        self._root.checkBlocks(recurse=True)
        
        for i in range(2):
            v = getattr(self.AmcCarrierBsa, f'BsaWaveformEngine[{i}]')
            v.WaveformEngineBuffers.Initialize()
        
        self.checkBlocks(recurse=True)
        
