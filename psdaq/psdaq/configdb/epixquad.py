import rogue
import pyrogue as pr
import epix

#
#  Need a reorganized epixQuad.Top (a la cameralink-gateway)
#    Factor into devTarget (KCU, PgpG4,...) and devRoot
#      devTarget(pr.Device) contains register description on host card
#      devRoot(shared.Root):
#        maps PCIe
#        instanciates devTarget
#        creates streams
#        instanciates and connects feb

import pyrogue.protocols

import surf.axi                     as axi
import surf.devices.analog_devices  as analog_devices
import surf.devices.cypress         as cypress
import surf.xilinx                  as xil
import surf.protocols.ssi           as ssi

import ePixAsics as epix

import ePixQuad

class EpixQuad(pr.Device):
   def __init__(   self,       
                   name        = "Top",
                   description = "Container for EpixQuad",
                   memMap      = None,
                   **kwargs):
      super().__init__(name=name, description=description, **kwargs)
      
      @self.command()
      def ClearAsicMatrix():
         # save TrigEn state and stop
         self.SystemRegs.enable.set(True)
         trigEn = self.SystemRegs.TrigEn.get()
         self.SystemRegs.TrigEn.set(False)
         # clear matrix in all enabled ASICs
         for i in range(16):
            self.Epix10kaSaci[i].ClearMatrix()
         # restore TrigEn state
         self.SystemRegs.TrigEn.set(trigEn)
      
      ######################################################################
      
      # Add devices
      self.add(ePixQuad.EpixVersion( 
         name    = 'AxiVersion', 
         memBase = memMap, 
         offset  = 0x00000000, 
         expand  = False,
      ))
      
      self.add(ePixQuad.SystemRegs( 
         name    = 'SystemRegs', 
         memBase = memMap, 
         offset  = 0x00100000, 
         expand  = False,
      ))
      
      self.add(ePixQuad.AcqCore( 
         name    = 'AcqCore', 
         memBase = memMap, 
         offset  = 0x01000000, 
         expand  = False,
      ))
      
      self.add(ePixQuad.RdoutCore( 
         name    = 'RdoutCore', 
         memBase = memMap, 
         offset  = 0x01100000, 
         expand  = False,
      ))
      
      self.add(axi.AxiStreamMonitoring( 
         name    = 'RdoutStreamMonitoring', 
         memBase = memMap, 
         offset  = 0x01300000, 
         expand  = False,
      ))
      
      self.add(ssi.SsiPrbsTx(
         name    = 'PrbsTx',
         memBase = memMap, 
         offset  = 0x01400000, 
         expand  = False, 
         enabled = False,
      ))
      
      self.add(ePixQuad.PseudoScopeCore( 
         name    = 'PseudoScopeCore', 
         memBase = memMap, 
         offset  = 0x01200000, 
         expand  = False,
      ))
      
      if (hwType != 'simulation'): 
         self.add(ePixQuad.VguardDac( 
            name    = 'VguardDac', 
            memBase = memMap, 
            offset  = 0x00500000, 
            expand  = False,
         ))
      
      self.add(ePixQuad.EpixQuadMonitor( 
         name    = 'EpixQuadMonitor', 
         memBase = memMap, 
         offset  = 0x00700000, 
         expand  = False,
      ))
      
      ##################################################
      ## DO NOT MAP. MICROBLAZE IS THE ONLY 
      ## AXI LITE MASTER THAT SHOULD ACCESS THIS DEVICE
      ##################################################
      #self.add(ePixQuad.AxiI2cMaster( 
      #    name    = 'AxiI2cMaster', 
      #    memBase = memMap, 
      #    offset  = 0x00600000, 
      #    expand  = False,
      #    hidden  = True,
      #))
      
      self.add(axi.AxiMemTester( 
         name    = 'AxiMemTester', 
         memBase = memMap, 
         offset  = 0x00400000, 
         expand  = False,
         enabled = False,
      ))
      
      for i in range(16):
         asicSaciAddr = [
            0x04000000, 0x04400000, 0x04800000, 0x04C00000,
            0x05000000, 0x05400000, 0x05800000, 0x05C00000,
            0x06000000, 0x06400000, 0x06800000, 0x06C00000,
            0x07000000, 0x07400000, 0x07800000, 0x07C00000
         ]
         self.add(epix.Epix10kaAsic(
            name    = ('Epix10kaSaci[%d]'%i),
            memBase = memMap, 
            offset  = asicSaciAddr[i], 
            enabled = False,
            expand  = False,
            size    = 0x3fffff,
         ))
      
      self.add(ePixQuad.SaciConfigCore( 
         name       = 'SaciConfigCore', 
         memBase    = memMap, 
         offset     = 0x08000000, 
         expand     = False,
         enabled    = False,
         simSpeedup = (hwType == 'simulation'),
      ))
      
      if (hwType != 'simulation'):     
      
         confAddr = [
            0x02A00000, 0x02A00800, 0x02A01000, 0x02A01800, 0x02B00000, 
            0x02B00800, 0x02B01000, 0x02B01800, 0x02C00000, 0x02C00800
         ]
         for i in range(10):      
            self.add(analog_devices.Ad9249ConfigGroup(
               name    = ('Ad9249Config[%d]'%i),
               memBase = memMap, 
               offset  = confAddr[i], 
               enabled = False,
               expand  = False,
            ))
      
      for i in range(10):      
         self.add(analog_devices.Ad9249ReadoutGroup(
            name    = ('Ad9249Readout[%d]'%i),
            memBase = memMap, 
            offset  = (0x02000000+i*0x00100000), 
            enabled = False,
            expand  = False,
            fpga    = 'ultrascale',
         ))
      
      self.add(ePixQuad.AdcTester( 
         name    = 'Ad9249Tester', 
         memBase = memMap, 
         offset  = 0x02D00000, 
         enabled = False,
         expand  = False,
         hidden  = False,
      ))
               
      if (hwType != 'simulation'):
      
         self.add(cypress.CypressS25Fl(
            offset   = 0x00300000, 
            memBase  = memMap,
            expand   = False, 
            addrMode = True, 
            hidden   = True, 
         ))                   
      
      
      # ADC startup parameters
      self.adcRstTime = 0.01
      self.serRstTime = 0.01
      self.retries = 5
      
      if path.exists('ePixQuadAdcTrainingData.txt'):
         with open('ePixQuadAdcTrainingData.txt') as f:
            self.allDelays = f.readlines()
         self.allDelays = [int(i) for i in self.allDelays] 
         if len(self.allDelays) < 90:
            self.allDelays = [-1] * 90
      else:
         self.allDelays = [-1] * 90
      
      @self.command()
      def AdcStartup():
         self.SystemRegs.enable.set(True)
         self.Ad9249Tester.enable.set(True)
         self.RdoutCore.enable.set(True)
         self.AcqCore.enable.set(True)
         self.PseudoScopeCore.enable.set(True)
         for adc in range(10):
            self.Ad9249Readout[adc].enable.set(True)
            self.Ad9249Config[adc].enable.set(True)
         
         # disable and stop all internal ADC startup activity
         self.SystemRegs.AdcBypass.set(True)
         # Wait 100 ms
         time.sleep(0.1)
         
         #load trained delays
         for adc in range(10):
            newDly = self.allDelays[adc*9]
            if newDly >= 0:
               self.Ad9249Readout[adc].FrameDelay.set(0x200+newDly)
            else:
               print("Bad stored delay. Train ADCs!")
            for lane in range(8):
               newDly = self.allDelays[adc*9+lane+1]
               if newDly >= 0:
                  self.Ad9249Readout[adc].ChannelDelay[lane].set(0x200+newDly)
               else:
                  print("Bad stored delay. Train ADCs!")
         
         # test ADCs and reset if needed
         for adc in range(10):
            while True:
               if self.testAdc(self, adc, 0) < 0 or self.testAdc(self, adc, 1) < 0:
                  self.resetAdc(self, adc)
               else:
                  break
         
         # re-enable internal ADC startup
         self.SystemRegs.AdcBypass.set(False)
         
         self.Ad9249Tester.enable.set(False)
         print('Done')
      
      @self.command()
      def AdcTrain():
         
         self.SystemRegs.enable.set(True)
         self.Ad9249Tester.enable.set(True)
         for adc in range(10):
            self.Ad9249Readout[adc].enable.set(True)
            self.Ad9249Config[adc].enable.set(True)
         
         # disable and stop all internal ADC startup activity
         self.SystemRegs.AdcBypass.set(True)
         # Wait 100 ms
         time.sleep(0.1)
         
         for adc in range(10):
            
            result = 0
            
            while True:
            
               self.resetAdc(self, adc)
               
               prevDly = self.Ad9249Readout[adc].FrameDelay.get()
               newDly = self.trainFrameAdc(self, adc, self.retries)
               
               # skip ADC if the frame training failed all times
               if newDly >= 0:
                  result = result + 1
                  print('Diff delay %d'%(prevDly-newDly))
                  self.Ad9249Readout[adc].FrameDelay.set(0x200+newDly)
                  self.allDelays[adc*9] = newDly
                  # otherwise train data lanes
                  for lane in range(8):
                     prevDly = self.Ad9249Readout[adc].ChannelDelay[lane].get()
                     newDly = self.trainDataLaneAdc(self, adc, lane, self.retries)
                     if newDly >= 0:
                        result = result + 1
                        print('Diff delay %d'%(prevDly-newDly))
                        self.Ad9249Readout[adc].ChannelDelay[lane].set(0x200+newDly)
                        self.allDelays[adc*9+lane+1] = newDly
               
               if result < 9:
                  print('ADC %d failed. Retrying forever.'%(adc))
                  result = 0
               else:
                  break
                     
         self.Ad9249Tester.enable.set(False)
         
         # flash training data
         self.flashAdcDelays(self)
         
         # save training data
         with open('ePixQuadAdcTrainingData.txt', 'w') as f:
            for item in self.allDelays:
               f.write("%s\n" % item)
         
         # re-enable internal ADC startup
         self.SystemRegs.AdcBypass.set(False)
   
      @self.command()
      def ClearAdcProm():
         self.CypressS25Fl.enable.set(True)
         self.CypressS25Fl.resetFlash()
         # erase 64kB per sector ERASE_SIZE = 0x10000
         # use space at 48MB (mcs size 16MB)
         # mcs end 0xf43efc
         self.CypressS25Fl.eraseCmd(0x3000000)
         
         # create empty prom data array
         writeArray = [0] * 64
         
         self.CypressS25Fl.setDataReg(writeArray)
         self.CypressS25Fl.writeCmd(0x3000000)
         
         # Wait for last transaction to finish
         self.CypressS25Fl.waitForFlashReady()
         
         # Start address of a burst transfer
         self.CypressS25Fl.readCmd(0x3000000)
         # Get the data
         readArray = self.CypressS25Fl.getDataReg()
         
         if readArray != writeArray:
            click.secho(
               "\n\n\
               ***************************************************\n\
               ***************************************************\n\
               Writing ADC constants to PROM failed !!!!!!        \n\
               ***************************************************\n\
               ***************************************************\n\n"
               , bg='red',
            )
         else:
            click.secho(
               "\n\n\
               ***************************************************\n\
               ***************************************************\n\
               Writing ADC constants to PROM done       \n\
               ***************************************************\n\
               ***************************************************\n\n"
               , bg='green',
            )
   
   @staticmethod
   def resetAdc(self, adc):
      
      print('Reseting ADC deserializer %d ... '%(adc), end='')
      self.SystemRegs.AdcClkRst.set(0x1<<adc)
      time.sleep(self.serRstTime)
      self.SystemRegs.AdcClkRst.set(0x0)
      time.sleep(self.serRstTime)
      print('Done')
      
      print('Reseting ADC %d ... '%(adc), end='')
      self.Ad9249Config[adc].InternalPdwnMode.set(3)
      time.sleep(self.adcRstTime)
      self.Ad9249Config[adc].InternalPdwnMode.set(0)
      time.sleep(self.adcRstTime)
      print('Done')
      
      print('Setting ADC in offset binary mode ...', end='')
      self.Ad9249Config[adc].OutputFormat.set(0)
      print('Done')
   
   @staticmethod
   def trainFrameAdc(self, adc, retry):
      print('ADC %d frame delay training'%adc)
      delayInd = []
      delayLen = []
      delayLoc = 0
      delayCnt = 0
      retryCnt = retry
      while retryCnt > 0:
         for delay in range(512):
            self.Ad9249Readout[adc].FrameDelay.set(0x200+delay)
            # Reset lost lock counter
            self.Ad9249Readout[adc].LostLockCountReset()
            # Wait 1 ms
            time.sleep(0.001)
            # Check lock status
            lostLockCountReg = self.Ad9249Readout[adc].LostLockCount.get()
            lockedReg = self.Ad9249Readout[adc].Locked.get()
            if (lostLockCountReg == 0) and (lockedReg == 1):
               print('1', end='')
               if delayLoc == 0:
                  delayInd.append(delay)
               delayLoc = 1
               delayCnt = delayCnt + 1
            else:
               print('0', end='')
               if delayLoc == 1:
                  delayLen.append(delayCnt)
               delayLoc = 0
               delayCnt = 0
            if delayLoc == 1 and delay == 511:
               delayLen.append(delayCnt)
               
         print(' ')
         if len(delayInd) > 0:
            #print(delayInd)
            #print(delayLen)
            delayIndMax = delayLen.index(max(delayLen))
            delaySet = int(delayInd[delayIndMax]+delayLen[delayIndMax]/2)
            print('Found delay %d'%(delaySet))
            break
         else:
            retryCnt = retryCnt - 1
            if retryCnt > 0:
               print('Failed. Retrying %d'%retryCnt)
               self.resetAdc(self, adc)
            else:
               print('Failed ADC %d'%(adc))
               delaySet = -1
      
      return delaySet
      
   @staticmethod
   def trainDataLaneAdc(self, adc, lane, retry):
      # enable mixed bit frequency pattern
      self.Ad9249Config[adc].OutputTestMode.set(12)
      # set the pattern tester
      self.Ad9249Tester.TestDataMask.set(0x3FFF)
      self.Ad9249Tester.TestPattern.set(0x2867)
      self.Ad9249Tester.TestSamples.set(10000)
      self.Ad9249Tester.TestTimeout.set(10000)
      self.Ad9249Tester.TestChannel.set(adc*8+lane)
      print('ADC %d data lane %d delay training'%(adc, lane))
      delayInd = []
      delayLen = []
      delayLoc = 0
      delayCnt = 0
      retryCnt = retry
      while retryCnt > 0:
         for delay in range(512):
            self.Ad9249Readout[adc].ChannelDelay[lane].set(0x200+delay)
            
            # start testing
            self.Ad9249Tester.TestRequest.set(True)
            self.Ad9249Tester.TestRequest.set(False)
            
            while (self.Ad9249Tester.TestPassed.get() != True) and (self.Ad9249Tester.TestFailed.get() != True):
               pass
            testPassed = self.Ad9249Tester.TestPassed.get()
            
            if testPassed == True:
               print('1', end='')
               if delayLoc == 0:
                  delayInd.append(delay)
               delayLoc = 1
               delayCnt = delayCnt + 1
            else:
               print('0', end='')
               if delayLoc == 1:
                  delayLen.append(delayCnt)
               delayLoc = 0
               delayCnt = 0
            if delayLoc == 1 and delay == 511:
               delayLen.append(delayCnt)
               
         print(' ')
         if len(delayInd) > 0:
            #print(delayInd)
            #print(delayLen)
            delayIndMax = delayLen.index(max(delayLen))
            delaySet = int(delayInd[delayIndMax]+delayLen[delayIndMax]/2)
            print('Found delay %d'%(delaySet))
            break
         else:
            retryCnt = retryCnt - 1
            if retryCnt > 0:
               print('Failed. Retrying %d'%retryCnt)
               self.resetAdc(self, adc)
            else:
               print('Failed ADC %d'%(adc))
               delaySet = -1
      
      # disable mixed bit frequency pattern
      self.Ad9249Config[adc].OutputTestMode.set(0)
      return delaySet
   
   @staticmethod
   def testAdc(self, adc, pattern):
      print('ADC %d testing'%adc)
      
      
      result = 0
      # Reset lost lock counter
      self.Ad9249Readout[adc].LostLockCountReset()
      # Wait 1 ms
      time.sleep(0.001)
      # Check lock status
      lostLockCountReg = self.Ad9249Readout[adc].LostLockCount.get()
      lockedReg = self.Ad9249Readout[adc].Locked.get()
      if (lostLockCountReg == 0) and (lockedReg == 1):
         result = result + 1
      else:
         print('ADC %d frame clock locking failed'%adc)
      
      
      # enable mixed bit frequency pattern
      if pattern == 0:
         self.Ad9249Config[adc].OutputTestMode.set(12)
      else:
         self.Ad9249Config[adc].OutputTestMode.set(8)
         self.Ad9249Config[adc].UserPatt1Lsb.set(0x00)
         self.Ad9249Config[adc].UserPatt1Msb.set(0x60)
      # set the pattern tester
      self.Ad9249Tester.TestDataMask.set(0x3FFF)
      if pattern == 0:
         self.Ad9249Tester.TestPattern.set(0x2867)
      else:
         self.Ad9249Tester.TestPattern.set(0x1800)
      self.Ad9249Tester.TestSamples.set(100000)
      self.Ad9249Tester.TestTimeout.set(10000)
      for lane in range(8):
         self.Ad9249Tester.TestChannel.set(adc*8+lane)
         # start testing
         self.Ad9249Tester.TestRequest.set(True)
         self.Ad9249Tester.TestRequest.set(False)
         
         while (self.Ad9249Tester.TestPassed.get() != True) and (self.Ad9249Tester.TestFailed.get() != True):
            pass
         testPassed = self.Ad9249Tester.TestPassed.get()
         
         if testPassed == True:
            result = result + 1
         else:
            print('ADC %d data lane %d locking failed'%(adc, lane))
         
      # disable mixed bit frequency pattern
      self.Ad9249Config[adc].OutputTestMode.set(0)
      
      
      if result < 9:
         return -1
      else:
         return 0
   
   
   @staticmethod
   def flashAdcDelays(self):
      self.CypressS25Fl.enable.set(True)
      self.CypressS25Fl.resetFlash()
      # erase 64kB per sector ERASE_SIZE = 0x10000
      # use space at 48MB (mcs size 16MB)
      # mcs end 0xf43efc
      self.CypressS25Fl.eraseCmd(0x3000000)
      
      
      # Create a burst data array
      #print(", ".join("0x{:04x}".format(num) for num in self.allDelays))  
      #print("-----------------------------------------------------------------------")

      # create prom data array
      writeArray = [0] * 64
      
      # copy ADC frame (1st) and lane (x8) delays words 0 to 44
      for adc in range(10):
         wordCnt = int((adc*9)/2) # 0 to 39
         shiftCnt = int((adc*9)%2)*16
         #print('wordCnt  %d'%wordCnt)
         #print('shiftCnt %d'%shiftCnt)
         writeArray[wordCnt] |= ((self.allDelays[adc*9]) & 0xffff) << shiftCnt
         for lane in range(8):
            wordCnt = int((adc*9+lane+1)/2) # 0 to 39
            shiftCnt = int((adc*9+lane+1)%2)*16
            #print('wordCnt  %d'%wordCnt)
            #print('shiftCnt %d'%shiftCnt)
            writeArray[wordCnt] |= ((self.allDelays[adc*9+lane+1]) & 0xffff) << shiftCnt
      
      #print(", ".join("0x{:04x}".format(num) for num in writeArray)) 
      #print("-----------------------------------------------------------------------")      
      
      self.CypressS25Fl.setDataReg(writeArray)
      self.CypressS25Fl.writeCmd(0x3000000)
      
      # Wait for last transaction to finish
      self.CypressS25Fl.waitForFlashReady()
      
      # Start address of a burst transfer
      self.CypressS25Fl.readCmd(0x3000000)
      # Get the data
      readArray = self.CypressS25Fl.getDataReg()
      
      if readArray != writeArray:
         click.secho(
            "\n\n\
            ***************************************************\n\
            ***************************************************\n\
            Writing ADC constants to PROM failed !!!!!!        \n\
            ***************************************************\n\
            ***************************************************\n\n"
            , bg='red',
         )
      else:
         click.secho(
            "\n\n\
            ***************************************************\n\
            ***************************************************\n\
            Writing ADC constants to PROM done       \n\
            ***************************************************\n\
            ***************************************************\n\n"
            , bg='green',
         )
      
      #print(", ".join("0x{:04x}".format(num) for num in readArray))
      #print("-----------------------------------------------------------------------")
      
      
      
