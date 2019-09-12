------------------------------------------------------------------------------
-- File       : AXILtoFIRcoef.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-12-04
-- Last update: 2017-12-04
-------------------------------------------------------------------------------
-- Description:
-------------------------------------------------------------------------------
-- This file is part of 'axi-pcie-core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'axi-pcie-core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
--use ieee.std_logic_unsigned.all;
--use ieee.std_logic_signed.all;
use ieee.numeric_std.ALL;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPkg.all;
use work.SsiPkg.all;
use work.AxiPciePkg.all;
use work.TimingPkg.all;
use work.Pgp2bPkg.all;

library unisim;
use unisim.vcomponents.all;

-------------------------------------------------------------------------------
-- This file performs the accumulation for the background subtraction
-------------------------------------------------------------------------------

entity AXILtoFIRcoef is
   generic (
      TPD_G             : time                := 1 ns;
      DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G           : boolean             := true );
   port (
      -- System Interface
      sysClk            : in    sl;
      sysRst            : in    sl;
      -- DMA Interfaces  (sysClk domain)
      dataOutMaster     : out   AxiStreamMasterType   := AXI_STREAM_MASTER_INIT_C;
      dataOutSlave      : in    AxiStreamSlaveType    := AXI_STREAM_SLAVE_INIT_C;
      configOutMaster   : out   AxiStreamMasterType   := AXI_STREAM_MASTER_INIT_C;
      configOutSlave    : in    AxiStreamSlaveType    := AXI_STREAM_SLAVE_INIT_C;

      -- AXI-Lite Interface
      axilReadMaster    : in    AxiLiteReadMasterType;
      axilReadSlave     : out   AxiLiteReadSlaveType;
      axilWriteMaster   : in    AxiLiteWriteMasterType;
      axilWriteSlave    : out   AxiLiteWriteSlaveType);
end AXILtoFIRcoef;

architecture mapping of AXILtoFIRcoef is

   constant PGP2BTXIN_LEN                  : integer             := 19;
   constant CAMERA_RESOLUTION_BITS         : positive            := 8;
   constant CAMERA_PIXEL_NUMBER            : positive            := 2048;
   constant FIR_COEFFICIENT_LENGTH         : positive            := 32;

   constant INT_CONFIG_C                   : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes=>16,tDestBits=>0);
   constant FIR_COEFICIENT_OUTPUT_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(FIR_COEFFICIENT_LENGTH, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 1, 2);
   constant DMA_AXIS_DOWNSIZED_CONFIG_G    : AxiStreamConfigType := ssiAxiStreamConfig(1, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 1, 2);

   type StateType is (
      IDLE_S,
      MOVE_S,
      MOVE_CONFIG_S);

   type RegType is record
      master          : AxiStreamMasterType;
      slave           : AxiStreamSlaveType;
      configMaster    : AxiStreamMasterType;
      configSlave     : AxiStreamSlaveType;
      axilReadSlave   : AxiLiteReadSlaveType;
      axilWriteSlave  : AxiLiteWriteSlaveType;
      scratchPad      : slv(31 downto 0);
      newCoefficients : sl;
      state           : StateType;
      counter         : natural range 0 to (CAMERA_PIXEL_NUMBER-1);
   end record RegType;

   constant REG_INIT_C : RegType := (
      master          => AXI_STREAM_MASTER_INIT_C,
      slave           => AXI_STREAM_SLAVE_INIT_C,
      configMaster    => AXI_STREAM_MASTER_INIT_C,
      configSlave     => AXI_STREAM_SLAVE_INIT_C,
      axilReadSlave   => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave  => AXI_LITE_WRITE_SLAVE_INIT_C,
      scratchPad      => (others =>'0' ),
      newCoefficients => '0',
      state           => IDLE_S,
      counter         => 0);


---------------------------------------
-------record intitial value-----------
---------------------------------------


   signal r                : RegType               := REG_INIT_C;
   signal rin              : RegType               := REG_INIT_C;

   signal outCtrl          : AxiStreamCtrlType     := AXI_STREAM_CTRL_INIT_C;
   signal configOutCtrl    : AxiStreamCtrlType     := AXI_STREAM_CTRL_INIT_C;

begin



   ---------------------------------
   -- Application
   ---------------------------------
   comb : process (r,rin, sysRst, axilReadMaster, axilWriteMaster, outCtrl,configOutCtrl) is
      variable v      : RegType := REG_INIT_C ;
      variable axilEp : AxiLiteEndpointType;
   begin

      -- Latch the current value
      v := r;

      ------------------------      
      -- AXI-Lite Transactions
      ------------------------      

      -- Determine the transaction type
      axiSlaveWaitTxn(axilEp, axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave);

      axiSlaveRegister (axilEp, x"000", 0, v.master.tData(31 downto 0));
      axiSlaveRegister (axilEp, x"004", 0, v.master.tData(63 downto 32));
      axiSlaveRegister (axilEp, x"008", 0, v.master.tData(95 downto 64));
      axiSlaveRegister (axilEp, x"00C", 0, v.master.tData(127 downto 96));
      axiSlaveRegister (axilEp, x"010", 0, v.master.tData(159 downto 128));
      axiSlaveRegister (axilEp, x"014", 0, v.master.tData(191 downto 160));
      axiSlaveRegister (axilEp, x"018", 0, v.master.tData(223 downto 192));
      axiSlaveRegister (axilEp, x"01C", 0, v.master.tData(255 downto 224));
      axiSlaveRegister (axilEp, x"020", 0, v.scratchpad);


      axiSlaveDefault(axilEp, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_DECERR_C);

      ------------------------      
      -- updating time constant
      ------------------------       

      ------------------------      
      -- Main Part of Code
      ------------------------ 

      v.slave.tReady                                        := not outCtrl.pause;
      v.master.tLast                                        := '0';
      v.master.tValid                                       := '0';
      --v.master.tData(FIR_COEFFICIENT_LENGTH*8 -1 downto 0)  := v.scratchPad(FIR_COEFFICIENT_LENGTH*8 -1 downto 0);

      v.configSlave.tReady                                  := not configOutCtrl.pause;
      v.configMaster.tLast                                  := '0';
      v.configMaster.tValid                                 := '0';

      case r.state is

            when IDLE_S =>
               if v.slave.tReady = '1' and v.scratchpad(0) = '1' then
                   v.state := MOVE_S;
               end if;
        

            when MOVE_S  => 

               v.scratchpad(0) := '0';
               
               v.master.tValid   := '1';
               v.master.tLast    := '1';

               if v.slave.tReady = '1' then
                
                  v.state           := MOVE_CONFIG_S;

               else

                  v.state           := MOVE_S;

               end if;     

            when MOVE_CONFIG_S  => 
            ------------------------------
            -- update slv logic array
            ------------------------------

               v.configMaster.tLast    := '1';
               v.configMaster.tValid   := '1';

               if v.configSlave.tReady = '1' then
                 
                  v.state                 := IDLE_S;
                  v.newCoefficients       :='0';

               else
                  v.state                := MOVE_CONFIG_S;
               end if;     



      end case;

      -------------
      -- Reset
      -------------
      if (sysRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs 
      axilReadSlave  <= r.axilReadSlave;
      axilWriteSlave <= r.axilWriteSlave;

   end process comb;

   seq : process (sysClk) is
   begin
      if (rising_edge(sysClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

   ---------------------------------
   -- Output FIFO
   ---------------------------------
   U_OutFifo: entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => false,
         GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_PAUSE_THRESH_G => 500,
         SLAVE_AXI_CONFIG_G  => FIR_COEFICIENT_OUTPUT_CONFIG_G,
         MASTER_AXI_CONFIG_G => DMA_AXIS_DOWNSIZED_CONFIG_G)
      port map (
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => r.master,
         sAxisCtrl   => outCtrl,
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => dataOutMaster,
         mAxisSlave  => dataOutSlave);

   ---------------------------------
   -- Config Output FIFO
   ---------------------------------
   U_OutFifo_2: entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => false,
         GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_PAUSE_THRESH_G => 500,
         SLAVE_AXI_CONFIG_G  => DMA_AXIS_DOWNSIZED_CONFIG_G,
         MASTER_AXI_CONFIG_G => DMA_AXIS_DOWNSIZED_CONFIG_G)
      port map (
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => r.configMaster,
         sAxisCtrl   => configOutCtrl,
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => configOutMaster,
         mAxisSlave  => configOutSlave);

end mapping;
