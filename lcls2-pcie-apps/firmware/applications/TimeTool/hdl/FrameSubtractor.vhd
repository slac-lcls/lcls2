------------------------------------------------------------------------------
-- File       : FrameSubtractor.vhd
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.ALL;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.AxiPkg.all;
use work.SsiPkg.all;
use work.Pgp2bPkg.all;

library unisim;
use unisim.vcomponents.all;

-------------------------------------------------------------------------------
-- This file performs the accumulation for the background subtraction
-------------------------------------------------------------------------------

entity FrameSubtractor is
   generic (
      TPD_G             : time                := 1 ns;
      DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G           : boolean             := true );
   port (
      -- System Interface
      sysClk              : in    sl;
      sysRst              : in    sl;
      -- DMA Interfaces  (sysClk domain)
      dataInMaster        : in    AxiStreamMasterType;
      dataInSlave         : out   AxiStreamSlaveType;
      dataOutMaster       : out   AxiStreamMasterType;
      dataOutSlave        : in    AxiStreamSlaveType;
      -- Pedestal DMA Interfaces  (sysClk domain)
      pedestalInMaster    : in    AxiStreamMasterType;
      pedestalInSlave     : out   AxiStreamSlaveType;
      -- AXI-Lite Interface
      axilReadMaster      : in    AxiLiteReadMasterType;
      axilReadSlave       : out   AxiLiteReadSlaveType;
      axilWriteMaster     : in    AxiLiteWriteMasterType;
      axilWriteSlave      : out   AxiLiteWriteSlaveType);
end FrameSubtractor;

architecture mapping of FrameSubtractor is

   constant INT_CONFIG_C                  : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes=>16,tDestBits=>0);
   constant PGP2BTXIN_LEN                 : integer             := 19;
   constant CAMERA_RESOLUTION_BITS        : positive            := 8;
   constant CAMERA_PIXEL_NUMBER           : positive            := 2048;
   constant PIXELS_PER_TRANSFER           : positive            := 16;

   --type CameraFrameBuffer is array (natural range<>) of slv(CAMERA_RESOLUTION_BITS-1 downto 0);
   type CameraFrameBuffer is array (natural range<>) of signed((CAMERA_RESOLUTION_BITS-1) downto 0);

   type StateType is (
      IDLE_S,
      MOVE_S);

   type RegType is record
      master                : AxiStreamMasterType;
      slave                 : AxiStreamSlaveType;
      pedestalMaster        : AxiStreamMasterType;
      pedestalSlave         : AxiStreamSlaveType;
      axilReadSlave         : AxiLiteReadSlaveType;
      axilWriteSlave        : AxiLiteWriteSlaveType;
      counter               : natural range 0 to (CAMERA_PIXEL_NUMBER-1);
      pedestal_counter      : natural range 0 to (CAMERA_PIXEL_NUMBER-1);
      scratchPad            : slv(31 downto 0);
      do_subtraction        : slv(31 downto 0);
      axi_test              : slv(31 downto 0);
      state                 : StateType;
      state_pedestal        : StateType;
      aSingleFrame          : CameraFrameBuffer((PIXELS_PER_TRANSFER-1) downto 0);
      storedPedestalFrame   : CameraFrameBuffer((CAMERA_PIXEL_NUMBER-1) downto 0);
   end record RegType;

   constant REG_INIT_C : RegType := (
      master                => AXI_STREAM_MASTER_INIT_C,
      slave                 => AXI_STREAM_SLAVE_INIT_C,
      pedestalMaster        => AXI_STREAM_MASTER_INIT_C,
      pedestalSlave         => AXI_STREAM_SLAVE_INIT_C,
      axilReadSlave         => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave        => AXI_LITE_WRITE_SLAVE_INIT_C,
      counter               => 0,
      pedestal_counter      => 0,
      scratchPad            => (others => '0'),
      do_subtraction        => (others=>'0'),
      axi_test              => (others=>'0'),
      state                 => IDLE_S,
      state_pedestal        => IDLE_S,
      aSingleFrame          => (others => (others => '0') ),
      storedPedestalFrame   => (others => (others => '0') ));

---------------------------------------
-------record intitial value-----------
---------------------------------------


   signal r                        : RegType     := REG_INIT_C;
   signal rin                      : RegType     := REG_INIT_C;

   signal inMaster                 : AxiStreamMasterType   :=    AXI_STREAM_MASTER_INIT_C;
   signal inSlave                  : AxiStreamSlaveType    :=    AXI_STREAM_SLAVE_INIT_C;  
   signal outCtrl                  : AxiStreamCtrlType     :=    AXI_STREAM_CTRL_INIT_C;

   signal pedestalInMasterBuf      : AxiStreamMasterType   :=    AXI_STREAM_MASTER_INIT_C;
   signal pedestalInSlaveBuf       : AxiStreamSlaveType    :=    AXI_STREAM_SLAVE_INIT_C;  



begin
   ---------------------------------
   -- No-Input FIFO. 
   ---------------------------------
   pedestalInMasterBuf    <= pedestalInMaster;     --may migrate to buffered input fifo 
   pedestalInSlave        <= pedestalInSlaveBuf;   --may migrate to buffered input fifo 

   ---------------------------------
   -- Input FIFO
   ---------------------------------
   U_InFifo: entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => 9,
         FIFO_PAUSE_THRESH_G => 500,
         SLAVE_AXI_CONFIG_G  => DMA_AXIS_CONFIG_G,
         MASTER_AXI_CONFIG_G => INT_CONFIG_C)
      port map (
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => dataInMaster,
         sAxisSlave  => dataInSlave,
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => inMaster,
         mAxisSlave  => inSlave);


   ---------------------------------
   -- Application
   ---------------------------------
   comb : process (r, sysRst, axilReadMaster, axilWriteMaster, inMaster,pedestalInMasterBuf, outCtrl) is
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

      axiSlaveRegister (axilEp, x"000", 0, v.scratchPad);
      axiSlaveRegister (axilEp, x"004", 0, v.do_subtraction);

      axiSlaveDefault(axilEp, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_DECERR_C);



      ------------------------      
      -- Main Part of Code
      ------------------------ 

      v.slave.tReady          := not outCtrl.pause;
      v.master.tLast          := '0';
      v.master.tValid         := '0';

      v.pedestalMaster.tValid := '0';
      v.pedestalMaster.tLast  := '0';

      ------------------------      
      -- First State Machine
      ------------------------ 

      case r.state is

            when IDLE_S =>
            ------------------------------
            -- check which state
            ------------------------------
            if v.slave.tReady = '1' and inMaster.tValid = '1' then   --if this one was first, pedestal may never update
                  v.state           := MOVE_S;
                  v.slave.tReady    :='0';

            else
                  v.state           := IDLE_S;
                  v.slave.tReady    :='0';

            end if;

            when MOVE_S  => 
            ------------------------------
            -- update slv logic array
            ------------------------------
               v.slave.tReady    :='1';
               if v.slave.tReady = '1' and inMaster.tValid = '1' then
                  v.master                   := inMaster;     --copies one 'transfer' (trasnfer is the AXI jargon for one TVALID/TREADY transaction)
                                                              --tReady is propogated from downstream to upstream


                  if v.do_subtraction(0) = '1' then

                      v.scratchPad(0) := '1';

                      for i in 0 to INT_CONFIG_C.TDATA_BYTES_C-1 loop


                            v.aSingleFrame(i)                := RESIZE((signed(inMaster.tdata(i*8+7 downto i*8))-r.storedPedestalFrame(r.counter + i)),8);
                            v.master.tData(i*8+7 downto i*8) := std_logic_vector(r.aSingleFrame(i));                       --output 

                      end loop;             

                  end if;
               
                                   
                  v.counter                  := r.counter+INT_CONFIG_C.TDATA_BYTES_C;

                  --the camera pixel number vs pedestal counter condition wasn't required in test bench.  worrisome and will need attention in future
                  if v.master.tLast = '1' or v.counter >= CAMERA_PIXEL_NUMBER then
                        v.counter            := 0;
                        --v.slave.tReady       := '0';
                        --v.state              := IDLE_S;
                  end if;
                  
                  



               else
                  v.master.tValid  := '0';   --message to downstream data processing that there's no valid data ready
                  v.slave.tReady   := '0';   --message to upstream that we're not ready
                  v.master.tLast   := '0';
                  v.state          := IDLE_S;

               end if; 
           
 

      end case;


      ------------------------      
      -- Second State Machine
      ------------------------ 
      case r.state_pedestal is

            when IDLE_S =>
            ------------------------------
            -- check which state
            ------------------------------
            if pedestalInMasterBuf.tValid ='1' then

                  v.state_pedestal         := MOVE_S;
                  v.pedestalSlave.tReady   := '0';

            else
                  v.state_pedestal         := IDLE_S;
                  v.pedestalSlave.tReady   := '0';

            end if;
           
            when MOVE_S  => 
            ------------------------------
            -- update slv logic array
            ------------------------------
               v.pedestalSlave.tReady   := '1';
               if pedestalInMasterBuf.tValid = '1' then


                  v.pedestalMaster  := pedestalInMasterBuf ;     --copies one 'transfer' (trasnfer is the AXI jargon for one TVALID/TREADY transaction)
                                                                 

                  for i in 0 to INT_CONFIG_C.TDATA_BYTES_C-1 loop

                        v.storedPedestalFrame(r.pedestal_counter + i)        := RESIZE(signed(pedestalInMasterBuf.tdata(i*8+7 downto i*8)),8);
                        

                  end loop;

                 
                  v.pedestal_counter                  := r.pedestal_counter+INT_CONFIG_C.TDATA_BYTES_C;

                  --the camera pixel number vs pedestal counter condition wasn't required in test bench.  worrisome and will need attention in future
                  if v.pedestalMaster.tLast = '1' or v.pedestal_counter >= CAMERA_PIXEL_NUMBER then 
                        v.pedestal_counter            := 0;
                        --v.pedestalSlave.tReady        := '0';
                        --v.state_pedestal              := IDLE_S;
                  end if;
                  



               else
                  v.pedestalMaster.tValid  := '0';   --message to downstream data processing that there's no valid data ready
                  v.pedestalSlave.tReady   := '0';   --message to upstream that we're not ready
                  v.pedestalMaster.tLast   := '0';
                  v.state_pedestal         := IDLE_S;
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
      axilReadSlave             <= r.axilReadSlave;
      axilWriteSlave            <= r.axilWriteSlave;
      inSlave                   <= v.slave;
      pedestalInSlaveBuf        <= v.pedestalSlave;




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
         SLAVE_AXI_CONFIG_G  => INT_CONFIG_C,
         MASTER_AXI_CONFIG_G => DMA_AXIS_CONFIG_G)
      port map (
         sAxisClk    => sysClk,
         sAxisRst    => sysRst,
         sAxisMaster => r.Master,
         sAxisCtrl   => outCtrl,
         mAxisClk    => sysClk,
         mAxisRst    => sysRst,
         mAxisMaster => dataOutMaster,
         mAxisSlave  => dataOutSlave);

end mapping;
