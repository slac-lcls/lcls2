------------------------------------------------------------------------------
-- File       : TimeToolCore.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-12-04
-- Last update: 2019-03-22
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
use ieee.numeric_std.all;

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
-- This file performs the the prescaling, or the amount of raw data which is stored
-------------------------------------------------------------------------------

entity TimeToolPrescaler is
   generic (
      TPD_G             : time                := 1 ns;
      DMA_AXIS_CONFIG_G : AxiStreamConfigType := ssiAxiStreamConfig(16, TKEEP_COMP_C, TUSER_FIRST_LAST_C, 8, 2);
      DEBUG_G           : boolean             := true);
   port (
      -- System Interface
      sysClk          : in  sl;
      sysRst          : in  sl;
      -- DMA Interfaces  (sysClk domain)
      dataInMaster    : in  AxiStreamMasterType;
      dataInSlave     : out AxiStreamSlaveType;
      dataOutMaster   : out AxiStreamMasterType;
      dataOutSlave    : in  AxiStreamSlaveType;
      -- AXI-Lite Interface
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType);
end TimeToolPrescaler;

architecture mapping of TimeToolPrescaler is

   constant INT_CONFIG_C  : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 16, tDestBits => 0);
   constant PGP2BTXIN_LEN : integer             := 19;

   type StateType is (
      IDLE_S,
      MOVE_S,
      SEND_NULL,
      BLOWOFF_S);

   type RegType is record
      master         : AxiStreamMasterType;
      slave          : AxiStreamSlaveType;
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
      counter        : slv(31 downto 0);
      prescalingRate : slv(31 downto 0);
      scratchPad     : slv(31 downto 0);
      state          : StateType;
      validate_state : slv(2 downto 0);
   end record RegType;

   constant REG_INIT_C : RegType := (
      master         => AXI_STREAM_MASTER_INIT_C,
      slave          => AXI_STREAM_SLAVE_INIT_C,
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C,
      counter        => (others => '0'),
      prescalingRate => (others => '0'),
      scratchPad     => (others => '0'),
      state          => IDLE_S,
      validate_state => (others => '0'));

---------------------------------------
-------record intitial value-----------
---------------------------------------


   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal inMaster : AxiStreamMasterType;
   signal inSlave  : AxiStreamSlaveType;
   signal outCtrl  : AxiStreamCtrlType;

begin

   ---------------------------------
   -- Input FIFO
   ---------------------------------
   U_InFifo : entity work.AxiStreamFifoV2
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
   comb : process (axilReadMaster, axilWriteMaster, inMaster, outCtrl, r,
                   sysRst) is
      variable v      : RegType;
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
      axiSlaveRegister (axilEp, x"004", 0, v.prescalingRate);

      axiSlaveDefault(axilEp, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_DECERR_C);

      v.slave.tReady  := not outCtrl.pause;
      v.master.tLast  := '0';
      v.master.tValid := '0';

      case r.state is

         when IDLE_S =>
            ------------------------------
            -- check which state
            ------------------------------
            v.validate_state := (others => '0');  --debugging signal
            if v.slave.tReady = '1' and inMaster.tValid = '1' then
               if v.counter = v.prescalingRate then
                  v.state := MOVE_S;
               else
                  v.state := SEND_NULL;
               end if;

            else
               v.state := IDLE_S;
            end if;

            v.slave.tReady  := '0';  --may need to be after v.state := IDLE_S statement

         when MOVE_S =>
            ------------------------------
            -- send regular frame
            ------------------------------
            
            ssiSetUserEofe(DMA_AXIS_CONFIG_G, v.master, '0'); -- side band data to batcher to indicate that the frame contains data

            v.validate_state(0) := '1';     --debugging signal
            v.slave.tReady  := not outCtrl.pause;
            if v.slave.tReady = '1' and inMaster.tValid = '1' and v.counter = v.prescalingRate then
               v.master            := inMaster;  --copies one 'transfer' (trasnfer is the AXI jargon for one TVALID/TREADY transaction)
               v.validate_state(1) := '1';  --debugging signal               

            else
               v.master.tValid     := '0';  --message to downstream data processing that there's no valid data ready
               v.slave.tReady      := '0';  --message to upstream that we're not ready
               v.master.tLast      := '0';
               v.state             := IDLE_S;
               v.validate_state(2) := '1';  --debugging signal

            end if;

         when SEND_NULL =>
            ------------------------------
            -- send null frame
            ------------------------------
            v.slave.tReady  := not outCtrl.pause;
            if v.slave.tReady = '1' and inMaster.tValid = '1' then
               v.master.tValid := '1';
               v.master.tLast  := '1';

               v.master.tKeep(DMA_AXIS_CONFIG_G.TDATA_BYTES_C-1 downto 0) := toSlv(1, DMA_AXIS_CONFIG_G.TDATA_BYTES_C);
               ssiSetUserEofe(DMA_AXIS_CONFIG_G, v.master, '1');

               v.state := BLOWOFF_S;
            else
               v.master.tValid := '0';  --message to downstream data processing that there's no valid data ready
               v.slave.tReady  := '0';  --message to upstream that we're not ready
               v.master.tLast  := '0';
               v.state         := IDLE_S;
            end if;

         when BLOWOFF_S =>
            if (inMaster.tValid = '1' and inMaster.tLast = '1') or (ssiGetUserEofe(DMA_AXIS_CONFIG_G, inMaster)='1') then
               v.state := IDLE_S;
            else
               v.master.tValid := '0';
               v.master.tLast  := '0';
               v.slave.tReady  := '1';
               v.state         := BLOWOFF_S;

            end if;



      end case;

      -----------------------------
      --increment prescaling counter. needs to be after the data mover in order for the last packet to be sent
      -----------------------------
      if inMaster.tLast = '1' and inMaster.tValid = '1' and v.slave.tReady = '1' then
         if v.counter >= v.prescalingRate then
            v.counter := (others => '0');
         else
            v.counter := v.counter + 1;
         end if;
      end if;

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
      inSlave        <= v.slave;

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
   U_OutFifo : entity work.AxiStreamFifoV2
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
