-------------------------------------------------------------------------------
-- File       : TimingPhyInit.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- This file is part of 'Camera link gateway'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'Camera link gateway', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;

entity TimingPhyInit is
   generic (
      TPD_G              : time := 1 ns;
      SIMULATION_G       : boolean;
      TIMING_BASE_ADDR_G : slv;
      AXIL_CLK_FREQ_G    : real);
   port (
      mmcmLocked       : in  slv(1 downto 0);
      -- AXI-Lite Register Interface (sysClk domain)
      axilClk          : in  sl;
      axilRst          : in  sl;
      mAxilReadMaster  : out AxiLiteReadMasterType;
      mAxilReadSlave   : in  AxiLiteReadSlaveType;
      mAxilWriteMaster : out AxiLiteWriteMasterType;
      mAxilWriteSlave  : in  AxiLiteWriteSlaveType);
end TimingPhyInit;

architecture rtl of TimingPhyInit is

   constant TIMEOUT_1SEC_C : positive := ite(SIMULATION_G, 1, getTimeRatio(AXIL_CLK_FREQ_G, 1.0));

   type StateType is (
      REQ_S,
      ACK_S,
      DONE_S);

   type RegType is record
      cnt   : natural range 0 to 3;
      timer : natural range 0 to TIMEOUT_1SEC_C;
      req   : AxiLiteReqType;
      state : StateType;
   end record;

   constant REG_INIT_C : RegType := (
      cnt   => 0,
      timer => 0,
      req   => AXI_LITE_REQ_INIT_C,
      state => REQ_S);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal ack : AxiLiteAckType;

begin

   U_AxiLiteMaster : entity work.AxiLiteMaster
      generic map (
         TPD_G => TPD_G)
      port map (
         req             => r.req,
         ack             => ack,
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilWriteMaster => mAxilWriteMaster,
         axilWriteSlave  => mAxilWriteSlave,
         axilReadMaster  => mAxilReadMaster,
         axilReadSlave   => mAxilReadSlave);

   --------------------- 
   -- AXI Lite Interface
   --------------------- 
   comb : process (ack, axilRst, mmcmLocked, r) is
      variable v      : RegType;
      variable regCon : AxiLiteEndPointType;
   begin
      -- Latch the current value
      v := r;

      -- Decrement the timer
      if (r.timer /= 0) then
         v.timer := r.timer -1;
      end if;

      -- State Machine
      case (r.state) is
         ----------------------------------------------------------------------
         when REQ_S =>
            -- Check if ready for next transaction
            if (ack.done = '0') and (r.timer = 0) then

               -- Setup the AXI-Lite Master request
               v.req.request := '1';
               v.req.rnw     := '0';

               -- self.TimingFrameRx.RxPllReset.set(1)
               if (r.cnt = 0) then
                  v.req.address := TIMING_BASE_ADDR_G + x"0000_0020";
                  v.req.wrData  := x"0000_0080";

               -- self.TimingFrameRx.RxPllReset.set(0)
               elsif (r.cnt = 1) then
                  v.req.address := TIMING_BASE_ADDR_G + x"0000_0020";
                  v.req.wrData  := x"0000_0000";

               -- self.TimingFrameRx.RxReset.set(1)
               elsif (r.cnt = 2) then
                  v.req.address := TIMING_BASE_ADDR_G + x"0000_0020";
                  v.req.wrData  := x"0000_0008";


               -- self.TimingFrameRx.RxReset.set(0)
               else
                  v.req.address := TIMING_BASE_ADDR_G + x"0000_0020";
                  v.req.wrData  := x"0000_0000";

               end if;
               
               -- Next state
               v.state := ACK_S;
            end if;
         ----------------------------------------------------------------------
         when ACK_S =>
            -- Wait for DONE to set
            if (ack.done = '1') then

               -- Reset the flag
               v.req.request := '0';

               -- Check if using timer
               if (r.cnt = 0) then
                  -- Arm the timer
                  v.timer := TIMEOUT_1SEC_C;
               end if;

               if (r.cnt /= 3) then

                  -- Increment the counter
                  v.cnt := r.cnt + 1;

                  -- Next state
                  v.state := REQ_S;

               else

                  -- Next state
                  v.state := DONE_S;

               end if;
            end if;
         ----------------------------------------------------------------------
         when DONE_S =>
            -- Done with power up initialization of the KCU1500 Timing RX PHY
            v.timer := 0;
      ----------------------------------------------------------------------
      end case;

      -- Reset
      if (axilRst = '1') or (mmcmLocked /= "11") then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

   end process comb;

   seq : process (axilClk) is
   begin
      if (rising_edge(axilClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end rtl;
