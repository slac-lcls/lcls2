-------------------------------------------------------------------------------
-- File       : SsiFrameLimiter.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: Limits the amount of data being sent across a SSI AXIS bus 
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Firmware Standard Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Firmware Standard Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
entity Prescaler is
   generic (
      TPD_G        : time                := 1 ns;
      MAX_CNT_G    : positive            := 1024;
      AXI_CONFIG_G : AxiStreamConfigType := AXI_STREAM_CONFIG_INIT_C);
   port (
      -- Clock and Reset
      axisClk  : in  sl;
      axisRst  : in  sl;
      -- Configurations
      maxCnt   : in  slv(bitsize(MAX_CNT_G)-1 downto 0);
      -- Slave Port
      rxMaster : in  AxiStreamMasterType;
      rxSlave  : out AxiStreamSlaveType;
      -- Master Port
      txMaster : out AxiStreamMasterType;
      txSlave  : in  AxiStreamSlaveType);
end Prescaler;
architecture rtl of Prescaler is
   type StateType is (
      IDLE_S,
      MOVE_S);
   type RegType is record
      cnt      : slv(bitsize(MAX_CNT_G)-1 downto 0);
      rxSlave  : AxiStreamSlaveType;
      txMaster : AxiStreamMasterType;
      state    : StateType;
   end record RegType;
   constant REG_INIT_C : RegType := (
      cnt      => (others => '0'),
      rxSlave  => AXI_STREAM_SLAVE_INIT_C,
      txMaster => AXI_STREAM_MASTER_INIT_C,
      state    => IDLE_S);
   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;
begin
   comb : process (axisRst, maxCnt, r, rxMaster, txSlave) is
      variable v : RegType;
   begin
      -- Latch the current value
      v := r;
      -- Reset the flags
      v.rxSlave.tReady := '0';
      if (txSlave.tReady = '1') then
         v.txMaster.tValid := '0';
         v.txMaster.tLast  := '0';
         v.txMaster.tUser  := (others => '0');
      end if;
      -- State Machine
      case r.state is
         ----------------------------------------------------------------------
         when IDLE_S =>
            -- Wait for next waveform and ready to move data
            if (rxMaster.tValid = '1') and (v.txMaster.tValid = '0') then
               -- Check if sending waveform
               if (r.cnt = maxCnt) then
                  -- Reset the counter
                  v.cnt       := (others => '0');
                  -- Set the flag
                  v.sendFrame := '1';
                  -- Next state
                  v.state     := MOVE_S;
               else
                  -- Increment the counter
                  v.cnt                                                    := r.cnt + 1;
                  -- Send the NULL frame
                  v.txMaster.tValid                                        := '1';
                  v.txMaster.tLast                                         := '1';
                  v.txMaster.tKeep(AXIS_CONFIG_G.TDATA_BYTES_C-1 downto 0) := toSlv(1, AXIS_CONFIG_G.TDATA_BYTES_C);
                  ssiSetUserEofe(AXI_CONFIG_G, v.txMaster, '1');
                  -- Check if inbound is not single word frame
                  if (rxMaster.tLast = '0') then
                     -- Set the flag
                     v.sendFrame := '0';
                     -- Next state
                     v.state     := MOVE_S;
                  else
                     -- Accept the data
                     v.rxSlave.tReady := '1';
                  end if;
               end if;
            end if;
         ----------------------------------------------------------------------
         when IDLE_S =>
            -- Wait for next waveform and ready to move data
            if (rxMaster.tValid = '1') and (v.txMaster.tValid = '0') then
               -- Accept the data
               v.rxSlave.tReady  := '1';
               -- Move the data
               v.txMaster        := rxMaster;
               -- Check if forwarding frame
               v.txMaster.tValid := r.sendFrame;
               -- Check for last word in the frame
               if (rxMaster.tLast = '1') then
                  -- Next state
                  v.state := IDLE_S;
               end if;
            end if;
      ----------------------------------------------------------------------
      end case;
      -- Outputs       
      rxSlave  <= v.rxSlave;            -- Combinatorial output
      txMaster <= r.txMaster;           -- Registered output
      -- Reset
      if (axisRst = '1') then
         v := REG_INIT_C;
      end if;
      -- Register the variable for next clock cycle
      rin <= v;
   end process comb;
   seq : process (axisClk) is
   begin
      if rising_edge(axisClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;
end rtl;
