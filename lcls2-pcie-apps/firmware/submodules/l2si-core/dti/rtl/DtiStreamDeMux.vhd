-------------------------------------------------------------------------------
-- Title      : DTI AXI Stream De-Multiplexer
-- Project    : General Purpose Core
-------------------------------------------------------------------------------
-- File       : DtiStreamDeMux.vhd
-- Author     : Ryan Herbst, rherbst@slac.stanford.edu
-- Created    : 2014-04-25
-- Last update: 2018-03-14
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description:
-- Block to connect a single incoming AXI stream to multiple outgoing AXI
-- streams based upon the incoming tDest value.
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Firmware Standard Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Firmware Standard Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
-- Modification history:
-- 04/25/2014: created.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.NUMERIC_STD.all;
use work.StdRtlPkg.all;
use work.ArbiterPkg.all;
use work.AxiStreamPkg.all;

entity DtiStreamDeMux is
   generic (
      TPD_G          : time                  := 1 ns;
      NUM_MASTERS_G  : integer range 1 to 32 := 12;
      TDEST_HIGH_G   : integer range 0 to 7  := 7;
      TDEST_LOW_G    : integer range 0 to 7  := 0);
   port (
      -- Slave
      sFlood       : in  sl;
      sFloodMask   : in  slv(NUM_MASTERS_G-1 downto 0);
      sAxisMaster  : in  AxiStreamMasterType;
      sAxisSlave   : out AxiStreamSlaveType;
      -- Masters
      mAxisMasters : out AxiStreamMasterArray(NUM_MASTERS_G-1 downto 0);
      mAxisSlaves  : in  AxiStreamSlaveArray(NUM_MASTERS_G-1 downto 0);
      -- Clock and reset
      axisClk      : in  sl;
      axisRst      : in  sl);
end DtiStreamDeMux;

architecture structure of DtiStreamDeMux is

  constant MODE_G         : string                := "INDEXED";          -- Or "ROUTED"
  constant PIPE_STAGES_G  : integer range 0 to 16 := 0;
  
   type RegType is record
      slave   : AxiStreamSlaveType;
      masters : AxiStreamMasterArray(NUM_MASTERS_G-1 downto 0);
   end record RegType;

   constant REG_INIT_C : RegType := (
      slave   => AXI_STREAM_SLAVE_INIT_C,
      masters => (others => AXI_STREAM_MASTER_INIT_C));

   signal pipeAxisMasters : AxiStreamMasterArray(NUM_MASTERS_G-1 downto 0);
   signal pipeAxisSlaves  : AxiStreamSlaveArray(NUM_MASTERS_G-1 downto 0);

   signal axisRst_d : sl;
  
   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   assert ((TDEST_HIGH_G - TDEST_LOW_G + 1 >= log2(NUM_MASTERS_G)))
      report "In INDEXED mode, TDest range " & integer'image(TDEST_HIGH_G) & " downto " & integer'image(TDEST_LOW_G) &
      " is too small for NUM_MASTERS_G=" & integer'image(NUM_MASTERS_G)
      severity error;

   comb : process (axisRst_d, pipeAxisSlaves, r, sAxisMaster, sFlood, sFloodMask) is
      variable v   : RegType;
      variable idx : natural;
      variable i   : natural;
      variable allReady : sl;
   begin
      -- Latch the current value
      v := r;

      -- Reset strobing signals
      v.slave.tReady := '0';

      -- Update tValid register
      allReady := '1';
      for i in 0 to NUM_MASTERS_G-1 loop
         if pipeAxisSlaves(i).tReady = '1' then
            v.masters(i).tValid := '0';
         end if;
         if (v.masters(i).tValid = '1' and sFloodMask(i) = '1') then
           allReady := '0';
         end if;
      end loop;

      if sFlood = '1' then
        if (allReady = '1') and (sAxisMaster.tValid = '1') then
          -- Accept the data
          v.slave.tReady := '1';
          -- Move the data
          for idx in 0 to NUM_MASTERS_G-1 loop
            if sFloodMask(idx) = '1' then
              v.masters(idx) := sAxisMaster;
              v.masters(idx).tDest(TDEST_HIGH_G downto TDEST_LOW_G) := (others=>'0');
            end if;
          end loop;
        end if;
      else
        -- TDEST indicates the output port
        idx := to_integer(unsigned(sAxisMaster.tDest(TDEST_HIGH_G downto TDEST_LOW_G)));

        -- Check for invalid destination
        if idx >= NUM_MASTERS_G then
          -- Blow off the data
          v.slave.tReady := '1';
        -- Check if ready to move data
        elsif (v.masters(idx).tValid = '0') and (sAxisMaster.tValid = '1') then
          -- Accept the data
          v.slave.tReady := '1';
          -- Move the data
          v.masters(idx) := sAxisMaster;
          v.masters(idx).tDest(TDEST_HIGH_G downto TDEST_LOW_G) := (others=>'0');
        end if;
      end if;

      -- Reset
      if (axisRst_d = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs
      sAxisSlave      <= v.slave;
      pipeAxisMasters <= r.masters;

   end process comb;

   GEN_VEC :
   for i in (NUM_MASTERS_G-1) downto 0 generate
      
      U_Pipeline : entity work.AxiStreamPipeline
         generic map (
            TPD_G         => TPD_G,
            PIPE_STAGES_G => PIPE_STAGES_G)
         port map (
            axisClk     => axisClk,
            axisRst     => axisRst,
            sAxisMaster => pipeAxisMasters(i),
            sAxisSlave  => pipeAxisSlaves(i),
            mAxisMaster => mAxisMasters(i),
            mAxisSlave  => mAxisSlaves(i));   

   end generate GEN_VEC;

   seq : process (axisClk) is
   begin
      if (rising_edge(axisClk)) then
         axisRst_d <= axisRst after TPD_G;
         r <= rin after TPD_G;
      end if;
   end process seq;

end structure;

