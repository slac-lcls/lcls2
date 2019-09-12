-------------------------------------------------------------------------------
-- Title         : EvrV2Trigger
-- Project       : LCLS-II Timing Pattern Generator
-------------------------------------------------------------------------------
-- File          : EvrV2Trigger.vhd
-- Author        : Matt Weaver, weaver@slac.stanford.edu
-- Created       : 01/23/2016
-------------------------------------------------------------------------------
-- Description:
-- Pipeline of trigger output state.  The trigger output activates config.delay
-- plus 5 clk ticks after fire is asserted and deasserts after config.width clk
-- ticks.  The FIFO allows up to 127 triggers to be pipelined.  Individual
-- trigger delays and widths have 20 bits range, provided the 127 trigger
-- pipelining is not exceeded.
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
-- Modification history:
-- 01/23/2016: created.
-------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.all;
use work.TPGPkg.all;
use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;

entity EvrV2Trigger is
  generic ( TPD_G        : time := 1 ns;
            CHANNELS_C   : integer := 1;
            TRIG_DEPTH_C : integer := 16;
            TRIG_WIDTH_C : integer := EVRV2_TRIG_WIDTH_C; -- bit size of
                                                        -- width,delay counters
            USE_MASK_G   : boolean := false;
            DEBUG_C      : boolean := false);
  port (
      clk        : in  sl;
      rst        : in  sl;
      config     : in  EvrV2TriggerConfigType;
      arm        : in  slv(CHANNELS_C-1 downto 0);
      fire       : in  sl;
      trigstate  : out sl );
end EvrV2Trigger;

architecture EvrV2Trigger of EvrV2Trigger is

   type RegType is record
     fifo_delay     : slv(TRIG_WIDTH_C-1 downto 0);      -- clks until trigger fifo is empty
     armed          : sl;
     delay          : slv(TRIG_WIDTH_C-1 downto 0);
     width          : slv(TRIG_WIDTH_C-1 downto 0);
     state          : sl;
     fifoReset      : sl;
     fifoWr         : sl;
     fifoRd         : sl;
     fifoDin        : slv(TRIG_WIDTH_C-1 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
     fifo_delay => (others=>'0'),
     armed      => '0',
     delay      => (others=>'0'),
     width      => (others=>'0'),
     state      => '0',
     fifoReset  => '1',
     fifoWr     => '0',
     fifoRd     => '0',
     fifoDin    => (others=>'0'));

   constant FIFO_AWIDTH_C : natural := bitSize( TRIG_DEPTH_C );
   
   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal fifoValid : sl;

   signal fifoDout  : slv(TRIG_WIDTH_C-1 downto 0);
   signal fifoCount : slv(FIFO_AWIDTH_C-1 downto 0);
   signal fifoEmpty : sl;
   signal fifoFull  : sl;


   signal fifoCountDbg : slv(6 downto 0);
   
begin


   trigstate <= r.state;

   GEN_NO_FIFO : if TRIG_DEPTH_C = 0 generate
     fifoDout  <= resize(config.delay,fifoDout'length);
     fifoEmpty <= not r.fifoWr;
   end generate;

   GEN_FIFO : if TRIG_DEPTH_C > 0 generate
     U_Fifo : entity work.FifoSync
       generic map ( TPD_G        => TPD_G,
                     DATA_WIDTH_G => TRIG_WIDTH_C,
                     ADDR_WIDTH_G => FIFO_AWIDTH_C,
                     FWFT_EN_G    => false )
       port map (    rst   => r.fifoReset,
                     clk   => clk,
                     wr_en => rin.fifoWr,
                     rd_en => rin.fifoRd,
                     din   => rin.fifoDin,
                     dout  => fifoDout,
                     valid => fifoValid,
                     empty => fifoEmpty,
                     full  => fifoFull,
                     data_count => fifoCount );
   end generate;

   process (r, arm, fire, rst, config, fifoValid, fifoDout, fifoEmpty)
      variable v : RegType;
   begin 
      v := r;

      v.state     := not config.polarity;
      v.fifoReset := '0';
      v.fifoRd    := '0';

      if allBits(r.delay,'0') then
        if allBits(r.width,'0') then
          if r.fifoRd='1' then
            v.width  := config.width(TRIG_WIDTH_C-1 downto 0);
            v.delay  := fifoDout;
          elsif fifoEmpty='0' then
            v.fifoRd := '1';
          end if;
        else
          v.width  := r.width-1;
          v.state  := config.polarity;
        end if;
      else
        v.delay  := r.delay-1;
      end if;
      
      if fire = '1' and r.armed = '1' then
         v.armed      := '0';
         v.fifoWr     := '1';
         v.fifoDin    := config.delay(TRIG_WIDTH_C-1 downto 0) - r.fifo_delay;
         v.fifo_delay := config.delay(TRIG_WIDTH_C-1 downto 0) + config.width(TRIG_WIDTH_C-1 downto 0) + 1;
      else
         v.fifoWr     := '0';
         if not allBits(r.fifo_delay,'0') then
           v.fifo_delay := r.fifo_delay - 1;
         end if;
      end if;

      if ((arm(conv_integer(config.channel)) = '1' and not USE_MASK_G) or
          ((arm and config.channels(CHANNELS_C-1 downto 0)) /= toSlv(0,CHANNELS_C) and USE_MASK_G)) then
         v.armed := '1';
      end if;

      if rst='1' or config.enabled='0' then
         v := REG_INIT_C;
      end if;
      
      rin <= v;
   end process;

   process (clk)
   begin
     if rising_edge(clk) then
       r <= rin;
     end if;
   end process;

end EvrV2Trigger;

