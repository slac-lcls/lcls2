-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : SeqReset.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2016-04-13
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Calculates automated jumps in sequencer instruction RAM.
--   Reacts to BCS fault state change, MPS state change, and manual reset.
--   The manual reset is highest priority, followed by BCS, and MPS.
--   Any state change that isn't acted upon because of a higher priority reaction
--   will be enacted on the following cycle.
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
LIBRARY ieee;
use work.all;

USE ieee.std_logic_1164.ALL;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
use work.TimingPkg.all;
use work.TPGPkg.all;
use work.StdRtlPkg.all;

entity SeqReset is
   generic ( TPD_G : time := 1 ns);
   port ( 
      -- Clock and reset
      clk                : in  sl;
      rst                : in  sl;
      config             : in  TPGJumpConfigType;
      frame              : in  TimingMessageType;
      strobe             : in  sl;
      resetReq           : in  sl;
      resetO             : out sl
      );
end SeqReset;

-- Define architecture for top level module
architecture mapping of SeqReset is 

  type RegType is record
     req     : sl;
     latch   : sl;
     resetO  : sl;
  end record;
  constant REG_INIT_C : RegType := (
     req     => '0',
     latch   => '0',
     resetO  => '0');
  
  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  
begin

  resetO <= rin.resetO;
  
  comb: process (r, config, resetReq, strobe, frame)
     variable v : RegType;
     variable rateSel : sl;
  begin  -- process
    v        := r;
    v.resetO := '0';
    v.req    := resetReq;
    
    --  Check for synchronized jump cycle
    case (config.syncSel(15 downto 14)) is
       when "00" => rateSel := frame.fixedRates(conv_integer(config.syncSel(3 downto 0)));
       when "01" => if (config.syncSel(conv_integer(frame.acTimeSlot)+3-1)='0') then
                      rateSel := '0';
                    else
                      rateSel := frame.acRates(conv_integer(config.syncSel(2 downto 0)));
                    end if;
       when others => rateSel := '0';
    end case;

    --  Read in the new configuration on manual reset
    if (resetReq='1' and r.req='0') then
       v.latch := '1';
    end if;

    if (r.latch='1' and rateSel='1' and strobe='1') then
       v.latch  :='0';
       v.resetO :='1';
    end if;   
      
    rin <= v;
  end process comb;

  seq: process (clk) is
  begin
    if rising_edge(clk) then
      r <= rin after TPD_G;
    end if;
  end process seq;
  
end mapping;
