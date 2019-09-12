------------------------------------------------------------------------------
-- Title         : Timing pattern generator
-- Project       : LCLS-II Timing System
-------------------------------------------------------------------------------
-- File          : TPGNotify.vhd
-- Author        : Matt Weaver, weaver@slac.stanford.edu
-- Created       : 05/19/2016
-------------------------------------------------------------------------------
-- Description:
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
-- 09/15/2015: created.
-------------------------------------------------------------------------------
library ieee;
use work.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;

entity TPGNotify is
  generic (
   TPD_G : time := 1 ns);
  port (
    -- Clock and reset
    txClk         : in  sl;
    txRst         : in  sl;

    irqReq        : in  sl;
    irqEnable     : in  sl;

    obDebugMaster : out AxiStreamMasterType );
end TPGNotify;


architecture TPGNotify of TPGNotify is

  type RegType is record
    count  : slv(31 downto 0);
    master : AxiStreamMasterType;
  end record;
  constant REG_INIT_C : RegType := (
    count  => (others=>'0'),
    master => AXI_STREAM_MASTER_INIT_C );

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  signal irq, irqS : sl;
  
  attribute use_dsp48      : string;
  attribute use_dsp48 of r : signal is "yes";   
  
begin

  obDebugMaster <= r.master;
  
  irq <= irqEnable and irqReq;
  
  U_SyncIrq : entity work.SynchronizerOneShot
    port map ( clk     => txClk,
               rst     => txRst,
               dataIn  => irq,
               dataOut => irqS );
  
  process ( r, irqS ) is
    variable v : RegType;
  begin
    v := r;
    if (irqS='1') then
      v.master.tValid := '1';
      v.master.tData(r.count'range) := r.count;
      v.master.tLast  := '1';
      v.master.tDest  := x"FF";
      v.count := r.count+1;
    end if;
    rin <= v;
  end process;

  process (txClk) is
  begin
    if rising_edge(txClk) then
      r <= rin after TPD_G;
    end if;
  end process;
end TPGNotify;
