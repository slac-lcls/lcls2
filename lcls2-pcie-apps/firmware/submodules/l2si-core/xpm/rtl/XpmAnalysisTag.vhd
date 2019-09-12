-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmAnalysisTag.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2016-09-09
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Analysis Tag Insertion
-- 
-- Analysis tag bits are segregated into sets where only one bit in a set may be
-- asserted at any time.  A set of tag bits represents a group of analysis nodes
-- that work together and consequently receive exclusive events.  The bit groupings
-- are configured programmably according to 'config'.  Each group contains its
-- own FIFO for ordering analysis event requests.
--
-- Requests to tag events for analysis are received through the programmable register
-- interface via 'push'.  Tag bits for the next event are asserted on the 'tag'
-- signal and incremented via the 'pop' input signal.
-- 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 XPM Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 XPM Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.XpmPkg.all;

entity XpmAnalysisTag is
   port (
      wrclk            : in  sl;
      config           : in  XpmAnalysisConfigType;
      rdclk            : in  sl;
      rden             : in  sl;
      rddone           : in  sl;
      rdvalid          : out slv(  NTagBytes-1 downto 0);
      tag              : out slv(8*NTagBytes-1 downto 0) );
end XpmAnalysisTag;

architecture rtl of XpmAnalysisTag is

  signal dout   : slv(8*NTagBytes-1 downto 0);
  signal dvalid : slv(  NTagBytes-1 downto 0);

  type RegType is record
    advance : slv(2*NTagBytes-1 downto 0);
    dvalid  : slv(  NTagBytes-1 downto 0);
    rdvalid : slv(  NTagBytes-1 downto 0);
    rden    : sl;
  end record;
  constant REG_INIT_C : RegType := (
    advance => (others=>'0'),
    dvalid  => (others=>'0'),
    rdvalid => (others=>'0'),
    rden    => '0' );
  
  signal r    : RegType := REG_INIT_C;
  signal rin  : RegType;

  signal tagb : slv(8*NTagBytes-1 downto 0);

begin

  tag     <= tagb;
  rdvalid <= r.rdvalid;

  GEN_FIFOS : for i in 0 to NTagBytes-1 generate
    U_TagFifo : entity work.FifoAsync
      generic map ( FWFT_EN_G    => false,
                    DATA_WIDTH_G => 8,
                    ADDR_WIDTH_G => 14,
                    INIT_G       => x"FF" )
      port map ( rst    => config.rst (i),
                 wr_clk => wrclk,
                 wr_en  => config.push(i),
                 din    => config.tag (8*i+7 downto 8*i),
                 rd_clk => rdclk,
                 rd_en  => r.advance(i+NTagBytes),
                 dout   => dout(8*i+7 downto 8*i),
                 valid  => dvalid(i) );
    tagb (8*i+7 downto 8*i) <= dout(8*i+7 downto 8*i) when r.dvalid(i)='1' else (others=>'1');
  end generate;
    
  process (r, rden, rddone, dvalid) is
    variable v : RegType;
    variable b : sl;
  begin
    v := r;
    v.advance(2*NTagBytes-1 downto NTagBytes) := (others=>'0');
    v.advance(NTagBytes-1 downto 0) := r.advance(2*NTagBytes-1 downto NTagBytes);
    v.rdvalid := (others=>'0');
    v.rden    := rddone;

    if r.rden='1' then
      if rden='1' then
        v.advance(2*NTagBytes-1 downto NTagBytes) := (others=>'1');
        v.rdvalid    := r.dvalid;
      else
        v.advance(2*NTagBytes-1 downto NTagBytes) := r.dvalid;
      end if;
    end if;

    for i in 0 to NTagBytes-1 loop
      if r.advance(i)='1' then
        v.dvalid(i) := dvalid(i);
      end if;
    end loop;
    
    rin <= v;
  end process;

  process (rdclk) is
  begin
    if rising_edge(rdclk) then
      r <= rin;
    end if;
  end process;

end rtl;
