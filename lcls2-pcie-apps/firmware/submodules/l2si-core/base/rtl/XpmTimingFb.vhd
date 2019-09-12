-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmTimingFb.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-08
-- Last update: 2018-09-05
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.XpmPkg.all;

entity XpmTimingFb is
   generic (
      DEBUG_G        : boolean := false );
   port (
      clk            : in  sl;
      rst            : in  sl;
      pllReset       : in  sl := '0';
      phyReset       : in  sl := '0';
      status         : in  TimingPhyStatusType := TIMING_PHY_STATUS_INIT_C;
      id             : in  slv(31 downto 0) := (others=>'1');
      l1input        : in  XpmL1InputArray(NPartitions-1 downto 0);
      full           : in  slv            (NPartitions-1 downto 0);
      l1ack          : out slv            (NPartitions-1 downto 0);
      phy            : out TimingPhyType );
end XpmTimingFb;

architecture rtl of XpmTimingFb is

  type StateType is (IDLE_S, PFULL_S, ID1_S, ID2_S, PDATA1_S, PDATA2_S, EOF_S);
  
  constant MAX_IDLE_C : slv(7 downto 0) := x"0F";

  type RegType is record
    state             : StateType;
    idleCnt           : slv(MAX_IDLE_C'range);
    txData            : slv(15 downto 0);
    txDataK           : slv( 1 downto 0);
    full              : slv(NPartitions-1 downto 0);
    strobe            : slv(NPartitions-1 downto 0);
    ready             : sl;
    partition         : integer range 0 to NPartitions-1;
    control           : TimingPhyControlType;
  end record;

  constant REG_INIT_C : RegType := (
    state             => IDLE_S,
    idleCnt           => (others=>'0'),
    txData            => (D_215_C & K_COM_C),
    txDataK           => "01",
    full              => (others=>'1'),
    strobe            => (others=>'0'),
    ready             => '0',
    partition         => 0,
    control           => TIMING_PHY_CONTROL_INIT_C );

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

  component ila_0
    port ( clk     : in  sl;
           probe0  : in  slv(255 downto 0) );
  end component;

  signal s_state : slv(2 downto 0);
  
begin

  GEN_DBUG : if DEBUG_G generate
    s_state <= "000" when r.state = IDLE_S else
               "001" when r.state = PFULL_S else
               "010" when r.state = PDATA1_S else
               "011" when r.state = PDATA2_S else
               "100";
    
    U_ILA : ila_0
      port map ( clk       => clk,
                 probe0(0) => rst,
                 probe0(3 downto 1) => s_state,
                 probe0(4) => r.ready,
                 probe0(12 downto 5) => r.idleCnt,
                 probe0(28 downto 13) => r.txData,
                 probe0(30 downto 29) => r.txDataK,
                 probe0(38 downto 31) => r.full(7 downto 0),
                 probe0(46 downto 39) => r.strobe(7 downto 0),
                 probe0(54 downto 47) => full(7 downto 0),
                 probe0(55)           => status.locked,
                 probe0(56)           => status.resetDone,
                 probe0(57)           => status.bufferByDone,
                 probe0(58)           => status.bufferByErr,
                 probe0(255 downto 59) => (others=>'0') );
  end generate;
  
  l1ack       <= r.strobe;
  phy.data    <= r.txData;
  phy.dataK   <= r.txDataK;
  phy.control.pllReset <= pllReset;
  phy.control.reset    <= phyReset;
  phy.control.inhibit  <= '0';
  phy.control.polarity <= '0';
  phy.control.bufferByRst <= '0';
  
  comb: process (r, full, l1input, rst, id) is
    variable v : RegType;
  begin
    v := r;

    v.txDataK := "01";
    v.strobe  := (others=>'0');
    v.ready   := '0';
    
    if (r.full/=full) then
      v.ready := '1';
    end if;
    if (r.idleCnt=MAX_IDLE_C) then
      v.ready := '1';
    end if;
    for i in 0 to NPartitions-1 loop
      if l1input(i).valid='1' then
        v.ready := '1';
      end if;
    end loop;
    
    case (r.state) is
      when IDLE_S =>
        v.idleCnt := r.idleCnt+1;
        if (r.ready='1') then
          v.idleCnt := (others=>'0');
          v.txData  := D_215_C & K_EOS_C;
          v.state   := PFULL_S;
        else
          v.txData  := D_215_C & K_COM_C;
        end if;
      when PFULL_S =>
        v.txDataK := "00";
        v.txData  := (others=>'0');
        v.txData(full'range) := full;
        v.full := full;
        v.state   := ID1_S;
      when ID1_S =>
        v.txDataK := "00";
        v.txData  := id(15 downto 0);
        v.state   := ID2_S;
      when ID2_S =>
        v.txDataK := "00";
        v.txData  := id(31 downto 16);
        v.state   := EOF_S;
        v.partition := 0;
        for i in 0 to NPartitions-1 loop
          if (l1input(i).valid='1') then
            v.partition := i;
            v.state     := PDATA1_S;
          end if;
        end loop;
      when PDATA1_S =>
        v.txDataK := "00";
        v.txData             := (others=>'0');
        v.txData(7 downto 4) := l1input(r.partition).trigsrc;
        v.txData(3 downto 0) := toSlv(r.partition,4);
        v.state   := PDATA2_S;
      when PDATA2_S =>
        v.txDataK := "00";
        v.txData              := (others=>'0');
        v.txData(14)          := '1';
        v.txData(13 downto 5) := l1input(r.partition).trigword;
        v.txData( 4 downto 0) := l1input(r.partition).tag;
        v.strobe(r.partition) := '1';
        v.state   := PFULL_S;
      when EOF_S =>
        v.txData  := D_215_C & K_EOF_C;
        v.state   := IDLE_S;
      when others => null;
    end case;

    if (rst='1') then
      v := REG_INIT_C;
    end if;
    
    rin <= v;

  end process comb;

  seq : process (clk) is
  begin
    if (rising_edge(clk)) then
      r <= rin;
    end if;
  end process seq;

end rtl;
