-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmRxLink.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-08-01
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Sensor link deserializer
--
-- This module receives the sensor link datastream and extracts the readout
-- status and event feedback information, if any.  The readout status is expressed
-- by the 'full' signal which is asserted either by the almost full status from
-- the link or the history of 'l0Accept' and 'l1Accept' signals with the given
-- link configuration limits 'config'.
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
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.XpmPkg.all;

entity XpmRxLink is
  port (
    clk              : in  sl;
    rst              : in  sl;
    config           : in  XpmLinkConfigType;
    rxData           : in  slv(15 downto 0);
    rxDataK          : in  slv( 1 downto 0);
    rxClk            : in  sl;
    rxRst            : in  sl;
    rxErr            : in  sl;
    isXpm            : out sl;
    id               : out slv(31 downto 0);
    rxRcvs           : out slv(31 downto 0);
    full             : out slv            (NPartitions-1 downto 0);
    l1Input          : out XpmL1InputArray(NPartitions-1 downto 0) );
end XpmRxLink;

architecture rtl of XpmRxLink is
  type RxStateType is (IDLE_S, PFULL_S, ID1_S, ID2_S, PDATA1_S, PDATA2_S, DDATA_S);
  
  type RegType is record
    state     : RxStateType;
    partition : integer range 0 to NPartitions-1;
    isXpm     : sl;
    id        : slv(31 downto 0);
    rxRcvs    : slv(31 downto 0);
    pfull     : slv(NPartitions-1 downto 0);
    l1input   : XpmL1InputType;
    strobe    : slv(NPartitions-1 downto 0);
  end record;
  constant REG_INIT_C : RegType := (
    state     => IDLE_S,
    partition => 0,
    isXpm     => '0',
    id        => (others=>'0'),
    rxRcvs    => (others=>'0'),
    pfull     => (others=>'1'),
    l1input   => XPM_L1_INPUT_INIT_C,
    strobe    => (others=>'0') );

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

  signal uconfig : XpmLinkConfigType := XPM_LINK_CONFIG_INIT_C;
  
begin

  isXpm  <= r.isXpm;
  id     <= r.id;
  rxRcvs <= r.rxRcvs;
  
  U_FIFO : for i in 0 to NPartitions-1 generate
    U_ASync : entity work.FifoAsync
      generic map ( FWFT_EN_G    => true,
                    DATA_WIDTH_G => 18,
                    ADDR_WIDTH_G =>  4 )
      port map ( rst                => rxRst,
                 wr_clk             => rxClk,
                 wr_en              => r.strobe(i),
                 din (17 downto  9) => r.l1input.trigword,
                 din ( 8 downto  4) => r.l1input.tag,
                 din ( 3 downto  0) => r.l1input.trigsrc,
                 rd_clk             => clk,
                 rd_en              => '1',
                 valid              => l1Input(i).valid,
                 dout(17 downto  9) => l1Input(i).trigword,
                 dout( 8 downto  4) => l1Input(i).tag,
                 dout( 3 downto  0) => l1Input(i).trigsrc );
  end generate;
  
  U_Enable : entity work.Synchronizer
    port map    ( clk    => rxClk,
                  dataIn => config.enable,
                  dataOut=> uconfig.enable );

  U_Full : entity work.SynchronizerVector
    generic map ( INIT_G  => toSlv(-1,NPartitions),
                  WIDTH_G => NPartitions )
    port map    ( clk    => clk,
                  dataIn => r.pfull,
                  dataOut=> full );

  U_Partition : entity work.SynchronizerVector
    generic map ( WIDTH_G => config.partition'length )
    port map   ( clk     => rxClk,
                 dataIn  => config.partition,
                 dataOut => uconfig.partition );

  U_TrigSrc : entity work.SynchronizerVector
    generic map ( WIDTH_G => config.trigsrc'length )
    port map   ( clk     => rxClk,
                 dataIn  => config.trigsrc,
                 dataOut => uconfig.trigsrc );

  comb: process (r, rxData, rxDataK, rxErr, rxRst, uconfig ) is
    variable v : RegType;
    variable p : integer range 0 to NPartitions-1;
  begin
    v := r;
    v.strobe := (others=>'0');

    if r.isXpm='0' then
      v.partition := conv_integer(uconfig.partition);
    end if;

    case (r.state) is
      when IDLE_S =>
        if (rxDataK="01") then
          if (rxData=(D_215_C & K_EOS_C)) then
            v.isXpm  := '1';
            v.rxRcvs := r.rxRcvs+1;
            v.state  := PFULL_S;
          elsif (rxData=(D_215_C & K_SOF_C)) then
            v.isXpm  := '0';
            v.rxRcvs := r.rxRcvs+1;
            v.state  := DDATA_S;
          end if;
        end if;
      when PFULL_S =>
        v.pfull := rxData(r.pfull'range);
        v.state := ID1_S;
      when ID1_S =>
        if (rxDataK="01" and rxData=(D_215_C & K_EOF_C)) then
          v.state := IDLE_S;
        else
          v.id(15 downto 0) := rxData;
          v.state := ID2_S;
        end if;
      when ID2_S =>
        v.id(31 downto 16) := rxData;
        v.state := PDATA1_S;
      when PDATA1_S =>
        if (rxDataK="01" and rxData=(D_215_C & K_EOF_C)) then
          v.state := IDLE_S;
        else
          v.l1input.trigsrc    := rxData( 7 downto 4);
          v.partition          := conv_integer(rxData( 3 downto 0));
          v.state := PDATA2_S;
        end if;
      when PDATA2_S =>
        if (rxDataK="01" and rxData=(D_215_C & K_EOF_C)) then
          v.state := IDLE_S;
        else
          v.strobe(r.partition) := rxData(14);
          v.l1input.trigword    := rxData(13 downto 5);
          v.l1input.tag         := rxData( 4 downto 0);
          v.state := PFULL_S;
        end if;

      when DDATA_S =>
        if (rxDataK="01" and rxData=(D_215_C & K_EOF_C)) then
          v.state := IDLE_S;
        else
          v.pfull               := (others=>'0');
          v.pfull (r.partition) := rxData(15);
          v.strobe(r.partition) := rxData(14);
          v.l1input.trigword    := rxData(13 downto 5);
          v.l1input.tag         := rxData( 4 downto 0);
          v.l1input.trigsrc     := uconfig.trigsrc;
        end if;
      when others => null;
    end case;

    if (rxRst='1' or rxErr='1') then
      v := REG_INIT_C;
    end if;

    if (uconfig.enable='0') then
      v.pfull  := (others=>'0');
      v.strobe := (others=>'0');
    end if;
    
    rin <= v;
  end process comb;

  seq: process (rxClk) is
  begin
    if rising_edge(rxClk) then
      r <= rin;
    end if;
  end process seq;

end rtl;
