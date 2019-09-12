-------------------------------------------------------------------------------
-- Title      : TimingStreamTx
-------------------------------------------------------------------------------
-- File       : TimingStreamTx.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2017-02-02
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Generates a 16b serial stream of the LCLS-II timing message.
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
library ieee;
use work.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.StdRtlPkg.all;
use work.TimingPkg.all;
use work.CrcPkg.all;

entity TimingStreamTx is
  generic (
    TPD_G : time := 1 ns);
   port (
      -- Clock and reset
      clk       : in  sl;
      rst       : in  sl;
      fiducial  : in  sl;
      dataBuff  : in  TimingDataBuffType;
      pulseId   : in  slv(31 downto 0);
      eventCodes: in  slv(255 downto 0);
      data      : out slv(15 downto 0);
      dataK     : out slv(1 downto 0));
end TimingStreamTx;

-- Define architecture for top level module
architecture TimingStreamTx of TimingStreamTx is

   type StateType is (IDLE_S, FRAME_S, PULSEID_S, ECODE_S);
   type RegType is
   record
      state         : StateType;
      dataBuffEn    : sl;
      dataBuff      : slv(TIMING_DATABUFF_BITS_C-1 downto 0);
      pulseId       : slv(31 downto 0);
      eventCodes    : slv(255 downto 0);
      wordCount     : slv( 7 downto 0);
      dbufData      : slv( 7 downto 0);
      ecodData      : slv( 7 downto 0);
      dataK         : slv( 1 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
      state         => IDLE_S,
      dataBuffEn    => '0',
      dataBuff      => (others=>'0'),
      pulseId       => (others=>'0'),
      eventCodes    => (others=>'0'),
      wordCount     => (others=>'0'),
      dbufData      => K_COM_C,
      ecodData      => x"00",
      dataK         => "10");

   constant NDATABUFF_WORDS : slv(7 downto 0) := toSlv(TIMING_DATABUFF_BITS_C/8,8);
   
  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  signal crc : slv(31 downto 0);
  
begin

  data     <= r.dbufData & r.ecodData;
  dataK    <= r.dataK;
  
  comb: process (rst, fiducial, pulseId, eventCodes, dataBuff, r)
    variable v    : RegType;
    variable istr : integer;
  begin 
      v := r;

      v.dataBuffEn := not r.dataBuffEn;
      v.dbufData   := D_215_C;
      v.ecodData   := K_COM_C;
      v.dataK      := "01";
      
      case (r.state) is
        when IDLE_S =>
          v.wordCount := (others=>'0');
          if fiducial = '1' then
            v.state      := FRAME_S;
            v.dataBuff   := toSlv(dataBuff);
            v.dataBuffEn := '0';
            v.pulseId    := pulseId;
            v.eventCodes := eventCodes;
            v.dataK      := "11";
            v.dbufData   := K_280_C;
          end if;
        when FRAME_S =>
          if r.wordCount=NDATABUFF_WORDS then
            v.state     := PULSEID_S;
            v.wordCount := (others=>'0');
            v.dataK      := "11";
            v.dbufData   := K_281_C;
          elsif r.dataBuffEn='1' then
            v.dbufData  := r.dataBuff(7 downto 0);
            v.dataBuff  := x"00" & r.dataBuff(r.dataBuff'left downto 8);
            v.wordCount := r.wordCount+1;
          end if;
        when PULSEID_S =>
          v.dataK     := "00";
          if r.wordCount=toSlv(32,r.wordCount'length) then
            v.state     := ECODE_S;
            v.wordCount := (others=>'0');
            v.ecodData  := x"7D";
          else
            v.wordCount := r.wordCount+1;
            v.pulseId   := r.pulseId(30 downto 0) & '0';
            if r.pulseId(31)='0' then
              v.ecodData := x"70";
            else
              v.ecodData := x"71";
            end if;
          end if;
        when ECODE_S =>
          if r.wordCount = toSlv(255,r.wordCount'length) then
            v.state := IDLE_S;
          else
            v.wordCount := r.wordCount+1;
            if r.eventCodes(conv_integer(r.wordCount))='1' then
              v.dataK    := "00";
              v.ecodData := r.wordCount;
            end if;
          end if;
        when others => null;
      end case;

      if (rst='1') then
        rin <= REG_INIT_C;
      end if;
      
      rin <= v;

   end process;

   process (clk)
   begin  -- process
      if rising_edge(clk) then
         r <= rin after TPD_G;
      end if;
   end process;

end TimingStreamTx;
