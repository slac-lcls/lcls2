-------------------------------------------------------------------------------
-- Title      : TimingDeserializer
-------------------------------------------------------------------------------
-- File       : TimingDeserializer.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2016-04-28
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

entity TimingDeserializer is
   generic (
      TPD_G     : time    := 1 ns;
      STREAMS_C : integer := 1 );
   port (
      -- Clock and reset
      clk       : in  sl;
      rst       : in  sl;
      fiducial  : out sl;
      streams   : out TimingSerialArray(STREAMS_C-1 downto 0);
      streamIds : in  Slv4Array        (STREAMS_C-1 downto 0);
      advance   : out slv              (STREAMS_C-1 downto 0);
      data      : in  TimingRxType;
      sof       : out sl;
      eof       : out sl;
      crcErr    : out sl );
end TimingDeserializer;

-- Define architecture for top level module
architecture TimingDeserializer of TimingDeserializer is

   type StateType is (IDLE_S, SOS_S, SEGMENT_S, SINK_S, CRC1_S, CRC2_S, EOF_S);
   
   type RegType is
   record
      state        : StateType;
      stream       : integer range 0 to STREAMS_C-1;
      streams      : TimingSerialArray(STREAMS_C-1 downto 0);
      served       : slv(STREAMS_C-1 downto 0);
      advance      : slv(STREAMS_C-1 downto 0);
      fiducial     : sl;
      crc          : slv(31 downto 0);
      crcReset     : sl;
      crcValid     : sl;
      sof          : sl;
      eof          : sl;
      crcErr       : sl;
   end record;

   constant REG_INIT_C : RegType := (
      state        => IDLE_S,
      stream       => 0,
      streams      => (others=>TIMING_SERIAL_INIT_C),
      served       => (others=>'0'),
      advance      => (others=>'0'),
      fiducial     => '0',
      crc          => (others=>'0'),
      crcReset     => '0',
      crcValid     => '0',
      sof          => '0',
      eof          => '0',
      crcErr       => '0');

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  signal crc : slv(31 downto 0);
  
begin

  fiducial <= r.fiducial;
  streams  <= r.streams;
  advance  <= r.advance;
  sof      <= r.sof;
  eof      <= r.eof;
  crcErr   <= r.crcErr;
  
  U_CRC : entity work.Crc32Parallel
    generic map ( TPD_G=>TPD_G, BYTE_WIDTH_G => 2, CRC_INIT_G => x"FFFFFFFF" )
    port map ( crcOut       => crc,
               crcClk       => clk,
               crcDataValid => rin.crcValid,
               crcDataWidth => "001",
               crcIn        => data.data,
               crcReset     => r.crcReset );
  
  comb: process (rst, r, crc, streamIds, data)
    variable v    : RegType;
    variable istr : integer;
  begin 
      v := r;

      v.fiducial := '0';
      v.crcReset := '0';
      v.crcValid := '0';
      v.sof      := '0';
      v.eof      := '0';
      v.crcErr   := '0';
      v.advance  := (others=>'0');
      --for i in 0 to STREAMS_C-1 loop
      --  v.streams(i).ready := '0';
      --end loop;
      
      case (r.state) is
        when IDLE_S => 
          if (data.dataK="01" and data.data=(D_215_C & K_SOF_C)) then
             for i in 0 to STREAMS_C-1 loop
               v.streams(i).ready := r.served(i);
             end loop;
             v.fiducial := '1';
             v.sof      := '1';
             v.crcReset := '1';
             v.served   := (others=>'0');
             v.state    := SOS_S;
          end if;
        when SOS_S =>
          if (data.dataK="01" and data.data=(D_215_C & K_EOF_C)) then
             v.eof      := '1';
             v.state    := EOF_S;
          else
            v.state := SINK_S;
            for i in 0 to STREAMS_C-1 loop
              if (streamIds(i)=data.data(15 downto 12)) then
                v.stream           := i;
                v.streams(i).last  := data.data(7);
                v.streams(i).offset:= data.data(6 downto 0);
                v.crcValid         := '1';
                v.served(i)        := '1';
                v.state            := SEGMENT_S;
              end if;
            end loop;
          end if;
          v.crcValid  := '1';
        when SEGMENT_S =>
          -- Check for end of stream
          if (data.dataK="01" and data.data=(D_215_C & K_EOS_C)) then
            v.state   := SOS_S;
          else
            -- Send next word in stream
            v.streams(r.stream).data  := data.data;
            v.advance(r.stream)       := '1';
          end if;
          v.crcValid  := '1';
        when SINK_S =>
          -- Check for end of stream
          if (data.dataK="01" and data.data=(D_215_C & K_EOS_C)) then
            v.state   := SOS_S;
          end if;
          v.crcValid  := '1';
        when EOF_S =>
          v.crc       := data.data & r.crc(31 downto 16);
          v.state     := CRC1_S;
        when CRC1_S =>
          v.crc       := data.data & r.crc(31 downto 16);
          v.state     := CRC2_S;
        when CRC2_S =>
          if (r.crc/=crc) then
            v.crcErr := '1';
            v.served := (others=>'0');
          end if;
          v.state     := IDLE_S;
        when others => null;
      end case;

      -- On reset or error, reset the streams to invalidate accumulated data
      if (rst='1' or data.decErr/="00" or data.dspErr/="00") then
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

end TimingDeserializer;
