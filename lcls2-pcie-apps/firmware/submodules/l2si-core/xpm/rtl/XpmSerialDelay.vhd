-------------------------------------------------------------------------------
-- Title      : XpmSerialDelay
-------------------------------------------------------------------------------
-- File       : XpmSerialDelay.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-07-07
-- Last update: 2017-08-21
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Delays a 16b serialized frame
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

entity XpmSerialDelay is
   generic ( TPD_G         : time    := 1 ns;
             DELAY_WIDTH_G : integer := 16;
             NWORDS_G      : integer := 16;    -- frame length in 16b words
             FDEPTH_G      : integer := 100 );  -- max depth of frame pipeline
   port (
      -- Clock and reset
      clk        : in  sl;
      rst        : in  sl;
      delay      : in  slv(DELAY_WIDTH_G-1 downto 0);
      delayRst   : in  sl;
      fiducial_i : in  sl;  -- must precede advance_i
      advance_i  : in  sl;  -- follows fiducial_i by 1
      stream_i   : in  TimingSerialType;
      reset_o    : out sl;
      fiducial_o : out sl;  -- precedes advance_o by 1
      advance_o  : out sl;  -- accompanies valid data
      stream_o   : out TimingSerialType;
      overflow_o : out sl);
end XpmSerialDelay;

-- Define architecture for top level module
architecture behavior of XpmSerialDelay is

  constant CADDR_WIDTH_C : integer := log2(FDEPTH_G);
  constant MADDR_WIDTH_C : integer := log2(NWORDS_G*FDEPTH_G);

  type StateType is ( IDLE_S, SHIFT_S, ARMED_S, INJECT_S, ERR_S );
  
  type RegType is record
    count  : slv(DELAY_WIDTH_G-1 downto 0);
    target : slv(DELAY_WIDTH_G-1 downto 0);
    stream : TimingSerialType;
    state  : StateType;
    running: sl;
    firstW : sl;
    rd_cnt : sl;
    rd_msg : sl;
    valid  : sl;
    delayRst: sl;
    fifoRst: sl;
    fiducial_inj : slv(3 downto 0);
    inject_fid : sl;
    inject_msg : sl;
    inject_cnt : slv(log2(NWORDS_G)-1 downto 0);
  end record;

  constant REG_INIT_C : RegType := (
    count  => (others=>'0'),
    target => (others=>'0'),
    stream => TIMING_SERIAL_INIT_C,
    state  => ERR_S,
    running=> '0',
    firstW => '0',
    rd_cnt => '0',
    rd_msg => '0',
    valid  => '0',
    delayRst => '0',
    fifoRst=> '1',
    fiducial_inj => (others=>'0'),
    inject_fid => '0',
    inject_msg => '0',
    inject_cnt => (others=>'0'));

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  
  signal valid_cnt : sl;
  signal valid_msg : sl;
  signal full_cnt  : sl;
  signal full_msg  : sl;
  signal dout_cnt  : slv(DELAY_WIDTH_G-1 downto 0);
  signal dout_rdy  : sl;
  signal dout_msg  : slv(15 downto 0);
  signal dout_offset : slv(6 downto 0);
  signal dout_last   : sl;
  signal firstW    : sl;
  
begin

   stream_o   <= r.stream;
   fiducial_o <= r.rd_cnt or r.inject_fid;
   advance_o  <= r.rd_msg or r.inject_msg;
   overflow_o <= full_cnt or full_msg;
   reset_o    <= not r.running;
   
   U_CntDelay : entity work.FifoSync
     generic map ( TPD_G        => TPD_G,
                   FWFT_EN_G    => true,
                   DATA_WIDTH_G => DELAY_WIDTH_G+9,
                   ADDR_WIDTH_G => CADDR_WIDTH_C )
     port map ( rst               => r.fifoRst,
                clk               => clk,
                wr_en             => fiducial_i,
                din(0)                         => stream_i.ready,
                din(7 downto 1)                => stream_i.offset,
                din(8)                         => stream_i.last,
                din(DELAY_WIDTH_G+8 downto 9)  => r.target,
                rd_en                          => r.rd_cnt,
                dout(0)                        => dout_rdy,
                dout(7 downto 1)               => dout_offset,
                dout(8)                        => dout_last,
                dout(DELAY_WIDTH_G+8 downto 9) => dout_cnt,
                valid             => valid_cnt,
                overflow          => full_cnt );
   
   U_MsgDelay : entity work.FifoSync
     generic map ( TPD_G        => TPD_G,
                   FWFT_EN_G    => true,
                   DATA_WIDTH_G => 17,
                   ADDR_WIDTH_G => MADDR_WIDTH_C )
     port map ( rst               => r.fifoRst,
                clk               => clk,
                wr_en             => advance_i,
                din(15 downto 0)  => stream_i.data,
                din(16)           => r.firstW,
                rd_en             => rin.rd_msg,
                dout(15 downto 0) => dout_msg,
                dout(16)          => firstW,
                valid             => valid_msg,
                overflow          => full_msg );

   process (r, rst, delay, delayRst, valid_cnt, valid_msg,
            dout_cnt, dout_msg, dout_offset, dout_last, dout_rdy,
            firstW, fiducial_i, advance_i ) is
     variable v : RegType;
   begin
     v := r;

     v.count  := r.count+1;
     v.target := r.count+5+delay; -- need extra fixed delay for cntdelay fifo
     v.rd_msg := '0';
     v.rd_cnt := '0';
     v.fifoRst := '0';
     v.fiducial_inj := fiducial_i & r.fiducial_inj(r.fiducial_inj'left downto 1);
     v.inject_fid := '0';
     v.inject_msg := '0';
     
     if fiducial_i='1' then
       v.firstW := '1';
     elsif advance_i='1' then
       v.firstW := '0';
     end if;

     if delayRst='1' then
       v.delayRst := '1';
     end if;
     
     case (r.state) is
       when ERR_S =>
         v.fifoRst := advance_i;
         if fiducial_i='1' then
           v.running := '1';
           v.state := ARMED_S;
         end if;
       when IDLE_S  =>
         if (valid_msg='1' and firstW='1') then
           v.stream.data  := dout_msg;
           v.rd_msg       := '1';
           v.state        := SHIFT_S;
         end if;
       when SHIFT_S =>
         if (valid_msg='0' or firstW='1') then
           v.stream.ready := '0';
           v.state  := ARMED_S;
         else
           v.stream.data  := dout_msg;
           v.rd_msg       := '1';
         end if;
       when ARMED_S =>
         v.stream.ready := '0';
         if r.delayRst='1' then
           v := REG_INIT_C;
         elsif (valid_cnt='1' and dout_cnt=r.count) then
           v.rd_cnt := '1';
           v.stream.ready  := dout_rdy;
           v.stream.offset := dout_offset;
           v.stream.last   := dout_last;
           v.state := IDLE_S;
         elsif (r.fiducial_inj(0)='1') then
           -- inject an empty frame
           v.inject_fid := '1';
           v.inject_cnt := (others=>'0');
           v.stream.ready  := '1';
           v.stream.offset := (others=>'0');
           v.stream.last   := '1';
           v.state := INJECT_S;
         end if;
       when INJECT_S =>
         v.inject_msg  := '1';
         v.inject_cnt  := r.inject_cnt+1;
         v.stream.data := (others=>'0');
         if (r.inject_cnt=NWORDS_G-1) then
           v.state := ARMED_S;
         end if;
       when others => NULL;
     end case;

     if rst='1' then
       v := REG_INIT_C;
     end if;
     
     rin <= v;
   end process;

   process ( clk ) is
   begin
     if rising_edge(clk) then
       r <= rin after TPD_G;
     end if;
   end process;

end behavior;
