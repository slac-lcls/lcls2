-------------------------------------------------------------------------------
-- Title      : TimingSerialDelay
-------------------------------------------------------------------------------
-- File       : TimingSerialDelay.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-07-07
-- Last update: 2018-02-15
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

entity TimingSerialDelay is
   generic ( TPD_G         : time    := 1 ns;
             NWORDS_G      : integer := 16;    -- frame length in 16b words
             FDEPTH_G      : integer := 100 );  -- max depth of frame pipeline
   port (
      -- Clock and reset
      clk        : in  sl;
      rst        : in  sl;
      delay      : in  slv(19 downto 0);
      fiducial_i : in  sl;
      advance_i  : in  sl;
      stream_i   : in  TimingSerialType;
      frame_o    : out slv( 16*NWORDS_G-1 downto 0 );
      strobe_o   : out sl;
      valid_o    : out sl;
      overflow_o : out sl);
end TimingSerialDelay;

-- Define architecture for top level module
architecture TimingSerialDelay of TimingSerialDelay is

  constant CADDR_WIDTH_C : integer := log2(FDEPTH_G);
  constant MADDR_WIDTH_C : integer := log2(NWORDS_G*FDEPTH_G);

  type StateType is ( IDLE_S, SHIFT_S, ARMED_S, ERR_S );
  
  type RegType is record
    count  : slv(19 downto 0);
    target : slv(19 downto 0);
    frame  : Slv16Array(NWORDS_G-1 downto 0);
    state  : StateType;
    firstW : sl;
    rd_cnt : sl;
    rd_msg : sl;
    strobe : sl;
    valid  : sl;
    fifoRst: sl;
  end record;

  constant REG_INIT_C : RegType := (
    count  => (others=>'0'),
    target => (others=>'0'),
    frame  => (others=>(others=>'0')),
    state  => ERR_S,
    firstW => '0',
    rd_cnt => '0',
    rd_msg => '0',
    strobe => '0',
    valid  => '0',
    fifoRst=> '1');

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  
  signal valid_cnt : sl;
  signal valid_msg : sl;
  signal full_cnt  : sl;
  signal full_msg  : sl;
  signal dout_cnt  : slv(19 downto 0);
  signal dout_rdy  : sl;
  signal dout_msg  : slv(15 downto 0);
  signal firstW    : sl;
  signal wr_cnt    : sl;

  attribute use_dsp48      : string;
  attribute use_dsp48 of r : signal is "yes";  
  
begin

   GEN_FRAME: for i in 0 to NWORDS_G-1 generate
     frame_o(16*i+15 downto 16*i) <= r.frame(i);
   end generate;

   strobe_o   <= r.strobe;
   valid_o    <= r.valid;
   overflow_o <= full_cnt or full_msg;
   
   wr_cnt     <= fiducial_i and not r.firstW;
   
   U_CntDelay : entity work.FifoSync
     generic map ( TPD_G        => TPD_G,
                   FWFT_EN_G    => true,
                   DATA_WIDTH_G => 21,
                   ADDR_WIDTH_G => CADDR_WIDTH_C )
     port map ( rst               => r.fifoRst,
                clk               => clk,
                wr_en             => wr_cnt,
                din(19 downto 0)  => r.target,
                din(20)           => stream_i.ready,
                rd_en             => r.rd_cnt,
                dout(19 downto 0) => dout_cnt,
                dout(20)          => dout_rdy,
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

   process (r, rst, delay, valid_cnt, dout_cnt, valid_msg, dout_msg, dout_rdy, firstW, fiducial_i, advance_i ) is
     variable v : RegType;
   begin
     v := r;

     v.count  := r.count+1;
     v.target := r.count+5+delay; -- need extra fixed delay for cntdelay fifo
     v.rd_msg := '0';
     v.rd_cnt := '0';
     v.strobe := '0';
     v.fifoRst := '0';

     if fiducial_i='1' then
       v.firstW := '1';
     elsif advance_i='1' then
       v.firstW := '0';
     end if;
       
     case (r.state) is
       when ERR_S =>
         if fiducial_i='1' then
           v.state := IDLE_S;
         else 
           v.fifoRst := '1';
         end if;
       when IDLE_S  =>
         if (valid_msg='1' and firstW='1') then
           v.frame  := dout_msg & r.frame(r.frame'left downto 1);
           v.rd_msg := '1';
           v.state  := SHIFT_S;
         end if;
       when SHIFT_S =>
         if (valid_msg='0' or firstW='1') then
           v.state  := ARMED_S;
         else
           v.frame  := dout_msg & r.frame(r.frame'left downto 1);
           v.rd_msg := '1';
         end if;
       when ARMED_S =>
         if (valid_cnt='1' and dout_cnt=r.count) then
           v.rd_cnt := '1';
           v.strobe := '1';
           v.valid  := dout_rdy;
           v.state := IDLE_S;
         end if;
       when others => NULL;
     end case;

     if (rst='1') then
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

end TimingSerialDelay;
