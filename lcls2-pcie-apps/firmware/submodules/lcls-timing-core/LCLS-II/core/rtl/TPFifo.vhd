-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TPFifo.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-15
-- Last update: 2015-10-28
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: FIFO for caching timing messages with triggering/sparsification.
--
-- Triggering is specified by 'trig' input which indicates the selection bit
-- Sparsification is done by comparing 'wrDataWord' input against mask 'wsel'.
-- The 'wsel' selection mask is written as the first word (along with sof marker).
-- The last word written is eof marker.  The FIFO full signal is latched until
-- reset, so the FIFO never records a partial frame or drops frames between resets.
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
library UNISIM;
use UNISIM.VCOMPONENTS.all;
use work.StdRtlPkg.all;

entity TPFifo is
   generic ( TPD_G : time := 1 ns; LOGDEPTH : integer := 10);
   port (
      -- Clock and reset
      rst        : in sl;
      wrClk      : in sl;
      sof        : in sl;
      eof        : in sl;
      wrData     : in slv(15 downto 0);
      wrDataWord : in slv(3 downto 0);
      trig       : in slv(11 downto 0);
      wsel       : in slv(15 downto 0);

      rdClk  : in  sl;
      rdEn   : in  sl;
      rdData : out slv(31 downto 0);

      full  : out sl;
      empty : out sl
      );
end TPFifo;

-- Define architecture for top level module
architecture TPFifo of TPFifo is

   type RegType is
   record
      srst    : sl;
      ready   : sl;
      full    : sl;
      wrEn    : sl;
      wselect : slv(wsel'range);
      uninit  : slv(7 downto 0);
      dwrAddr : slv(7 downto 0);
      drdAddr : slv(7 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
      srst    => '1',
      ready   => '0',
      full    => '0',
      wrEn    => '0',
      wselect => (others => '0'),
      uninit  => (others => '1'),
      dwrAddr => (others => '0'),
      drdAddr => (others => '0'));

   type StatusType is record
      empty   : sl;
      full    : sl;
      trig    : sl;
      dwrData : slv(18 downto 0);
   end record;

   signal r      : RegType := REG_INIT_C;
   signal rin    : RegType;
   signal status : StatusType;

   signal wrEnQ, rdEnQ : sl;
   signal wselq        : sl;
   signal dwrDataQ     : slv(17 downto 0);

begin

   rdData(31 downto 18) <= (others => '0');

   full  <= r.full;
   empty <= status.empty;
   rdEnQ <= rdEn and not status.empty;

   wrEnQ <= (status.trig and not r.full) or
            (r.wrEn and (status.dwrData(18) or status.dwrData(17)));
   wselq    <= r.wselect(conv_integer(wrDataWord));
   dwrDataQ <= ("01" & r.wselect) when status.trig = '1' else
               status.dwrData(dwrDataQ'range);
   status.trig <= '1' when (status.dwrData(16) = '1' and r.ready = '1') else
                  '0';

   U_tpfifo_delay : entity work.SimpleDualPortRam
      generic map (
         TPD_G        => TPD_G,
         DATA_WIDTH_G => 19,
         ADDR_WIDTH_G => 8)
      port map (
         clka              => wrClk,
         ena               => '1',
         wea               => '1',
         addra             => r.dwrAddr,
         dina(15 downto 0) => wrData,
         dina(16)          => sof,
         dina(17)          => eof,
         dina(18)          => wselq,
         clkb              => wrClk,
         enb               => '1',
         rstb              => r.srst,
         addrb             => r.drdAddr,
         doutb             => status.dwrData);

   U_tpfifo_async : entity work.FifoAsync
      generic map (
         TPD_G        => TPD_G,
         FWFT_EN_G    => true,
         DATA_WIDTH_G => 18,
         ADDR_WIDTH_G => LOGDEPTH,
         FULL_THRES_G => (2**LOGDEPTH)-64)
      port map (
         rst           => r.srst,
         wr_clk        => wrClk,
         wr_en         => wrEnQ,
         din           => dwrDataQ(17 downto 0),
         wr_data_count => open,
         wr_ack        => open,
         overflow      => open,
         prog_full     => status.full,
         almost_full   => open,
         full          => open,
         not_full      => open,
         rd_clk        => rdClk,
         rd_en         => rdEnQ,
         dout          => rdData(17 downto 0),
         rd_data_count => open,
         valid         => open,
         underflow     => open,
         prog_empty    => open,
         almost_empty  => open,
         empty         => status.empty);

   comb : process (r, status, trig, wsel, rst, wrData)
      variable v : RegType;
   begin  -- process
      v := r;

      v.srst    := '0';
      v.ready   := '0';
      v.dwrAddr := r.dwrAddr+1;
      v.drdAddr := r.drdAddr+1;
      if allBits(r.uninit, '0') then
         v.ready := wrData(conv_integer(trig(3 downto 0)));
      else
         v.uninit := r.uninit-1;
      end if;

      if status.trig = '1' then
         if status.full = '1' then
            v.wrEn := '0';
            v.full := '1';
         elsif r.full = '0' then
            v.wrEn := '1';
         end if;
      elsif status.dwrData(17) = '1' then
         v.wrEn := '0';
      end if;

      if r.srst = '1' then
         v.dwrAddr := trig(11 downto 4);
         v.drdAddr := (others => '0');
         v.uninit  := trig(11 downto 4);
         v.wselect := wsel;
         v.ready   := '0';
      end if;

      rin <= v;
   end process;

   process (rst, wrClk)
   begin
      if rst = '1' then
         r.ready <= '0';
         r.srst  <= '1';
         r.full  <= '0';
         r.wrEn  <= '0';
      elsif rising_edge(wrClk) then
         r <= rin after TPD_G;
      end if;
   end process;

end TPFifo;
