-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiHeaderCache.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-07-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.ArbiterPkg.all;
use work.TimingExtnPkg.all;
use work.TimingPkg.all;
use work.XpmPkg.all;
use work.DtiPkg.all;

library unisim;
use unisim.vcomponents.all;

entity DtiHeaderCache is
   generic (
      TPD_G               : time                := 1 ns );
   port (
     rst             : in  sl;
     --  Cache Input
     wrclk           : in  sl;
     enable          : in  sl;
     timingBus       : in  TimingBusType;
     exptBus         : in  ExptBusType;
     triggerBus      : in  ExptBusType;
     partition       : in  slv(2 downto 0);
     pdata           : out XpmPartitionDataType;
     pdataV          : out sl;
     cntL0           : out slv(19 downto 0);
     cntL1A          : out slv(19 downto 0);
     cntL1R          : out slv(19 downto 0);
     cntWrFifo       : out slv( 3 downto 0);
     --  Cache Output
     rdclk           : in  sl;
     entag           : in  sl;
     l0tag           : in  slv(4 downto 0);
     advance         : in  sl;
     valid           : out sl;
     pmsg            : out sl;
     phdr            : out sl;
     cntRdFifo       : out slv( 3 downto 0);
     hdrOut          : out DtiEventHeaderType );
end DtiHeaderCache;

architecture rtl of DtiHeaderCache is

  type WrRegType is record
    rden   : sl;
    wren   : sl;
    pvec   : slv(47 downto 0);
    pmsg   : slv( 5 downto 0);
    phdr   : slv( 5 downto 0);
    tword  : XpmPartitionDataType;
    twordV : sl;
    pword  : XpmPartitionDataType;
    pwordV : sl;
    cntL0  : slv(19 downto 0);
    cntL1A : slv(19 downto 0);
    cntL1R : slv(19 downto 0);
    cntWrF : slv( 3 downto 0);
    rstF   : sl;
  end record;

  constant WR_REG_INIT_C : WrRegType := (
    rden   => '0',
    wren   => '0',
    pvec   => (others=>'1'),
    pmsg   => (others=>'0'),
    phdr   => (others=>'0'),
    tword  => XPM_PARTITION_DATA_INIT_C,
    twordV => '0',
    pword  => XPM_PARTITION_DATA_INIT_C,
    pwordV => '0',
    cntL0  => (others=>'0'),
    cntL1A => (others=>'0'),
    cntL1R => (others=>'0'),
    cntWrF => (others=>'0'),
    rstF   => '1' );

  signal wr    : WrRegType := WR_REG_INIT_C;
  signal wr_in : WrRegType;

  type RdRegType is record
    cntRdF : slv( 3 downto 0);
  end record;

  constant RD_REG_INIT_C : RdRegType := (
    cntRdF => (others=>'0') );

  signal rd    : RdRegType := RD_REG_INIT_C;
  signal rd_in : RdRegType;
  
  signal wrrst, rdrst : sl;
  signal entagw, entagr : sl;
  signal daddr        : slv(  4 downto 0);
  signal doutf        : slv(  6 downto 0);
  signal doutb        : slv(191 downto 0);
  signal spartition   : slv(partition'range);
  signal wr_data_count: slv( 3 downto 0);
  signal rd_data_count: slv( 3 downto 0);

  signal pword        : slv(47 downto 0);
  signal gword        : slv(15 downto 0);
  signal ip           : integer;
  signal ptag         : slv(4 downto 0);
  signal hdrWe        : sl;
begin

  --  trigger bus
  pdata            <= wr.tword;
  pdataV           <= wr.twordV;
  cntL0            <= wr.cntL0;
  cntL1A           <= wr.cntL1A;
  cntL1R           <= wr.cntL1R;
  cntWrFifo        <= wr.cntWrF;
  cntRdFifo        <= rd.cntRdF;

  hdrOut.pulseId   <= doutb( 63 downto   0);
  hdrOut.timeStamp <= doutb(127 downto  64);
  hdrOut.evttag    <= doutb(191 downto 128);
  pmsg             <= doutf(5);
  phdr             <= doutf(6);
  
  GEN_GROUPS : for i in 0 to NPartitions-1 generate
    gword(i) <= '1' when (toPartitionWord(exptBus.message.partitionWord(i)).l0a='1') else
                '0';
  end generate;
  gword(15 downto 8) <= (others=>'0');
  
  daddr <= l0tag when entagr='1' else
           doutf(4 downto 0);

  hdrWe <= wr_in.pmsg(0) or wr_in.phdr(0);
  pword <= exptBus.message.partitionWord(conv_integer(spartition));
  ptag  <= toPartitionWord(pword).l0tag;
  
  U_RstIn  : entity work.RstSync
    port map ( clk      => wrclk,
               asyncRst => rst,
               syncRst  => wrrst );
  
  U_RstOut  : entity work.RstSync
    port map ( clk      => rdclk,
               asyncRst => rst,
               syncRst  => rdrst );

  U_EntagR : entity work.Synchronizer
    port map ( clk      => rdclk,
               dataIn   => entag,
               dataOut  => entagr );
  
  U_TagRam : entity work.SimpleDualPortRam
    generic map ( DATA_WIDTH_G => 192,
                  ADDR_WIDTH_G => 5 )
    port map ( clka   => wrclk,
               ena    => '1',
               wea    => hdrWe,
               addra  => ptag,
               dina( 63 downto   0) => timingBus.message.pulseId,
               dina(127 downto  64) => timingBus.message.timeStamp,
               dina(143 downto 128) => gword,
               dina(159 downto 144) => pword(15 downto 0),
               dina(191 downto 160) => pword(47 downto 16),
               clkb   => rdclk,
               enb    => '1',
               addrb  => daddr,
               doutb  => doutb );

  U_TagFifo : entity work.FifoAsync
    generic map ( ADDR_WIDTH_G => 4,
                  DATA_WIDTH_G => 7,
                  FWFT_EN_G    => true )
    port map ( rst           => wr.rstF,
               wr_clk        => wrclk,
               wr_en         => hdrWe,
               wr_data_count => wr_data_count,
               din(4 downto 0) => ptag,
               din(5)          => wr_in.pmsg(0),
               din(6)          => wr_in.phdr(0),
               rd_clk        => rdclk,
               rd_en         => advance,
               rd_data_count => rd_data_count,
               dout          => doutf,
               valid         => valid );

  U_SPartition : entity work.SynchronizerVector
    generic map ( WIDTH_G => 3 )
    port map ( clk     => wrclk,
               dataIn  => partition,
               dataOut => spartition );
  
  comb : process( wr, wrrst, timingBus, triggerBus, exptBus, spartition, enable, wr_data_count ) is
    variable v  : WrRegType;
    variable ip : integer;
  begin
    v := wr;

    v.rden      := '0';
    v.wren      := '0';
    v.twordV    := '0';
    v.pwordV    := '0';
    v.pmsg      := wr.pmsg(wr.pmsg'left-1 downto 0) & '0';
    v.phdr      := wr.phdr(wr.phdr'left-1 downto 0) & '0';
    v.rstF      := '0';
    
    ip := conv_integer(spartition);

    if timingBus.strobe = '1' then
      v.rden   := '1';
      --  Prompt trigger
      if triggerBus.valid='1' then
        v.tword  := toPartitionWord(triggerBus.message.partitionWord(ip));
        v.twordV := enable and triggerBus.message.partitionWord(ip)(15);
      end if;
      --  Delayed event header
      if exptBus.valid='1' then
        v.pword  := toPartitionWord(exptBus.message.partitionWord(ip));
        v.pwordV := enable and exptBus.message.partitionWord(ip)(15);
        v.pvec   := exptBus.message.partitionWord(ip);
        v.pmsg(0) := not exptBus.message.partitionWord(ip)(15);
        v.phdr(0) := enable and     exptBus.message.partitionWord(ip)(15) and
                     toPartitionWord(exptBus.message.partitionWord(ip)).l0a;
      end if;
    end if;

    if wr.pmsg /= 0 and toPartitionMsg(wr.pvec).hdr = MSG_CLEAR_FIFO then
      v.rstF := '1';
    end if;
      
    if wr.rden = '1' then
      v.wren  := wr.pword.l0a or not wr.pvec(15);
    end if;
    
    if wrrst = '1' then
      v := WR_REG_INIT_C;
    end if;

    if wr.pwordV = '1' then
      if wr.pword.l0a = '1' then
        v.cntL0 := wr.cntL0 + 1;
      end if;

      if wr.pword.l1e = '1' then
        if wr.pword.l1a = '1' then
          v.cntWrF := wr_data_count;
          v.cntL1A := wr.cntL1A + 1;
        else
          v.cntL1R := wr.cntL1R + 1;
        end if;
      end if;
    end if;
    
    wr_in <= v;
  end process;
  
  seq : process (wrclk) is
  begin
    if rising_edge(wrclk) then
      wr <= wr_in;
    end if;
  end process;

  rdcomb : process( rd, rdrst, advance, rd_data_count ) is
    variable v  : RdRegType;
  begin
    v := rd;

    if advance = '1' then
      v.cntRdF := rd_data_count;
    end if;

    rd_in <= v;
  end process;

  rdseq : process (rdclk) is
  begin
    if rising_edge(rdclk) then
      rd <= rd_in;
    end if;
  end process;
  
end rtl;
