-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2Core.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-01-04
-- Last update: 2017-04-27
-- Platform   : 
-- Standard   : VHDL'93/02
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

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.NUMERIC_STD.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;

entity EvrV2Axi is
  generic (
    TPD_G        : time    := 1 ns;
    DMA_ENABLE_G : boolean := false;
    CHANNELS_C   : integer := 1 );
  port (
    -- AXI-Lite and IRQ Interface
    axiClk              : in  sl;
    axiRst              : in  sl;
    axilWriteMaster     : in  AxiLiteWriteMasterType;
    axilWriteSlave      : out AxiLiteWriteSlaveType;
    axilReadMaster      : in  AxiLiteReadMasterType;
    axilReadSlave       : out AxiLiteReadSlaveType;
    -- configuration
    irqEnable           : out sl;
    channelConfig       : out EvrV2ChannelConfigArray(CHANNELS_C-1 downto 0);
    trigSel             : out sl;
    dmaFullThr          : out slv(23 downto 0);
    -- status
    irqReq              : in  sl := '0';
    partitionAddr       : in  slv(PADDR_LEN-1 downto 0) := (others=>'0');
    rstCount            : out sl;
    eventCount          : in  SlVectorArray(CHANNELS_C downto 0,31 downto 0);
    gtxDebug            : in  slv(7 downto 0) := (others=>'0') );
end EvrV2Axi;

architecture mapping of EvrV2Axi is

  type RegType is record
    axilReadSlave  : AxiLiteReadSlaveType;
    axilWriteSlave : AxiLiteWriteSlaveType;
    irqEnable      : sl;
    countReset     : sl;
    trigSel        : sl;
    channelConfig  : EvrV2ChannelConfigArray(CHANNELS_C-1 downto 0);
    dmaFullThr     : slv(23 downto 0);
  end record;
  constant REG_INIT_C : RegType := (
    axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
    axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C,
    irqEnable      => '0',
    countReset     => '0',
    trigSel        => '0',
    channelConfig  => (others=>EVRV2_CHANNEL_CONFIG_INIT_C),
    dmaFullThr     => (others=>'1') );
  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

begin  -- mapping

  channelConfig  <= r.channelConfig;
  axilReadSlave  <= r.axilReadSlave;
  axilWriteSlave <= r.axilWriteSlave;
  irqEnable      <= r.irqEnable;
  trigSel        <= r.trigSel;
  rstCount       <= r.countReset;
  dmaFullThr     <= r.dmaFullThr;

  process (axiClk)
  begin  -- process
    if rising_edge(axiClk) then
      r <= rin;
    end if;
  end process;

  process (r,axilReadMaster,axilWriteMaster,axiRst,gtxDebug,eventCount,irqReq,partitionAddr)
    variable v : RegType;
    variable sReg : slv(0 downto 0);
    variable axilStatus : AxiLiteStatusType;
    procedure axilSlaveRegisterR (addr : in slv; reg : in slv) is
    begin
      axiSlaveRegister(axilReadMaster, v.axilReadSlave, axilStatus, addr, 0, reg);
    end procedure;
    procedure axilSlaveRegisterR (addr : in slv; reg : in slv; ack : out sl) is
    begin
      if (axilStatus.readEnable = '1') then
         if (std_match(axilReadMaster.araddr(addr'length-1 downto 0), addr)) then
            v.axilReadSlave.rdata(reg'range) := reg;
            axiSlaveReadResponse(v.axilReadSlave);
            ack := '1';
         end if;
      end if;
    end procedure;
    procedure axilSlaveRegisterW (addr : in slv; offset : in integer; reg : inout slv) is
    begin
      axiSlaveRegister(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, addr, offset, reg);
    end procedure;
    procedure axilSlaveRegisterW (addr : in slv; offset : in integer; reg : inout sl) is
    begin
      axiSlaveRegister(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, addr, offset, reg);
    end procedure;
    procedure axilSlaveDefault (
      axilResp : in slv(1 downto 0)) is
    begin
      axiSlaveDefault(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, axilResp);
    end procedure;
  begin  -- process
    v  := r;
    sReg(0) := irqReq;
    axiSlaveWaitTxn(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus);
    axilSlaveRegisterW(X"010", 0, v.countReset);
    axilSlaveRegisterW(X"014", 0, v.trigSel);
    for i in 0 to CHANNELS_C-1 loop
      axilSlaveRegisterW(slv(conv_unsigned(i*32+32,9)),  0, v.channelConfig(i).enabled);
      axilSlaveRegisterW(slv(conv_unsigned(i*32+36,9)),  0, v.channelConfig(i).rateSel);
      axilSlaveRegisterW(slv(conv_unsigned(i*32+36,9)), 13, v.channelConfig(i).destSel);
      axilSlaveRegisterR(slv(conv_unsigned(i*32+40,9)), muxSlVectorArray(eventCount, i));
    end loop;  -- i
    axilSlaveRegisterR(slv(conv_unsigned(12*32+40,9)), muxSlVectorArray(eventCount, CHANNELS_C));

    if DMA_ENABLE_G then
      axilSlaveRegisterW(X"000", 0, v.irqEnable);
      axilSlaveRegisterR(X"004", sReg);
      axilSlaveRegisterR(X"008", partitionAddr);
      axilSlaveRegisterR(X"00C", gtxDebug);
      axilSlaveRegisterW(X"018", 0, v.dmaFullThr);
      for i in 0 to CHANNELS_C-1 loop
        axilSlaveRegisterW(slv(conv_unsigned(i*32+32,9)),  1, v.channelConfig(i).bsaEnabled);
        axilSlaveRegisterW(slv(conv_unsigned(i*32+32,9)),  2, v.channelConfig(i).dmaEnabled);
        axilSlaveRegisterW(slv(conv_unsigned(i*32+44,9)),  0, v.channelConfig(i).bsaActiveDelay);
        axilSlaveRegisterW(slv(conv_unsigned(i*32+44,9)), 20, v.channelConfig(i).bsaActiveSetup);
        axilSlaveRegisterW(slv(conv_unsigned(i*32+48,9)),  0, v.channelConfig(i).bsaActiveWidth);
      end loop;  -- i
    end if;
    
    axilSlaveDefault(AXI_RESP_OK_C);
    rin <= v;
  end process;

end mapping;
