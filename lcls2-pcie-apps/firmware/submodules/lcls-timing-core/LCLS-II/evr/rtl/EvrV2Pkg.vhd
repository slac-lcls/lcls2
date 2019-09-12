-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2Pkg.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-01-04
-- Last update: 2018-02-15
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

use work.StdRtlPkg.all;
use work.TimingPkg.all;

package EvrV2Pkg is

  constant TriggerOutputs  : integer := 12;
--  constant ReadoutChannels : integer := 10;
  constant ReadoutChannels : integer := 12;
  
  -- pipeline depth of frames for integrating BSA active signals:
  -- BSA active signals are integrating from <BsaActiveSetup> frames
  -- prior to the <eventSelect> until <BsaActiveDelay> + <BsaActiveWidth> after
  -- the <eventSelect>
--  constant BsaActiveSetup : integer := 108;

  constant EVRV2_CHANNEL_CONFIG_BITS_C : integer  := 81;
  
  type EvrV2ChannelConfig is record
    enabled          : sl;
    -- EventSelection
    rateSel          : slv(12 downto 0);
    -- Bits(12:11)=(fixed,AC,seq,reserved)
    -- fixed:  marker = 3:0
    -- AC   :  marker = 2:0;  TS = 8:3 (mask)
    -- seq  :  bit    = 3:0;  seq = 8:4
    destSel : slv(18 downto 0);
    -- Bits(17:16)=(Beam,NoBeam,DONT_CARE,reserved)
    -- Bits(15:0)=Mask of Destinations (when Beam)
    -- BSA
    bsaEnabled       : sl;              -- participate in BSA
    bsaActiveSetup   : slv( 5 downto 0);
    bsaActiveDelay   : slv(19 downto 0);
    bsaActiveWidth   : slv(19 downto 0);
    dmaEnabled       : sl;
  end record;

  constant EVRV2_CHANNEL_CONFIG_INIT_C : EvrV2ChannelConfig := (
    enabled        => '0',
    rateSel        => (others=>'0'),
    destSel        => (others=>'0'),
    bsaEnabled     => '0',
    bsaActiveSetup => (others=>'0'),
    bsaActiveDelay => (others=>'0'),
    bsaActiveWidth => toSlv(1,20),
    dmaEnabled     => '0' );

  type EvrV2ChannelConfigArray is array (natural range<>) of EvrV2ChannelConfig;

  constant EVRV2_TRIG_WIDTH_C : integer := 28;
  constant EVRV2_TRIGGER_CONFIG_BITS_C : integer := 29+2*EVRV2_TRIG_WIDTH_C;
  
  type EvrV2TriggerConfigType is record
    enabled  : sl;
    polarity : sl;
    delay    : slv(EVRV2_TRIG_WIDTH_C-1 downto 0);
    width    : slv(EVRV2_TRIG_WIDTH_C-1 downto 0);
    channel  : slv( 3 downto 0);
    channels : slv(15 downto 0);  -- mask of channels that arm trigger
    delayTap : slv( 5 downto 0);
    loadTap  : sl;
  end record;

  constant EVRV2_TRIGGER_CONFIG_INIT_C : EvrV2TriggerConfigType := (
    enabled   => '0',
    polarity  => '1',
    delay     => (others=>'0'),
    width     => (others=>'0'),
    channel   => (others=>'0'),
    channels  => (others=>'0'),
    delayTap  => (others=>'0'),
    loadTap   => '0' );

  type EvrV2TriggerConfigArray is array (natural range<>) of EvrV2TriggerConfigType;

  type EvrV2BsaControlType is record
    timeStamp  : slv(63 downto 0);
    bsaInit    : slv(63 downto 0);
    bsaDone    : slv(63 downto 0);
  end record;
  
  constant EVRV2_BSA_CONTROL_INIT_C : EvrV2BsaControlType := (
    timeStamp  => (others=>'0'),
    bsaInit    => (others=>'0'),
    bsaDone    => (others=>'0') );

  type EvrV2BsaChannelType is record
    pulseId    : slv(63 downto 0);
    bsaActive  : slv(63 downto 0);
    avgDoneId  : slv(63 downto 0);
    bsaAvgDone : slv(63 downto 0);
    bsaDone    : slv(63 downto 0);
  end record;

  constant EVRV2_BSA_CHANNEL_INIT_C : EvrV2BsaChannelType := (
    pulseId   => (others=>'0'),
    bsaActive => (others=>'0'),
    avgDoneId => (others=>'0'),
    bsaAvgDone=> (others=>'0'),
    bsaDone   => (others=>'0') );

  type EvrV2BsaChannelArray is array (natural range<>) of EvrV2BsaChannelType;
  
  type EvrV2DataType is record
    strobe      : sl;
    timingFrame : TimingMessageType;
  end record;

  type EvrV2CacheControlType is record
    reset       : sl;
    rdclk       : sl;
    advance     : sl;
  end record;

  constant EVRV2_CACHE_CONTROL_INIT_C : EvrV2CacheControlType := (
    reset   => '0',
    rdclk   => '0',
    advance => '0' );
  
  type EvrV2CacheControlArray is array (natural range<>) of EvrV2CacheControlType;
  
  type EvrV2CacheDataType is record
    empty       : sl;
    count       : slv(31 downto 0);
    data        : slv(31 downto 0);
  end record;

  constant EVRV2_CACHE_DATA_INIT_C : EvrV2CacheDataType := (
    empty   => '0',
    count   => (others=>'0'),
    data    => (others=>'0') );

  type EvrV2CacheDataArray is array (natural range<>) of EvrV2CacheDataType;

  type EvrV2DmaControlType is record
    ready    : sl;
    testData : slv(23 downto 0);
    fullThr  : slv(23 downto 0);
  end record;

  constant EVRV2_DMA_CONTROL_INIT_C : EvrV2DmaControlType := (
    ready    => '0',
    testData => (others=>'0'),
    fullThr  => (others=>'1'));

  type EvrV2DmaControlArray is array (natural range<>) of EvrV2DmaControlType;

  type EvrV2DmaDataType is record
    tValid : sl;
    tData  : slv(31 downto 0);
  end record;

  constant EVRV2_DMA_DATA_INIT_C : EvrV2DmaDataType := (
    tValid  => '0',
    tData   => (others=>'0') );

  type EvrV2DmaDataArray is array (natural range<>) of EvrV2DmaDataType;

  constant EVRV2_EVENT_TAG       : slv(15 downto 0) := x"0000";
  constant EVRV2_BSA_CONTROL_TAG : slv(15 downto 0) := x"0001";
  constant EVRV2_BSA_CHANNEL_TAG : slv(15 downto 0) := x"0002";
  constant EVRV2_END_TAG         : slv(15 downto 0) := x"000F";
  constant EVRV2_LCLS_TAG_BIT    : integer := 6;
  constant EVRV2_DROP_TAG_BIT    : integer := 7;

  function toSlv( cfg : EvrV2ChannelConfig ) return slv;
  function toSlv( cfg : EvrV2TriggerConfigType ) return slv;
  function toChannelConfig( vector : slv ) return EvrV2ChannelConfig;
  function toTriggerConfig( vector : slv ) return EvrV2TriggerConfigType;

end EvrV2Pkg;
  
package body EvrV2Pkg is

  function toSlv( cfg : EvrV2ChannelConfig) return slv is
    variable vector : slv(EVRV2_CHANNEL_CONFIG_BITS_C-1 downto 0) := (others=>'0');
    variable i      : integer := 0;
  begin
    assignSlv(i, vector, cfg.enabled);
    assignSlv(i, vector, cfg.rateSel);
    assignSlv(i, vector, cfg.destSel);
    assignSlv(i, vector, cfg.bsaEnabled);
    assignSlv(i, vector, cfg.bsaActiveSetup);
    assignSlv(i, vector, cfg.bsaActiveDelay);
    assignSlv(i, vector, cfg.bsaActiveWidth);
    assignSlv(i, vector, cfg.dmaEnabled);
    return vector;
  end function;

  function toSlv( cfg : EvrV2TriggerConfigType) return slv is
    variable vector : slv(EVRV2_TRIGGER_CONFIG_BITS_C-1 downto 0) := (others=>'0');
    variable i      : integer := 0;
  begin
    assignSlv(i, vector, cfg.enabled);
    assignSlv(i, vector, cfg.polarity);
    assignSlv(i, vector, cfg.delay);
    assignSlv(i, vector, cfg.width);
    assignSlv(i, vector, cfg.channel);
    assignSlv(i, vector, cfg.channels);
    assignSlv(i, vector, cfg.delayTap);
    assignSlv(i, vector, cfg.loadTap);
    return vector;
  end function;

  function toChannelConfig( vector : slv ) return EvrV2ChannelConfig is
    variable cfg : EvrV2ChannelConfig := EVRV2_CHANNEL_CONFIG_INIT_C;
    variable i   : integer                := vector'right;
  begin
    assignRecord(i, vector, cfg.enabled);
    assignRecord(i, vector, cfg.rateSel);
    assignRecord(i, vector, cfg.destSel);
    assignRecord(i, vector, cfg.bsaEnabled);
    assignRecord(i, vector, cfg.bsaActiveSetup);
    assignRecord(i, vector, cfg.bsaActiveDelay);
    assignRecord(i, vector, cfg.bsaActiveWidth);
    assignRecord(i, vector, cfg.dmaEnabled);
    return cfg;
  end function;
  
  function toTriggerConfig( vector : slv ) return EvrV2TriggerConfigType is
    variable cfg : EvrV2TriggerConfigType := EVRV2_TRIGGER_CONFIG_INIT_C;
    variable i      : integer := vector'right;
  begin
    assignRecord(i, vector, cfg.enabled);
    assignRecord(i, vector, cfg.polarity);
    assignRecord(i, vector, cfg.delay);
    assignRecord(i, vector, cfg.width);
    assignRecord(i, vector, cfg.channel);
    assignRecord(i, vector, cfg.channels);
    assignRecord(i, vector, cfg.delayTap);
    assignRecord(i, vector, cfg.loadTap);
    return cfg;
  end function;
end package body;
  
    
