-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmApp.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-09-06
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: XpmApp's Top Level
-- 
-- Note: Common-to-XpmApp interface defined here (see URL below)
--       https://confluence.slac.stanford.edu/x/rLyMCw
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
use work.TimingExtnPkg.all;
use work.TimingPkg.all;
--use work.AmcCarrierPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmApp is
   generic (
      TPD_G               : time                := 1 ns;
      NDsLinks            : integer             := 7;
      NBpLinks            : integer             := 14 );
   port (
      -----------------------
      -- XpmApp Ports --
      -----------------------
      regclk            : in  sl;
      update            : in  slv(NPartitions-1 downto 0);
      config            : in  XpmConfigType;
      status            : out XpmStatusType;
      -- AMC's DS Ports
      dsLinkStatus      : in  XpmLinkStatusArray(NDsLinks-1 downto 0);
      dsRxData          : in  Slv16Array        (NDsLinks-1 downto 0);
      dsRxDataK         : in  Slv2Array         (NDsLinks-1 downto 0);
      dsTxData          : out Slv16Array        (NDsLinks-1 downto 0);
      dsTxDataK         : out Slv2Array         (NDsLinks-1 downto 0);
      dsRxErr           : in  slv               (NDsLinks-1 downto 0);
      dsRxClk           : in  slv               (NDsLinks-1 downto 0);
      dsRxRst           : in  slv               (NDsLinks-1 downto 0);
      --  BP DS Ports
      bpTxData          : out slv(15 downto 0);
      bpTxDataK         : out slv( 1 downto 0);
      bpStatus          : in  XpmBpLinkStatusArray(NBpLinks   downto 0);
      bpRxLinkFull      : in  Slv16Array          (NBpLinks-1 downto 0);
      -- Timing Interface (timingClk domain) 
      timingClk         : in  sl;
      timingRst         : in  sl;
      timingIn          : in  TimingRxType;
      timingFbClk       : in  sl;
      timingFbRst       : in  sl;
      timingFbId        : in  slv(31 downto 0);
      timingFb          : out TimingPhyType );
end XpmApp;

architecture top_level_app of XpmApp is

  type LinkFullArray  is array (natural range<>) of slv(26 downto 0);
  type LinkL1InpArray is array (natural range<>) of XpmL1InputArray(NDsLinks-1 downto 0);
    
  type StateType is (INIT_S, PADDR_S, EWORD_S, EOS_S);
  type RegType is record
    full       : LinkFullArray (NPartitions-1 downto 0);
    l1input    : LinkL1InpArray(NPartitions-1 downto 0);
    fiducial   : sl;
    source     : sl;
    paddr      : slv(PADDR_LEN-1 downto 0);
    taddr      : slv(PADDR_LEN-1 downto 0);
    streams    : TimingSerialArray(1 downto 0);
    advance    : slv( 1 downto 0);
    state      : StateType;
    aword      : integer range 0 to paddr'left/16;
    eword      : integer range 0 to (NTagBytes+1)/2;
    ipart      : integer range 0 to 2*NPartitions-1;
  end record;
  constant REG_INIT_C : RegType := (
    full       => (others=>(others=>'0')),
    l1input    => (others=>(others=>XPM_L1_INPUT_INIT_C)),
    fiducial   => '0',
    source     => '1',
    paddr      => (others=>'1'),
    taddr      => (others=>'1'),
    streams    => (others=>TIMING_SERIAL_INIT_C),
    advance    => (others=>'0'),
    state      => INIT_S,
    aword      => 0,
    eword      => 0,
    ipart      => 0 );

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  

  --  input data from sensor links
  type L1InputArray is array (natural range<>) of XpmL1InputArray(NPartitions-1 downto 0);
  type FullArray    is array (natural range<>) of slv            (NPartitions-1 downto 0);

  signal l1Input        : L1InputArray(NDsLinks-1 downto 0);
  signal isXpm          : slv         (NDsLinks-1 downto 0);
  signal dsFull         : FullArray   (NDsLinks-1 downto 0);
  signal dsRxRcvs       : Slv32Array  (NDsLinks-1 downto 0);
  signal dsId           : Slv32Array  (NDsLinks-1 downto 0);
  signal bpRxLinkFullS  : Slv16Array        (NBpLinks-1 downto 0);
  
  --  Serialized data to sensor links
  signal txData         : slv(15 downto 0);
  signal txDataK        : slv( 1 downto 0);

  constant STREAMS_C : integer := 2;
  signal streams   : TimingSerialArray(STREAMS_C-1 downto 0);
  signal streamIds : Slv4Array        (STREAMS_C-1 downto 0) := (x"1",x"0");
  signal advance   : slv              (STREAMS_C-1 downto 0);
  signal fiducial  : sl;
  signal sof, eof, crcErr : sl;
  signal pmaster     : slv(NPartitions-1 downto 0);
  signal expWord     : Slv48Array(NPartitions-1 downto 0);

begin

  linkstatp: process (bpStatus, dsLinkStatus, dsRxRcvs, isXpm, dsId) is
    variable linkStat : XpmLinkStatusType;
  begin
    for i in 0 to NDsLinks-1 loop
      linkStat           := dsLinkStatus(i);
      linkStat.rxRcvCnts := dsRxRcvs(i);
      linkStat.rxIsXpm   := isXpm   (i);
      linkStat.rxId      := dsId    (i);
      status.dsLink(i)   <= linkStat;
    end loop;
    status.bpLink <= bpStatus;
  end process;

  GEN_SYNCBP : for i in 0 to NBpLinks-1 generate
    U_SyncFull : entity work.SynchronizerVector
      generic map ( WIDTH_G => 16 )
      port map ( clk     => timingClk,
                 dataIn  => bpRxLinkFull(i),
                 dataOut => bpRxLinkFullS(i) );
  end generate;
  
  U_SyncPaddr : entity work.SynchronizerVector
    generic map ( WIDTH_G => status.paddr'length )
    port map ( clk     => regclk,
               dataIn  => r.paddr,
               dataOut => status.paddr );
  
  U_TimingFb : entity work.XpmTimingFb
    port map ( clk        => timingFbClk,
               rst        => timingFbRst,
               id         => timingFbId,
               l1input    => (others=>XPM_L1_INPUT_INIT_C),
               full       => (others=>'0'),
               phy        => timingFb );
               
  GEN_DSLINK: for i in 0 to NDsLinks-1 generate
    U_TxLink : entity work.XpmTxLink
      generic map ( ADDR => i )
      port map ( clk             => timingClk,
                 rst             => timingRst,
                 config          => config.dsLink(i),
                 isXpm           => isXpm(i),
                 streams         => r.streams,
                 streamIds       => streamIds,
                 paddr           => r.paddr,
                 advance_i       => r.advance,
                 fiducial        => r.fiducial,
                 sof             => sof,
                 eof             => eof,
                 crcErr          => crcErr,
                 txData          => dsTxData (i),
                 txDataK         => dsTxDataK(i) );
    U_RxLink : entity work.XpmRxLink
      port map ( clk             => timingClk,
                 rst             => timingRst,
                 config          => config.dsLink(i),
                 rxData          => dsRxData (i),
                 rxDataK         => dsRxDataK(i),
                 rxErr           => dsRxErr  (i),
                 rxClk           => dsRxClk  (i),
                 rxRst           => dsRxRst  (i),
                 isXpm           => isXpm    (i),
                 id              => dsId     (i),
                 rxRcvs          => dsRxRcvs (i),
                 full            => dsFull   (i),
                 l1Input         => l1Input  (i) );
  end generate GEN_DSLINK;

  U_BpTx : entity work.XpmTxLink
    generic map ( ADDR    => 15,
                  DEBUG_G => true )
    port map ( clk             => timingClk,
               rst             => timingRst,
               config          => config.bpLink(0),
               isXpm           => '1',
               streams         => r.streams,
               streamIds       => streamIds,
               paddr           => r.paddr,
               advance_i       => r.advance,
               fiducial        => r.fiducial,
               sof             => sof,
               eof             => eof,
               crcErr          => crcErr,
               txData          => bpTxData ,
               txDataK         => bpTxDataK );

  U_Deserializer : entity work.TimingDeserializer
    generic map ( STREAMS_C => 2 )
    port map ( clk       => timingClk,
               rst       => timingRst,
               fiducial  => fiducial,
               streams   => streams,
               streamIds => streamIds,
               advance   => advance,
               data      => timingIn,
               sof       => sof,
               eof       => eof,
               crcErr    => crcErr );

  GEN_PART : for i in 0 to NPartitions-1 generate

    U_Master : entity work.XpmAppMaster
      generic map ( NDsLinks   => NDsLinks,
                    DEBUG_G    => ite(i>0, false, true) )
      port map ( regclk        => regclk,
                 update        => update          (i),
                 config        => config.partition(i),
                 status        => status.partition(i),
                 timingClk     => timingClk,
                 timingRst     => timingRst,
                 streams       => streams,
                 streamIds     => streamIds,
                 advance       => advance,
                 fiducial      => fiducial,
                 sof           => sof,
                 eof           => eof,
                 crcErr        => crcErr,
                 full          => r.full          (i),
                 l1Input       => r.l1input       (i),
                 result        => expWord         (i) );

    U_SyncMaster : entity work.Synchronizer
      port map ( clk     => timingClk,
                 dataIn  => config.partition(i).l0Select.enabled,
                 dataOut => pmaster(i) );
  end generate;

  comb : process ( r, timingRst, dsFull, bpRxLinkFullS, l1Input, fiducial, advance, expWord, streams, pmaster ) is
    variable v : RegType;
  begin
    v := r;
    v.streams    := streams;
    v.streams(0).ready := '1';
    v.streams(1).ready := '1';
    v.advance    := advance;
    v.fiducial   := fiducial;
    
    --  test if we are the top of the hierarchy
    if streams(1).ready='1' then
      v.source        := '0';
    end if;

    if (advance(0)='0' and r.advance(0)='1') then
      v.streams(0).ready := '0';
    end if;
    
    case r.state is
      when INIT_S =>
        v.aword := 0;
        if (r.source='0' and advance(1)='1') then
          v.taddr := v.streams(1).data & r.taddr(r.paddr'left downto r.paddr'left-15);
          v.aword           := r.aword+1;
          v.state           := PADDR_S;
        elsif (r.source='1' and advance(0)='0' and r.advance(0)='1') then
          v.advance(1)      := '1';
          v.streams(1).data := r.paddr(15 downto 0);
          v.aword           := r.aword+1;
          v.state           := PADDR_S;
        end if;
      when PADDR_S =>
        if r.source='1' then
          v.advance(1)      := '1';
          v.streams(1).data := r.paddr(r.aword*16+15 downto r.aword*16);
        else
          v.taddr := v.streams(1).data & r.taddr(r.paddr'left downto r.paddr'left-15);
        end if;
        if (r.aword=r.paddr'left/16) then
          v.ipart := 0;
          v.eword := 0;
          v.state := EWORD_S;
        else
          v.aword := r.aword+1;
        end if;
      when EWORD_S =>
        v.eword      := r.eword+1;
        v.advance(1) := '1';
        if r.source='1' or pmaster(r.ipart)='1' then
          v.streams(1).data := expWord(r.ipart)(r.eword*16+15 downto r.eword*16);
        end if;
        if (r.eword=(NTagBytes+1)/2) then
          if (r.ipart=NPartitions-1) then
            v.state := EOS_S;
          else
            v.ipart := r.ipart+1;
            v.eword := 0;
          end if;
        end if;
      when EOS_S =>
        v.streams(1).ready := '0';
        v.paddr := r.taddr;
        v.aword := 0;
        v.state := INIT_S;
      when others => NULL;
    end case;

    for i in 0 to NPartitions-1 loop
      for j in 0 to NDsLinks-1 loop
        v.full   (i)(j) := dsFull (j)(i);
        v.l1input(i)(j) := l1Input(j)(i);
      end loop;
      for j in 0 to NBpLinks-1 loop
        v.full   (i)(j+16) := bpRxLinkFullS(j)(i);
      end loop;
    end loop;

    if timingRst='1' then
      v := REG_INIT_C;
    end if;
    
    rin <= v;
  end process;

  seq : process ( timingClk) is
  begin
    if rising_edge(timingClk) then
      r <= rin;
    end if;
  end process;
  
end top_level_app;
