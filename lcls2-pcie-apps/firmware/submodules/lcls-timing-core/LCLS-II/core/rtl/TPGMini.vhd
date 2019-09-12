-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TPGMini.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-11-09
-- Last update: 2018-04-20
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
use work.all;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use work.TPGPkg.all;
use work.StdRtlPkg.all;
use work.TimingPkg.all;

entity TPGMini is
  generic (
    TPD_G : time := 1 ns;
    NARRAYSBSA   : integer := 2
    );
  port (
    statusO : out TPGStatusType;
    configI : in  TPGConfigType;

    txClk      : in  sl;
    txRst      : in  sl;
    txRdy      : in  sl;
    txData     : out slv(15 downto 0);
    txDataK    : out slv(1 downto 0)
    );
end TPGMini;


-- Define architecture for top level module
architecture TPGMini of TPGMini is

  type RegType is record
    bsaComplete : slv(63 downto 0);
    countUpdate : slv( 1 downto 0);
  end record;
  constant REG_INIT_C : RegType := (
    bsaComplete => (others=>'0'),
    countUpdate => (others=>'0'));

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;
  
  signal frame : TimingMessageType := TIMING_MESSAGE_INIT_C;

  signal baseEnable  : sl;
  signal baseEnabled : slv(4 downto 0);

  signal pulseIdn : slv(63 downto 0);

  signal pulseIdWr : sl;

  signal acTSn      : slv(2 downto 0);
  signal acTSPhasen : slv(11 downto 0);

  constant ACRateWidth : integer := 8;
  constant ACRateDepth : integer := TPGPkg.ACRATEDEPTH;

  constant FixedRateWidth : integer := 20;
  constant FixedRateDepth : integer := TPGPkg.FIXEDRATEDEPTH;

  signal syncReset : sl;

  signal pllChanged : slv(31 downto 0) := (others => '0');
  signal count186M  : slv(31 downto 0);
  signal countSyncE : slv(31 downto 0);

  -- Interval counters
  signal countRst              : sl;
  signal intervalCnt           : slv(31 downto 0);
  signal countBRT, countBRTn   : slv(31 downto 0);
  signal countSeq              : Slv32Array(MAXSEQDEPTH-1 downto 0);

  -- Delay registers (for closing timing)
  signal status : TPGStatusType := TPG_STATUS_INIT_C;
  signal config : TPGConfigType;

  constant TPG_ID : integer := 0;
  
  signal streams   : TimingSerialArray(0 downto 0);
  signal streamIds : Slv4Array(0 downto 0);
  signal advance   : slv(0 downto 0);
  
  attribute use_dsp48                : string;
  attribute use_dsp48 of intervalCnt : signal is "yes";   
  attribute use_dsp48 of pllChanged  : signal is "yes";   
  attribute use_dsp48 of countSyncE  : signal is "yes";   
  
begin

  -- Dont know about these inputs yet
  frame.bcsFault <= (others => '0');

  frame.mpsLimit       <= (others => '0');
  frame.mpsClass       <= (others => (others => '0'));
  frame.mpsValid       <= '0';

  -- resources
  status.nbeamseq    <= toSlv(0, status.nbeamseq'length);
  status.nexptseq    <= toSlv(0, status.nexptseq'length);
  status.narraysbsa  <= toSlv(NARRAYSBSA, status.narraysbsa'length);
  status.seqaddrlen  <= toSlv(0, status.seqaddrlen'length);
  status.nallowseq   <= toSlv(0, status.nallowseq'length);

  status.pulseId    <= frame.pulseId;
  status.outOfSync  <= frame.syncStatus;
  status.bcsFault   <= frame.bcsFault;
  status.pllChanged <= pllChanged;
  status.count186M  <= count186M;
  status.countSyncE <= countSyncE;

  syncReset        <= '0';
  frame.resync     <= '0';
  frame.syncStatus <= '0';

  BaseEnableDivider : entity work.Divider
    generic map (
      TPD_G => TPD_G,
      Width => 16)
    port map (
      sysClk   => txClk,
      sysReset => syncReset,
      enable   => '1',
      clear    => '0',
      divisor  => config.baseDivisor,
      trigO    => baseEnable);

  frame.acRates <= (others=>'0');

  FixedDivider_loop : for i in 0 to FixedRateDepth-1 generate
    U_FixedDivider_1 : entity work.Divider
      generic map (
        TPD_G => TPD_G,
        Width => FixedRateWidth)
      port map (
        sysClk   => txClk,
        sysReset => txRst,
        enable   => baseEnable,
        clear    => '0',
        divisor  => config.FixedRateDivisors(i),
        trigO    => frame.fixedRates(i));
  end generate FixedDivider_loop;

  status.seqRdData <= (others=>(others=>'0'));
  status.seqState  <= (others=>SEQUENCER_STATE_INIT_C);
  countSeq         <= (others=>(others=>'0'));

  frame.control     <= (others=>(others=>'0'));
  frame.beamRequest <= (others=>'0');
  
  BsaLoop : for i in 0 to NARRAYSBSA-1 generate
    U_BsaControl : entity work.BsaControl
      generic map (TPD_G => TPD_G, ASYNC_REGCLK_G => false)
      port map (
        sysclk     => txClk,
        sysrst     => txRst,
        bsadef     => config.bsadefv(i),
        nToAvgOut  => status.bsaStatus(i)(15 downto 0),
        avgToWrOut => status.bsaStatus(i)(31 downto 16),
        txclk      => txClk,
        txrst      => txRst,
        enable     => baseEnabled(4),
        fixedRate  => frame.fixedRates,
        acRate     => frame.acRates,
        acTS       => frame.acTimeSlot,
        beamSeq    => frame.beamRequest,
        expSeq     => frame.control,
        bsaInit    => frame.bsaInit(i),
        bsaActive  => frame.bsaActive(i),
        bsaAvgDone => frame.bsaAvgDone(i),
        bsaDone    => frame.bsaDone(i));
  end generate BsaLoop;

  GEN_NULL_BSA: if NARRAYSBSA<64 generate
    status.bsaStatus(63 downto NARRAYSBSA) <= (others => (others => '0'));
    frame.bsaInit   (63 downto NARRAYSBSA) <= (others => '0');
    frame.bsaActive (63 downto NARRAYSBSA) <= (others => '0');
    frame.bsaAvgDone(63 downto NARRAYSBSA) <= (others => '0');
    frame.bsaDone   (63 downto NARRAYSBSA) <= (others => '0');
  end generate GEN_NULL_BSA;

  U_TSerializer : entity work.TimingSerializer
    generic map ( TPD_G => TPD_G, STREAMS_C => 1 )
    port map ( clk       => txClk,
               rst       => txRst,
               fiducial  => baseEnabled(0),
               streams   => streams,
               streamIds => streamIds,
               advance   => advance,
               data      => txData,
               dataK     => txDataK );
  
  U_TPSerializer : entity work.TPSerializer
    generic map ( TPD_G => TPD_G, Id => TPG_ID )
    port map ( txClk      => txClk,
               txRst      => txRst,
               fiducial   => baseEnable,
               msg        => frame,
               advance    => advance  (0),
               stream     => streams  (0),
               streamId   => streamIds(0) );

  status.irqFifoData  <= (others=>'0');
  status.irqFifoFull  <= '0';
  status.irqFifoEmpty <= '1';

  pulseIdn <= config.pulseId when pulseIdWr = '1' else
              frame.pulseId+1 when baseEnable = '1' else
              frame.pulseId;

  acTSn      <= "001";
  acTSPhasen <= (others => '0');

  countBRTn <= (others => '0') when countRst = '1' else
               countBRT+1 when baseEnable = '1' else
               countBRT;

  process (txClk, txRst, txRdy, config)
    variable outOfSyncd : sl;
    variable txRdyd     : sl;
  begin  -- process
    if rising_edge(txClk) then
      frame.pulseId         <= pulseIdn                                              after TPD_G;
      pulseIdWr             <= '0';
      frame.acTimeSlot      <= acTSn                                                 after TPD_G;
      frame.acTimeSlotPhase <= acTSPhasen                                            after TPD_G;
      baseEnabled           <= baseEnabled(baseEnabled'left-1 downto 0) & baseEnable after TPD_G;
      count186M             <= count186M+1;
      if (frame.syncStatus = '1' and outOfSyncd = '0') then
        countSyncE <= countSyncE+1;
      end if;
      if (txRdy /= txRdyd) then
        pllChanged <= pllChanged+1;
      end if;
      outOfSyncd := frame.syncStatus;
      txRdyd     := txRdy;
      countBRT   <= countBRTn;
      if allBits(intervalCnt, '0') then  -- need to execute this when
                                         -- intervalReg is changed
        countRst         <= '1';
        status.countBRT  <= countBRT;
        status.countSeq  <= countSeq;
        intervalCnt      <= config.interval;
      else
        countRst    <= '0';
        intervalCnt <= intervalCnt-1;
      end if;
      -- synchronous reset
      if txRst = '1' then
        frame.acTimeSlot      <= "001";
        frame.acTimeSlotPhase <= (others => '0');
        baseEnabled           <= (others => '0');
        count186M             <= (others => '0');
        countSyncE            <= (others => '0');
        outOfSyncd            := '1';
        countRst              <= '1';
        status.countTrig      <= (others => (others => '0'));
        status.countBRT       <= (others => '0');
        status.countSeq       <= (others => (others => '0'));
      end if;
      if config.intervalRst = '1' then
        intervalCnt <= (others => '0');
      end if;
      pulseIdWr <= config.pulseIdWrEn;
    end if;
  end process;

  comb: process(r, txRst, countRst, frame, baseEnable) is
    variable v : RegType;
  begin
    v := r;

    v.bsaComplete := (others=>'0');
    v.countUpdate := v.countUpdate(0) & '0';

    if baseEnable='1' then
      v.bsaComplete(frame.bsaDone'range) := frame.bsaDone;
    end if;

    if countRst='1' then
      v.countUpdate := "01";
    end if;

    if txRst='1' then
      v := REG_INIT_C;
    end if;
    
    rin <= v;

    status.bsaComplete <= r.bsaComplete;
    status.countUpdate <= r.countUpdate(1);
  end process;

  seq: process (txClk) is
  begin
    if rising_edge(txClk) then
      r <= rin after TPD_G;
    end if;
  end process;

  U_ClockTime : entity work.ClockTime
    generic map (
      TPD_G =>   TPD_G)
    port map (
      rst    => txRst,
      clkA   => txClk,
      wrEnA  => config.timeStampWrEn,
      wrData => config.timeStamp,
      rdData => status.timeStamp,
      clkB   => txClk,
      wrEnB  => baseEnable,
      dataO  => frame.timeStamp);

  statusO <= status;
  config  <= configI;
  
end TPGMini;
