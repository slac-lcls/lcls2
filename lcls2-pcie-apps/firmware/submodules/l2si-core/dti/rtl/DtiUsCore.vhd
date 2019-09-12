-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiUsCore.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-07-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
-- Note: Common-to-DtiApp interface defined here (see URL below)
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
use work.EventPkg.all;
use work.XpmPkg.all;
use work.DtiPkg.all;
use work.ArbiterPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;

entity DtiUsCore is
   generic (
      TPD_G               : time                := 1 ns;
      DEBUG_G             : boolean             := false );
   port (
     --  Core Interface
     sysClk          : in  sl;
     sysRst          : in  sl;
     clear           : in  sl := '0';
     update          : in  sl := '1';
     config          : in  DtiUsLinkConfigType;
     remLinkID       : in  slv(31 downto 0);
     status          : out DtiUsLinkStatusType;
     fullOut         : out slv(15 downto 0);
     msgDelay        : out slv( 6 downto 0);
     --  Ethernet control interface
     --ctlClk          : in  sl;
     --ctlRst          : in  sl;
     --ctlRxMaster     : in  AxiStreamMasterType;
     --ctlRxSlave      : out AxiStreamSlaveType;
     --ctlTxMaster     : out AxiStreamMasterType;
     --ctlTxSlave      : in  AxiStreamSlaveType;
     --  Timing interface
     timingClk       : in  sl;
     timingRst       : in  sl;
     timingHdr       : in  TimingHeaderType; -- delayed
     exptBus         : in  ExptBusType;      -- delayed
     timingHdrP      : in  TimingHeaderType; -- prompt
     triggerBus      : in  ExptBusType;      -- prompt
     --  DSLinks interface
     eventClk        : in  sl;
     eventRst        : in  sl;
     eventMasters    : out AxiStreamMasterArray(MaxDsLinks-1 downto 0);
     eventSlaves     : in  AxiStreamSlaveArray (MaxDsLinks-1 downto 0);
     dsFull          : in  slv(MaxDsLinks-1 downto 0);
     --  App Interface
     --  In from detector
     ibClk           : out sl;
     ibLinkUp        : in  sl;
     ibErrs          : in  slv(31 downto 0) := (others=>'0');
     ibFull          : in  sl;
     ibMaster        : in  AxiStreamMasterType;
     ibSlave         : out AxiStreamSlaveType;
     --  Out to detector
     obClk           : out sl;
     obTrigValid     : out sl;
     obTrig          : out XpmPartitionDataType);
     --obMaster        : out AxiStreamMasterType;
     --obSlave         : in  AxiStreamSlaveType );
end DtiUsCore;

architecture rtl of DtiUsCore is

  constant MAX_BIT : integer := bitSize(MaxDsLinks)-1;

  type StateType is (S_IDLE, S_EVHDR1, S_EVHDR2, S_EVHDR3, S_EVHDR4, S_FIRST_PAYLOAD, S_PAYLOAD,
                     S_DUMP, S_RSTFIFO);
  
  type RegType is record
    ena     : sl;
    state   : StateType;
    msg     : sl;
    hdr     : sl;
    dest    : slv(MAX_BIT downto 0);
    mask    : slv(MaxDsLinks-1 downto 0);
    ack     : slv(MaxDsLinks-1 downto 0);
    full    : slv(15 downto 0);
    hdrRd   : sl;
    master  : AxiStreamMasterType;
    slave   : AxiStreamSlaveType;
    rxFull  : slv(31 downto 0);
    rxFullO : slv(31 downto 0);
    ibRecv  : slv(47 downto 0);
    ibRecvO : slv(47 downto 0);
    ibEvt   : slv(31 downto 0);
    ibDump  : slv(31 downto 0);
  end record;
  
  constant REG_INIT_C : RegType := (
    ena     => '0',
    state   => S_IDLE,
    msg     => '0',
    hdr     => '0',
    dest    => toSlv(MaxDsLinks-1,MAX_BIT+1),
    mask    => (others=>'0'),
    ack     => (others=>'0'),
    full    => (others=>'0'),
    hdrRd   => '0',
    master  => AXI_STREAM_MASTER_INIT_C,
    slave   => AXI_STREAM_SLAVE_INIT_C,
    rxFull  => (others=>'0'),
    rxFullO => (others=>'0'),
    ibRecv  => (others=>'0'),
    ibRecvO => (others=>'0'),
    ibEvt   => (others=>'0'),
    ibDump  => (others=>'0') );
  
  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

  type TRegType is record
    full    : Slv16Array(3 downto 0);
    ninh    : slv(31 downto 0);
  end record;

  constant TREG_INIT_C : TRegType := (
    full    => (others=>(others=>'0')),
    ninh    => (others=>'0') );

  signal t   : TRegType := TREG_INIT_C;
  signal tin : TRegType;

  signal eventTag : slv(4 downto 0);
  signal pmsg     : sl;
  signal phdr     : sl;
  signal pdata    : XpmPartitionDataType;
  signal pdataV   : sl;
  signal eventHeader    : EventHeaderType;
  signal eventHeaderVec : slv(191 downto 0);
  signal eventHeaderV   : sl;
  
  signal configV, configSV, configTV : slv(DTI_US_LINK_CONFIG_BITS_C-1 downto 0);
  signal configS, configT : DtiUsLinkConfigType;

  signal ibEvtMaster  : AxiStreamMasterType;
  signal ibEvtSlave   : AxiStreamSlaveType;
  
  signal tMaster      : AxiStreamMasterType;
  signal tSlave       : AxiStreamSlaveType;

  signal urst    : sl;
  signal supdate : sl := '0';
  signal sclear  : sl;
  signal tclear  : sl;
  signal senable : sl;
  signal shdrOnly : sl;
  signal tfull   : slv(15 downto 0);
  signal wrFifoD : slv( 3 downto 0);
  signal rdFifoD : slv( 3 downto 0);
  signal rstFifo : sl;
  
  component ila_0
    port ( clk    : sl;
           probe0 : slv(255 downto 0) );
  end component;

  signal r_state : slv(3 downto 0);
  signal dbgl0r  : sl;
  signal eready  : slv(MaxDsLinks-1 downto 0);
  signal evalid  : slv(MaxDsLinks-1 downto 0);
  signal ieventM : AxiStreamMasterArray(MaxDsLinks-1 downto 0);
begin

  eventMasters <= ieventM;
  
  GEN_DEBUG : if DEBUG_G generate
    dbgl0r <= toPartitionWord(exptBus.message.partitionWord(0)).l0r;
    U_ILA : ila_0
      port map ( clk   => timingClk,
                 probe0(0) => timingHdr.strobe,
                 probe0(1) => dbgl0r,
                 probe0(17 downto 2) => tfull,
                 probe0(33 downto 18) => t.full(0),
                 probe0(49 downto 34) => t.full(1),
                 probe0(81 downto 50) => t.ninh,
                 probe0(129 downto 82) => exptBus.message.partitionWord(0),
                 probe0(255 downto 130) => (others=>'0') );

    r_state <= x"0" when r.state = S_IDLE else
               x"1" when r.state = S_EVHDR1 else
               x"2" when r.state = S_EVHDR2 else
               x"3" when r.state = S_EVHDR3 else
               x"4" when r.state = S_EVHDR4 else
               x"5" when r.state = S_FIRST_PAYLOAD else
               x"6" when r.state = S_PAYLOAD else
               x"7" when r.state = S_DUMP else
               x"8";

    GEN_EREADY : for i in 0 to MaxDsLinks-1 generate
      eready(i) <= eventSlaves(i).tReady;
      evalid(i) <= ieventM    (i).tValid;
    end generate;
    
    U_ILA_HDR : ila_0
      port map ( clk   => eventClk,
                 probe0(0) => pmsg,
                 probe0(1) => phdr,
                 probe0(2) => r.hdrRd,
                 probe0(3) => eventHeaderV,
                 probe0(19 downto 4) => eventHeader.l1t,
                 probe0(23 downto 20) => r_state,
                 probe0(24) => tMaster.tValid,
                 probe0(25) => tSlave .tReady,
                 probe0( 89 downto  26) => tMaster.tData(63 downto 0),
                 probe0( 97 downto  90) => resize(eready,8),
                 probe0(105 downto  98) => resize(evalid,8),
                 probe0(109 downto 106) => tMaster   .tDest(3 downto 0),
                 probe0(113 downto 110) => ieventM(0).tDest(3 downto 0),
                 probe0(255 downto 114) => (others=>'0') );
  end generate;
  
  obClk         <= timingClk;
  ibClk         <= eventClk;
  obTrig        <= pdata;
  obTrigValid   <= pdataV;
  fullOut       <= r.full;
  urst          <= clear and not r.ena;

  status.linkUp     <= ibLinkUp;
  status.remLinkID  <= remLinkID;
  status.rxErrs     <= ibErrs;
  status.rxFull     <= r.rxFull;
  status.ibRecv     <= r.ibRecv;
  status.ibEvt      <= r.ibEvt;
  status.ibDump     <= r.ibDump;
  status.rxInh      <= t.ninh;
  status.wrFifoD    <= wrFifoD;
  status.rdFifoD    <= rdFifoD;
  
  eventTag        <= ibMaster.tId(eventTag'range);

  U_Mux : entity work.DtiStreamDeMux
    generic map ( NUM_MASTERS_G  => MaxDsLinks,
                  TDEST_HIGH_G   => MAX_BIT,
                  TDEST_LOW_G    => 0 )
    port map ( sAxisMaster  => tMaster,
               sAxisSlave   => tSlave,
               sFlood       => r.msg,
               sFloodMask   => configS.fwdMask,
--               mAxisMasters => eventMasters,
               mAxisMasters => ieventM,
               mAxisSlaves  => eventSlaves,
               axisClk      => eventClk,
               axisRst      => eventRst );

  U_HdrCache : entity work.EventHeaderCache
    port map ( rst       => urst,
               --  Cache Input
               wrclk     => timingClk,
               --  configuration
               enable    => senable,
               partition => config.partition(2 downto 0),
               -- event input
               timing_prompt  => timingHdrP,
               expt_prompt    => triggerBus,
               timing_aligned => timingHdr,
               expt_aligned   => exptBus,
               -- trigger output
               pdata     => pdata,
               pdataV    => pdataV,
               cntL0     => status.obL0,
               cntL1A    => status.obL1A,
               cntL1R    => status.obL1R,
               cntWrFifo => wrFifoD,
               rstFifo   => rstFifo,
               msgDelay  => msgDelay,
               -- Cache Output
               rdclk     => eventClk,
               entag     => config.tagEnable,
               l0tag     => eventTag,
               advance   => r.hdrRd,
               pmsg      => pmsg,
               phdr      => phdr,
               cntRdFifo => rdFifoD,
               hdrOut    => eventHeader,
               valid     => eventHeaderV );

  eventHeaderVec <= toSlv(eventHeader);
  configV        <= toSlv(config);
  
  U_SyncConfig : entity work.SynchronizerVector
    generic map ( WIDTH_G => configV'length )
    port map ( clk     => eventClk,
               dataIn  => configV,
               dataOut => configSV );

  configS <= toUsLinkConfig(configSV);

  U_SyncConfigT : entity work.SynchronizerVector
    generic map ( WIDTH_G => configV'length )
    port map ( clk     => timingClk,
               dataIn  => configV,
               dataOut => configTV );

  configT <= toUsLinkConfig(configTV);

  U_ClearS : entity work.Synchronizer
    port map ( clk     => eventClk,
               dataIn  => clear,
               dataOut => sclear );
  
  U_ClearT : entity work.Synchronizer
    port map ( clk     => timingClk,
               dataIn  => clear,
               dataOut => tclear );
  
  U_FullT : entity work.SynchronizerVector
    generic map ( WIDTH_G => 16 )
    port map ( clk     => timingClk,
               dataIn  => r.full,
               dataOut => tfull );
  
  U_EnableS : entity work.Synchronizer
    port map ( clk     => timingClk,
               dataIn  => config.enable,
               dataOut => senable );
  
  U_HdrOnlyS : entity work.Synchronizer
    port map ( clk     => eventClk,
               dataIn  => config.hdrOnly,
               dataOut => shdrOnly );
  
  --
  --  For event traffic:
  --    Arbitrate through forwarding mask
  --    Add event header
  --
  comb : process ( r, ibMaster, tSlave, configS, eventRst, sclear, supdate, eventHeaderVec, eventHeaderV,
                   dsfull, ibFull, wrFifoD,
                   pmsg, phdr, rin, shdrOnly, rstFifo ) is
    variable v : RegType;
    variable selv : sl;
    variable fwd  : slv(MAX_BIT downto 0);
    variable isFull : sl;
  begin
    v := r;

    v.slave.tReady := '0';
    v.ena  := configS.enable;
    v.hdrRd          := '0';
    
    arbitrate(r.mask, r.dest, fwd, selv, v.ack);

    if tSlave.tReady='1' then
      v.master.tValid := '0';
    end if;

    case r.state is
      when S_IDLE =>
        if v.master.tValid='0' and eventHeaderV='1' then
          v.msg         := '0';
          if pmsg='1' then
            v.msg          := '1';
            v.state        := S_EVHDR1;
            v.ibEvt        := r.ibEvt+1;
          elsif phdr='1' then
            if selv = '0' then
              v.hdr          := '0';
              v.hdrRd        := '1';
              v.state        := S_DUMP;
              v.ibDump       := r.ibDump+1;
            else
              v.hdr          := shdrOnly;
              v.state        := S_EVHDR1;
              v.ibEvt        := r.ibEvt+1;
            end if;
          end if;
        end if;
      when S_EVHDR1 =>
        if v.master.tValid='0' then
          v.dest          := fwd;
          v.master.tValid := '1';
          v.master.tData(63 downto 0) := eventHeaderVec(63 downto 0);
          v.master.tKeep  := genTKeep(US_IB_CONFIG_C);
          v.master.tLast  := '0';
          if r.msg = '1' or r.hdr = '1' then
            v.master.tDest  := resize(fwd,r.master.tDest'length);
            ssiSetUserSof(US_IB_CONFIG_C, v.master, '1');
          else
            v.master.tDest  := resize(fwd,r.master.tDest'length);
            v.master.tId    := ibMaster.tId;
            v.master.tUser  := ibMaster.tUser;  -- SOF sometimes goes here
          end if;
          v.state         := S_EVHDR2;
        end if;
      when S_EVHDR2 =>
        if v.master.tValid='0' then
          v.master.tValid := '1';
          v.master.tData(63 downto 0) := eventHeaderVec(127 downto 64);
          v.master.tUser  := (others=>'0');
          v.state         := S_EVHDR3;
        end if;
      when S_EVHDR3 =>
        if v.master.tValid='0' then
          v.master.tValid := '1';
          v.master.tData(63 downto 0) := eventHeaderVec(191 downto 128);
          v.hdrRd         := '1';
          v.state         := S_EVHDR4;
        end if;
      when S_EVHDR4 =>
        if v.master.tValid='0' then
          v.master.tValid := '1';
          v.master.tData(63 downto 0) := configS.dataSrc & configS.dataType;
          if r.msg = '1' or r.hdr = '1' then
            v.hdr          := '0';
            v.master.tLast := '1';
            ssiSetUserEofe(US_IB_CONFIG_C, v.master, '0');
            v.state        := S_IDLE;
          else
            v.state        := S_FIRST_PAYLOAD;
          end if;
        end if;
      when S_FIRST_PAYLOAD =>
        if v.master.tValid='0' and ibMaster.tValid='1' then
          -- preserve tDest
          v.slave.tReady  := '1';
          v.master        := ibMaster;
          v.master.tUser  := (others=>'0');  -- already in EVHDR1
          v.master.tDest  := r.master.tDest;
          v.state         := S_PAYLOAD;
          if ibMaster.tLast='1' then -- maybe missing EOFE
            v.state  := S_IDLE;
          end if;
        end if;
      when S_PAYLOAD =>
        if v.master.tValid='0' and ibMaster.tValid='1' then
          -- preserve tDest
          v.slave.tReady  := '1';
          v.master        := ibMaster;
          v.master.tDest  := r.master.tDest;
          if ibMaster.tLast='1' then
            v.state  := S_IDLE;
          end if;
        end if;
      when S_DUMP =>
        if ibMaster.tValid='1' then
          v.slave.tReady := '1';
          if ibMaster.tLast='1' then
            v.state  := S_IDLE;
          end if;
        end if;
      when S_RSTFIFO =>
        --  terminate outgoing stream
        if r.master.tLast = '0' then
          if v.master.tValid = '0' then
            v.master.tValid := '1';
            v.master.tLast  := '1';
            ssiSetUserEofe(US_IB_CONFIG_C, v.master, '1');
          end if;
        --  drain the incoming stream (no guaranteed way of emptying)
        elsif ibMaster.tValid = '1' then
          v.slave.tReady := '1';
        else
          v.state := S_IDLE;
        end if;
      when others =>
        null;
    end case;

    if rstFifo = '1' then
      v.state := S_RSTFIFO;
    end if;
    
    v.full := (others=>'0');
    isFull := not selv;                -- downstream full
    if configS.fwdMode = '0' then      -- Round robin mode
      v.mask := configS.fwdMask;
      isFull := isFull or dsFull(conv_integer(fwd));
    else                               -- Next not full
      v.mask := configS.fwdMask and not dsFull;
    end if;
    isFull := isFull or ibFull;        -- detector full (direct)
    if wrFifoD > configS.afdepth then  -- detector pipeline full 
      isFull := '1';
    end if;
    
    if isFull='1' and r.ena='1' then
      v.full(conv_integer(configS.partition)) := isFull;
      v.rxFull  := r.rxFull+1;
    end if;

    if r.slave.tReady='1' and r.state/=S_IDLE then
      v.ibRecv := r.ibRecv+1;
    end if;

    if sclear = '1' then
      v.rxFull := (others=>'0');
      v.ibRecv := (others=>'0');
      v.ibEvt  := (others=>'0');
      v.ibDump := (others=>'0');
    end if;
    
    if eventRst = '1' then
      v := REG_INIT_C;
    end if;

    if supdate = '1' then
      v.rxFullO := r.rxFull;
      v.ibRecvO := r.ibRecv;
    end if;
    
    rin <= v;

    tMaster    <= r.master;
    ibSlave    <= rin.slave;
  end process;
  
  seq : process (eventClk) is
  begin
    if rising_edge(eventClk) then
      r <= rin;
    end if;
  end process;

  tcomb: process (t, timingRst, timingHdrP, tclear, tfull, configT, triggerBus) is
    variable v : TRegType;
    variable ip : integer;
  begin
    v := t;

    ip := conv_integer(configT.partition);

    if timingHdrP.strobe = '1' then
      if (toPartitionWord(triggerBus.message.partitionWord(ip)).l0r = '1' and
          t.full(t.full'left)(ip)='1') then
        v.ninh := t.ninh+1;
      end if;
      v.full := t.full(t.full'left-1 downto 0) & toSlv(0,16);
    else
      v.full(0) := t.full(0) or tfull;
    end if;
    
    if timingRst='1' then
      v := TREG_INIT_C;
    end if;
    
    tin <= v;
  end process;

  tseq : process (timingClk) is
  begin
    if rising_edge(timingClk) then
      t <= tin;
    end if;
  end process;
  
end rtl;
