------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiUsGbE.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-05-15
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
--   Application interface to JungFrau.  Uses 10GbE.  Trigger is external TTL
--   (L0 only?). Control register access is external 1GbE link.
--
--   Intercept out-bound messages as register transactions for 10GbE core.
--   Use simulation embedding: ADDR(31:1) & RNW & DATA(31:0).
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
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.XpmPkg.all;
use work.DtiPkg.all;
use work.DtiSimPkg.all;
use work.SsiPkg.all;

entity DtiUsGbE is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : slv(7 downto 0)     := (others=>'0');
      ENABLE_TAG_G        : boolean             := false ;
      DEBUG_G             : boolean             := false );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     status          : out DtiUsAppStatusType;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     -- 
     sysClk125       : in  sl;
     sysRst125       : in  sl;
     sysClk62        : in  sl;
     sysRst62        : in  sl;
     --
     ibClk           : in  sl;
     ibRst           : in  sl;
     ibMaster        : out AxiStreamMasterType;
     ibSlave         : in  AxiStreamSlaveType;
     linkUp          : out sl;
     rxErr           : out sl;
     --
     obClk           : in  sl;
     obRst           : in  sl;
     obTrig          : in  XpmPartitionDataType;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiUsGbE;

architecture top_level_app of DtiUsGbE is

  type StateType is (S_IDLE, S_READ_HDR, S_READ_BID, S_PAYLOAD, S_DUMP, S_SKIP);
  type AxilStateType is (S_AXIL_IDLE, S_WR_ACK_WAIT, S_RD_ACK_WAIT, S_IB_ACK_WAIT);

  type ARegType is record
    --  Register state
    astate     : AxilStateType;
    rMaster    : AxiLiteReadMasterType;
    wMaster    : AxiLiteWriteMasterType;
    regmaster  : AxiStreamMasterType;
    regslave   : AxiStreamSlaveType;
    status     : DtiUsAppStatusType;
  end record;
  
  constant AREG_INIT_C : ARegType := (
    astate     => S_AXIL_IDLE,
    rMaster    => AXI_LITE_READ_MASTER_INIT_C,
    wMaster    => AXI_LITE_WRITE_MASTER_INIT_C,
    regmaster  => AXI_STREAM_MASTER_INIT_C,
    regslave   => AXI_STREAM_SLAVE_INIT_C,
    status     => DTI_US_APP_STATUS_INIT_C );

  constant MACADDR : slv(47 downto 0) := (x"0203005644" & ID_G);
  
  constant TRGEND : integer := 5;
  constant TRGDATA : Slv64Array(0 to TRGEND) := (
    (MACADDR(39 downto 32) & MACADDR(47 downto 40) &
     x"00407f5e0001"),  -- dst = 01005e7f4000 (239.255.48.0)
    (x"00450008" & MACADDR(7 downto 0) &
     MACADDR(15 downto 8) & MACADDR(23 downto 16) &
     MACADDR(31 downto 24)),  -- src = 020300564400; IP v4
    x"1101000000002200",  -- IP header (UDP)
    x"ffef0100a8c00000",
    x"0600002000000030",
    x"0504030201000000" );
  
  type RegType is record
    --  Event state
    state      : StateType;
    evtmaster  : AxiStreamMasterType;
    evtslave   : AxiStreamSlaveType;
    triggered  : sl;
    firstframe : sl;
    packetnum  : slv( 7 downto 0);
    framenum   : slv(23 downto 0);
    ibcount    : slv( 2 downto 0);
    -- Outbound triggers
    l0         : sl;
    tbusy      : sl;
    trgmaster  : AxiStreamMasterType;
    trgword    : integer range 0 to TRGEND;
    trgcount   : slv(31 downto 0);
  end record;
  
  constant REG_INIT_C : RegType := (
    state      => S_IDLE,
    evtmaster  => AXI_STREAM_MASTER_INIT_C,
    evtslave   => AXI_STREAM_SLAVE_INIT_C,
    triggered  => '0',
    firstframe => '1',
    packetnum  => (others=>'0'),
    framenum   => (others=>'0'),
    ibcount    => (others=>'0'),
    l0         => '0',
    tbusy      => '0',
    trgmaster  => AXI_STREAM_MASTER_INIT_C,
    trgword    => 0,
    trgcount   => (others=>'0') );
  
  signal a   : ARegType := AREG_INIT_C;
  signal ain : ARegType;

  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

  constant FIRST_PACKET_C : slv(7 downto 0) := (others=>'1');
  constant HDR_WORDS_C    : integer         := 6;
    
  constant AXIS_CONFIG_C : AxiStreamConfigType := (
    TSTRB_EN_C    => false,
    TDATA_BYTES_C => 8,
    TID_BITS_C    => 5,
    TDEST_BITS_C  => 1,
    TKEEP_MODE_C  => TKEEP_COMP_C,
    TUSER_BITS_C  => 2,
    TUSER_MODE_C  => TUSER_FIRST_LAST_C );
  
  signal amcIbMaster : AxiStreamMasterType;
  signal amcIbSlave  : AxiStreamSlaveType;

  signal l0S, l1S, l1aS : sl;

  component ila_0
    port ( clk    : sl;
           probe0 : slv(255 downto 0) );
  end component;

  signal iobSlave  : AxiStreamSlaveType;
  signal iibMaster : AxiStreamMasterType;
  signal r_state   : slv(1 downto 0);

  signal regReadSlave   : AxiLiteReadSlaveType;
  signal regWriteSlave  : AxiLiteWriteSlaveType;
  signal ibSaxisMasters : AxiStreamMasterArray(1 downto 0);
  signal ibSaxisSlaves  : AxiStreamSlaveArray (1 downto 0);
  signal dmaIbMaster    : AxiStreamMasterType;
  signal dmaIbSlave     : AxiStreamSlaveType;
  signal dmaObMaster    : AxiStreamMasterType;
  signal dmaObSlave     : AxiStreamSlaveType;

  signal regMaster      : AxiStreamMasterType;
  signal regSlave       : AxiStreamSlaveType;

begin

  obSlave <= iobSlave;

  U_GigEth : entity work.GigEthGthUltraScale
    generic map (
      EN_AXI_REG_G => true )
    port map (
      localMac            => MACADDR,
      -- Streaming DMA Interface 
      dmaClk              => amcClk,
      dmaRst              => amcRst,
      dmaIbMaster         => dmaIbMaster,
      dmaIbSlave          => dmaIbSlave,
      dmaObMaster         => dmaObMaster,
      dmaObSlave          => dmaObSlave,
      -- Slave AXI-Lite Interface 
      axiLiteClk          => obClk,
      axiLiteRst          => obRst,
      axiLiteReadMaster   => a.rMaster,
      axiLiteReadSlave    => regReadSlave,
      axiLiteWriteMaster  => a.wMaster,
      axiLiteWriteSlave   => regWriteSlave,
      -- PHY + MAC signals
      sysClk62            => sysClk62,
      sysClk125           => sysClk125,
      sysRst125           => sysRst125,
      extRst              => '0',
      phyReady            => linkUp,
      -- MGT Ports
      gtTxP               => amcTxP,
      gtTxN               => amcTxN,
      gtRxP               => amcRxP,
      gtRxN               => amcRxN );

  GEN_DEBUG : if DEBUG_G generate
    U_ILA : ila_0
      port map ( clk    => amcClk,
                 probe0(0) => dmaIbMaster.tValid,
                 probe0(1) => dmaIbMaster.tLast,
                 probe0( 3 downto 2) => dmaIbMaster.tUser( 1 downto 0),
                 probe0(67 downto 4) => dmaIbMaster.tData(63 downto 0),
                 probe0(68)          => dmaIbSlave .tReady,
                 --
                 probe0(69)             => r.evtmaster.tValid,
                 probe0(70)             => r.evtmaster.tLast,
                 probe0(133 downto  71) => r.evtmaster.tData(63 downto 0),
                 probe0(134)            => '0',
                 probe0(135)            => '0',
                 probe0(136)            => amcRst,
                 --
                 probe0(137)            => dmaObMaster.tValid,
                 probe0(138)            => dmaObMaster.tLast,
                 probe0(202 downto 139) => dmaObMaster.tData,
                 probe0(255 downto 203) => (others=>'0') );
  end generate;

  ibMaster <= iibMaster;

  U_IbMux  : entity work.AxiStreamMux
    generic map ( NUM_SLAVES_G => 2 )
    port map ( axisClk  => ibClk,
               axisRst  => ibRst,
               sAxisMasters => ibSaxisMasters,
               sAxisSlaves  => ibSaxisSlaves,
               mAxisMaster  => ibMaster,
               mAxisSlave   => ibSlave );
  
  U_IbFifo : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => AXIS_CONFIG_C,
                  MASTER_AXI_CONFIG_G => US_IB_CONFIG_C )
    port map ( sAxisClk    => amcClk,
               sAxisRst    => amcRst,
               sAxisMaster => amcIbMaster,
               sAxisSlave  => amcIbSlave,
               mAxisClk    => ibClk,
               mAxisRst    => ibRst,
               mAxisMaster => ibSaxisMasters(0),
               mAxisSlave  => ibSaxisSlaves (0) );

  U_ObFifo : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
                  MASTER_AXI_CONFIG_G => AXIS_CONFIG_C )
    port map ( sAxisClk    => obClk,
               sAxisRst    => obRst,
               sAxisMaster => regMaster,
               sAxisSlave  => regSlave,
               mAxisClk    => ibClk,
               mAxisRst    => ibRst,
               mAxisMaster => ibSaxisMasters(1),
               mAxisSlave  => ibSaxisSlaves (1) );

  U_SyncL0 : entity work.SynchronizerOneShot
    port map ( clk     => amcClk,
               rst     => fifoRst,
               dataIn  => obTrig.l0a,
               dataOut => l0S );
  
  U_SyncL1 : entity work.SynchronizerOneShot
    port map ( clk     => amcClk,
               rst     => fifoRst,
               dataIn  => obTrig.l1e,
               dataOut => l1S );
  
  U_SyncL1A : entity work.Synchronizer
    port map ( clk     => amcClk,
               rst     => fifoRst,
               dataIn  => obTrig.l1a,
               dataOut => l1aS );
  
  --
  --  Parse amcOb stream for register transactions or obTrig
  --
  acomb : process ( fifoRst, a, obMaster, regReadSlave, regWriteSlave, regSlave ) is
    variable v   : ARegType;
    variable reg : RegTransactionType;
  begin
    v := a;
    v.regmaster.tDest(0) := '1';
    v.regslave .tReady   := '0';

    reg := toRegTransType(obMaster.tData(63 downto 0));

    case a.astate is
      when S_AXIL_IDLE =>
        if obMaster.tValid = '1' then
          v.regslave.tReady   := '1';
          if reg.rnw = '0' then  -- write
            v.wMaster.awaddr  := reg.address & '0';
            v.wMaster.wdata   := reg.data;
            v.wMaster.awprot  := (others=>'0');
            v.wMaster.wstrb   := (others=>'1');
            v.wMaster.awvalid := '1';
            v.wMaster.wvalid  := '1';
            v.wMaster.bready  := '1';
            v.astate          := S_WR_ACK_WAIT;
          else -- read
            v.rMaster.araddr  := reg.address & '0';
            v.rMaster.arprot  := (others=>'0');
            v.rMaster.arvalid := '1';
            v.rMaster.rready  := '1';
            v.astate          := S_RD_ACK_WAIT;
          end if;
        else
          v.regmaster.tValid := '0';
        end if;
      when S_WR_ACK_WAIT =>
        if regWriteSlave.awready = '1' then
          v.wMaster.awvalid := '0';
        end if;
        if regWriteSlave.wready = '1' then
          v.wMaster.wvalid := '0';
        end if;
        if regWriteSlave.bvalid = '1' then
          v.wMaster.bready  := '0';
          v.regmaster.tValid   := '1';
          v.regmaster.tLast    := '1';
          v.regmaster.tDest(0) := '1';
          ssiSetUserSof (AXIS_CONFIG_C,v.regmaster,'1');
          ssiSetUserEofe(AXIS_CONFIG_C,v.regmaster,'0');
          reg.data    := a.wMaster.wdata;
          reg.address := a.wMaster.awaddr(31 downto 1);
          reg.rnw     := '0';
          v.regmaster.tData(63 downto 0) := toSlv(reg);
          if regSlave.tReady = '1' then
            v.astate := S_AXIL_IDLE;
          else
            v.astate := S_IB_ACK_WAIT;
          end if;
        end if;
      when S_RD_ACK_WAIT =>
        if regReadSlave.arready = '1' then
          v.rMaster.arvalid := '0';
        end if;
        if regReadSlave.rvalid = '1' then
          v.rMaster.rready := '0';
          v.regmaster.tValid   := '1';
          v.regmaster.tLast    := '1';
          ssiSetUserSof (AXIS_CONFIG_C,v.regmaster,'1');
          ssiSetUserEofe(AXIS_CONFIG_C,v.regmaster,'0');
          reg.data    := regReadSlave.rdata;
          reg.address := a.rMaster.araddr(31 downto 1);
          reg.rnw     := '1';
          v.regmaster.tData(63 downto 0) := toSlv(reg);
          if regSlave.tReady = '1' then
            v.astate := S_AXIL_IDLE;
          else
            v.astate := S_IB_ACK_WAIT;
          end if;
        end if;
      when S_IB_ACK_WAIT =>
        if regSlave.tReady = '1' then
          v.astate := S_AXIL_IDLE;
        end if;
      when others => null;
    end case;            
    
    if obMaster.tValid = '1' and a.regSlave.tReady = '1' then
      v.status.obReceived := a.status.obReceived+1;
    end if;

    if ((a.rMaster.rready = '1' and regReadSlave.rvalid = '1') or
        (a.wMaster.wvalid = '1' and regWriteSlave.wready = '1')) then
      v.status.obSent := a.status.obSent+1;
    end if;

    if fifoRst = '1' then
      v := AREG_INIT_C;
    end if;

    ain <= v;

    regMaster  <= a.regmaster;
    obSlave    <= v.regslave;
    status     <= a.status;

  end process;

  aseq : process (obClk) is
  begin
    if rising_edge(obClk) then
      a <= ain;
    end if;
  end process;
  
  
  comb : process ( fifoRst, r, amcIbSlave, dmaIbMaster, dmaObSlave,
                   l0S, l1S, l1aS ) is
    variable v   : RegType;
    variable reg : RegTransactionType;
  begin
    v := r;
    v.evtmaster.tDest(0) := '0';
    v.evtslave .tReady   := '0';

    if l1S = '1' then
      v.triggered := '1';
    end if;
    
    --
    --  assemble packets into datagrams
    --
    if amcIbSlave.tReady = '1' then
      v.evtmaster.tValid := '0';
    end if;
    
    case r.state is
      when S_IDLE =>
        if dmaIbMaster.tValid = '1' then
          v.evtslave.tReady := '1';
          if r.triggered = '1' then
            v.triggered    := '0';
            v.ibcount      := (others=>'0');
            v.state        := S_READ_HDR;
          else
            v.state        := S_DUMP;
          end if;
        end if;
      when S_READ_HDR =>
        if dmaIbMaster.tValid = '1' then
          v.evtslave.tReady := '1';
          v.ibcount         := r.ibcount+1;
          if r.ibcount = HDR_WORDS_C-1 then
            v.firstframe := '0';
            v.packetnum  := dmaIbMaster.tData(39 downto 32);
            v.framenum   := dmaIbMaster.tData(63 downto 40);
            --  Validate packetnum and framenum
            if r.firstframe = '1' or v.framenum = r.framenum then
              if v.packetnum = r.packetnum-1 then
                v.state      := S_READ_BID;
              else 
                v.state     := S_DUMP; -- bad packet number
              end if;
            else
              v.state := S_SKIP; -- bad frame number
            end if;
          end if;
        end if;
      when S_READ_BID =>
        if dmaIbMaster.tValid = '1' then
          v.state := S_PAYLOAD;
        end if;
      when S_PAYLOAD =>
        if dmaIbMaster.tValid = '1' then
          if v.evtmaster.tValid = '0' then
            v.evtslave.tReady  := '1';
            v.evtmaster.tValid := '1';
            v.evtmaster.tData(63 downto 0) := dmaIbMaster.tData(63 downto 0);
            if r.packetnum = FIRST_PACKET_C then
              ssiSetUserSof( AXIS_CONFIG_C, v.evtmaster, '1');
            end if;
            if dmaIbMaster.tLast = '1' then
              if r.packetnum = 0 then
                ssiSetUserEofe(AXIS_CONFIG_C, v.evtmaster, '0');
              end if;
              v.state := S_IDLE;
            end if;
          end if;
        end if;
      when S_DUMP =>
        if v.evtmaster.tValid = '0' then
          v.evtmaster.tValid := '1';
          v.evtmaster.tLast  := '1';
          if r.packetnum = FIRST_PACKET_C then
            ssiSetUserSof( AXIS_CONFIG_C, v.evtmaster, '1');
          end if;
          ssiSetUserEofe(AXIS_CONFIG_C, v.evtmaster, '1');
          v.framenum         := r.framenum+1;
          v.packetnum        := (others=>'0');
          v.state            := S_SKIP;
        end if;
      when S_SKIP =>
        if dmaIbMaster.tValid = '1' then
          v.evtslave.tReady := '1';
          if dmaIbMaster.tLast = '1' then
            v.state := S_IDLE;
          end if;
        end if;
      when others =>
        null;
    end case;

    if l0S = '1' then
      v.l0       := '1';
      v.trgcount := r.trgcount+1;
    end if;
    
    if dmaObSlave.tReady = '1' then
      v.trgmaster.tValid := '0';
    end if;

    if r.tbusy = '0' then
      if r.l0 = '1' then
        v.trgword := 0;
        v.l0      := '0';
        v.tbusy   := '1';
      end if;
    else
      if v.trgmaster.tValid='0' then
        v.trgmaster.tValid := '1';
        v.trgmaster.tData(63 downto 0) := TRGDATA(r.trgword);
        if r.trgword = TRGEND then
          v.trgmaster.tData(63 downto 32) := r.trgcount;
          v.trgmaster.tLast  := '1';
          v.tbusy := '0';
        else
          v.trgword := r.trgword+1;
          v.trgmaster.tLast  := '0';
        end if;
      end if;
    end if;
    
    rin <= v;

    amcIbMaster <= r.evtmaster;
    dmaIbSlave  <= v.evtslave;
    dmaObMaster <= r.trgmaster;
    
  end process;
            
  seq : process (amcClk) is
  begin
    if rising_edge(amcClk) then
      r <= rin;
    end if;
  end process;
  
end top_level_app;
