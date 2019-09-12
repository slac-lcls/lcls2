------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiUsPgp3.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-08-19
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
use work.Pgp3Pkg.all;
use work.SsiPkg.all;

entity DtiUsPgp3 is
   generic (
      TPD_G               : time                := 1 ns;
      ID_G                : slv(7 downto 0)     := (others=>'0');
      ENABLE_TAG_G        : boolean             := false ;
      EN_AXIL_G           : boolean             := false ;
      DEBUG_G             : boolean             := false ;
      AXIL_BASE_ADDR_G    : slv(31 downto 0)    := (others=>'0') );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     status          : out DtiUsAppStatusType;
     amcRxP          : in  sl;
     amcRxN          : in  sl;
     amcTxP          : out sl;
     amcTxN          : out sl;
     fifoRst         : in  sl;
     -- Quad PLL Ports
     qplllock        : in  slv(1 downto 0);
     qplloutclk      : in  slv(1 downto 0);
     qplloutrefclk   : in  slv(1 downto 0);
     qpllRst         : out slv(1 downto 0);
     --
     axilClk         : in  sl;
     axilRst         : in  sl;
     axilReadMaster  : in  AxiLiteReadMasterType;
     axilReadSlave   : out AxiLiteReadSlaveType;
     axilWriteMaster : in  AxiLiteWriteMasterType;
     axilWriteSlave  : out AxiLiteWriteSlaveType;
     --
     ibClk           : in  sl;
     ibRst           : in  sl;
     ibMaster        : out AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0);
     ibSlave         : in  AxiStreamSlaveArray (NUM_DTI_VC_C-1 downto 0);
     loopback        : in  sl;
     linkUp          : out sl;
     locLinkID       : in  slv(31 downto 0);
     remLinkID       : out slv(31 downto 0);
     rxErrs          : out slv(31 downto 0);
     txFull          : out sl;
     monClk          : out sl;
     --
     obClk           : in  sl;
     obRst           : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType;
     --
     timingClk       : in  sl;
     timingRst       : in  sl;
     obTrig          : in  XpmPartitionDataType;
     obTrigValid     : in  sl );
end DtiUsPgp3;

architecture top_level_app of DtiUsPgp3 is

  type RegType is record
    --  Event state
    opCodeEn   : sl;
    opCode     : slv( 7 downto 0);
    --  Ctl Statistics
    obSent     : slv(31 downto 0);
    obReceived : slv(31 downto 0);
    --  SOF fixup
    tlast      : sl;
  end record;
  
  constant REG_INIT_C : RegType := (
    opCodeEn   => '0',
    opCode     => (others=>'0'),
    obSent     => (others=>'0'),
    obReceived => (others=>'0'),
    tlast      => '1' );

  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;

  signal pgpTrig      : slv(2 downto 0);
  signal pgpTrigValid : sl;

  signal locTxIn        : Pgp3TxInType := PGP3_TX_IN_INIT_C;
  signal pgpTxIn        : Pgp3TxInType;
  signal pgpTxOut       : Pgp3TxOutType;
  signal pgpRxIn        : Pgp3RxInType := PGP3_RX_IN_INIT_C;
  signal pgpRxOut       : Pgp3RxOutType;
  signal pgpTxMasters   : AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0) := (others=>AXI_STREAM_MASTER_INIT_C);
  signal pgpTxSlaves    : AxiStreamSlaveArray (NUM_DTI_VC_C-1 downto 0);
  signal pgpRxMasters   : AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0);
  signal pgpRxCtrls     : AxiStreamCtrlArray  (NUM_DTI_VC_C-1 downto 0) := (others=>AXI_STREAM_CTRL_UNUSED_C);

  signal pgpClk         : sl;
  signal pgpRst         : sl;

  signal itxfull        : sl;
  signal slocLinkID     : slv(31 downto 0);
  
  signal iibMaster      : AxiStreamMasterArray(NUM_DTI_VC_C-1 downto 0);
  signal iobSlave       : AxiStreamSlaveType;

  component ila_0
    port ( clk : in sl;
           probe0 : in slv(255 downto 0) );
  end component;
  
begin

  GEN_DBUG : if DEBUG_G generate
    U_ILA_IB : ila_0
      port map ( clk                  => ibClk,
                 probe0(0)            => ibSlave  (VC_CTL).tReady,
                 probe0(1)            => iibMaster(VC_CTL).tValid,
                 probe0(65 downto  2) => iibMaster(VC_CTL).tData(63 downto 0),
                 probe0(67 downto 66) => iibMaster(VC_CTL).tUser( 1 downto 0),
                 probe0(68)           => iibMaster(VC_CTL).tLast,
                 probe0(255 downto 69) => (others=>'0') );
    U_ILA_OB : ila_0
      port map ( clk                  => obClk,
                 probe0(0)            => iobSlave.tReady,
                 probe0(1)            => obMaster.tValid,
                 probe0(65 downto  2) => obMaster.tData(63 downto 0),
                 probe0(67 downto 66) => obMaster.tUser( 1 downto 0),
                 probe0(68)           => obMaster.tLast,
                 probe0(255 downto 69) => (others=>'0') );
  --  U_ILA_P : ila_0
  --    port map ( clk                  => pgpClk,
  --               probe0(0)            => pgpRxCtrls(0).pause,
  --               probe0(1)            => pgpRxMasters(0).tValid,
  --               probe0(17 downto  2) => pgpRxMasters(0).tData(15 downto 0),
  --               probe0(19 downto 18) => pgpRxMasters(0).tUser( 1 downto 0),
  --               probe0(255 downto 20) => (others=>'0') );
  end generate;

  obSlave  <= iobSlave;
  
  linkUp                   <= pgpRxOut.linkReady;
  pgpRxIn.loopback         <= '0' & loopback & '0';
--  remLinkID                <= pgpRxOut.remLinkData;
  
  locTxIn.disable          <= '0';
  locTxIn.flowCntlDis      <= '1';
--  locTxIn.skpInterval      <= (others=>'0');
  locTxIn.skpInterval      <= X"0000FFF0";  -- override bad default
  locTxIn.opCodeEn         <= r.opCodeEn;
  locTxIn.opCodeNumber     <= toSlv(6,3);
  locTxIn.opCodeData(15 downto  0) <= resize(r.opCode,16);
  locTxIn.opCodeData(47 downto 16) <= slocLinkId;
--  locTxIn.locData          <= ID_G;
  pgpTxIn                  <= locTxIn;
  
  monClk                   <= pgpClk;
  
  U_PgpFb : entity work.DtiPgp3Fb
    port map ( pgpClk       => pgpClk,
               pgpRst       => pgpRst,
               pgpRxOut     => pgpRxOut,
               rxLinkId     => remLinkID,
               txAlmostFull => itxFull );

  U_Pgp3 : entity work.Pgp3GthUs
    generic map ( NUM_VC_G     => NUM_DTI_VC_C,
--                  DEBUG_G      => DEBUG_G,
                  EN_DRP_G     => EN_AXIL_G,
                  EN_PGP_MON_G => EN_AXIL_G,
                  AXIL_CLK_FREQ_G  => 156.25e+6,
                  AXIL_BASE_ADDR_G => AXIL_BASE_ADDR_G )
    port map ( -- Stable Clock and Reset
               stableClk    => axilClk,
               stableRst    => axilRst,
               -- QPLL Interface
               qpllLock     => qplllock,
               qpllclk      => qplloutclk,
               qpllrefclk   => qplloutrefclk,
               qpllRst      => qpllRst,
               -- Gt Serial IO
               pgpGtTxP     => amcTxP,
               pgpGtTxN     => amcTxN,
               pgpGtRxP     => amcRxP,
               pgpGtRxN     => amcRxN,
               -- Clocking
               pgpClk       => pgpClk,
               pgpClkRst    => pgpRst,
               -- Non VC Tx Signals
               pgpTxIn      => pgpTxIn,
               pgpTxOut     => pgpTxOut,
               -- Non VC Rx Signals
               pgpRxIn      => pgpRxIn,
               pgpRxOut     => pgpRxOut,
               -- Frame TX Interface
               pgpTxMasters => pgpTxMasters,
               pgpTxSlaves  => pgpTxSlaves,
               -- Frame RX Interface
               pgpRxMasters => pgpRxMasters,
               pgpRxCtrl    => pgpRxCtrls,
               -- AXI-Lite Register Interface
               axilClk         => axilClk,
               axilRst         => axilRst,
               axilReadMaster  => axilReadMaster,
               axilReadSlave   => axilReadSlave,
               axilWriteMaster => axilWriteMaster,
               axilWriteSlave  => axilWriteSlave );

  U_RXERR : entity work.SynchronizerOneShotCnt
    generic map ( CNT_WIDTH_G => 32 )
    port map ( wrClk   => pgpClk,
               rdClk   => axilClk,
               cntRst  => fifoRst,
               rollOverEn => '1',
               dataIn  => pgpRxOut.linkError,
               dataOut => open,
               cntOut  => rxErrs );
  
  U_TXFULL : entity work.Synchronizer
    port map ( clk     => ibClk,
               dataIn  => itxFull,
               dataOut => txFull );
  
--  U_ObToAmc : entity work.AxiStreamFifoV2
  U_ObToAmc : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
                  MASTER_AXI_CONFIG_G => PGP3_AXIS_CONFIG_C )
    port map ( sAxisClk    => obClk,
               sAxisRst    => obRst,
               sAxisMaster => obMaster,
               sAxisSlave  => iobSlave,
               mAxisClk    => pgpClk,
               mAxisRst    => pgpRst,
               mAxisMaster => pgpTxMasters(VC_CTL),
               mAxisSlave  => pgpTxSlaves (VC_CTL));

  GEN_AMCTOIB : for i in 0 to NUM_DTI_VC_C-1 generate
--    U_AmcToIb : entity work.AxiStreamFifoV2
    U_AmcToIb : entity work.AxiStreamFifo
      generic map ( SLAVE_AXI_CONFIG_G  => PGP3_AXIS_CONFIG_C,
                    MASTER_AXI_CONFIG_G => US_IB_CONFIG_C,
                    FIFO_ADDR_WIDTH_G   => 9,
                    FIFO_PAUSE_THRESH_G => 256 )
      port map ( sAxisClk    => pgpClk,
                 sAxisRst    => pgpRst,
                 sAxisMaster => pgpRxMasters(i),
                 sAxisSlave  => open,
                 sAxisCtrl   => pgpRxCtrls  (i),
                 mAxisClk    => ibClk,
                 mAxisRst    => ibRst,
                 mAxisMaster => iibMaster   (i),
                 mAxisSlave  => ibSlave     (i));
  end generate;
  
  U_SyncObTrig : entity work.SynchronizerFifo
    generic map ( DATA_WIDTH_G  => 3 )
    port map ( rst     => timingRst,
               wr_clk  => timingClk,
               wr_en   => obTrigValid,
               din(0)  => obTrig.l0a,
               din(1)  => obTrig.l1e,
               din(2)  => obTrig.l1a,
               rd_clk  => pgpClk,
               valid   => pgpTrigValid,
               dout    => pgpTrig );

  U_SyncLocLinkID : entity work.SynchronizerVector
    generic map ( DATA_WIDTH_G => 32 )
    port map ( clk     => pgpClk,
               dataIn  => locLinkID,
               dataOut => slocLinkID );
  
  comb : process ( fifoRst, r, pgpTrig, pgpTrigValid,
                   pgpTxMasters, pgpTxSlaves,
                   pgpRxMasters,
                   iibMaster ) is
    variable v   : RegType;
    variable m   : AxiStreamMasterType;
  begin
    v := r;

    -- Need to transmit three bits: L0 and L1(A/R)
    v.opCodeEn := '0';
    if pgpTrigValid = '1' then
      if pgpTrig /= 0 then
        v.opCodeEn := '1';
      end if;
      v.opCode(7)          := '1';  -- currently required by pgpcard
      v.opCode(2 downto 0) := pgpTrig;
    end if;

    if pgpTxMasters(VC_CTL).tValid='1' and pgpTxSlaves(VC_CTL).tReady='1' then
      v.obSent := r.obSent + 1;
    end if;

    if pgpRxMasters(VC_CTL).tValid='1' then
      v.obReceived := r.obReceived + 1;
    end if;
    
    if fifoRst = '1' then
      v := REG_INIT_C;
    end if;

    status.obSent     <= r.obSent;
    status.obReceived <= r.obReceived;

    --  SOF fixup
    ibMaster <= iibMaster;
    m := iibMaster(VC_CTL);
    if m.tValid = '1' then
      m.tKeep := genTKeep(8);
      if r.tlast = '1' then
        axiStreamSetUserBit(PGP3_AXIS_CONFIG_C, m, SSI_SOF_C, '1', 0);
      end if;
      v.tlast := m.tLast;
    end if;
    ibMaster(VC_CTL) <= m;
    
    r_in <= v;
  end process;
            
  seq : process (pgpClk) is
  begin
    if rising_edge(pgpClk) then
      r <= r_in;
    end if;
  end process;
  
end top_level_app;
