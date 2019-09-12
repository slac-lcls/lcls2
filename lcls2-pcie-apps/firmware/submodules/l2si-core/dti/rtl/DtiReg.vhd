------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiReg.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-07-26
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Software programmable register interface
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 XPM Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the
-- top-level directory of this distribution and at:
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
-- No part of 'LCLS2 XPM Core', including this file,
-- may be copied, modified, propagated, or distributed except according to
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use ieee.numeric_std.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.AmcCarrierPkg.all;  -- ETH_AXIS_CONFIG_C
use work.DtiPkg.all;
use work.XpmPkg.all;

entity DtiReg is
   generic ( AXIL_BASE_ADDR_G : slv(31 downto 0) := x"00000000" );
   port (
      axilClk          : in  sl;
      axilRst          : in  sl;
      axilClear        : out sl;
      axilUpdate       : out sl;
      axilWriteMaster  : in  AxiLiteWriteMasterType;  
      axilWriteSlave   : out AxiLiteWriteSlaveType;  
      axilReadMaster   : in  AxiLiteReadMasterType;  
      axilReadSlave    : out AxiLiteReadSlaveType;
      --
      status           : in  DtiStatusType;
      config           : out DtiConfigType;
      monclk           : in  slv(3 downto 0) );
end DtiReg;

architecture rtl of DtiReg is

  constant AXIL_XBAR_CONFIG_C : AxiLiteCrossbarMasterConfigArray(1 downto 0) := genAxiLiteConfig(2, AXIL_BASE_ADDR_G, 8, 7);
  signal axilWriteMasters : AxiLiteWriteMasterArray(1 downto 0);
  signal axilWriteSlaves  : AxiLiteWriteSlaveArray (1 downto 0);
  signal axilReadMasters  : AxiLiteReadMasterArray (1 downto 0);
  signal axilReadSlaves   : AxiLiteReadSlaveArray  (1 downto 0);
  
  type StateType is (IDLE_S, READING_S);
  
  type RegType is record
    update          : sl;
    clear           : sl;
    config          : DtiConfigType;
    qplllock        : slv(status.qplllock'range);
    usLink          : slv(3 downto 0);
    dsLink          : slv(3 downto 0);
    axilReadSlaves  : AxiLiteReadSlaveArray (1 downto 0);
    axilWriteSlaves : AxiLiteWriteSlaveArray(1 downto 0);
  end record RegType;

  constant REG_INIT_C : RegType := (
    update          => '1',
    clear           => '0',
    config          => DTI_CONFIG_INIT_C,
    qplllock        => (others=>'0'),
    usLink          => (others=>'0'),
    dsLink          => (others=>'0'),
    axilReadSlaves  => (others=>AXI_LITE_READ_SLAVE_INIT_C),
    axilWriteSlaves => (others=>AXI_LITE_WRITE_SLAVE_INIT_C) );

  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;

  signal usStatus, iusStatus : DtiUsLinkStatusType;
  signal dsStatus, idsStatus : DtiDsLinkStatusType;
  signal usApp   , iusApp    : DtiUsAppStatusType;
  signal usLinkUp : slv(MaxUsLinks-1 downto 0);
  signal dsLinkUp : slv(MaxDsLinks-1 downto 0);
  signal bpStatus : DtiBpLinkStatusType;
  signal qplllock : slv(status.qplllock'range);

  signal monClkRate : Slv32Array(3 downto 0);
  signal monClkLock : slv       (3 downto 0);
  signal monClkFast : slv       (3 downto 0);
  signal monClkSlow : slv       (3 downto 0);

  signal pllStat     : slv(3 downto 0);
  signal pllCount    : SlVectorArray(3 downto 0, 2 downto 0);

  signal msgDelaySetS : Slv7Array(NPartitions-1 downto 0);
  signal msgDelayGetS : Slv7Array(MaxUsLinks-1 downto 0);

begin

  config         <= r.config;
  axilClear      <= r.clear;
  axilUpdate     <= r.update;

  iusStatus      <= status.usLink(conv_integer(r.usLink));
  idsStatus      <= status.dsLink(conv_integer(r.dsLink));
  iusApp         <= status.usApp (conv_integer(r.usLink));

  U_AXIL_XBAR : entity work.AxiLiteCrossbar
      generic map (
         NUM_SLAVE_SLOTS_G  => 1,
         NUM_MASTER_SLOTS_G => 2,
         MASTERS_CONFIG_G   => AXIL_XBAR_CONFIG_C)
      port map (
         axiClk              => axilClk,
         axiClkRst           => axilRst,
         sAxiWriteMasters(0) => axilWriteMaster,
         sAxiWriteSlaves(0)  => axilWriteSlave,
         sAxiReadMasters(0)  => axilReadMaster,
         sAxiReadSlaves(0)   => axilReadSlave,
         mAxiWriteMasters    => axilWriteMasters,
         mAxiWriteSlaves     => r.axilWriteSlaves,
         mAxiReadMasters     => axilReadMasters,
         mAxiReadSlaves      => r.axilReadSlaves);

  GEN_USLINKUP : for i in 0 to MaxUsLinks-1 generate
    U_SYNC : entity work.Synchronizer
      port map ( clk     => axilClk,
                 dataIn  => status.usLink(i).linkUp,
                 dataOut => usLinkUp(i) );
  end generate;
  
  GEN_DSLINKUP : for i in 0 to MaxDsLinks-1 generate
    U_SYNC : entity work.Synchronizer
      port map ( clk     => axilClk,
                 dataIn  => status.dsLink(i).linkUp,
                 dataOut => dsLinkUp(i) );
  end generate;

  U_BPLINKUP : entity work.Synchronizer
    port map ( clk     => axilClk,
               dataIn  => status.bpLink.linkUp,
               dataOut => bpStatus.linkUp );

  U_BPSent : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => status.bpLink.obSent,
               dataOut => bpStatus.obSent );

  usStatus.rxErrs  <= iusStatus.rxErrs;

  U_UsRxInh : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.rxInh,
               dataOut => usStatus.rxInh );
  
  U_UsRemLinkID : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.remLinkID,
               dataOut => usStatus.remLinkID );
  
  U_UsRxFull : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.rxFull,
               dataOut => usStatus.rxFull );
  
  U_UsIbReceived : entity work.SynchronizerVector
    generic map ( WIDTH_G => 48 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.ibRecv,
               dataOut => usStatus.ibRecv );
  
  U_UsIbEvt : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.ibEvt,
               dataOut => usStatus.ibEvt );
  
  U_UsIbDump : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.ibDump,
               dataOut => usStatus.ibDump );
  
  U_UsObL0 : entity work.SynchronizerVector
    generic map ( WIDTH_G => 20 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.obL0,
               dataOut => usStatus.obL0 );
  
  U_UsObL1A : entity work.SynchronizerVector
    generic map ( WIDTH_G => 20 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.obL1A,
               dataOut => usStatus.obL1A );
  
  U_UsObL1R : entity work.SynchronizerVector
    generic map ( WIDTH_G => 20 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.obL1R,
               dataOut => usStatus.obL1R );

  U_UsWrFifoD : entity work.SynchronizerVector
    generic map ( WIDTH_G => 4 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.wrFifoD,
               dataOut => usStatus.wrFifoD );

  U_UsRdFifoD : entity work.SynchronizerVector
    generic map ( WIDTH_G => 4 )
    port map ( clk     => axilClk,
               dataIn  => iusStatus.rdFifoD,
               dataOut => usStatus.rdFifoD );
  
  dsStatus.rxErrs <= idsStatus.rxErrs;
  
  U_DsRemLinkID : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => idsStatus.remLinkID,
               dataOut => dsStatus.remLinkID );
  
  U_DsRxFull : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => idsStatus.rxFull,
               dataOut => dsStatus.rxFull );
  
  U_DsObSent : entity work.SynchronizerVector
    generic map ( WIDTH_G => 48 )
    port map ( clk     => axilClk,
               dataIn  => idsStatus.obSent,
               dataOut => dsStatus.obSent );
  
  U_AppObRecd : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusApp.obReceived,
               dataOut => usApp.obReceived );
  
  U_AppObSent : entity work.SynchronizerVector
    generic map ( WIDTH_G => 32 )
    port map ( clk     => axilClk,
               dataIn  => iusApp.obSent,
               dataOut => usApp.obSent );

  U_QpllLock : entity work.SynchronizerVector
    generic map ( WIDTH_G => status.qplllock'length )
    port map ( clk     => axilClk,
               dataIn  => status.qplllock,
               dataOut => qplllock );

  GEN_MONCLK : for i in 0 to 3 generate
    U_SYNC : entity work.SyncClockFreq
      generic map ( REF_CLK_FREQ_G => 156.25E+6,
                    COMMON_CLK_G   => true,
                    CLK_LOWER_LIMIT_G =>  95.0E+6,
                    CLK_UPPER_LIMIT_G => 186.0E+6 )
      port map ( freqOut     => monClkRate(i),
                 freqUpdated => open,
                 locked      => monClkLock(i),
                 tooFast     => monClkFast(i),
                 tooSlow     => monClkSlow(i),
                 clkIn       => monClk(i),
                 locClk      => axilClk,
                 refClk      => axilClk );
  end generate;

  U_StatLol : entity work.SyncStatusVector
    generic map ( COMMON_CLK_G => true,
                  WIDTH_G      => 4,
                  CNT_WIDTH_G  => 3 )
    port map ( statusIn(0) => status.amcPll(0).los,
               statusIn(1) => status.amcPll(0).lol,
               statusIn(2) => status.amcPll(1).los,
               statusIn(3) => status.amcPll(1).lol,
               statusOut => pllStat,
               cntRstIn  => '0',
               rollOverEnIn => (others=>'1'),
               cntOut    => pllCount,
               wrClk     => axilClk,
               rdClk     => axilClk );

  GEN_MSGDELAYSET : for i in 0 to NPartitions-1 generate
    U_MsgDelaySetS : entity work.SynchronizerVector
      generic map ( WIDTH_G => 7 )
      port map ( clk     => axilClk,
                 dataIn  => status.msgDelaySet(i),
                 dataOut => msgDelaySetS(i) );
  end generate;

  GEN_MSGDELAYGET : for i in 0 to MaxUsLinks-1 generate
    U_MsgDelayGetS : entity work.SynchronizerVector
      generic map ( WIDTH_G => 7 )
      port map ( clk     => axilClk,
                 dataIn  => status.msgDelayGet(i),
                 dataOut => msgDelayGetS(i) );
  end generate;

  comb : process (r, axilRst, axilReadMasters, axilWriteMasters, usApp,
                  usLinkUp, dsLinkUp, usStatus, dsStatus, bpStatus, qplllock,
                  monClkRate, monClkLock, monClkFast, monClkSlow,
                  pllStat, pllCount, msgDelaySetS, msgDelayGetS) is
    variable v          : RegType;
    variable ra         : integer;
    variable ep         : AxiLiteEndpointType;
    variable ia         : integer;
    
    -- Shorthand procedures for read/write register
    procedure axilRegRW(addr : in slv; offset : in integer; reg : inout slv) is
    begin
      axiSlaveRegister(ep, addr, offset, reg, "0");
    end procedure;
    procedure axilRegRW(addr : in slv; offset : in integer; reg : inout sl) is
    begin
      axiSlaveRegister(ep, addr, offset, reg, '0');
    end procedure;
    -- Shorthand procedures for read only registers
    procedure axilRegR (addr : in slv; offset : in integer; reg : in slv) is
    begin
      axiSlaveRegisterR(ep, addr, offset, reg);
    end procedure;
    procedure axilRegR (addr : in slv; offset : in integer; reg : in sl) is
    begin
      axiSlaveRegisterR(ep, addr, offset, reg);
    end procedure;

  begin
    v := r;

    -- Determine the transaction type
    axiSlaveWaitTxn(ep, axilWriteMasters(0), axilReadMasters(0),
                    v.axilWriteSlaves(0),  v.axilReadSlaves(0) );
    v.axilReadSlaves(0).rdata := (others=>'0');

    for i in 0 to MaxUsLinks-1 loop
      axilRegRW (toSlv(  16*i+0,7),  0, v.config.usLink(i).enable );
      axilRegRW (toSlv(  16*i+0,7),  1, v.config.usLink(i).tagEnable );
      axilRegRW (toSlv(  16*i+0,7),  2, v.config.usLink(i).l1Enable );
      axilRegRW (toSlv(  16*i+0,7),  3, v.config.usLink(i).hdrOnly );
      axilRegRW (toSlv(  16*i+0,7),  4, v.config.usLink(i).partition );
      axilRegRW (toSlv(  16*i+0,7),  8, v.config.usLink(i).afdepth );
--      axilRegRW (toSlv(  16*i+0,7),  8, v.config.usLink(i).trigDelay );
      axilRegRW (toSlv(  16*i+0,7), 16, v.config.usLink(i).fwdMask );
      axilRegRW (toSlv(  16*i+0,7), 31, v.config.usLink(i).fwdMode );
      axilRegRW (toSlv(  16*i+4,7),  0, v.config.usLink(i).dataSrc );
      axilRegRW (toSlv(  16*i+8,7),  0, v.config.usLink(i).dataType );
    end loop;

    for i in 0 to MaxUsLinks-1 loop
      axilRegR (toSlv( 16*7+0,7),  i, usLinkUp(i) );
    end loop;    
    axilRegR (toSlv( 16*7+0,7),  15, bpStatus.linkUp);
    for i in 0 to MaxDsLinks-1 loop
      axilRegR (toSlv( 16*7+0,7), 16+i, dsLinkUp(i) );
    end loop;    

    axilRegRW (toSlv( 16*7+4,7),  0, v.usLink);
    axilRegRW (toSlv( 16*7+4,7), 16, v.dsLink);
    axilRegRW (toSlv( 16*7+4,7), 30, v.clear );
    axilRegRW (toSlv( 16*7+4,7), 31, v.update);

    axilRegR (toSlv( 16*7+8,7), 0, bpStatus.obSent);
    axilRegR (toSlv( 16*7+12,7), 0, AXIL_BASE_ADDR_G);

    -- Set the status
    axiSlaveDefault(ep, v.axilWriteSlaves(0), v.axilReadSlaves(0),
                    AXI_RESP_OK_C);

    -- Determine the transaction type
    axiSlaveWaitTxn(ep, axilWriteMasters(1), axilReadMasters(1),
                    v.axilWriteSlaves(1),  v.axilReadSlaves(1));
    v.axilReadSlaves(1).rdata := (others=>'0');
    
    axilRegR (toSlv( 16*0+0 ,7),  0, usStatus.remLinkID );
    axilRegR (toSlv( 16*0+4 ,7),  0, usStatus.rxFull );
--    axilRegR (toSlv( 16*0+8 ,7),  0, usStatus.ibRecv (31 downto 0));
    axilRegR (toSlv( 16*0+8 ,7),  0, usStatus.rxInh(23 downto 0) );
    axilRegR (toSlv( 16*0+8 ,7), 24, usStatus.wrFifoD );
    axilRegR (toSlv( 16*0+8 ,7), 28, usStatus.rdFifoD );
    axilRegR (toSlv( 16*0+12,7),  0, usStatus.ibEvt );

    axilRegR (toSlv( 16*1+0 ,7),  0, dsStatus.remLinkID);
    axilRegR (toSlv( 16*1+4 ,7),  0, dsStatus.rxFull);
    axilRegR (toSlv( 16*1+8 ,7),  0, dsStatus.obSent(31 downto 0));
    axilRegR (toSlv( 16*1+12,7),  0, dsStatus.obSent(47 downto 32));

    axilRegR (toSlv( 16*2+0 ,7),  0, usApp.obReceived);
    axilRegR (toSlv( 16*2+8 ,7),  0, usApp.obSent);

    axilRegRW(toSlv( 16*3+0, 7),  0, v.qplllock);
    axilRegRW(toSlv( 16*3+0 ,7), 16, v.config.bpPeriod );
    v.qplllock := qplllock;
    
    for i in 0 to 3 loop
      axilRegR (toSlv( 16*3+4*i+4, 7),  0, monClkRate(i)(28 downto 0));
      axilRegR (toSlv( 16*3+4*i+4, 7), 29, monClkSlow(i));
      axilRegR (toSlv( 16*3+4*i+4, 7), 30, monClkFast(i));
      axilRegR (toSlv( 16*3+4*i+4, 7), 31, monClkLock(i));
    end loop;

    for i in 0 to 1 loop
      ra := 16*5+i*4;
      axilRegRW(toSlv(ra,7),  0, v.config.amcPll(i).bwSel);
      axilRegRW(toSlv(ra,7),  4, v.config.amcPll(i).frqTbl);
      axilRegRW(toSlv(ra,7),  8, v.config.amcPll(i).frqSel);
      axilRegRW(toSlv(ra,7), 16, v.config.amcPll(i).rate);
      axilRegRW(toSlv(ra,7), 20, v.config.amcPll(i).inc);
      axilRegRW(toSlv(ra,7), 21, v.config.amcPll(i).dec);
      axilRegRW(toSlv(ra,7), 22, v.config.amcPll(i).bypass);
      axilRegRW(toSlv(ra,7), 23, v.config.amcPll(i).rstn);
      axilRegR (toSlv(ra,7), 24, muxSlVectorArray( pllCount, 2*i+0));
      axilRegR (toSlv(ra,7), 27, pllStat(2*i+0));
      axilRegR (toSlv(ra,7), 28, muxSlVectorArray( pllCount, 2*i+1));
      axilRegR (toSlv(ra,7), 31, pllStat(2*i+1));
    end loop;

    ra := 16*6;
    for i in 0 to 3 loop
      axilRegR (toSlv( ra+0,7), i*8, msgDelaySetS(i));
      axilRegR (toSlv( ra+4,7), i*8, msgDelaySetS(i+4));
    end loop;

    axilRegR (toSlv( ra +8,7),  0, msgDelayGetS(0) );
    axilRegR (toSlv( ra +8,7),  8, msgDelayGetS(1) );
    axilRegR (toSlv( ra +8,7), 16, msgDelayGetS(2) );
    axilRegR (toSlv( ra +8,7), 24, msgDelayGetS(3) );
    axilRegR (toSlv( ra+12,7),  0, msgDelayGetS(4) );
    axilRegR (toSlv( ra+12,7),  8, msgDelayGetS(5) );
    axilRegR (toSlv( ra+12,7), 16, msgDelayGetS(6) );

    -- Set the status
    axiSlaveDefault(ep, v.axilWriteSlaves(1), v.axilReadSlaves(1),
                    AXI_RESP_OK_C);
    
    ----------------------------------------------------------------------------------------------
    -- Reset
    ----------------------------------------------------------------------------------------------
    if (axilRst = '1') then
      v := REG_INIT_C;
    end if;

    r_in <= v;
  end process;

  seq : process (axilClk) is
  begin
    if rising_edge(axilClk) then
      r <= r_in;
    end if;
  end process;

end rtl;
