-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmGthUltrascaleWrapper.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-09-06
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Interface to sensor link MGT
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

use work.StdRtlPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;


entity XpmGthUltrascaleWrapper is
   generic ( GTGCLKRX   : boolean := true;
             NLINKS_G   : integer := 7;
             USE_IBUFDS : boolean := true );
   port (
      gtTxP            : out slv(NLINKS_G-1 downto 0);
      gtTxN            : out slv(NLINKS_G-1 downto 0);
      gtRxP            : in  slv(NLINKS_G-1 downto 0);
      gtRxN            : in  slv(NLINKS_G-1 downto 0);
      devClkP          : in  sl := '0';
      devClkN          : in  sl := '0';
      devClkIn         : in  sl := '0';
      devClkOut        : out sl;
      stableClk        : in  sl;
      txData           : in  Slv16Array(NLINKS_G-1 downto 0);
      txDataK          : in  Slv2Array (NLINKS_G-1 downto 0);
      rxData           : out Slv16Array(NLINKS_G-1 downto 0);
      rxDataK          : out Slv2Array (NLINKS_G-1 downto 0);
      rxClk            : out slv       (NLINKS_G-1 downto 0);
      rxRst            : out slv       (NLINKS_G-1 downto 0);
      rxErr            : out slv       (NLINKS_G-1 downto 0);
      txClk            : out sl;
      txClkIn          : in  sl;
      config           : in  XpmLinkConfigArray(NLINKS_G-1 downto 0);
      status           : out XpmLinkStatusArray(NLINKS_G-1 downto 0) );
end XpmGthUltrascaleWrapper;

architecture rtl of XpmGthUltrascaleWrapper is

COMPONENT gt_dslink_ss_nophase_amc0
  PORT (
    gtwiz_userclk_tx_active_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_userclk_rx_active_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_tx_reset_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_tx_start_user_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_tx_done_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_tx_error_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_rx_reset_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_rx_start_user_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_rx_done_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_buffbypass_rx_error_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_clk_freerun_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_all_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_tx_pll_and_datapath_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_tx_datapath_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_rx_pll_and_datapath_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_rx_datapath_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_rx_cdr_stable_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_tx_done_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_reset_rx_done_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtwiz_userdata_tx_in : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    gtwiz_userdata_rx_out : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
--    cpllrefclksel_in : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
    drpclk_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
--    gtgrefclk_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gthrxn_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gthrxp_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gtrefclk0_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    loopback_in : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
    rx8b10ben_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxcommadeten_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxmcommaalignen_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxpcommaalignen_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxusrclk_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxusrclk2_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    tx8b10ben_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    txctrl0_in : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    txctrl1_in : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    txctrl2_in : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
    txusrclk_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    txusrclk2_in : IN STD_LOGIC_VECTOR(0 DOWNTO 0);
    gthtxn_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    gthtxp_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxbyteisaligned_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxbyterealign_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxcommadet_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxctrl0_out : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    rxctrl1_out : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    rxctrl2_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    rxctrl3_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    rxoutclk_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    rxpmaresetdone_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    txoutclk_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    txpmaresetdone_out : OUT STD_LOGIC_VECTOR(0 DOWNTO 0)
  );
END COMPONENT;

  type RegType is record
    clkcnt  : slv(5 downto 0);
    errdet  : sl;
    reset   : sl;
  end record;
  constant REG_INIT_C : RegType := (
    clkcnt  => (others=>'0'),
    errdet  => '0',
    reset   => '0' );
  type RegTypeArray is array(natural range<>) of RegType;

  constant ERR_INTVL : slv(5 downto 0) := (others=>'1');
  
  signal r    : RegTypeArray(NLINKS_G-1 downto 0) := (others=>REG_INIT_C);
  signal rin  : RegTypeArray(NLINKS_G-1 downto 0);
  
  signal txCtrl2In  : Slv8Array (NLINKS_G-1 downto 0);
  signal rxCtrl0Out : Slv16Array(NLINKS_G-1 downto 0);
  signal rxCtrl1Out : Slv16Array(NLINKS_G-1 downto 0);
  signal rxCtrl3Out : Slv8Array (NLINKS_G-1 downto 0);

  signal txOutClk   : slv(NLINKS_G-1 downto 0);
  signal txUsrClk   : slv(NLINKS_G-1 downto 0);
  signal gtRefClk   : sl;

  signal rxErrL    : slv(NLINKS_G-1 downto 0);
  signal rxErrS    : slv(NLINKS_G-1 downto 0);
  signal rxErrCnts : Slv16Array(NLINKS_G-1 downto 0);

  signal rxOutClk  : slv(NLINKS_G-1 downto 0);
  signal rxUsrClk  : slv(NLINKS_G-1 downto 0);
  signal rxFifoRst : slv(NLINKS_G-1 downto 0);
  signal rxErrIn   : slv(NLINKS_G-1 downto 0);

  signal rxReset      : slv(NLINKS_G-1 downto 0);
  signal rxResetDone  : slv(NLINKS_G-1 downto 0);
 
  signal rxbypassrst  : slv(NLINKS_G-1 downto 0);
  signal txbypassrst  : slv(NLINKS_G-1 downto 0);

  signal loopback  : Slv3Array(NLINKS_G-1 downto 0);

  component ila_1x256x1024
    port ( clk : in  sl;
           probe0 : in slv(255 downto 0) );
  end component;

begin

  rxClk   <= rxUsrClk;
  rxRst   <= rxFifoRst;
  rxErr   <= rxErrL;
  txClk   <= txUsrClk(0);

  GEN_IBUFDS : if USE_IBUFDS generate
    DEVCLK_IBUFDS_GTE3 : IBUFDS_GTE3
      generic map (
        REFCLK_EN_TX_PATH  => '0',
        REFCLK_HROW_CK_SEL => "01",    -- 2'b01: ODIV2 = Divide-by-2 version of O
        REFCLK_ICNTL_RX    => "00")
      port map (
        I     => devClkP,
        IB    => devClkN,
        CEB   => '0',
        ODIV2 => open,
        O     => gtRefClk);
  end generate;

  NO_GEN_IBUFDS : if not USE_IBUFDS generate
    gtRefClk <= devClkIn;
  end generate;

  devClkOut  <= gtRefClk;
  
  GEN_CTRL : for i in 0 to NLINKS_G-1 generate
    txCtrl2In (i) <= "000000" & txDataK(i);
    rxErrIn   (i) <= '0' when (rxCtrl1Out(i)(1 downto 0)="00" and rxCtrl3Out(i)(1 downto 0)="00") else '1';
    rxFifoRst (i) <= not rxResetDone(i);
    loopback  (i) <= "0" & config(i).loopback & "0";
    status    (i).rxErr       <= rxErrS(i);
    status    (i).rxErrCnts   <= rxErrCnts(i);
    status    (i).rxReady     <= rxResetDone(i);
    rxReset   (i) <= config(i).rxReset or r(i).reset;
    
    U_STATUS : entity work.SynchronizerOneShotCnt
      generic map ( CNT_WIDTH_G => 16 )
      port map ( dataIn       => rxErrL(i),
                 dataOut      => rxErrS(i),
                 rollOverEn   => '1',
                 cntOut       => rxErrCnts(i),
                 wrClk        => rxUsrClk (i),
                 rdClk        => stableClk );

    U_BUFG  : BUFG_GT
      port map (  I       => rxOutClk(i),
                  CE      => '1',
                  CEMASK  => '1',
                  CLR     => '0',
                  CLRMASK => '1',
                  DIV     => "000",
                  O       => rxUsrClk(i) );

    --U_TXBUFG  : BUFG_GT
    --  port map (  I       => txOutClk(0),
    --              CE      => '1',
    --              CEMASK  => '1',
    --              CLR     => '0',
    --              CLRMASK => '1',
    --              DIV     => "000",
    --              O       => txUsrClk(i) );
    txUsrClk(i) <= txClkIn;
    
    rxErrL (i)  <= rxErrIn(i);
    rxClk  (i)  <= rxUsrClk(i);
    rxRst  (i)  <= rxFifoRst(i);
    rxDataK(i)  <= rxCtrl0Out(i);

    U_RstSyncTx : entity work.RstSync
      port map ( clk      => txUsrClk(i),
                 asyncRst => config(i).txReset,
                 syncRst  => txbypassrst(i) );

    U_RstSyncRx : entity work.RstSync
      port map ( clk      => rxUsrClk(i),
                 asyncRst => rxReset(i),
                 syncRst  => rxbypassrst(i) );

    U_GthCore : gt_dslink_ss_nophase_amc0 -- 1 RTM link
      PORT MAP (
        gtwiz_userclk_tx_active_in           => "1",
        gtwiz_userclk_rx_active_in           => "1",
        gtwiz_buffbypass_tx_reset_in     (0) => txbypassrst(i),
        gtwiz_buffbypass_tx_start_user_in    => "0",
        gtwiz_buffbypass_tx_done_out     (0) => status(i).txResetDone,
        gtwiz_buffbypass_tx_error_out        => open,
        gtwiz_buffbypass_rx_reset_in     (0) => rxbypassrst(i),
        gtwiz_buffbypass_rx_start_user_in    => "0",
        gtwiz_buffbypass_rx_done_out     (0) => status(i).rxResetDone,
        gtwiz_buffbypass_rx_error_out        => open,  -- Might need this
        gtwiz_reset_clk_freerun_in(0)        => stableClk,
        gtwiz_reset_all_in                   => "0",
        gtwiz_reset_tx_pll_and_datapath_in(0)=> config(i).txPllReset,
        gtwiz_reset_tx_datapath_in        (0)=> config(i).txReset,
        gtwiz_reset_rx_pll_and_datapath_in(0)=> config(i).rxPllReset,
        gtwiz_reset_rx_datapath_in        (0)=> rxReset(i),
        gtwiz_reset_rx_cdr_stable_out        => open,
        gtwiz_reset_tx_done_out           (0)=> status(i).txReady,
        gtwiz_reset_rx_done_out           (0)=> rxResetDone(i),
        gtwiz_userdata_tx_in                 => txData(i),
        gtwiz_userdata_rx_out                => rxData(i),
        -- CPLL
--        cpllrefclksel_in                     => (others=>'1'),
        drpclk_in                         (0)=> stableClk,
--        gtgrefclk_in                      (0)=> txUsrClk(i),
        gthrxn_in                         (0)=> gtRxN(i),
        gthrxp_in                         (0)=> gtRxP(i),
        gtrefclk0_in                      (0)=> gtRefClk,
        loopback_in                          => loopback(i),
        rx8b10ben_in                         => (others=>'1'),
        rxcommadeten_in                      => (others=>'1'),
        rxmcommaalignen_in                   => (others=>'1'),
        rxpcommaalignen_in                   => (others=>'1'),
        rxusrclk_in                       (0)=> rxUsrClk(i),
        rxusrclk2_in                      (0)=> rxUsrClk(i),
        tx8b10ben_in                         => (others=>'1'),
        txctrl0_in                           => (others=>'0'),
        txctrl1_in                           => (others=>'0'),
        txctrl2_in                           => txCtrl2In(i),
        txusrclk_in                       (0)=> txUsrClk(i),
        txusrclk2_in                      (0)=> txUsrClk(i),
        gthtxn_out                        (0)=> gtTxN(i),
        gthtxp_out                        (0)=> gtTxP(i),
        rxbyteisaligned_out                  => open,
        rxbyterealign_out                    => open,
        rxcommadet_out                       => open,
        rxctrl0_out                          => rxCtrl0Out(i),
        rxctrl1_out                          => rxCtrl1Out(i),
        rxctrl2_out                          => open,
        rxctrl3_out                          => rxCtrl3Out(i),
        rxoutclk_out                      (0)=> rxOutClk(i),
        rxpmaresetdone_out                   => open,
        txoutclk_out                      (0)=> txOutClk(i),
        txpmaresetdone_out                   => open
        );

    comb : process ( r, rxResetDone, rxErrIn ) is
      variable v : RegType;
    begin
      v := r(i);

      if rxErrIn(i)='1' then
        if r(i).errdet='1' then
          v.reset := '1';
        else
          v.errdet := '1';
        end if;
      end if;

      if r(i).reset='0' then
        v.clkcnt := r(i).clkcnt+1;
        if r(i).clkcnt=ERR_INTVL then
          v.errdet := '0';
          v.clkcnt := (others=>'0');
        end if;
      end if;

      if rxResetDone(i)='0' then
        v := REG_INIT_C;
      end if;
      
      rin(i) <= v;
    end process comb;

    seq : process ( rxUsrClk ) is
    begin
      if rising_edge(rxUsrClk(i)) then
        r(i) <= rin(i);
      end if;
    end process seq;
    
  end generate GEN_CTRL;

end rtl;
