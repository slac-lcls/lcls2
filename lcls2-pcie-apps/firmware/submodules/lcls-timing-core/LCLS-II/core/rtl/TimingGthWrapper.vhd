-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingGthWrapper.vhd
-- Author     : Benjamin Reese  <bareese@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-16
-- Last update: 2015-10-05
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
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;

entity TimingGthWrapper is

   generic (
      TPD_G : time := 1 ns);

   port (
      stableClk : in sl;

      gtRefClk : in  sl;
      gtRxP    : in  sl;
      gtRxN    : in  sl;
      gtTxP    : out sl;
      gtTxN    : out sl;

      txInhibit : in sl;
--       txUsrClk  : in  sl;
--       txRstDone : out sl;
--       txData    : in  slv(15 downto 0);
--       txDataK   : in  slv(1 downto 0);

      rxRecClk  : out sl;
      rxUsrClk  : in  sl;
      rxReset   : in  sl;
      rxRstDone : out sl;
      rxData    : out slv(15 downto 0);
      rxDataK   : out slv(1 downto 0);
      rxDispErr : out slv(1 downto 0);
      rxDecErr  : out slv(1 downto 0)
      );

end entity TimingGthWrapper;

architecture rtl of TimingGthWrapper is

   component TimingGth
      port (
         gtwiz_userclk_tx_reset_in          : in  std_logic_vector(0 downto 0);
         gtwiz_userclk_tx_active_in         : in  std_logic_vector(0 downto 0);
         gtwiz_userclk_rx_active_in         : in  std_logic_vector(0 downto 0);
         gtwiz_buffbypass_tx_reset_in       : in  std_logic_vector(0 downto 0);
         gtwiz_buffbypass_tx_start_user_in  : in  std_logic_vector(0 downto 0);
         gtwiz_buffbypass_tx_done_out       : out std_logic_vector(0 downto 0);
         gtwiz_buffbypass_tx_error_out      : out std_logic_vector(0 downto 0);
         gtwiz_buffbypass_rx_reset_in       : in  std_logic_vector(0 downto 0);
         gtwiz_buffbypass_rx_start_user_in  : in  std_logic_vector(0 downto 0);
         gtwiz_buffbypass_rx_done_out       : out std_logic_vector(0 downto 0);
         gtwiz_buffbypass_rx_error_out      : out std_logic_vector(0 downto 0);
         gtwiz_reset_clk_freerun_in         : in  std_logic_vector(0 downto 0);
         gtwiz_reset_all_in                 : in  std_logic_vector(0 downto 0);
         gtwiz_reset_tx_pll_and_datapath_in : in  std_logic_vector(0 downto 0);
         gtwiz_reset_tx_datapath_in         : in  std_logic_vector(0 downto 0);
         gtwiz_reset_rx_pll_and_datapath_in : in  std_logic_vector(0 downto 0);
         gtwiz_reset_rx_datapath_in         : in  std_logic_vector(0 downto 0);
         gtwiz_reset_rx_cdr_stable_out      : out std_logic_vector(0 downto 0);
         gtwiz_reset_tx_done_out            : out std_logic_vector(0 downto 0);
         gtwiz_reset_rx_done_out            : out std_logic_vector(0 downto 0);
         gtwiz_userdata_tx_in               : in  std_logic_vector(15 downto 0);
         gtwiz_userdata_rx_out              : out std_logic_vector(15 downto 0);
         drpclk_in                          : in  std_logic_vector(0 downto 0);
         gthrxn_in                          : in  std_logic_vector(0 downto 0);
         gthrxp_in                          : in  std_logic_vector(0 downto 0);
         gtrefclk0_in                       : in  std_logic_vector(0 downto 0);
         loopback_in                        : in  std_logic_vector(2 downto 0);
         rx8b10ben_in                       : in  std_logic_vector(0 downto 0);
         rxcommadeten_in                    : in  std_logic_vector(0 downto 0);
         rxmcommaalignen_in                 : in  std_logic_vector(0 downto 0);
         rxpcommaalignen_in                 : in  std_logic_vector(0 downto 0);
         rxusrclk_in                        : in  std_logic_vector(0 downto 0);
         rxusrclk2_in                       : in  std_logic_vector(0 downto 0);
         tx8b10ben_in                       : in  std_logic_vector(0 downto 0);
         txctrl0_in                         : in  std_logic_vector(15 downto 0);
         txctrl1_in                         : in  std_logic_vector(15 downto 0);
         txctrl2_in                         : in  std_logic_vector(7 downto 0);
         txinhibit_in                       : in  std_logic_vector(0 downto 0);
         txusrclk_in                        : in  std_logic_vector(0 downto 0);
         txusrclk2_in                       : in  std_logic_vector(0 downto 0);
         gthtxn_out                         : out std_logic_vector(0 downto 0);
         gthtxp_out                         : out std_logic_vector(0 downto 0);
         rxbyteisaligned_out                : out std_logic_vector(0 downto 0);
         rxbyterealign_out                  : out std_logic_vector(0 downto 0);
         rxcommadet_out                     : out std_logic_vector(0 downto 0);
         rxctrl0_out                        : out std_logic_vector(15 downto 0);
         rxctrl1_out                        : out std_logic_vector(15 downto 0);
         rxctrl2_out                        : out std_logic_vector(7 downto 0);
         rxctrl3_out                        : out std_logic_vector(7 downto 0);
         rxoutclk_out                       : out std_logic_vector(0 downto 0);
         rxpmaresetdone_out                 : out std_logic_vector(0 downto 0);
         txoutclk_out                       : out std_logic_vector(0 downto 0);
         txpmaresetdone_out                 : out std_logic_vector(0 downto 0)
         );
   end component;

begin

   your_instance_name : TimingGth
      port map (
         gtwiz_userclk_tx_reset_in(0)       => gtReset,
         gtwiz_userclk_tx_active_in(0)      => '1',
         gtwiz_userclk_rx_active_in(0)      => '1',
         gtwiz_buffbypass_tx_reset_in(0)    => '1',
         gtwiz_buffbypass_tx_start_user_in  => '0',
         gtwiz_buffbypass_tx_done_out       => open,
         gtwiz_buffbypass_tx_error_out      => open,
         gtwiz_buffbypass_rx_reset_in       => '0',
         gtwiz_buffbypass_rx_start_user_in  => '0',
         gtwiz_buffbypass_rx_done_out       => buffbypassRxDone,
         gtwiz_buffbypass_rx_error_out      => buffbypassRxError,
         gtwiz_reset_clk_freerun_in         => stableClk,
         gtwiz_reset_all_in                 => '0',
         gtwiz_reset_tx_pll_and_datapath_in => '0',
         gtwiz_reset_tx_datapath_in         => gtReset,
         gtwiz_reset_rx_pll_and_datapath_in => '0',
         gtwiz_reset_rx_datapath_in         => gtReset,
         gtwiz_reset_rx_cdr_stable_out      => open,
         gtwiz_reset_tx_done_out            => open,
         gtwiz_reset_rx_done_out            => rxDone,
         gtwiz_userdata_tx_in               => (others => '0'),
         gtwiz_userdata_rx_out              => rxData,
         drpclk_in                          => stableClk,
         gthrxn_in                          => gtRxN,
         gthrxp_in                          => gtRxP,
         gtrefclk0_in                       => gtRefClk,
         rx8b10ben_in                       => "11",
         rxcommadeten_in                    => "11",
         rxmcommaalignen_in                 => "11",
         rxpcommaalignen_in                 => "11",
         rxusrclk_in                        => rxUsrClk,
         rxusrclk2_in                       => rxUsrClk,
         tx8b10ben_in                       => "11",
         txctrl0_in                         => (others => '0'),
         txctrl1_in                         => (others => '0'),
         txctrl2_in                         => (others => '0'),
         txinhibit_in(0)                    => txInhibit,
         txusrclk_in                        => txUsrClk,
         txusrclk2_in                       => txUsrClk,
         gthtxn_out                         => gtTxN,
         gthtxp_out                         => gtTxP,
         rxbyteisaligned_out                => open,
         rxbyterealign_out                  => open,
         rxcommadet_out                     => open,
         rxctrl0_out(1 downto 0)            => rxDataK,
         rxctrl0_out(15 downto 2)           => (others => '0'),
         rxctrl1_out(1 downto 0)            => dispErr,
         rxctrl1_out(15 downto 2)           => open,
         rxctrl2_out                        => open,
         rxctrl3_out(1 downto 0)            => decErr,
         rxctrl3_out(7 downto 2)            => open,
         rxoutclk_out                       => rxoutclk_out,
         rxpmaresetdone_out                 => rxpmaresetdone_out,
         txoutclk_out                       => txoutclk_out,
         txpmaresetdone_out                 => txpmaresetdone_out
         );

   PGPREFCLK_BUFG_GT : BUFG_GT
      port map (
         I       => txOutClk,
         CE      => '1',
         CLR     => '0',
         CEMASK  => '1',
         CLRMASK => '1',
         DIV     => "000",
         O       => txUsrClk);

end architecture rtl;


-- COMP_TAG_END ------ End COMPONENT Declaration ------------

-- The following code must appear in the VHDL architecture
-- body. Substitute your own instance name and net names.

------------- Begin Cut here for INSTANTIATION Template ----- INST_TAG

-- INST_TAG_END ------ E
