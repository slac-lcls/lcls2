-------------------------------------------------------------------------------
-- File       : TimeToolKcu1500VcsTb.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: Simulation Testbed for testing the FPGA module
-------------------------------------------------------------------------------
-- This file is part of 'Camera link gateway'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'Camera link gateway', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.BuildInfoPkg.all;

entity TimeToolKcu1500VcsTb is end TimeToolKcu1500VcsTb;

architecture testbed of TimeToolKcu1500VcsTb is

   constant TPD_G : time := 1 ns;

   signal userClkP : sl := '0';
   signal userClkN : sl := '1';

begin

   U_ClkPgp : entity work.ClkRst
      generic map (
         CLK_PERIOD_G      => 6.4 ns,   -- 156.25 MHz
         RST_START_DELAY_G => 0 ns,
         RST_HOLD_TIME_G   => 1000 ns)
      port map (
         clkP => userClkP,
         clkN => userClkN);

   U_Fpga : entity work.TimeToolKcu1500
      generic map (
         TPD_G          => TPD_G,
         ROGUE_SIM_EN_G => true,
         BUILD_INFO_G   => BUILD_INFO_C)
      port map (
         ---------------------
         --  Application Ports
         ---------------------
         -- QSFP[0] Ports
         qsfp0RefClkP => (others => '0'),
         qsfp0RefClkN => (others => '1'),
         qsfp0RxP     => (others => '0'),
         qsfp0RxN     => (others => '1'),
         qsfp0TxP     => open,
         qsfp0TxN     => open,
         -- QSFP[1] Ports
         qsfp1RefClkP => (others => '0'),
         qsfp1RefClkN => (others => '1'),
         qsfp1RxP     => (others => '0'),
         qsfp1RxN     => (others => '1'),
         qsfp1TxP     => open,
         qsfp1TxN     => open,
         --------------
         --  Core Ports
         --------------
         -- System Ports
         emcClk       => '0',
         userClkP     => userClkP,
         userClkN     => userClkN,
         -- QSFP[0] Ports
         qsfp0RstL    => open,
         qsfp0LpMode  => open,
         qsfp0ModSelL => open,
         qsfp0ModPrsL => '1',
         -- QSFP[1] Ports
         qsfp1RstL    => open,
         qsfp1LpMode  => open,
         qsfp1ModSelL => open,
         qsfp1ModPrsL => '1',
         -- Boot Memory Ports 
         flashCsL     => open,
         flashMosi    => open,
         flashMiso    => '1',
         flashHoldL   => open,
         flashWp      => open,
         -- PCIe Ports
         pciRstL      => '1',
         pciRefClkP   => '0',
         pciRefClkN   => '1',
         pciRxP       => (others => '0'),
         pciRxN       => (others => '1'),
         pciTxP       => open,
         pciTxN       => open);

end testbed;
