-------------------------------------------------------------------------------
-- File       : XpmBpClk.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-08
-- Last update: 2017-10-04
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- Note: Do not forget to configure the ATCA crate to drive the clock from the slot#2 MPS link node
-- For the 7-slot crate:
--    $ ipmitool -I lan -H ${SELF_MANAGER} -t 0x84 -b 0 -A NONE raw 0x2e 0x39 0x0a 0x40 0x00 0x00 0x00 0x31 0x01
-- For the 16-slot crate:
--    $ ipmitool -I lan -H ${SELF_MANAGER} -t 0x84 -b 0 -A NONE raw 0x2e 0x39 0x0a 0x40 0x00 0x00 0x00 0x31 0x01
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Common Carrier Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Common Carrier Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmBpClk is
   generic (
      TPD_G         : time    := 1 ns;
      MPS_SLOT_G    : boolean := false;
      SIM_SPEEDUP_G : boolean := false;
      PHASE_500M_G  : real    := 0.0 );
   port (
      -- Stable Clock and Reset 
      refClk       : in sl;
      refRst       : in sl;
      -- MPS Clocks and Resets
      mps100MHzClk : out sl;
      mps100MHzRst : out sl;
      mps250MHzClk : out sl;
      mps250MHzRst : out sl;
      mps500MHzClk : out sl;
      mps500MHzRst : out sl;
      mpsPllLocked : out sl;
      ----------------
      -- Core Ports --
      ----------------
      mpsClkOut    : out sl);
end XpmBpClk;

architecture mapping of XpmBpClk is

   signal mpsClkBuf     : sl;
   signal mpsRefClk     : sl;
   signal mpsClk        : sl;
   signal mpsRst        : sl;
   signal mpsMmcmClkOut : slv(2 downto 0);
   signal mpsMmcmRstOut : slv(2 downto 0);
   signal locked        : sl;

begin

  mpsClk <= refClk;
  mpsRst <= refRst;

   U_ClkManagerMps : entity work.ClockManagerUltraScale
      generic map(
         TPD_G              => TPD_G,
         TYPE_G             => "MMCM",
--         INPUT_BUFG_G       => ite(MPS_SLOT_G, false, true),
         INPUT_BUFG_G       => false,
         FB_BUFG_G          => true,
         RST_IN_POLARITY_G  => '1',
         NUM_CLOCKS_G       => 3,
         -- MMCM attributes
         BANDWIDTH_G        => "OPTIMIZED",
         CLKIN_PERIOD_G     => ite(MPS_SLOT_G, 8.0, 10.0),
         DIVCLK_DIVIDE_G    => 1,
         CLKFBOUT_MULT_F_G  => ite(MPS_SLOT_G, 8.0, 10.0),  -- 1.00 GHz
         CLKOUT0_DIVIDE_F_G => 2.0,                         -- 500 MHz = 1.00 GHz/2.0
         CLKOUT0_RST_HOLD_G => 4,
         CLKOUT0_PHASE_G    => PHASE_500M_G,
         CLKOUT1_DIVIDE_G   => 4,                           -- 250 MHz = 1.00 GHz/4
         CLKOUT2_DIVIDE_G   => 10)                          -- 100 MHz = 1.00 GHz/10
      port map(
         -- Clock Input
         clkIn  => mpsClk,
         rstIn  => mpsRst,
         -- Clock Outputs
         clkOut => mpsMmcmClkOut,
         -- Reset Outputs
         rstOut => mpsMmcmRstOut,
         -- Locked Status
         locked => locked);

   Sync_locked : entity work.Synchronizer
      generic map (
         TPD_G => TPD_G)
      port map (
         clk     => refClk,
         dataIn  => locked,
         dataOut => mpsPllLocked);            

   mps100MHzClk <= mpsMmcmClkOut(2);
   mps100MHzRst <= mpsMmcmRstOut(2);

   mps250MHzClk <= mpsMmcmClkOut(1);
   mps250MHzRst <= mpsMmcmRstOut(1);

   mps500MHzClk <= mpsMmcmClkOut(0);
   mps500MHzRst <= mpsMmcmRstOut(0);

   U_ClkOutBufSingle : entity work.ClkOutBufSingle
      generic map(
         TPD_G        => TPD_G,
         XIL_DEVICE_G => "ULTRASCALE")
      port map (
         outEnL => ite(MPS_SLOT_G, '0', '1'),
         clkIn  => mpsMmcmClkOut(2),
         clkOut => mpsClkOut);

end mapping;
