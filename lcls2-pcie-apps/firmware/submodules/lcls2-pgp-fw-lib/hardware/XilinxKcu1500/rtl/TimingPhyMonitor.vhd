-------------------------------------------------------------------------------
-- File       : TimingPhyMonitor.vhd
-- Company    : SLAC National Accelerator Laboratory
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
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;

entity TimingPhyMonitor is
   generic (
      TPD_G           : time    := 1 ns;
      SIMULATION_G    : boolean := false;
      AXIL_CLK_FREQ_G : real    := 156.25E+6);  -- units of Hz
   port (
      rxUserRst       : out sl;
      txUserRst       : out sl;
      useMiniTpg      : out Sl;
      mmcmRst         : out sl;
      loopback        : out slv(2 downto 0);
      remTrig         : in  slv(3 downto 0);
      remTrigDrop     : in  slv(3 downto 0);
      locTrig         : in  slv(3 downto 0);
      locTrigDrop     : in  slv(3 downto 0);
      mmcmLocked      : in  slv(1 downto 0);
      refClk          : in  slv(1 downto 0);
      refRst          : in  slv(1 downto 0);
      txClk           : in  sl;
      txRst           : in  sl;
      rxClk           : in  sl;
      rxRst           : in  sl;
      -- AXI-Lite Register Interface (sysClk domain)
      axilClk         : in  sl;
      axilRst         : in  sl;
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType);
end TimingPhyMonitor;

architecture rtl of TimingPhyMonitor is

   type RegType is record
      locTrigCnt     : Slv16Array(3 downto 0);
      locTrigDropCnt : Slv16Array(3 downto 0);
      remTrigCnt     : Slv16Array(3 downto 0);
      remTrigDropCnt : Slv16Array(3 downto 0);
      loopback       : slv(2 downto 0);
      cntRst         : sl;
      mmcmRst        : sl;
      rxUserRst      : sl;
      txUserRst      : sl;
      useMiniTpg     : sl;
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
   end record;

   constant REG_INIT_C : RegType := (
      locTrigCnt     => (others => (others => '0')),
      locTrigDropCnt => (others => (others => '0')),
      remTrigCnt     => (others => (others => '0')),
      remTrigDropCnt => (others => (others => '0')),
      loopback       => "000",
      cntRst         => '0',
      mmcmRst        => '0',
      rxUserRst      => '0',
      txUserRst      => '0',
      useMiniTpg     => '0',
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal remTrigFreq     : Slv32Array(3 downto 0);
   signal remTrigDropFreq : Slv32Array(3 downto 0);

   signal locTrigFreq     : Slv32Array(3 downto 0);
   signal locTrigDropFreq : Slv32Array(3 downto 0);

   signal refClkFreq : Slv32Array(1 downto 0);

   signal txReset   : sl;
   signal txClkFreq : slv(31 downto 0);

   signal rxReset   : sl;
   signal rxClkFreq : slv(31 downto 0);

begin

   GEN_TRIG_FREQ :
   for i in 3 downto 0 generate

      U_remTrigFreq : entity work.SyncTrigRate
         generic map (
            TPD_G          => TPD_G,
            COMMON_CLK_G   => true,
            ONE_SHOT_G     => false,
            REF_CLK_FREQ_G => AXIL_CLK_FREQ_G)
         port map (
            -- Trigger Input (locClk domain)
            trigIn      => remTrig(i),
            -- Trigger Rate Output (locClk domain)
            trigRateOut => remTrigFreq(i),
            -- Clocks
            locClk      => axilClk,
            refClk      => axilClk);

      U_remTrigDropFreq : entity work.SyncTrigRate
         generic map (
            TPD_G          => TPD_G,
            COMMON_CLK_G   => true,
            ONE_SHOT_G     => false,
            REF_CLK_FREQ_G => AXIL_CLK_FREQ_G)
         port map (
            -- Trigger Input (locClk domain)
            trigIn      => remTrigDrop(i),
            -- Trigger Rate Output (locClk domain)
            trigRateOut => remTrigDropFreq(i),
            -- Clocks
            locClk      => axilClk,
            refClk      => axilClk);

      U_locTrigFreq : entity work.SyncTrigRate
         generic map (
            TPD_G          => TPD_G,
            COMMON_CLK_G   => true,
            ONE_SHOT_G     => false,
            REF_CLK_FREQ_G => AXIL_CLK_FREQ_G)
         port map (
            -- Trigger Input (locClk domain)
            trigIn      => locTrig(i),
            -- Trigger Rate Output (locClk domain)
            trigRateOut => locTrigFreq(i),
            -- Clocks
            locClk      => axilClk,
            refClk      => axilClk);

      U_locTrigDropFreq : entity work.SyncTrigRate
         generic map (
            TPD_G          => TPD_G,
            COMMON_CLK_G   => true,
            ONE_SHOT_G     => false,
            REF_CLK_FREQ_G => AXIL_CLK_FREQ_G)
         port map (
            -- Trigger Input (locClk domain)
            trigIn      => locTrigDrop(i),
            -- Trigger Rate Output (locClk domain)
            trigRateOut => locTrigDropFreq(i),
            -- Clocks
            locClk      => axilClk,
            refClk      => axilClk);

   end generate GEN_TRIG_FREQ;

   GEN_REFCLK_FREQ :
   for i in 1 downto 0 generate
      U_refClk : entity work.SyncClockFreq
         generic map (
            TPD_G          => TPD_G,
            REF_CLK_FREQ_G => AXIL_CLK_FREQ_G,
            REFRESH_RATE_G => 1.0,
            CNT_WIDTH_G    => 32)
         port map (
            -- Frequency Measurement (locClk domain)
            freqOut => refClkFreq(i),
            -- Clocks
            clkIn   => refClk(i),
            locClk  => axilClk,
            refClk  => axilClk);
   end generate GEN_REFCLK_FREQ;

   Sync_txRst : entity work.Synchronizer
      generic map (
         TPD_G => TPD_G)
      port map (
         clk     => axilClk,
         dataIn  => txRst,
         dataOut => txReset);

   U_txClkFreq : entity work.SyncClockFreq
      generic map (
         TPD_G          => TPD_G,
         REF_CLK_FREQ_G => AXIL_CLK_FREQ_G,
         REFRESH_RATE_G => 1.0,
         CNT_WIDTH_G    => 32)
      port map (
         -- Frequency Measurement (locClk domain)
         freqOut => txClkFreq,
         -- Clocks
         clkIn   => txClk,
         locClk  => axilClk,
         refClk  => axilClk);

   Sync_rxRst : entity work.Synchronizer
      generic map (
         TPD_G => TPD_G)
      port map (
         clk     => axilClk,
         dataIn  => rxRst,
         dataOut => rxReset);

   U_rxClkFreq : entity work.SyncClockFreq
      generic map (
         TPD_G          => TPD_G,
         REF_CLK_FREQ_G => AXIL_CLK_FREQ_G,
         REFRESH_RATE_G => 1.0,
         CNT_WIDTH_G    => 32)
      port map (
         -- Frequency Measurement (locClk domain)
         freqOut => rxClkFreq,
         -- Clocks
         clkIn   => rxClk,
         locClk  => axilClk,
         refClk  => axilClk);

   --------------------- 
   -- AXI Lite Interface
   --------------------- 
   comb : process (axilReadMaster, axilRst, axilWriteMaster, locTrig,
                   locTrigDrop, locTrigDropFreq, locTrigFreq, mmcmLocked, r,
                   refClkFreq, refRst, remTrig, remTrigDrop, remTrigDropFreq,
                   remTrigFreq, rxClkFreq, rxReset, txClkFreq, txReset) is
      variable v      : RegType;
      variable regCon : AxiLiteEndPointType;
   begin
      -- Latch the current value
      v := r;

      -- Reset the strobes
      v.rxUserRst := '0';
      v.txUserRst := '0';
      v.mmcmRst   := '0';
      v.cntRst    := '0';

      -- Check for counter reset
      if (r.cntRst = '1') then
         v.locTrigCnt     := (others => (others => '0'));
         v.locTrigDropCnt := (others => (others => '0'));
         v.remTrigCnt     := (others => (others => '0'));
         v.remTrigDropCnt := (others => (others => '0'));
      else
         for i in 3 downto 0 loop
            if locTrig(i) = '1' then
               v.locTrigCnt(i) := r.locTrigCnt(i) + 1;
            end if;
            if locTrigDrop(i) = '1' then
               v.locTrigDropCnt(i) := r.locTrigDropCnt(i) + 1;
            end if;
            if remTrig(i) = '1' then
               v.remTrigCnt(i) := r.remTrigCnt(i) + 1;
            end if;
            if remTrigDrop(i) = '1' then
               v.remTrigDropCnt(i) := r.remTrigDropCnt(i) + 1;
            end if;

         end loop;
      end if;

      -- Determine the transaction type
      axiSlaveWaitTxn(regCon, axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave);

      -- Map the read registers
      axiSlaveRegister (regCon, x"00", 0, v.mmcmRst);
      axiSlaveRegisterR(regCon, x"04", 0, mmcmLocked);
      axiSlaveRegisterR(regCon, x"08", 0, refRst);
      axiSlaveRegister (regCon, x"0C", 0, v.loopback);

      axiSlaveRegister (regCon, x"10", 0, v.useMiniTpg);
      axiSlaveRegister (regCon, x"14", 0, v.rxUserRst);
      axiSlaveRegister (regCon, x"18", 0, v.txUserRst);

      axiSlaveRegisterR(regCon, x"20", 0, txReset);
      axiSlaveRegisterR(regCon, x"24", 0, rxReset);

      axiSlaveRegisterR(regCon, x"30", 0, refClkFreq(0));
      axiSlaveRegisterR(regCon, x"34", 0, refClkFreq(1));
      axiSlaveRegisterR(regCon, x"38", 0, txClkFreq);
      axiSlaveRegisterR(regCon, x"3C", 0, rxClkFreq);

      axiSlaveRegisterR(regCon, x"40", 0, locTrigFreq(0));
      axiSlaveRegisterR(regCon, x"44", 0, locTrigFreq(1));
      axiSlaveRegisterR(regCon, x"48", 0, locTrigFreq(2));
      axiSlaveRegisterR(regCon, x"4C", 0, locTrigFreq(3));

      axiSlaveRegisterR(regCon, x"50", 0, remTrigFreq(0));
      axiSlaveRegisterR(regCon, x"54", 0, remTrigFreq(1));
      axiSlaveRegisterR(regCon, x"58", 0, remTrigFreq(2));
      axiSlaveRegisterR(regCon, x"5C", 0, remTrigFreq(3));

      axiSlaveRegisterR(regCon, x"60", 0, locTrigDropFreq(0));
      axiSlaveRegisterR(regCon, x"64", 0, locTrigDropFreq(1));
      axiSlaveRegisterR(regCon, x"68", 0, locTrigDropFreq(2));
      axiSlaveRegisterR(regCon, x"6C", 0, locTrigDropFreq(3));

      axiSlaveRegisterR(regCon, x"70", 0, remTrigDropFreq(0));
      axiSlaveRegisterR(regCon, x"74", 0, remTrigDropFreq(1));
      axiSlaveRegisterR(regCon, x"78", 0, remTrigDropFreq(2));
      axiSlaveRegisterR(regCon, x"7C", 0, remTrigDropFreq(3));

      axiSlaveRegisterR(regCon, x"80", 0, r.locTrigCnt(0));
      axiSlaveRegisterR(regCon, x"80", 16, r.locTrigCnt(1));
      axiSlaveRegisterR(regCon, x"84", 0, r.locTrigCnt(2));
      axiSlaveRegisterR(regCon, x"84", 16, r.locTrigCnt(3));

      axiSlaveRegisterR(regCon, x"88", 0, r.remTrigCnt(0));
      axiSlaveRegisterR(regCon, x"88", 16, r.remTrigCnt(1));
      axiSlaveRegisterR(regCon, x"8C", 0, r.remTrigCnt(2));
      axiSlaveRegisterR(regCon, x"8C", 16, r.remTrigCnt(3));

      axiSlaveRegisterR(regCon, x"90", 0, r.locTrigDropCnt(0));
      axiSlaveRegisterR(regCon, x"90", 16, r.locTrigDropCnt(1));
      axiSlaveRegisterR(regCon, x"94", 0, r.locTrigDropCnt(2));
      axiSlaveRegisterR(regCon, x"94", 16, r.locTrigDropCnt(3));

      axiSlaveRegisterR(regCon, x"98", 0, r.remTrigDropCnt(0));
      axiSlaveRegisterR(regCon, x"98", 16, r.remTrigDropCnt(1));
      axiSlaveRegisterR(regCon, x"9C", 0, r.remTrigDropCnt(2));
      axiSlaveRegisterR(regCon, x"9C", 16, r.remTrigDropCnt(3));

      axiSlaveRegister (regCon, x"FC", 0, v.cntRst);

      -- Closeout the transaction
      axiSlaveDefault(regCon, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_DECERR_C);

      -- Outputs
      axilWriteSlave <= r.axilWriteSlave;
      axilReadSlave  <= r.axilReadSlave;
      useMiniTpg     <= r.useMiniTpg;
      loopback       <= r.loopback;

      -- Reset
      if (axilRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

   end process comb;

   seq : process (axilClk) is
   begin
      if (rising_edge(axilClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

   U_mmcmRst : entity work.PwrUpRst
      generic map (
         TPD_G         => TPD_G,
         SIM_SPEEDUP_G => SIMULATION_G,
         DURATION_G    => 156000000)
      port map (
         arst   => r.mmcmRst,
         clk    => axilClk,
         rstOut => mmcmRst);

   U_rxUserRst : entity work.PwrUpRst
      generic map (
         TPD_G         => TPD_G,
         SIM_SPEEDUP_G => SIMULATION_G,
         DURATION_G    => 125000000)
      port map (
         arst   => r.rxUserRst,
         clk    => axilClk,
         rstOut => rxUserRst);

   U_txUserRst : entity work.PwrUpRst
      generic map (
         TPD_G         => TPD_G,
         SIM_SPEEDUP_G => SIMULATION_G,
         DURATION_G    => 125000000)
      port map (
         arst   => r.txUserRst,
         clk    => axilClk,
         rstOut => txUserRst);

end rtl;
