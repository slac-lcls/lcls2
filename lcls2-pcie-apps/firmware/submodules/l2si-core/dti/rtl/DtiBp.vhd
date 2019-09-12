-------------------------------------------------------------------------------
-- File       : DtiBp.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-04
-- Last update: 2017-12-11
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.DtiPkg.all;
use work.TimingPkg.all;
use work.EventPkg.all;

library unisim;
use unisim.vcomponents.all;

entity DtiBp is
   generic ( TPD_G : time := 1 ns );
   port (
      ----------------------
      -- Top Level Interface
      ----------------------
      ref125MHzClk    : in  sl;
      ref125MHzRst    : in  sl;
      rxFull          : in  Slv16Array(0 downto 0);
      bpPeriod        : in  slv(7 downto 0);
      status          : out DtiBpLinkStatusType;
      monClk          : out slv(1 downto 0);
      --
      timingClk       : in  sl;
      timingRst       : in  sl;
      timingHdr       : in  TimingHeaderType;
      ----------------
      -- Core Ports --
      ----------------
      -- Backplane Ports
      bpClkIn         : in  sl;
      bpClkOut        : out sl;
      bpBusRxP        : in  sl;
      bpBusRxN        : in  sl;
      bpBusTxP        : out sl;
      bpBusTxN        : out sl );
end DtiBp;

architecture mapping of DtiBp is

   type RegType is record
     master : AxiStreamMasterType;
     full   : slv(31 downto 0);
     sent   : slv(47 downto 0);
     ticks  : slv( 7 downto 0);
     cnt    : slv( 7 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
     master => AXI_STREAM_MASTER_INIT_C,
     full   => (others=>'0'),
     sent   => (others=>'0'),
     ticks  => (others=>'0'),
     cnt    => (others=>'0') );

   signal r     : RegType := REG_INIT_C;
   signal rin   : RegType;
   
   signal bp100MHzClk : sl;
   signal bp100MHzRst : sl;
   signal bp250MHzClk : sl;
   signal bp250MHzRst : sl;
   signal bp500MHzClk : sl;
   signal bp500MHzRst : sl;
   signal bpPllLocked : sl;
   signal bpClkBuf    : sl;
   signal bpRefClk    : sl;

   signal txHeaderT   : slv(7 downto 0);
   signal txHeader    : slv(7 downto 0);
   signal txStrobe    : sl;
   signal sbpPeriod   : slv(7 downto 0);
   
   constant BP_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(4);
   
   signal bpSlave  : AxiStreamSlaveType;

begin

  monClk(0) <= bpRefClk;
  monClk(1) <= bp100MHzClk;
  
   U_IBUF : IBUF
      port map ( I => bpClkIn,
                 O => bpClkBuf);

   U_BUFG : BUFG
     port map ( I => bpClkBuf,
                O => bpRefClk );

   ------------------------------
   -- Backplane Clocks and Resets
   ------------------------------
   U_Clk : entity work.XpmBpClk
      generic map (
         TPD_G         => TPD_G,
         MPS_SLOT_G    => false,
         PHASE_500M_G  => 45.0 )  -- Correct skew btw clk500,clk250
      port map (
         -- Stable Clock and Reset 
         refClk       => bpRefClk,
         refRst       => '0',
         -- BP Clocks and Resets
         mps100MHzClk => bp100MHzClk,
         mps100MHzRst => bp100MHzRst,
         mps250MHzClk => bp250MHzClk,
         mps250MHzRst => bp250MHzRst,
         mps500MHzClk => bp500MHzClk,
         mps500MHzRst => bp500MHzRst,
         mpsPllLocked => bpPllLocked,
         ----------------
         -- Core Ports --
         ----------------   
         -- Backplane BP Ports
         mpsClkOut    => bpClkOut);

   U_SaltUltraScale : entity work.SaltUltraScale
     generic map (
       TPD_G               => TPD_G,
       TX_ENABLE_G         => true,
       RX_ENABLE_G         => false,
       COMMON_TX_CLK_G     => false,
       COMMON_RX_CLK_G     => false,
       SLAVE_AXI_CONFIG_G  => BP_CONFIG_C,
       MASTER_AXI_CONFIG_G => BP_CONFIG_C )
--       DEBUG_G             => true )
     port map (
       -- TX Serial Stream
       txP           => bpBusTxP,
       txN           => bpBusTxN,
       -- RX Serial Stream
       rxP           => '1',
       rxN           => '0',
       -- Reference Signals
       clk125MHz     => bp100MHzClk,
       rst125MHz     => bp100MHzRst,
       clk312MHz     => bp250MHzClk,
       clk625MHz     => bp500MHzClk,
       iDelayCtrlRdy => '1',
       linkUp        => status.linkUp,
       -- Slave Port
       sAxisClk      => ref125MHzClk,
       sAxisRst      => ref125MHzRst,
       sAxisMaster   => r.master,
       sAxisSlave    => bpSlave,
       -- Master Port
       mAxisClk      => ref125MHzClk,
       mAxisRst      => ref125MHzRst,
       mAxisMaster   => open,
       mAxisSlave    => AXI_STREAM_SLAVE_FORCE_C );

   U_IBUFDS : IBUFDS
     generic map (
       DIFF_TERM => true)
     port map(
       I  => bpBusRxP,
       IB => bpBusRxN,
       O  => open);

   U_TxHeader : entity work.SynchronizerVector
     generic map ( WIDTH_G => 8 )
     port map ( clk     => ref125MHzClk,
                dataIn  => txHeaderT,
                dataOut => txHeader );

   U_TxStrobe : entity work.SynchronizerOneShot
     port map ( clk     => ref125MHzClk,
                dataIn  => timingHdr.strobe,
                dataOut => txStrobe );

   tstrobe : process ( timingClk ) is
   begin
     if rising_edge(timingClk) then
       txHeaderT <= timingHdr.pulseId(7 downto 0);
     end if;
   end process;

   U_BpPeriod : entity work.SynchronizerVector
     generic map ( WIDTH_G => 8 )
     port map ( clk     => ref125MHzClk,
                dataIn  => bpPeriod,
                dataOut => sbpPeriod );
  
   comb: process ( r, ref125MHzRst, rxFull, txHeader, txStrobe, bpSlave, sbpPeriod ) is
     variable v : RegType;
   begin
     v := r;

     v.master.tValid := '0';
     v.ticks         := r.ticks + 1;
     v.cnt           := r.cnt + 1;

     if r.cnt = sbpPeriod then
       v.master.tValid := '1';
       v.cnt := (others=>'0');
     end if;
     v.master.tLast  := '1';
     v.master.tData(31 downto 0) := txHeader & r.ticks & rxFull(0);
     ssiSetUserSof(BP_CONFIG_C, v.master, '1');

     if r.master.tValid = '1' and bpSlave.tReady = '1' then
       v.sent := r.sent + 1;
     end if;

     if txStrobe = '1' then
       v.ticks := (others=>'0');
     end if;
     
     if ref125MHzRst='1' then
       v := REG_INIT_C;
     end if;
     
     rin <= v;

     status.obSent <= resize(r.sent,status.obSent'length);
   end process;
   
   seq: process (ref125MHzClk) is
   begin
     if rising_edge(ref125MHzClk) then
       r <= rin;
     end if;
   end process;

end mapping;
