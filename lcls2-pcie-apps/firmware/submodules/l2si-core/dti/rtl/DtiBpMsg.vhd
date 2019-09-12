-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : AmcCarrierMpsSalt.vhd
-- Author     : Larry Ruckman  <ruckman@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-04
-- Last update: 2016-09-29
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
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
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.AmcCarrierPkg.all;

library unisim;
use unisim.vcomponents.all;

entity AmcCarrierMpsSalt is
   generic (
      TPD_G            : time            := 1 ns;
      APP_TYPE_G       : AppType         := APP_NULL_TYPE_C;
      AXI_ERROR_RESP_G : slv(1 downto 0) := AXI_RESP_DECERR_C;
      MPS_SLOT_G       : boolean         := false);
   port (
      -- SALT Reference clocks
      mps125MHzClk      : in  sl;
      mps125MHzRst      : in  sl;
      mps312MHzClk      : in  sl;
      mps312MHzRst      : in  sl;
      mps625MHzClk      : in  sl;
      mps625MHzRst      : in  sl;
      mpsPllLocked      : in  sl;
      -- AXI-Lite Interface
      axilClk           : in  sl;
      axilRst           : in  sl;
      axilReadMaster    : in  AxiLiteReadMasterType;
      axilReadSlave     : out AxiLiteReadSlaveType;
      axilWriteMaster   : in  AxiLiteWriteMasterType;
      axilWriteSlave    : out AxiLiteWriteSlaveType;
      -- MPS Interface
      mpsIbMaster       : in  AxiStreamMasterType;
      mpsIbSlave        : out AxiStreamSlaveType;
      -- MPS/BP_MSG configuration/status signals
      appId             : in  slv(15 downto 0);
      mpsEnable         : out sl;
      mpsTestMode       : out sl;
      bpMsgEnable       : out sl;
      bpMsgTestMode     : out sl;
      timeStrbRate      : in  slv(31 downto 0);
      diagnosticClkFreq : in  slv(31 downto 0);
      ----------------------
      -- Top Level Interface
      ----------------------
      -- MPS Interface
      mpsObMasters      : out AxiStreamMasterArray(14 downto 0);
      mpsObSlaves       : in  AxiStreamSlaveArray(14 downto 0);
      ----------------
      -- Core Ports --
      ----------------
      -- Backplane MPS Ports
      mpsBusRxP         : in  slv(14 downto 1);
      mpsBusRxN         : in  slv(14 downto 1);
      mpsTxP            : out sl;
      mpsTxN            : out sl);
end AmcCarrierMpsSalt;

architecture mapping of AmcCarrierMpsSalt is

   constant STATUS_SIZE_C     : natural                := 15;
   constant MPS_CHANNELS_C    : natural range 0 to 32  := getMpsChCnt(APP_TYPE_G);
   constant MPS_THRESHOLD_C   : natural range 0 to 256 := getMpsThresholdCnt(APP_TYPE_G);
   constant BP_MSG_CHANNELS_C : natural range 0 to 32  := getBpMsgChCnt(APP_TYPE_G);

   type RegType is record
      mpsEnable      : sl;
      mpsTestMode    : sl;
      bpMsgEnable    : sl;
      bpMsgTestMode  : sl;
      cntRst         : sl;
      rollOverEn     : slv(STATUS_SIZE_C-1 downto 0);
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
   end record;

   constant REG_INIT_C : RegType := (
      mpsEnable      => '0',
      mpsTestMode    => '0',
      bpMsgEnable    => '0',
      bpMsgTestMode  => '0',
      cntRst         => '1',
      rollOverEn     => (others => '0'),
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal iDelayCtrlRdy : sl;
   signal mpsTxLinkUp   : sl;
   signal mpsRxLinkUp   : slv(14 downto 1);
   signal statusOut     : slv(STATUS_SIZE_C-1 downto 0);
   signal cntOut        : SlVectorArray(STATUS_SIZE_C-1 downto 0, 31 downto 0);

begin

   APP_SLOT : if (MPS_SLOT_G = false) generate
      
      mpsRxLinkUp  <= (others => '0');
      mpsObMasters <= (others => AXI_STREAM_MASTER_INIT_C);

      U_SaltUltraScale : entity work.SaltUltraScale
         generic map (
            TPD_G               => TPD_G,
            TX_ENABLE_G         => true,
            RX_ENABLE_G         => false,
            COMMON_TX_CLK_G     => false,
            COMMON_RX_CLK_G     => false,
            SLAVE_AXI_CONFIG_G  => MPS_CONFIG_C,
            MASTER_AXI_CONFIG_G => MPS_CONFIG_C)
         port map (
            -- TX Serial Stream
            txP           => mpsTxP,
            txN           => mpsTxN,
            -- RX Serial Stream
            rxP           => mpsBusRxP(1),
            rxN           => mpsBusRxN(1),
            -- Reference Signals
            clk125MHz     => mps125MHzClk,
            rst125MHz     => mps125MHzRst,
            clk312MHz     => mps312MHzClk,
            clk625MHz     => mps625MHzClk,
            iDelayCtrlRdy => iDelayCtrlRdy,
            linkUp        => mpsTxLinkUp,
            -- Slave Port
            sAxisClk      => axilClk,
            sAxisRst      => axilRst,
            sAxisMaster   => mpsIbMaster,
            sAxisSlave    => mpsIbSlave,
            -- Master Port
            mAxisClk      => axilClk,
            mAxisRst      => axilRst,
            mAxisMaster   => open,
            mAxisSlave    => AXI_STREAM_SLAVE_FORCE_C);            

      GEN_VEC :
      for i in 14 downto 2 generate
         U_IBUFDS : IBUFDS
            generic map (
               DIFF_TERM => true) 
            port map(
               I  => mpsBusRxP(i),
               IB => mpsBusRxN(i),
               O  => open);
      end generate GEN_VEC;
      
   end generate;

   MPS_SLOT : if (MPS_SLOT_G = true) generate
      
      U_SaltDelayCtrl : entity work.SaltDelayCtrl
         generic map (
            TPD_G           => TPD_G,
            SIM_DEVICE_G    => "ULTRASCALE",
            IODELAY_GROUP_G => "MPS_IODELAY_GRP")
         port map (
            iDelayCtrlRdy => iDelayCtrlRdy,
            refClk        => mps625MHzClk,
            refRst        => mps625MHzRst);     

      mpsTxLinkUp     <= '0';
      mpsObMasters(0) <= mpsIbMaster;
      mpsIbSlave      <= mpsObSlaves(0);

      U_OBUFDS : OBUFDS
         port map (
            I  => '0',
            O  => mpsTxP,
            OB => mpsTxN);      

      GEN_VEC :
      for i in 14 downto 1 generate
         U_SaltUltraScale : entity work.SaltUltraScale
            generic map (
               TPD_G               => TPD_G,
               TX_ENABLE_G         => false,
               RX_ENABLE_G         => true,
               COMMON_TX_CLK_G     => false,
               COMMON_RX_CLK_G     => false,
               SLAVE_AXI_CONFIG_G  => MPS_CONFIG_C,
               MASTER_AXI_CONFIG_G => MPS_CONFIG_C)
            port map (
               -- TX Serial Stream
               txP           => open,
               txN           => open,
               -- RX Serial Stream
               rxP           => mpsBusRxP(i),
               rxN           => mpsBusRxN(i),
               -- Reference Signals
               clk125MHz     => mps125MHzClk,
               rst125MHz     => mps125MHzRst,
               clk312MHz     => mps312MHzClk,
               clk625MHz     => mps625MHzClk,
               iDelayCtrlRdy => iDelayCtrlRdy,
               linkUp        => mpsRxLinkUp(i),
               -- Slave Port
               sAxisClk      => axilClk,
               sAxisRst      => axilRst,
               sAxisMaster   => AXI_STREAM_MASTER_INIT_C,
               sAxisSlave    => open,
               -- Master Port
               mAxisClk      => axilClk,
               mAxisRst      => axilRst,
               mAxisMaster   => mpsObMasters(i),
               mAxisSlave    => mpsObSlaves(i));             
      end generate GEN_VEC;
      
   end generate;

   comb : process (appId, axilReadMaster, axilRst, axilWriteMaster, cntOut, diagnosticClkFreq,
                   mpsPllLocked, r, statusOut, timeStrbRate) is
      variable v      : RegType;
      variable regCon : AxiLiteEndPointType;
   begin
      -- Latch the current value
      v := r;

      -- Reset strobe signals
      v.cntRst := '0';

      -- Determine the transaction type
      axiSlaveWaitTxn(regCon, axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave);

      -- Map the read registers
      axiSlaveRegisterR(regCon, x"000", 0, muxSlVectorArray(cntOut, 0));
      axiSlaveRegisterR(regCon, x"004", 0, muxSlVectorArray(cntOut, 1));
      axiSlaveRegisterR(regCon, x"008", 0, muxSlVectorArray(cntOut, 2));
      axiSlaveRegisterR(regCon, x"00C", 0, muxSlVectorArray(cntOut, 3));
      axiSlaveRegisterR(regCon, x"010", 0, muxSlVectorArray(cntOut, 4));
      axiSlaveRegisterR(regCon, x"014", 0, muxSlVectorArray(cntOut, 5));
      axiSlaveRegisterR(regCon, x"018", 0, muxSlVectorArray(cntOut, 6));
      axiSlaveRegisterR(regCon, x"01C", 0, muxSlVectorArray(cntOut, 7));
      axiSlaveRegisterR(regCon, x"020", 0, muxSlVectorArray(cntOut, 8));
      axiSlaveRegisterR(regCon, x"024", 0, muxSlVectorArray(cntOut, 9));
      axiSlaveRegisterR(regCon, x"028", 0, muxSlVectorArray(cntOut, 10));
      axiSlaveRegisterR(regCon, x"02C", 0, muxSlVectorArray(cntOut, 11));
      axiSlaveRegisterR(regCon, x"030", 0, muxSlVectorArray(cntOut, 12));
      axiSlaveRegisterR(regCon, x"034", 0, muxSlVectorArray(cntOut, 13));
      axiSlaveRegisterR(regCon, x"038", 0, muxSlVectorArray(cntOut, 14));
      axiSlaveRegisterR(regCon, x"100", 0, statusOut);
      axiSlaveRegisterR(regCon, x"104", 0, ite(MPS_SLOT_G, x"00000001", x"00000000"));
      axiSlaveRegisterR(regCon, x"108", 0, timeStrbRate);
      axiSlaveRegisterR(regCon, x"10C", 0, diagnosticClkFreq);
      axiSlaveRegisterR(regCon, x"110", 0, APP_TYPE_G);
      axiSlaveRegisterR(regCon, x"114", 0, toSlv(MPS_CHANNELS_C, 32));
      axiSlaveRegisterR(regCon, x"118", 0, toSlv(MPS_THRESHOLD_C, 32));
      axiSlaveRegisterR(regCon, x"11C", 0, toSlv(BP_MSG_CHANNELS_C, 32));
      axiSlaveRegisterR(regCon, x"120", 0, appId);
      axiSlaveRegisterR(regCon, x"124", 0, mpsPllLocked);

      -- Map the write registers
      axiSlaveRegister(regCon, x"200", 0, v.mpsEnable);
      axiSlaveRegister(regCon, x"204", 0, v.mpsTestMode);
      axiSlaveRegister(regCon, x"208", 0, v.bpMsgEnable);
      axiSlaveRegister(regCon, x"20C", 0, v.bpMsgTestMode);
      axiSlaveRegister(regCon, x"3F0", 0, v.rollOverEn);
      axiSlaveRegister(regCon, x"3FC", 0, v.cntRst);

      -- Closeout the transaction
      axiSlaveDefault(regCon, v.axilWriteSlave, v.axilReadSlave, AXI_ERROR_RESP_G);

      -- Synchronous Reset
      if (axilRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs
      axilWriteSlave <= r.axilWriteSlave;
      axilReadSlave  <= r.axilReadSlave;
      mpsEnable      <= r.mpsEnable;
      mpsTestMode    <= r.mpsTestMode;
      bpMsgEnable    <= r.bpMsgEnable;
      bpMsgTestMode  <= r.bpMsgTestMode;
      
   end process comb;

   seq : process (axilClk) is
   begin
      if (rising_edge(axilClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

   SyncStatusVec_Inst : entity work.SyncStatusVector
      generic map (
         TPD_G          => TPD_G,
         OUT_POLARITY_G => '1',
         CNT_RST_EDGE_G => true,
         CNT_WIDTH_G    => 32,
         WIDTH_G        => STATUS_SIZE_C)     
      port map (
         -- Input Status bit Signals (wrClk domain)                  
         statusIn(14 downto 1) => mpsRxLinkUp,
         statusIn(0)           => mpsTxLinkUp,
         -- Output Status bit Signals (rdClk domain)           
         statusOut             => statusOut,
         -- Status Bit Counters Signals (rdClk domain) 
         cntRstIn              => r.cntRst,
         rollOverEnIn          => r.rollOverEn,
         cntOut                => cntOut,
         -- Clocks and Reset Ports
         wrClk                 => mps125MHzClk,
         rdClk                 => axilClk);   

end mapping;
