-------------------------------------------------------------------------------
-- File       : CLinkWrapper.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description: 
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
use IEEE.std_logic_unsigned.all;
use IEEE.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.AxiLitePkg.all;
use work.SsiPkg.all;
use work.Pgp3Pkg.all;
use work.ClinkPkg.all;

entity CLinkWrapper is
   generic (
      TPD_G            : time                 := 1 ns;
      CHAN_COUNT_G     : integer range 1 to 2 := 1;
      AXIL_BASE_ADDR_G : slv(31 downto 0));
   port (
      -- Clink Ports
      cbl0Half0P      : inout slv(4 downto 0);  --  2,  4,  5,  6, 3
      cbl0Half0M      : inout slv(4 downto 0);  -- 15, 17, 18, 19 16
      cbl0Half1P      : inout slv(4 downto 0);  --  8, 10, 11, 12,  9
      cbl0Half1M      : inout slv(4 downto 0);  -- 21, 23, 24, 25, 22
      cbl0SerP        : out   sl;               -- 20
      cbl0SerM        : out   sl;               -- 7
      cbl1Half0P      : inout slv(4 downto 0);  --  2,  4,  5,  6, 3
      cbl1Half0M      : inout slv(4 downto 0);  -- 15, 17, 18, 19 16
      cbl1Half1P      : inout slv(4 downto 0);  --  8, 10, 11, 12,  9
      cbl1Half1M      : inout slv(4 downto 0);  -- 21, 23, 24, 25, 22
      cbl1SerP        : out   sl;               -- 20
      cbl1SerM        : out   sl;               -- 7
      -- LEDs
      ledRed          : out   slv(1 downto 0);
      ledGrn          : out   slv(1 downto 0);
      ledBlu          : out   slv(1 downto 0);
      -- Stable Reference IDELAY Clock and Reset
      refClk200MHz    : in    sl;
      refRst200MHz    : in    sl;
      -- Camera Control Bits
      camCtrl         : in    Slv4Array(1 downto 0);
      -- Camera Data Interface
      dataMasters     : out   AxiStreamMasterArray(1 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
      dataSlaves      : in    AxiStreamSlaveArray(1 downto 0);
      -- UART Interface
      rxUartMasters   : in    AxiStreamMasterArray(1 downto 0);
      rxUartSlaves    : out   AxiStreamSlaveArray(1 downto 0)  := (others => AXI_STREAM_SLAVE_FORCE_C);
      txUartMasters   : out   AxiStreamMasterArray(1 downto 0) := (others => AXI_STREAM_MASTER_INIT_C);
      txUartSlaves    : in    AxiStreamSlaveArray(1 downto 0);
      -- Axi-Lite Interface
      axilClk         : in    sl;
      axilRst         : in    sl;
      axilReadMaster  : in    AxiLiteReadMasterType;
      axilReadSlave   : out   AxiLiteReadSlaveType;
      axilWriteMaster : in    AxiLiteWriteMasterType;
      axilWriteSlave  : out   AxiLiteWriteSlaveType);
end CLinkWrapper;

architecture mapping of CLinkWrapper is

   constant AXIS_128_C : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 16, tDestBits => 0);
   constant AXIS_32_C  : AxiStreamConfigType := ssiAxiStreamConfig(dataBytes => 4, tDestBits => 0);

   signal txMasterA : AxiStreamMasterArray(CHAN_COUNT_G-1 downto 0);
   signal txSlaveA  : AxiStreamSlaveArray(CHAN_COUNT_G-1 downto 0);

   signal txMasterB : AxiStreamMasterArray(CHAN_COUNT_G-1 downto 0);
   signal txSlaveB  : AxiStreamSlaveArray(CHAN_COUNT_G-1 downto 0);

   signal txMasterC : AxiStreamMasterArray(CHAN_COUNT_G-1 downto 0);
   signal txSlaveC  : AxiStreamSlaveArray(CHAN_COUNT_G-1 downto 0);

   signal camStatus : ClChanStatusArray(1 downto 0);

begin

   U_ClinkTop : entity work.ClinkTop
      generic map (
         TPD_G              => TPD_G,
         CHAN_COUNT_G       => CHAN_COUNT_G,
         UART_READY_EN_G    => true,
         COMMON_AXIL_CLK_G  => true,
         COMMON_DATA_CLK_G  => true,
         DATA_AXIS_CONFIG_G => AXIS_128_C,
         UART_AXIS_CONFIG_G => PGP3_AXIS_CONFIG_C,
         AXIL_BASE_ADDR_G   => AXIL_BASE_ADDR_G)
      port map (
         -- Clink Ports
         cbl0Half0P      => cbl0Half0P,
         cbl0Half0M      => cbl0Half0M,
         cbl0Half1P      => cbl0Half1P,
         cbl0Half1M      => cbl0Half1M,
         cbl0SerP        => cbl0SerP,
         cbl0SerM        => cbl0SerM,
         cbl1Half0P      => cbl1Half0P,
         cbl1Half0M      => cbl1Half0M,
         cbl1Half1P      => cbl1Half1P,
         cbl1Half1M      => cbl1Half1M,
         cbl1SerP        => cbl1SerP,
         cbl1SerM        => cbl1SerM,
         -- Delay clock and reset, 200Mhz
         dlyClk          => refClk200MHz,
         dlyRst          => refRst200MHz,
         -- System clock and reset, > 100 Mhz
         sysClk          => axilClk,
         sysRst          => axilRst,
         -- Camera Control Bits & status, async
         camCtrl         => camCtrl(CHAN_COUNT_G-1 downto 0),
         camStatus       => camStatus,
         -- Camera data
         dataClk         => axilClk,
         dataRst         => axilRst,
         dataMasters     => txMasterA(CHAN_COUNT_G-1 downto 0),
         dataSlaves      => txSlaveA(CHAN_COUNT_G-1 downto 0),
         -- UART data
         uartClk         => axilClk,
         uartRst         => axilRst,
         sUartMasters    => rxUartMasters(CHAN_COUNT_G-1 downto 0),
         sUartSlaves     => rxUartSlaves(CHAN_COUNT_G-1 downto 0),
         mUartMasters    => txUartMasters(CHAN_COUNT_G-1 downto 0),
         mUartSlaves     => txUartSlaves(CHAN_COUNT_G-1 downto 0),
         -- Axi-Lite Interface
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMaster,
         axilReadSlave   => axilReadSlave,
         axilWriteMaster => axilWriteMaster,
         axilWriteSlave  => axilWriteSlave);

   GEN_VEC :
   for i in (CHAN_COUNT_G-1) downto 0 generate

      U_DataFifoA : entity work.AxiStreamFifoV2
         generic map (
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 12,
            FIFO_PAUSE_THRESH_G => 500,
            SLAVE_AXI_CONFIG_G  => AXIS_128_C,
            MASTER_AXI_CONFIG_G => AXIS_32_C)
         port map (
            sAxisClk    => axilClk,
            sAxisRst    => axilRst,
            sAxisMaster => txMasterA(i),
            sAxisSlave  => txSlaveA(i),
            mAxisClk    => axilClk,
            mAxisRst    => axilRst,
            mAxisMaster => txMasterB(i),
            mAxisSlave  => txSlaveB(i));

      -- Force 32-bit alignment
      process(txMasterB, txSlaveC)
      begin
         txMasterC(i)       <= txMasterB(i);
         txMasterC(i).tKeep <= (others => '1');
         txSlaveB(i)        <= txSlaveC(i);
      end process;

      U_DataFifoB : entity work.AxiStreamFifoV2
         generic map (
            TPD_G               => TPD_G,
            SLAVE_READY_EN_G    => true,
            GEN_SYNC_FIFO_G     => true,
            FIFO_ADDR_WIDTH_G   => 9,
            FIFO_PAUSE_THRESH_G => 500,
            SLAVE_AXI_CONFIG_G  => AXIS_32_C,
            MASTER_AXI_CONFIG_G => PGP3_AXIS_CONFIG_C)
         port map (
            sAxisClk    => axilClk,
            sAxisRst    => axilRst,
            sAxisMaster => txMasterC(i),
            sAxisSlave  => txSlaveC(i),
            mAxisClk    => axilClk,
            mAxisRst    => axilRst,
            mAxisMaster => dataMasters(i),
            mAxisSlave  => dataSlaves(i));

   end generate GEN_VEC;

   ----------------
   -- Misc. Signals
   ----------------
   process (camStatus)
   begin

      if camStatus(0).running = '1' then
         ledRed(0) <= '1';
         ledGrn(0) <= '0';
         ledBlu(0) <= '1';
      else
         ledRed(0) <= '0';
         ledGrn(0) <= '1';
         ledBlu(0) <= '1';
      end if;

      if CHAN_COUNT_G = 1 then
         ledRed(1) <= '1';
         ledGrn(1) <= '1';
         ledBlu(1) <= '1';
      elsif camStatus(1).running = '1' then
         ledRed(1) <= '1';
         ledGrn(1) <= '0';
         ledBlu(1) <= '1';
      else
         ledRed(1) <= '0';
         ledGrn(1) <= '1';
         ledBlu(1) <= '1';
      end if;

   end process;

end mapping;
