-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingMsgAxiRingBuffer.vhd
-- Author     : 
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2014-05-02
-- Last update: 2016-02-28
-- Platform   : Vivado 2013.3
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
--
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.TimingPkg.all;

entity TimingMsgAxiRingBuffer is
   generic (
      -- General Configurations
      TPD_G            : time                        := 1 ns;
      BRAM_EN_G        : boolean                     := true;
      REG_EN_G         : boolean                     := true;
      RAM_ADDR_WIDTH_G : positive range 1 to (2**24) := 10;
      VECTOR_SIZE_G    : integer );

   port (
      -- Timing Message interface
      timingClk       : in sl;
      timingRst       : in sl;
      timingMessage       : in slv(VECTOR_SIZE_G-1 downto 0);
      timingMessageStrobe : in sl;

      -- Axi Lite interface for readout
      axilClk         : in  sl;
      axilRst         : in  sl;
      axilReadMaster  : in  AxiLiteReadMasterType  := AXI_LITE_READ_MASTER_INIT_C;
      axilReadSlave   : out AxiLiteReadSlaveType;
      axilWriteMaster : in  AxiLiteWriteMasterType := AXI_LITE_WRITE_MASTER_INIT_C;
      axilWriteSlave  : out AxiLiteWriteSlaveType);

end TimingMsgAxiRingBuffer;

architecture rtl of TimingMsgAxiRingBuffer is

   signal axisMaster : AxiStreamMasterType;

begin

   -- Convert to AxiStream. Easiest way to chunk the timing message into 32 bit segments
   TimingMsgToAxiStream_1 : entity work.TimingMsgToAxiStream
      generic map (
         TPD_G          => TPD_G,
         COMMON_CLOCK_G => true,
         SHIFT_SIZE_G   => 32,
         AXIS_CONFIG_G  => ssiAxiStreamConfig(4),
         VECTOR_SIZE_G  => VECTOR_SIZE_G )
      port map (
         timingClk       => timingClk,
         timingRst       => timingRst,
         timingMessage       => timingMessage,
         timingMessageStrobe => timingMessageStrobe,
         axisClk         => timingClk,
         axisRst         => timingRst,
         axisMaster      => axisMaster);

      -- Pipe into AxiRingBuffer
      AxiLiteRingBuffer_1 : entity work.AxiLiteRingBuffer
      generic map (
         TPD_G            => TPD_G,
         BRAM_EN_G        => BRAM_EN_G,
         REG_EN_G         => REG_EN_G,
         DATA_WIDTH_G     => 32,
         RAM_ADDR_WIDTH_G => RAM_ADDR_WIDTH_G)
      port map (
         dataClk         => timingClk,
         dataRst         => timingRst,
         dataValid       => axisMaster.tvalid,
         dataValue       => axisMaster.tdata(31 downto 0),
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilReadMaster  => axilReadMaster,
         axilReadSlave   => axilReadSlave,
         axilWriteMaster => axilWriteMaster,
         axilWriteSlave  => axilWriteSlave);

end rtl;
