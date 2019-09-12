-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TimingMsgToAxiStream.vhd
-- Author     : Benjamin Reese  <bareese@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-01
-- Last update: 2016-03-11
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Convert timing messages into an (SSI) AxiStream
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;

use work.TimingPkg.all;

entity TimingMsgToAxiStream is

   generic (
      TPD_G          : time                := 1 ns;
      COMMON_CLOCK_G : boolean             := false;  -- Set true if timingClk=axisClk
      SHIFT_SIZE_G   : integer range 16 to 128;
      AXIS_CONFIG_G  : AxiStreamConfigType := ssiAxiStreamConfig(8, TKEEP_NORMAL_C);
      VECTOR_SIZE_G  : integer );

   port (
      timingClk       : in sl;
      timingRst       : in sl;
      timingMessage       : in slv(VECTOR_SIZE_G-1 downto 0);
      timingMessageStrobe : in sl;

      axisClk    : in  sl;
      axisRst    : in  sl;
      axisMaster : out AxiStreamMasterType;
      axisSlave  : in  AxiStreamSlaveType := AXI_STREAM_SLAVE_FORCE_C);

end entity TimingMsgToAxiStream;

architecture rtl of TimingMsgToAxiStream is

   -------------------------------------------------------------------------------------------------
   -- timingClk Domain
   -------------------------------------------------------------------------------------------------
   constant INT_AXIS_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(SHIFT_SIZE_G/8, AXIS_CONFIG_G.TKEEP_MODE_C);
   constant SHIFT_COUNT_MAX_C : integer             := VECTOR_SIZE_G/SHIFT_SIZE_G;
   constant COUNT_SIZE_C      : integer             := bitSize(SHIFT_COUNT_MAX_C);


   type RegType is record
      ssiMaster : SsiMasterType;
      message       : slv(VECTOR_SIZE_G-1 downto 0);
      active    : sl;
      count     : slv(COUNT_SIZE_C-1 downto 0);
   end record;

   constant REG_INIT_C : RegType := (
      ssiMaster => ssiMasterInit(INT_AXIS_CONFIG_C),
      message       => (others => '0'),
      active    => '0',
      count     => (others => '0'));

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal timingAxisMaster : AxiStreamMasterType;

begin

   comb : process (r, timingMessage, timingMessageStrobe, timingRst) is
      variable v : RegType;
   begin
      v := r;

      v.count := (others => '0');

      if (timingMessageStrobe = '1') then
         v.message    := timingMessage;
         v.active := '1';
      end if;

      v.ssiMaster     := ssiMasterInit(INT_AXIS_CONFIG_C);
      v.ssiMaster.sof := ite(r.count = 0, '1', '0');
      v.ssiMaster.eof := ite(r.count = SHIFT_COUNT_MAX_C, '1', '0');

      if (r.active = '1') then
         v.ssiMaster.data(SHIFT_SIZE_G-1 downto 0) := r.message(SHIFT_SIZE_G-1 downto 0);

         v.ssiMaster.valid := '1';
         v.active          := not v.ssiMaster.eof;
         v.message             := slvZero(SHIFT_SIZE_G) & r.message(VECTOR_SIZE_G-1 downto SHIFT_SIZE_G);
      end if;


      if (timingRst = '1') then
         v := REG_INIT_C;
      end if;

      timingAxisMaster <= ssi2AxisMaster(INT_AXIS_CONFIG_C, r.ssiMaster);

      rin <= v;

   end process comb;

   seq : process (timingClk) is
   begin
      if (rising_edge(timingClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

   AxiStreamFifo_1 : entity work.AxiStreamFifoV2
      generic map (
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => false,
         BRAM_EN_G           => false,
         USE_BUILT_IN_G      => false,
         GEN_SYNC_FIFO_G     => COMMON_CLOCK_G,
         FIFO_ADDR_WIDTH_G   => 4,
         FIFO_FIXED_THRESH_G => true,
--         FIFO_PAUSE_THRESH_G => FIFO_PAUSE_THRESH_G,
         SLAVE_AXI_CONFIG_G  => INT_AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => AXIS_CONFIG_G)
      port map (
         sAxisClk    => timingClk,
         sAxisRst    => timingRst,
         sAxisMaster => timingAxisMaster,
         sAxisSlave  => open,
         sAxisCtrl   => open,
         mAxisClk    => axisClk,
         mAxisRst    => axisRst,
         mAxisMaster => axisMaster,
         mAxisSlave  => axisSlave);

end architecture rtl;

