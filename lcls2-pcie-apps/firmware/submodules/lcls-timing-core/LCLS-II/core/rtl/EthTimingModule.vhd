-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EthTimingModule.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-02-13
-- Last update: 2018-02-26
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
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.TimingPkg.all;

entity EthTimingModule is

   generic (TPD_G             : time                := 1 ns;
            STREAM_L1_G       : boolean             := false;
            BUILD_ILA_G       : boolean             := false;
            ETHMSG_AXIS_CFG_G : AxiStreamConfigType := AXI_STREAM_CONFIG_INIT_C);
   port (
      -- Interface to GT
      timingClk      : in  sl;
      timingRst      : in  sl;
      timingStrobe   : in  sl;
      timingStream   : in  TimingStreamType;
      --
      ethClk         : in  sl;
      ethRst         : in  sl;
      ibEthMsgMaster : in  AxiStreamMasterType;
      ibEthMsgSlave  : out AxiStreamSlaveType;
      obEthMsgMaster : out AxiStreamMasterType;
      obEthMsgSlave  : in  AxiStreamSlaveType);

end entity EthTimingModule;

architecture rtl of EthTimingModule is

   constant ETH_WORD_SZ : integer := 8*ETHMSG_AXIS_CFG_G.TDATA_BYTES_C;
   constant ETH_WORDS   : integer := wordCount(TIMING_STREAM_BITS_C, ETH_WORD_SZ);
   constant ETH_REM     : integer := TIMING_STREAM_BITS_C/8 mod 16;
   constant ETH_TKEEP : slv(15 downto 0) := ite(ETH_REM > 0,
                                                (slvZero(16-ETH_REM) & slvOne(ETH_REM)),
                                                slvOne(16));
   type TmoType is record
      tmo : slv(29 downto 0);
   end record;
   constant TMO_INIT_C : TmoType := (
      tmo => (others => '1'));

   signal t   : TmoType := TMO_INIT_C;
   signal tin : TmoType;

   type RegType is record
      valid  : slv(ETH_WORDS-2 downto 0);
      data   : slv(TIMING_STREAM_BITS_C-1-ETH_WORD_SZ downto 0);
      master : AxiStreamMasterType;
   end record;

   constant REG_INIT_C : RegType := (
      valid  => (others => '0'),
      data   => (others => '0'),
      master => AXI_STREAM_MASTER_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal intObEthMsgSlave : AxiStreamSlaveType;
   signal tmo_s            : sl;

   component ila_0
      port (clk    : in sl;
            probe0 : in slv(255 downto 0));
   end component;

   signal intObEthMsgMaster : AxiStreamMasterType := AXI_STREAM_MASTER_INIT_C;

begin

   GEN_ILA : if BUILD_ILA_G generate
      U_ILA : ila_0
         port map (clk                  => timingClk,
                   probe0(0)            => timingRst,
                   probe0(1)            => tmo_s,
                   probe0(2)            => r.master.tValid,
                   probe0(3)            => r.master.tLast,
                   probe0(4)            => intObEthMsgSlave.tReady,
                   probe0(5)            => timingStrobe,
                   probe0(255 downto 6) => (others => '0'));

      U_ILA_ETH : ila_0
         port map (clk                   => ethClk,
                   probe0(0)             => ethRst,
                   probe0(1)             => obEthMsgSlave.tReady,
                   probe0(2)             => ibEthMsgMaster.tValid,
                   probe0(3)             => ibEthMsgMaster.tLast,
                   probe0(4)             => intObEthMsgMaster.tValid,
                   probe0(5)             => intObEthMsgMaster.tLast,
                   probe0(35 downto 6)   => t.tmo,
                   probe0(255 downto 36) => (others => '0'));
   end generate;

   ibEthMsgSlave  <= AXI_STREAM_SLAVE_FORCE_C;
   obEthMsgMaster <= intObEthMsgMaster;

   GEN_STREAM : if STREAM_L1_G generate
      U_TMO : entity work.Synchronizer
         port map (clk     => timingClk,
                   dataIn  => t.tmo(t.tmo'left),
                   dataOut => tmo_s);

      SynchronizerFifo_EthMsg : entity work.AxiStreamFifoV2
         generic map (TPD_G               => TPD_G,
                      SLAVE_AXI_CONFIG_G  => ETHMSG_AXIS_CFG_G,
                      MASTER_AXI_CONFIG_G => ETHMSG_AXIS_CFG_G)
         port map (sAxisClk    => timingClk,
                   sAxisRst    => timingRst,
                   sAxisMaster => r.master,
                   sAxisSlave  => intObEthMsgSlave,
                   mAxisClk    => ethClk,
                   mAxisRst    => ethRst,
                   mAxisMaster => intObEthMsgMaster,
                   mAxisSlave  => obEthMsgSlave);

      tmo : process(ethRst, ibEthMsgMaster, t) is
         variable v : TmoType;
      begin
         v := t;

         if ibEthMsgMaster.tValid = '1' then
            v.tmo := (others => '0');
         elsif t.tmo(t.tmo'left) = '0' then
            v.tmo := t.tmo + 1;
         end if;

         if ethRst = '1' then
            v := TMO_INIT_C;
         end if;

         tin <= v;
      end process tmo;

      seq_tmo : process (ethClk) is
      begin
         if rising_edge(ethClk) then
            t <= tin;
         end if;
      end process seq_tmo;

      comb : process(intObEthMsgSlave, r, timingRst, timingStream,
                     timingStrobe, tmo_s) is
         variable v : RegType;
      begin
         v := r;

         if intObEthMsgSlave.tReady = '1' then
            v.valid                                := '0' & r.valid(r.valid'left downto 1);
            v.data                                 := toSlv(0, ETH_WORD_SZ) & r.data(r.data'left downto ETH_WORD_SZ);
            v.master.tData(ETH_WORD_SZ-1 downto 0) := r.data(ETH_WORD_SZ-1 downto 0);
            v.master.tValid                        := r.valid(0);
            ssiSetUserSof(ETHMSG_AXIS_CFG_G, v.master, '0');
            if r.valid(1) = '1' then
               v.master.tLast := '0';
               v.master.tKeep := resize(x"FFFF",AXI_STREAM_MAX_TKEEP_WIDTH_C);
            else
               v.master.tLast := '1';
               v.master.tKeep := resize(ETH_TKEEP,AXI_STREAM_MAX_TKEEP_WIDTH_C);
               ssiSetUserEofe(ETHMSG_AXIS_CFG_G, v.master, '0');
            end if;
         end if;

         if timingStrobe = '1' and r.valid(0) = '0' and tmo_s = '0' then
            v.valid                                := (others => '1');
            v.data                                 := toSlv(timingStream)(TIMING_STREAM_BITS_C-1 downto ETH_WORD_SZ);
            v.master.tData(ETH_WORD_SZ-1 downto 0) := toSlv(timingStream)(ETH_WORD_SZ-1 downto 0);
            v.master.tValid                        := '1';
            v.master.tLast                         := '0';
            v.master.tKeep                         := resize(x"FFFF",AXI_STREAM_MAX_TKEEP_WIDTH_C);
            ssiSetUserSof(ETHMSG_AXIS_CFG_G, v.master, '1');
         end if;

         if timingRst = '1' then
            v := REG_INIT_C;
         end if;

         rin <= v;
      end process;

      seq : process(timingClk) is
      begin
         if rising_edge(timingClk) then
            r <= rin;
         end if;
      end process;
   end generate;

end rtl;
