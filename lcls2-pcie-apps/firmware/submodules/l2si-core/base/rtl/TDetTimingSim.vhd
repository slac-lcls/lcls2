-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TDetTiming.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-08
-- Last update: 2018-11-02
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 XPM Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the
-- top-level directory of this distribution and at:
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
-- No part of 'LCLS2 XPM Core', including this file,
-- may be copied, modified, propagated, or distributed except according to
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------
 
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.TimingPkg.all;
use work.TimingExtnPkg.all;
use work.EventPkg.all;
use work.TDetPkg.all;
use work.XpmPkg.all;

entity TDetTimingSim is
   generic (
      TPD_G               : time             := 1 ns;
      NDET_G              : natural          := 1;
      AXIL_BASEADDR_G     : slv(31 downto 0) := (others=>'0');
      AXIL_RINGB_G        : boolean          := false );
   port (
      --------------------------------------------
      -- Trigger Interface (Timing clock domain)
      --------------------------------------------
      trigClk          : out sl;
      trigBus          : out TDetTrigArray       (NDET_G-1 downto 0);
      --------------------------------------------
      -- Readout Interface
      --------------------------------------------
      tdetClk          : in  sl;
      tdetRst          : in  sl := '0';
      tdetTiming       : in  TDetTimingArray     (NDET_G-1 downto 0);
      tdetStatus       : out TDetStatusArray     (NDET_G-1 downto 0);
      -- Event stream
      tdetEventMaster  : out AxiStreamMasterArray(NDET_G-1 downto 0);
      tdetEventSlave   : in  AxiStreamSlaveArray (NDET_G-1 downto 0);
      -- Transition stream
      tdetTransMaster  : out AxiStreamMasterArray(NDET_G-1 downto 0);
      tdetTransSlave   : in  AxiStreamSlaveArray (NDET_G-1 downto 0);
      ----------------
      -- Core Ports --
      ----------------   
      -- LCLS Timing Ports
      timingRefClkOut  : out sl;
      timingRecClkOut  : out sl;
      timingBusOut     : out TimingBusType );
end TDetTimingSim;

architecture mapping of TDetTimingSim is

   signal timingRefClk   : sl;
   signal timingRefClkDiv: sl;
   signal rxControl      : TimingPhyControlType;
   signal rxStatus       : TimingPhyStatusType := (
     locked       => '1',
     resetDone    => '1',
     bufferByDone => '1',
     bufferByErr  => '0' );
   signal rxCdrStable    : sl;
   signal rxUsrClk       : sl;
   signal rxData         : slv(15 downto 0);
   signal rxDataK        : slv(1 downto 0);
   signal rxDispErr      : slv(1 downto 0);
   signal rxDecErr       : slv(1 downto 0);
   signal rxOutClk       : sl;
   signal rxRst          : sl;
   signal txStatus       : TimingPhyStatusType := (
     locked       => '1',
     resetDone    => '1',
     bufferByDone => '1',
     bufferByErr  => '0' );
   signal txUsrClk       : sl;
   signal txUsrRst       : sl;
   signal txOutClk       : sl;
   signal loopback       : slv(2 downto 0);
   signal timingPhy      : TimingPhyType;
   signal timingBus      : TimingBusType := TIMING_BUS_INIT_C;

   signal appTimingHdr   : TimingHeaderType; -- aligned
   signal appExptBus     : ExptBusType;      -- aligned
   signal timingHdr      : TimingHeaderType; -- prompt
   signal triggerBus     : ExptBusType;      -- prompt
   signal fullOut        : slv(NPartitions-1 downto 0);

   signal pdata          : XpmPartitionDataArray(NDET_G-1 downto 0);
   signal pdataV         : slv                  (NDET_G-1 downto 0);
   signal tdetMaster     : AxiStreamMasterArray (NDET_G-1 downto 0);
   signal tdetSlave      : AxiStreamSlaveArray  (NDET_G-1 downto 0);
   signal hdrOut         : EventHeaderArray     (NDET_G-1 downto 0);

   signal xpmClk       : slv       (NDSLinks-1 downto 0);
   signal xpmDsRxData  : Slv16Array(NDSLinks-1 downto 0) := (others=>x"0000");
   signal xpmDsRxDataK : Slv2Array (NDSLinks-1 downto 0) := (others=>"00");
   signal xpmDsTxData  : Slv16Array(NDSLinks-1 downto 0);
   signal xpmDsTxDataK : Slv2Array (NDSLinks-1 downto 0);
   
begin

   trigClk         <= rxOutClk;
   timingRecClkOut <= rxOutClk;
   timingBusOut    <= timingBus;
   rxRst           <= tdetRst;
   
   process is
   begin
     rxOutClk <= '1';
     wait for 2.69 ns;
     rxOutClk <= '0';
     wait for 2.69 ns;
   end process;

   --  Need timingBus with extn
   process is
     variable pulseId : slv(63 downto 0) := (others=>'0');
     variable anatag  : slv(23 downto 0) := (others=>'0');
     variable pmsg    : XpmPartitionMsgType  := XPM_PARTITION_MSG_INIT_C;
     variable pdat    : XpmPartitionDataType := XPM_PARTITION_DATA_INIT_C;
     variable frame   : slv( 3 downto 0) := (others=>'0');
   begin
     timingBus <= TIMING_BUS_INIT_C;
     timingBus.valid     <= '1';
     timingBus.modesel   <= '1';
     timingBus.message.version    <= toSlv(1,16);

     wait for 1 us;
     wait until rxOutClk = '0';

     for j in 0 to 99 loop
       timingBus.message.pulseId    <= pulseId;
       timingBus.message.timeStamp  <= pulseId;
       timingBus.strobe    <= '1';

       timingBus.extn.partitionWord(0)(0)  <= '0'; -- No L0
       timingBus.extn.partitionWord(0)(15) <= '1'; -- No Msg
       if frame = x"0" then
         pmsg.hdr     := MSG_DELAY_PWORD;
         pmsg.payload := toSlv(3,8);
         pmsg.anatag  := anatag;
         anatag       := anatag+1;
         timingBus.extn.partitionWord(0) <= toSlv(pmsg);
         timingBus.extnValid             <= '1';
       elsif frame = x"8" then
         pmsg.l0tag   := anatag(4 downto 0);
         pmsg.hdr     := toSlv(2,8);
         pmsg.payload := x"FE";
         pmsg.anatag  := anatag;
         anatag       := anatag+1;
         timingBus.extn.partitionWord(0) <= toSlv(pmsg);
       elsif frame = x"F" then
         pdat.l0a    := '1';
         pdat.l0tag  := anatag(4 downto 0);
         pdat.anatag := anatag;
         anatag      := anatag+1;
         timingBus.extn.partitionWord(0) <= toSlv(pdat);
       end if;
       if frame /= x"F" then
         frame := frame+1;
       end if;

       pulseId := pulseId+1;
       for i in 0 to 199 loop
         wait until rxOutClk = '1';
         wait until rxOutClk = '0';
         timingBus.strobe <= '0';
       end loop;
     end loop;
     wait;
   end process;
   
   timingHdr          <= toTimingHeader (timingBus);
   triggerBus.message <= ExptMessageType(timingBus.extn);
   triggerBus.valid   <= timingBus.extnValid;

   
   U_Realign : entity work.EventRealign
     generic map ( TF_DELAY_G => toSlv(5,7) )
     port map ( clk            => rxOutClk,
                rst            => rxRst,
                timingI        => timingHdr,
                exptBusI       => triggerBus,
                timingO        => appTimingHdr,
                exptBusO       => appExptBus,
                delay          => open );

   GEN_DET : for i in 0 to NDET_G-1 generate
     
     trigBus(i).l0a   <= pdata (i).l0a;
     trigBus(i).l0tag <= pdata (i).l0tag;
     trigBus(i).valid <= pdataV(i);

     U_HeaderCache : entity work.EventHeaderCache
       port map ( rst             => rxRst,
                  --  Cache Input
                  wrclk           => rxOutClk,
                  -- configuration
                  enable          => tdetTiming(i).enable,
--                cacheenable     : in  sl := '1';     -- caches headers --
                  partition       => tdetTiming(i).partition,
                  -- event input
                  timing_prompt   => timingHdr,
                  expt_prompt     => triggerBus,
                  timing_aligned  => appTimingHdr,
                  expt_aligned    => appExptBus,
                  -- trigger output
                  pdata           => pdata     (i),
                  pdataV          => pdataV    (i),
                  -- status
                  cntL0           => tdetStatus(i).cntL0,
                  cntL1A          => tdetStatus(i).cntL1A,
                  cntL1R          => tdetStatus(i).cntL1R,
                  cntWrFifo       => tdetStatus(i).cntWrFifo,
                  rstFifo         => open,
                  msgDelay        => tdetStatus(i).msgDelay,
                  cntOflow        => tdetStatus(i).cntOflow,
                  --  Cache Output
                  rdclk           => tdetClk,
                  advance         => tdetSlave (i).tReady,
                  valid           => tdetMaster(i).tValid,
                  pmsg            => tdetMaster(i).tDest(0),
                  cntRdFifo       => tdetStatus(i).cntRdFifo,
                  hdrOut          => hdrOut    (i) );

     tdetMaster(i).tData(8*TDET_AXIS_CONFIG_C.TDATA_BYTES_C-1 downto 0) <= toSlv(hdrOut(i));
     tdetMaster(i).tLast  <= '1';
     tdetMaster(i).tKeep  <= genTKeep(TDET_AXIS_CONFIG_C);

     U_DeMux : entity work.AxiStreamDeMux
       generic map ( NUM_MASTERS_G => 2,
                     TDEST_HIGH_G  => TDET_AXIS_CONFIG_C.TDEST_BITS_C-1 )
       port map ( axisClk         => tdetClk,
                  axisRst         => tdetRst,
                  sAxisMaster     => tdetMaster     (i),
                  sAxisSlave      => tdetSlave      (i),
                  mAxisMasters(0) => tdetEventMaster(i),
                  mAxisMasters(1) => tdetTransMaster(i),
                  mAxisSlaves (0) => tdetEventSlave (i),
                  mAxisSlaves (1) => tdetTransSlave (i) );
     
   end generate;

   fullp : process ( tdetTiming ) is
     variable vfull : slv(NPartitions-1 downto 0);
   begin
     vfull := (others=>'0');
     for i in 0 to NDET_G-1 loop
       if tdetTiming(i).enable='1' and tdetTiming(i).afull='1' then
         vfull(conv_integer(tdetTiming(i).partition)) := '1';
       end if;
     end loop;
     fullOut <= vfull;
   end process fullp;
     
   U_TimingFb : entity work.XpmTimingFb
     generic map ( DEBUG_G => true )
     port map ( clk            => txUsrClk,
                rst            => txUsrRst,
                status         => txStatus,
                pllReset       => rxControl.pllReset,
                phyReset       => rxControl.reset,
                id             => tdetTiming(0).id,
                l1input        => (others=>XPM_L1_INPUT_INIT_C),
                full           => fullOut,
                phy            => timingPhy );

   p_PAddr : process (rxOutClk) is
   begin
     if rising_edge(rxOutClk) then
       if (triggerBus.valid = '1' and timingBus.strobe = '1') then
         for i in 0 to NDET_G-1 loop
           tdetStatus(i).partitionAddr <= triggerBus.message.partitionAddr;
         end loop;
       end if;
     end if;
   end process p_PAddr;

end mapping;
