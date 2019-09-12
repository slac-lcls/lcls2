-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TPGMiniReg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-11-09
-- Last update: 2018-04-20
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
use work.AxiLitePkg.all;
use work.TPGPkg.all;
use work.TPGMiniEdefPkg.all;

entity TPGMiniReg is
   generic (
      TPD_G            : time            := 1 ns;
      NARRAYS_BSA      : integer         := 1;
      USE_WSTRB_G      : boolean         := false);
   port (
      -- PCIe Interface
      irqActive      : in  sl;
      irqEnable      : out sl;
      irqReq         : out sl;
      -- AXI-Lite Interface
      axiReadMaster  : in  AxiLiteReadMasterType;
      axiReadSlave   : out AxiLiteReadSlaveType;
      axiWriteMaster : in  AxiLiteWriteMasterType;
      axiWriteSlave  : out AxiLiteWriteSlaveType;
      -- EVR Interface      
      status         : in  TPGStatusType;
      config         : out TPGConfigType;
      edefConfig     : out TPGMiniEdefConfigType;
      txReset        : out sl;
      txLoopback     : out slv(2 downto 0);
      txInhibit      : out sl;
      -- Clock and Reset
      axiClk         : in  sl;
      axiRst         : in  sl);
end TPGMiniReg;

architecture rtl of TPGMiniReg is

   constant CLKSEL     : integer := 0;
   constant BASE_CNTL  : integer := 1;
   constant PULSEIDL   : integer := 2;
   constant PULSEIDU   : integer := 3;
   constant TSTAMPL    : integer := 4;
   constant TSTAMPU    : integer := 5;
   constant FIXEDRATE0 : integer := 6;   -- 10 registers
   constant FIXEDRATE9 : integer := 15;
   constant RATERELOAD : integer := 16;
   constant HIST_CNTL  : integer := 17;
   constant FWVERSION  : integer := 18;
   constant RESOURCES  : integer := 19;
   constant BSACMPLL   : integer := 20;
   constant BSACMPLU   : integer := 21;
   constant PULSEIDLW  : integer := 22;
   constant PULSEIDUW  : integer := 23;
   constant TSTAMPLW   : integer := 24;
   constant TSTAMPUW   : integer := 25;
   constant TXRST      : integer := 26;
   constant INTVLRST   : integer := 27;
   constant PIDSET     : integer := 28;
   constant TSSET      : integer := 29;
   constant BSA1_EDEF  : integer := 30;
   constant BSA1_INIT  : integer := 31;
   constant BSACTRL    : integer := 127;
   constant BSADEF     : integer := 128;  -- 128 registers
   constant BSADEF_END : integer := BSADEF+4*NARRAYS_BSA;
   constant BSASTATUS  : integer := 256;  -- 64 registers
   constant BSASTATUS_END  : integer := 319;
   constant CNTPLL     : integer := 320;
   constant CNT186M    : integer := 321;
   constant CNTSYNCE   : integer := 322;
   constant CNTINTVL   : integer := 323;
   constant CNTBRT     : integer := 324;
  
   type RegType is record
                     pulseId           : slv(31 downto 0);
                     timeStamp         : slv(31 downto 0);
                     bsadefRateMode    : Slv2Array (NARRAYS_BSA-1 downto 0);
                     bsadefFixedRate   : Slv4Array (NARRAYS_BSA-1 downto 0);
                     bsadefACRate      : Slv3Array (NARRAYS_BSA-1 downto 0);
                     bsadefACTSMask    : Slv6Array (NARRAYS_BSA-1 downto 0);
                     bsadefSeqSel      : Slv5Array (NARRAYS_BSA-1 downto 0);
                     bsadefSeqBit      : Slv4Array (NARRAYS_BSA-1 downto 0);
                     bsadefDestMode    : Slv2Array (NARRAYS_BSA-1 downto 0);
                     bsadefDestInclM   : Slv16Array(NARRAYS_BSA-1 downto 0);
                     bsadefDestExclM   : Slv16Array(NARRAYS_BSA-1 downto 0);
                     bsadefNToAvg      : Slv13Array(NARRAYS_BSA-1 downto 0);
                     bsadefAvgToWr     : Slv16Array(NARRAYS_BSA-1 downto 0);
                     bsadefMaxSevr     : Slv2Array (NARRAYS_BSA-1 downto 0);
                     bsaComplete       : slv(63 downto 0);
                     bsaCompleteQ      : sl;
                     countUpdate       : sl;
                     FixedRateDivisors : Slv20Array(9 downto 0);
                     config            : TPGConfigType;
                     edefConfig        : TPGMiniEdefConfigType;
                     txReset           : sl;
                     txLoopback        : slv( 2 downto 0);
                     txInhibit         : sl;
                     rdData            : slv(31 downto 0);
                     axiReadSlave      : AxiLiteReadSlaveType;
                     axiWriteSlave     : AxiLiteWriteSlaveType;
   end record RegType;
   
   constant REG_INIT_C : RegType := (
     pulseId           => (others=>'0'),
     timeStamp         => (others=>'0'),
     bsadefRateMode    => (others=>(others=>'0')),
     bsadefFixedRate   => (others=>(others=>'0')),
     bsadefACRate      => (others=>(others=>'0')),
     bsadefACTSMask    => (others=>(others=>'1')),
     bsadefSeqSel      => (others=>(others=>'0')),
     bsadefSeqBit      => (others=>(others=>'0')),
     bsadefDestMode    => (others=>(others=>'0')),
     bsadefDestInclM   => (others=>(others=>'1')),
     bsadefDestExclM   => (others=>(others=>'1')),
     bsadefNToAvg      => (others=>toSlv(1,13)),
     bsadefAvgToWr     => (others=>toSlv(100,16)),
     bsadefMaxSevr     => (others=>(others=>'1')),
     bsaComplete       => (others=>'0'),
     bsaCompleteQ      => '0',
     countUpdate       => '0',
     FixedRateDivisors => TPG_CONFIG_INIT_C.FixedRateDivisors,
     config            => TPG_CONFIG_INIT_C,
     edefConfig        => TPG_MINI_EDEF_CONFIG_INIT_C,
     txReset           => '0',
     txLoopback        => "000",
     txInhibit         => '0',
     rdData            => (others=>'0'),
     axiReadSlave      => AXI_LITE_READ_SLAVE_INIT_C,
     axiWriteSlave     => AXI_LITE_WRITE_SLAVE_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   assert NARRAYS_BSA < 33
     report "NARRAYS_BSA (" & integer'image(NARRAYS_BSA) & ") limit is 32 for TPGMini" severity failure;
   
   -------------------------------
   -- Configuration Register
   -------------------------------  
   comb : process (axiReadMaster, axiRst, axiWriteMaster, irqActive, r, status) is
      variable v            : RegType;
      variable axiStatus    : AxiLiteStatusType;
      variable axiWriteResp : slv(1 downto 0);
      variable axiReadResp  : slv(1 downto 0);
      variable rdPntr       : natural;
      variable wrPntr       : natural;
      variable iseq         : natural;
      variable ichn         : natural;
      variable regWrData    : slv(31 downto 0);
      variable tmpRdData    : slv(31 downto 0);
      variable regAddr      : slv(31 downto 2);
      variable bsaClear     : slv(63 downto 0);
   begin
      -- Latch the current value
      v := r;

      -- Calculate the address pointers
      wrPntr := conv_integer(axiWriteMaster.awaddr(10 downto 2));
      rdPntr := conv_integer(axiReadMaster .araddr(10 downto 2));
      regWrData := axiWriteMaster.wdata;
      
      -- Reset strobing signals
      v.config.pulseIdWrEn   := '0';
      v.config.timeStampWrEn := '0';
      v.config.intervalRst   := '0';
      v.edefConfig.wrEn      := '0';
      v.txReset              := '0';
      bsaClear               := (others=>'0');
      
      -- Determine the transaction type

      -----------------------------
      -- AXI-Lite Write Logic
      -----------------------------      
      axiSlaveWaitWriteTxn(axiWriteMaster,v.axiWriteSlave,axiStatus.writeEnable);

      if (axiStatus.writeEnable = '1') then
        regAddr := axiWriteMaster.awaddr(regAddr'range);
        -- Check for alignment
        if axiWriteMaster.awaddr(1 downto 0) = "00" then
          -- Address is aligned
          axiWriteResp          := AXI_RESP_OK_C;

          case wrPntr is
            when TXRST     => v.txReset                        := regWrData(0);
            when INTVLRST  => v.config.intervalRst             := regWrData(0);
            when PIDSET    => v.config.pulseIdWrEn             := regWrData(0);
            when TSSET     => v.config.timeStampWrEn           := regWrData(0);
            when CLKSEL    => v.config.txPolarity              := regWrData(1);
--                              v.txReset                        := regWrData(0);
                              v.txLoopback                     := regWrData(4 downto 2);
                              v.txInhibit                      := regWrData(5);
            when BASE_CNTL => v.config.baseDivisor             := regWrData(15 downto 0);
            when PULSEIDLW => v.config.pulseId(31 downto  0)   := regWrData;
            when PULSEIDUW => v.config.pulseId(63 downto 32)   := regWrData;
--                              v.config.pulseIdWrEn             := '1';
            when TSTAMPLW  => v.config.timeStamp(31 downto  0) := regWrData;
            when TSTAMPUW  => v.config.timeStamp(63 downto 32) := regWrData;
--                              v.config.timeStampWrEn           := '1'                 ;
            when FIXEDRATE0+0 => v.FixedRateDivisors(0)        := regWrData(19 downto 0);
            when FIXEDRATE0+1 => v.FixedRateDivisors(1)        := regWrData(19 downto 0);
            when FIXEDRATE0+2 => v.FixedRateDivisors(2)        := regWrData(19 downto 0);
            when FIXEDRATE0+3 => v.FixedRateDivisors(3)        := regWrData(19 downto 0);
            when FIXEDRATE0+4 => v.FixedRateDivisors(4)        := regWrData(19 downto 0);
            when FIXEDRATE0+5 => v.FixedRateDivisors(5)        := regWrData(19 downto 0);
            when FIXEDRATE0+6 => v.FixedRateDivisors(6)        := regWrData(19 downto 0);
            when FIXEDRATE0+7 => v.FixedRateDivisors(7)        := regWrData(19 downto 0);
            when FIXEDRATE0+8 => v.FixedRateDivisors(8)        := regWrData(19 downto 0);
            when FIXEDRATE0+9 => v.FixedRateDivisors(9)        := regWrData(19 downto 0);
            when RATERELOAD => v.config.FixedRateDivisors      := r.FixedRateDivisors;
            when BSACMPLL   => bsaClear(31 downto  0)          := regWrData;
            when BSACMPLU   => bsaClear(63 downto 32)          := regWrData;
            when BSA1_EDEF  => v.edefConfig                    := fromSlv( regWrData, '0' );
            when BSA1_INIT  => v.edefConfig.wrEn               := '1';
            when BSACTRL    => 
              for i in 0 to NARRAYS_BSA-1 loop
                v.config.bsadefv(i).init := regWrData(i);
              end loop;
            when BSADEF to BSADEF_END =>
              iseq               := conv_integer(regAddr(8 downto 4));
              case regAddr(3 downto 2) is
                when "00" => v.bsadefRateMode (iseq) := regWrData( 1 downto  0);
                             v.bsaDefFixedRate(iseq) := regWrData( 5 downto  2);
                             v.bsaDefACRate   (iseq) := regWrData( 8 downto  6);
                             v.bsaDefACTSMask (iseq) := regWrData(14 downto  9);
                             v.bsaDefSeqSel   (iseq) := regWrData(19 downto 15);
                             v.bsaDefSeqBit   (iseq) := regWrData(23 downto 20);
                             v.bsaDefDestMode (iseq) := regWrData(25 downto 24);
                when "01" => v.bsadefDestInclM(iseq) := regWrData(15 downto  0);
                             v.bsaDefDestExclM(iseq) := regWrData(31 downto 16);
                when "10" => v.bsadefNToAvg   (iseq) := regWrData(12 downto  0);
                             v.bsaDefMaxSevr  (iseq) := regWrData(15 downto 14);
                             v.bsaDefAvgToWr  (iseq) := regWrData(31 downto 16);
                when others => null;
              end case;
            when CNTINTVL   => v.config.interval    := regWrData;
--                               v.config.intervalRst := '1';
            when others  => axiWriteResp := AXI_RESP_DECERR_C;
          end case;
        else
          axiWriteResp := AXI_RESP_DECERR_C;
        end if;
        -- Send AXI response
        axiSlaveWriteResponse(v.axiWriteSlave, axiWriteResp);
      end if;
      
      -----------------------------
      -- AXI-Lite Read Logic
      -----------------------------      

      axiSlaveWaitReadTxn(axiReadMaster,v.axiReadSlave,axiStatus.readEnable);

      if (axiStatus.readEnable = '1') then
        regAddr   := axiReadMaster.araddr(regAddr'range);
        tmpRdData := (others=>'0');
        -- Check for alignment
        if axiReadMaster.araddr(1 downto 0) = "00" then
          -- Address is aligned
          axiReadResp           := AXI_RESP_OK_C;
          -- Decode the read address
          case rdPntr is
            when TXRST      => null;
            when INTVLRST   => null;
            when PIDSET     => null;
            when TSSET      => null;
            when CLKSEL     => tmpRdData(1)           := r.config.txPolarity;
                               tmpRdData(4 downto 2)  := r.txLoopback;
                               tmpRdData(5)           := r.txInhibit;
            when BASE_CNTL  => tmpRdData(15 downto 0) := r.config.baseDivisor;
            when PULSEIDL   => tmpRdData              := status.pulseId(31 downto  0);
                               v.pulseId              := status.pulseId(63 downto 32);
            when PULSEIDU   => tmpRdData              := r.pulseId;
            when TSTAMPL    => tmpRdData              := status.timeStamp(31 downto  0);
                               v.timeStamp            := status.timeStamp(63 downto 32);
            when TSTAMPU    => tmpRdData              := r.timeStamp;
            when PULSEIDLW  => tmpRdData              := r.config.pulseId(31 downto  0);
            when PULSEIDUW  => tmpRdData              := r.config.pulseId(63 downto 32);
            when TSTAMPLW   => tmpRdData              := r.config.timeStamp(31 downto  0);
            when TSTAMPUW   => tmpRdData              := r.config.timeStamp(63 downto 32);
            when FIXEDRATE0+0 => tmpRdData(19 downto 0) := r.FixedRateDivisors(0);
            when FIXEDRATE0+1 => tmpRdData(19 downto 0) := r.FixedRateDivisors(1);
            when FIXEDRATE0+2 => tmpRdData(19 downto 0) := r.FixedRateDivisors(2);
            when FIXEDRATE0+3 => tmpRdData(19 downto 0) := r.FixedRateDivisors(3);
            when FIXEDRATE0+4 => tmpRdData(19 downto 0) := r.FixedRateDivisors(4);
            when FIXEDRATE0+5 => tmpRdData(19 downto 0) := r.FixedRateDivisors(5);
            when FIXEDRATE0+6 => tmpRdData(19 downto 0) := r.FixedRateDivisors(6);
            when FIXEDRATE0+7 => tmpRdData(19 downto 0) := r.FixedRateDivisors(7);
            when FIXEDRATE0+8 => tmpRdData(19 downto 0) := r.FixedRateDivisors(8);
            when FIXEDRATE0+9 => tmpRdData(19 downto 0) := r.FixedRateDivisors(9);
            when RATERELOAD   => null;
--  Version found in common registers
--            when FWVERSION  => tmpRdData                      := FPGA_VERSION_C;
            when RESOURCES  => tmpRdData              := status.nallowseq &
                                                         status.seqaddrlen &
                                                         status.narraysbsa &
                                                         status.nexptseq &
                                                         status.nbeamseq;
            when BSACMPLU   => tmpRdData              := r.bsaComplete(63 downto 32);
            when BSACMPLL   => tmpRdData              := r.bsaComplete(31 downto  0);
            when BSA1_EDEF  => tmpRdData              := toSlv( r.edefConfig );
            when BSA1_INIT  => tmpRdData              := (others => '0');
            when BSACTRL    =>
              for i in 0 to NARRAYS_BSA-1 loop
                tmpRdData(i) := r.config.bsadefv(i).init;
              end loop;
            when BSADEF to BSADEF_END =>
              iseq      := conv_integer(regAddr(8 downto 4));
              case regAddr(3 downto 2) is
                when "00" => tmpRdData( 1 downto  0) := r.bsadefRateMode (iseq);
                             tmpRdData( 5 downto  2) := r.bsaDefFixedRate(iseq);
                             tmpRdData( 8 downto  6) := r.bsaDefACRate   (iseq);
                             tmpRdData(14 downto  9) := r.bsaDefACTSMask (iseq); 
                             tmpRdData(19 downto 15) := r.bsaDefSeqSel   (iseq); 
                             tmpRdData(23 downto 20) := r.bsaDefSeqBit   (iseq); 
                             tmpRdData(25 downto 24) := r.bsaDefDestMode (iseq); 
                when "01" => tmpRdData(15 downto  0) := r.bsadefDestInclM(iseq); 
                             tmpRdData(31 downto 16) := r.bsaDefDestExclM(iseq); 
                when "10" => tmpRdData(12 downto  0) := r.bsadefNToAvg   (iseq); 
                             tmpRdData(15 downto 14) := r.bsaDefMaxSevr  (iseq); 
                             tmpRdData(31 downto 16) := r.bsaDefAvgToWr  (iseq); 
                when others => null;
              end case;
            when BSASTATUS to BSASTATUS_END =>
              iseq      := conv_integer(regAddr(7 downto 2));
              tmpRdData := status.bsastatus(iseq);
            when CNTPLL     => tmpRdData := status.pllChanged;
            when CNT186M    => tmpRdData := status.count186M;
            when CNTSYNCE   => tmpRdData := status.countSyncE;
            when CNTINTVL   => tmpRdData := r.config.interval;
            when CNTBRT     => tmpRdData := status.countBRT;
            when others     => axiReadResp := AXI_RESP_DECERR_C;
          end case;
          v.axiReadSlave.rdata := tmpRdData;
          -- Send AXI response
          axiSlaveReadResponse(v.axiReadSlave, axiReadResp);
        else
          axiSlaveReadResponse(v.axiReadSlave, AXI_RESP_DECERR_C);
        end if;
      end if;

      for i in 0 to NARRAYS_BSA-1 loop
        if (v.config.bsadefv(i).init ='1' and r.config.bsadefv(i).init='0') then
          v.config.bsadefv(i).rateSel(12 downto 11) := r.bsadefRateMode(i);
          case r.bsadefRateMode(i) is
            when "00"   => v.config.bsadefv(i).rateSel( 3 downto 0) := r.bsadefFixedRate(i);
            when "01"   => v.config.bsadefv(i).rateSel( 8 downto 3) := r.bsadefACTSMask (i);
                           v.config.bsadefv(i).rateSel( 2 downto 0) := r.bsadefACRate   (i);
            when others => v.config.bsadefv(i).rateSel(10 downto 6) := r.bsadefSeqSel   (i);
                           v.config.bsadefv(i).rateSel( 3 downto 0) := r.bsadefSeqBit   (i);
          end case;
          v.config.bsadefv(i).destSel(17 downto 16) := r.bsadefDestMode(i);
          case r.bsadefDestMode(i) is
            when "00"   => v.config.bsadefv(i).destSel(15 downto 0) := r.bsadefDestInclM(i);
            when "01"   => v.config.bsadefv(i).destSel(15 downto 0) := r.bsadefDestExclM(i);
            when others => null;
          end case;
          v.config.bsadefv(i).nToAvg  := r.bsadefNToAvg (i);
          v.config.bsadefv(i).avgToWr := r.bsadefAvgToWr(i);
          v.config.bsadefv(i).maxSevr := r.bsadefMaxSevr(i);
        end if;
      end loop;
      
      -- Misc. Mapping and Logic
      v.bsaComplete := (r.bsaComplete and not bsaClear) or status.bsaComplete;
      if allBits(r.bsaComplete,'0') then
        v.bsaCompleteQ := '0';
      else
        v.bsaCompleteQ := '1';
      end if;

      -- Synchronous Reset
      --if axiRst = '1' then
      --  v := REG_INIT_C;
      --end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs
      axiWriteSlave   <= r.axiWriteSlave;
      axiReadSlave    <= r.axiReadSlave;
      config          <= r.config;
      edefConfig      <= r.edefConfig;
      txReset         <= r.txReset;
      txLoopback      <= r.txLoopback;
      txInhibit       <= r.txInhibit;
      
      irqEnable       <= '0';
      irqReq          <= '0';
   end process comb;

   seq : process (axiClk) is
   begin
      if rising_edge(axiClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end rtl;
