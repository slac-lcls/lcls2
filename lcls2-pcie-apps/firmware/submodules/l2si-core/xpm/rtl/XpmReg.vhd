-----------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmReg.vhd
-- Author     : Matt Weaver
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-12-14
-- Last update: 2018-08-02
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Software programmable register interface
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
use ieee.numeric_std.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.AmcCarrierPkg.all;  -- ETH_AXIS_CONFIG_C
use work.XpmPkg.all;

entity XpmReg is
   port (
      axilClk          : in  sl;
      axilRst          : in  sl;
      axilUpdate       : out slv(NPartitions-1 downto 0);
      axilWriteMaster  : in  AxiLiteWriteMasterType;  
      axilWriteSlave   : out AxiLiteWriteSlaveType;  
      axilReadMaster   : in  AxiLiteReadMasterType;  
      axilReadSlave    : out AxiLiteReadSlaveType;
      -- Application Debug Interface (sysclk domain)
      ibDebugMaster    : in  AxiStreamMasterType;
      ibDebugSlave     : out AxiStreamSlaveType;
      --
      staClk           : in  sl;
      pllStatus        : in  XpmPllStatusArray(NAmcs-1 downto 0);
      status           : in  XpmStatusType;
      monClk           : in  slv(3 downto 0) := (others=>'0');
      config           : out XpmConfigType;
      dbgChan          : out slv(4 downto 0) );
end XpmReg;

architecture rtl of XpmReg is

  type StateType is (IDLE_S, READING_S);
  
  type RegType is record
    state          : StateType;
    tagSlave       : AxiStreamSlaveType;
    load           : sl;
    config         : XpmConfigType;
    partition      : slv(3 downto 0);
    link           : slv(4 downto 0);
    amc            : slv(0 downto 0);
    inhibit        : slv(1 downto 0);
    linkCfg        : XpmLinkConfigType;
    linkStat       : XpmLinkStatusType;
    partitionCfg   : XpmPartitionConfigType;
    partitionStat  : XpmPartitionStatusType;
    pllCfg         : XpmPllConfigType;
    pllStat        : XpmPllStatusType;
    inhibitCfg     : XpmInhibitConfigType;
    axilReadSlave  : AxiLiteReadSlaveType;
    axilWriteSlave : AxiLiteWriteSlaveType;
    axilRdEn       : slv(NPartitions-1 downto 0);
    linkDebug      : slv(4 downto 0);
    tagStream      : sl;
    anaWrCount     : Slv32Array(NPartitions-1 downto 0);
  end record RegType;

  constant REG_INIT_C : RegType := (
    state          => IDLE_S,
    tagSlave       => AXI_STREAM_SLAVE_INIT_C,
    load           => '1',
    config         => XPM_CONFIG_INIT_C,
    partition      => (others=>'0'),
    link           => (others=>'0'),
    amc            => (others=>'0'),
    inhibit        => (others=>'0'),
    linkCfg        => XPM_LINK_CONFIG_INIT_C,
    linkStat       => XPM_LINK_STATUS_INIT_C,
    partitionCfg   => XPM_PARTITION_CONFIG_INIT_C,
    partitionStat  => XPM_PARTITION_STATUS_INIT_C,
    pllCfg         => XPM_PLL_CONFIG_INIT_C,
    pllStat        => XPM_PLL_STATUS_INIT_C,
    inhibitCfg     => XPM_INHIBIT_CONFIG_INIT_C,
    axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
    axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C,
    axilRdEn       => (others=>'1'),
    linkDebug      => (others=>'0'),
    tagStream      => '0',
    anaWrCount     => (others=>(others=>'0')));

  constant TAG_AXIS_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(8);
  signal tagMaster : AxiStreamMasterType;
  
  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;

  signal pll_stat    : slv(2*NAmcs-1 downto 0);
  signal pllStat     : slv(2*NAmcs-1 downto 0);
  signal pllCount    : SlVectorArray(2*NAmcs-1 downto 0, 2 downto 0);

  type AnaRdArray is array (natural range<>) of SlVectorArray(0 downto 0,31 downto 0);
  signal anaRdCount  : AnaRdArray(NPartitions-1 downto 0);
  
  signal s    : XpmStatusType;
  signal linkStat, slinkStat  : XpmLinkStatusType;

  signal monClkRate : Slv32Array(3 downto 0);
  signal monClkLock : slv       (3 downto 0);
  signal monClkSlow : slv       (3 downto 0);
  signal monClkFast : slv       (3 downto 0);

  constant DEBUG_C : boolean := true;

  signal p0InhCh  : sl;
  signal p0InhErr : sl;
  signal pInhV    : slv(NPartitions-1 downto 0);
  
  component ila_0
    port ( clk : in sl;
           probe0 : in slv(255 downto 0) );
  end component;
  
begin

  GEN_DBUG : if DEBUG_C generate
    U_ILA : ila_0
      port map ( clk  => axilClk,
                 probe0(0) => pInhV(0),
                 probe0(1) => p0InhCh,
                 probe0(2) => p0InhErr,
                 probe0(66 downto 3) => s.partition(0).l0Select.inhibited,
                 probe0(255 downto 67) => (others=>'0') );
    process (axilClk) is
      variable p0Inh, p0Inhi : slv(15 downto 0);
    begin
      if rising_edge(axilClk) then
        p0Inhi := s.partition(0).l0Select.inhibited(p0Inh'range);
        if p0Inh/=p0Inhi then
          p0InhCh <= '1';
        else
          p0InhCh <= '0';
        end if;
        if p0Inh>p0Inhi then
          p0InhErr <= '1';
        else
          p0InhErr <= '0';
        end if;
        p0Inh := p0Inhi;
      end if;
    end process;
  end generate;
    
  dbgChan        <= r.linkDebug(dbgChan'range);
  config         <= r.config;
  axilReadSlave  <= r.axilReadSlave;
  axilWriteSlave <= r.axilWriteSlave;
  axilUpdate     <= r.axilRdEn;

  GEN_MONCLK : for i in 0 to 3 generate
    U_SYNC : entity work.SyncClockFreq
      generic map ( REF_CLK_FREQ_G    => 125.00E+6,
                    CLK_LOWER_LIMIT_G =>  95.0E+6,
                    CLK_UPPER_LIMIT_G => 186.0E+6 )
      port map ( freqOut     => monClkRate(i),
                 freqUpdated => open,
                 locked      => monClkLock(i),
                 tooFast     => monClkFast(i),
                 tooSlow     => monClkSlow(i),
                 clkIn       => monClk(i),
                 locClk      => axilClk,
                 refClk      => axilClk );
  end generate;

  --
  --  Still need to cross clock-domains for register readout of:
  --    link status (32 links)
  --    partition inhibit counts (from 32 links for each partition)
  --

  GEN_BP : for i in 0 to NBpLinks generate
    U_LinkUp : entity work.Synchronizer
      port map ( clk     => axilClk,
                 dataIn  => status.bpLink(i).linkUp,
                 dataOut => s.bpLink(i).linkUp );
    U_IbRecv : entity work.SynchronizerVector
      generic map ( WIDTH_G => 32 )
      port map ( clk     => axilClk,
                 dataIn  => status.bpLink(i).ibRecv,
                 dataOut => s.bpLink(i).ibRecv );
    U_RxLate : entity work.SynchronizerVector
      generic map ( WIDTH_G => 16 )
      port map ( clk     => axilClk,
                 dataIn  => status.bpLink(i).rxLate,
                 dataOut => s.bpLink(i).rxLate );
  end generate;
               
  GEN_PART : for i in 0 to NPartitions-1 generate
    U_Sync64_ena : entity work.SynchronizerFifo
      generic map ( DATA_WIDTH_G => 64 )
      port map ( wr_clk => staClk, rd_clk=> axilClk, rd_en=> r.axilRdEn(i),
                 din  => status.partition(i).l0Select.enabled  ,
                 dout => s.partition(i).l0Select.enabled);
    U_Sync64_inh : entity work.SynchronizerFifo
      generic map ( DATA_WIDTH_G => 64 )
      port map ( wr_clk => staClk, rd_clk=> axilClk, rd_en=> r.axilRdEn(i),
                 din  => status.partition(i).l0Select.inhibited  ,
                 valid => pInhV(i),
                 dout => s.partition(i).l0Select.inhibited);
    U_Sync64_num : entity work.SynchronizerFifo
      generic map ( DATA_WIDTH_G => 64 )
      port map ( wr_clk => staClk, rd_clk=> axilClk, rd_en=> r.axilRdEn(i),
                 din  => status.partition(i).l0Select.num  ,
                 dout => s.partition(i).l0Select.num);
    U_Sync64_nin : entity work.SynchronizerFifo
      generic map ( DATA_WIDTH_G => 64 )
      port map ( wr_clk => staClk, rd_clk=> axilClk, rd_en=> r.axilRdEn(i),
                 din  => status.partition(i).l0Select.numInh  ,
                 dout => s.partition(i).l0Select.numInh);
    U_Sync64_nac : entity work.SynchronizerFifo
      generic map ( DATA_WIDTH_G => 64 )
      port map ( wr_clk => staClk, rd_clk=> axilClk, rd_en=> r.axilRdEn(i),
                 din  => status.partition(i).l0Select.numAcc  ,
                 dout => s.partition(i).l0Select.numAcc);
    U_SyncAna : entity work.SyncStatusVector
      generic map ( WIDTH_G     => 1,
                    CNT_WIDTH_G => 32 )
      port map ( statusIn  => status.partition(i).anaRd(1 downto 1),
                 statusOut => open,
                 cntRstIn  => r.config.partition(i).analysis.rst(1),
                 rollOverEnIn => (others=>'1'),
                 cntOut    => anaRdCount(i),
                 wrClk     => staClk,
                 rdClk     => axilClk );
  end generate;
  
  GEN_LOL : for i in 0 to NAmcs-1 generate
    pll_stat(2*i+0) <= pllStatus(i).los;
    pll_stat(2*i+1) <= pllStatus(i).lol;
  end generate;
    
  U_StatLol : entity work.SyncStatusVector
     generic map ( COMMON_CLK_G => true,
                   WIDTH_G      => 2*NAmcs,
                   CNT_WIDTH_G  => 3 )
     port map ( statusIn  => pll_stat,
                statusOut => pllStat,
                cntRstIn  => '0',
                rollOverEnIn => (others=>'1'),
                cntOut    => pllCount,
                wrClk     => axilClk,
                rdClk     => axilClk );
                
  U_AnalysisFifo : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => ETH_AXIS_CONFIG_C,
                  MASTER_AXI_CONFIG_G => TAG_AXIS_CONFIG_C )
    port map ( sAxisClk     => axilClk,
               sAxisRst     => axilRst,
               sAxisMaster  => ibDebugMaster,
               sAxisSlave   => ibDebugSlave,
               mAxisClk     => axilClk,
               mAxisRst     => axilRst,
               mAxisMaster  => tagMaster,
               mAxisSlave   => r_in.tagSlave );

  comb : process (r, axilReadMaster, axilWriteMaster, tagMaster, status, s, axilRst,
                  pllStatus, pllCount, pllStat, anaRdCount,
                  monClkRate, monClkLock, monClkFast, monClkSlow) is
    variable v          : RegType;
    variable axilStatus : AxiLiteStatusType;
    variable ip         : integer;
    variable il         : integer;
    variable ia         : integer;
    variable ra         : integer;
    -- Shorthand procedures for read/write register
    procedure axilRegRW(addr : in slv; offset : in integer; reg : inout slv) is
    begin
      axiSlaveRegister(axilWriteMaster, axilReadMaster,
                       v.axilWriteSlave, v.axilReadSlave, axilStatus,
                       addr, offset, reg, false, "0");
    end procedure;
    procedure axilRegRW(addr : in slv; offset : in integer; reg : inout sl) is
    begin
      axiSlaveRegister(axilWriteMaster, axilReadMaster,
                       v.axilWriteSlave, v.axilReadSlave, axilStatus,
                       addr, offset, reg, false, '0');
    end procedure;
    -- Shorthand procedures for read only registers
    procedure axilRegR (addr : in slv; offset : in integer; reg : in slv) is
    begin
      axiSlaveRegister(axilReadMaster, v.axilReadSlave, axilStatus,
                       addr, offset, reg);
    end procedure;
    procedure axilRegR (addr : in slv; offset : in integer; reg : in sl) is
    begin
      axiSlaveRegister(axilReadMaster, v.axilReadSlave, axilStatus,
                       addr, offset, reg);
    end procedure;
    procedure axilRegR64 (addr : in slv; reg : in slv) is
    begin
      axilRegR(addr+0,0,reg(31 downto  0));
      axilRegR(addr+4,0,reg(63 downto 32));
    end procedure;
  begin
    v := r;
    -- reset strobing signals
    v.axilReadSlave.rdata := (others=>'0');
    v.partitionCfg.message.insert := '0';
    v.tagSlave.tReady      := '1';

    ip := conv_integer(r.partition);
    il := conv_integer(r.link(3 downto 0));
    ia := conv_integer(r.amc);
    
    if r.load='1' then
      if r.link(4)='0' then
        v.linkCfg      := r.config.dsLink(il);
      else
        v.linkCfg      := r.config.bpLink(il);
      end if;
      v.partitionCfg := r.config.partition(ip);
      v.pllCfg       := r.config.pll      (ia);
    else
      if r.link(4)='0' then
        v.config.dsLink (il)      := r.linkCfg;
      else
        v.config.bpLink (il)      := r.linkCfg;
      end if;
      v.config.partition(ip)      := r.partitionCfg;
      v.config.pll      (ia)      := r.pllCfg;
    end if;

    if r.link(4)='0' then
      v.linkStat         := status.dsLink (il);  -- clock-domain?
    elsif r.link(3 downto 0)=toSlv(0,4) then
      v.linkStat         := XPM_LINK_STATUS_INIT_C;
      v.linkStat.txReady := s.bpLink (il).linkUp;
    elsif r.link(3)='0' then
      v.linkStat           := XPM_LINK_STATUS_INIT_C;
      v.linkStat.rxReady   := s.bpLink (il).linkUp;
      v.linkStat.rxRcvCnts := s.bpLink (il).ibRecv;
      v.linkStat.rxErrCnts := s.bpLink (il).rxErrs;
    else
      v.linkStat           := XPM_LINK_STATUS_INIT_C;
      v.linkStat.rxErrCnts := s.bpLink (conv_integer(r.link(2 downto 0))).rxLate;
    end if;
    v.partitionStat := status.partition(ip);
    v.pllStat       := pllStatus       (ia);
    
    -- Determine the transaction type
    axiSlaveWaitTxn(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus);
    v.axilReadSlave.rdata := (others=>'0');
    
    -- Read/write to the configuration registers
    -- Read only from status registers

    ra := 0;
    axilRegR  (toSlv( ra,12),  0, status.paddr);
    ra := ra+4;
    axilRegRW (toSlv( ra,12),  0, v.partition);
    axilRegRW (toSlv( ra,12),  4, v.link);
    axilRegRW (toSlv( ra,12), 10, v.linkDebug);
    axilRegRW (toSlv( ra,12), 16, v.amc);
    axilRegRW (toSlv( ra,12), 20, v.inhibit);
    axilRegRW (toSlv( ra,12), 24, v.tagStream);
    
    v.load := '0';
    v.linkCfg.txDelayRst   := '0';

    if axilStatus.writeEnable='1' then
      if std_match(axilWriteMaster.awaddr(11 downto 0),toSlv(4,12)) then
        v.load := '1';
      end if;
        
      if std_match(axilWriteMaster.awaddr(11 downto 0),toSlv(8,12)) then
        v.linkCfg.txDelayRst := '1';
      end if;
    end if;

    ra := ra+4;
    axilRegRW(toSlv(ra,12),    0, v.linkCfg.txDelay);
    axilRegRW(toSlv(ra,12),   18, v.linkCfg.txPllReset);
    axilRegRW(toSlv(ra,12),   19, v.linkCfg.rxPllReset);
    axilRegRW(toSlv(ra,12),   20, v.linkCfg.partition);
    axilRegRW(toSlv(ra,12),   24, v.linkCfg.trigsrc);
    axilRegRW(toSlv(ra,12),   28, v.linkCfg.loopback);
    axilRegRW(toSlv(ra,12),   29, v.linkCfg.txReset);
    axilRegRW(toSlv(ra,12),   30, v.linkCfg.rxReset);
    axilRegRW(toSlv(ra,12),   31, v.linkCfg.enable);

    ra := ra+4;
    axilRegR (toSlv(ra,12),   0, r.linkStat.rxErrCnts);
    axilRegR (toSlv(ra,12),  16, r.linkStat.txResetDone);
    axilRegR (toSlv(ra,12),  17, r.linkStat.txReady);
    axilRegR (toSlv(ra,12),  18, r.linkStat.rxResetDone);
    axilRegR (toSlv(ra,12),  19, r.linkStat.rxReady);
    axilRegR (toSlv(ra,12),  20, r.linkStat.rxIsXpm);

    ra := ra+4;
    axilRegR (toSlv(ra,12),  0, r.linkStat.rxRcvCnts);

    ra := ra+4;
    axilRegRW(toSlv(ra,12),  0, v.pllCfg.bwSel);
    axilRegRW(toSlv(ra,12),  4, v.pllCfg.frqTbl);
    axilRegRW(toSlv(ra,12),  8, v.pllCfg.frqSel);
    axilRegRW(toSlv(ra,12), 16, v.pllCfg.rate);
    axilRegRW(toSlv(ra,12), 20, v.pllCfg.inc);
    axilRegRW(toSlv(ra,12), 21, v.pllCfg.dec);
    axilRegRW(toSlv(ra,12), 22, v.pllCfg.bypass);
    axilRegRW(toSlv(ra,12), 23, v.pllCfg.rstn);
    axilRegR (toSlv(ra,12), 24, muxSlVectorArray( pllCount, 2*ia+0));
    axilRegR (toSlv(ra,12), 27, pllStat(2*ia+0));
    axilRegR (toSlv(ra,12), 28, muxSlVectorArray( pllCount, 2*ia+1));
    axilRegR (toSlv(ra,12), 31, pllStat(2*ia+1));

    ra := ra+4;
    axilRegRW (toSlv(ra,12), 0, v.partitionCfg.l0Select.reset);
    axilRegRW (toSlv(ra,12),16, v.partitionCfg.l0Select.enabled);
    axilRegRW (toSlv(ra,12),31, v.axilRdEn(ip));

    ra := ra+4;
    axilRegRW (toSlv(ra,12), 0, v.partitionCfg.l0Select.rateSel);
    axilRegRW (toSlv(ra,12),16, v.partitionCfg.l0Select.destSel);

    axilRegR64(toSlv(ra+4,12), s.partition(ip).l0Select.enabled);
    axilRegR64(toSlv(ra+12,12), s.partition(ip).l0Select.inhibited);
    axilRegR64(toSlv(ra+20,12), s.partition(ip).l0Select.num);
    axilRegR64(toSlv(ra+28,12), s.partition(ip).l0Select.numInh);
    axilRegR64(toSlv(ra+36,12), s.partition(ip).l0Select.numAcc);
    axilRegR64(toSlv(ra+44,12), s.partition(ip).l1Select.numAcc);

    axilRegRW (toSlv(ra+52,12),  0, v.partitionCfg.l1Select.clear);
    axilRegRW (toSlv(ra+52,12), 16, v.partitionCfg.l1Select.enable);

    ra := ra+56;
    axilRegRW (toSlv(ra,12),  0, v.partitionCfg.l1Select.trigsrc);
    axilRegRW (toSlv(ra,12),  4, v.partitionCfg.l1Select.trigword);
    axilRegRW (toSlv(ra,12), 16, v.partitionCfg.l1Select.trigwr);
      
    axilRegRW (toSlv(ra+4,12), 0, v.partitionCfg.analysis.rst);
    axilRegRW (toSlv(ra+8,12), 0, v.partitionCfg.analysis.tag);
    axilRegRW (toSlv(ra+12,12), 0, v.partitionCfg.analysis.push);
    axilRegR  (toSlv(ra+16,12), 0, r.anaWrCount(ip));
    axilRegR  (toSlv(ra+20,12), 0, muxSlVectorArray( anaRdCount(ip), 0));

    axilRegRW (toSlv(ra+24,12), 0, v.partitionCfg.pipeline.depth);

    axilRegRW (toSlv(ra+28,12),15, v.partitionCfg.message.insert);
    axilRegRW (toSlv(ra+28,12), 0, v.partitionCfg.message.hdr);
    axilRegRW (toSlv(ra+32,12), 0, v.partitionCfg.message.payload);

    ra := ra+36;
    axilRegR (toSlv(ra,12),  0, r.linkStat.rxId);
    
    for j in r.partitionCfg.inhibit.setup'range loop
      axilRegRW (toSlv(128+j*4,12),  0, v.partitionCfg.inhibit.setup(j).interval);
      axilRegRW (toSlv(128+j*4,12), 12, v.partitionCfg.inhibit.setup(j).limit);
      axilRegRW (toSlv(128+j*4,12), 31, v.partitionCfg.inhibit.setup(j).enable);
    end loop;

    for j in 0 to 31 loop
      axilRegR (toSlv(144+j*4,12), 0, r.partitionStat.inhibit.counts(j));
    end loop;

    for j in 0 to 3 loop
      axilRegR (toSlv(272+j*4, 12),  0, monClkRate(j)(28 downto 0));
      axilRegR (toSlv(272+j*4, 12), 29, monClkSlow(j));
      axilRegR (toSlv(272+j*4, 12), 30, monClkFast(j));
      axilRegR (toSlv(272+j*4, 12), 31, monClkLock(j));
   end loop;

    if r.partitionCfg.analysis.rst(1)='1' then
      v.anaWrCount(ip) := (others=>'0');
    elsif r.partitionCfg.analysis.push(1)='1' then
      v.anaWrCount(ip) := r.anaWrCount(ip)+1;
    end if;
      
    --if r.config.tagstream='0' then
    --  v.tagSlave.tReady := '0';
    --elsif tagMaster.tValid='1' then
    --  ip := conv_integer(tagMaster.tDest(3 downto 0));
    --  v.config.partition(ip).analysis.tag  := tagMaster.tData(31 downto  0);
    --  v.config.partition(ip).analysis.push := tagMaster.tData(35 downto 34);
    --end if;
    v.tagSlave.tReady := '0';
    
    -- Set the status
    axiSlaveDefault(axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave, axilStatus, AXI_RESP_OK_C);

    ----------------------------------------------------------------------------------------------
    -- Reset
    ----------------------------------------------------------------------------------------------
    if (axilRst = '1') then
      v := REG_INIT_C;
    end if;

    r_in <= v;
  end process;

  seq : process (axilClk) is
  begin
    if rising_edge(axilClk) then
      r <= r_in;
    end if;
  end process;
end rtl;
