------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiUsSimApp.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-10-03
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: DtiApp's Top Level
-- 
-- Note: Common-to-DtiApp interface defined here (see URL below)
--       https://confluence.slac.stanford.edu/x/rLyMCw
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.XpmPkg.all;
use work.DtiPkg.all;
use work.DtiSimPkg.all;
use work.SsiPkg.all;

entity DtiUsSimApp is
   generic (
      TPD_G               : time                := 1 ns;
      SERIAL_ID_G         : slv(31 downto 0)    := (others=>'0');
      ENABLE_TAG_G        : boolean             := false ;
      DEBUG_G             : boolean             := false );

   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     status          : out DtiUsAppStatusType;
     --amcRxP          : in  sl;
     --amcRxN          : in  sl;
     --amcTxP          : out sl;
     --amcTxN          : out sl;
     fifoRst         : in  sl;
     --
     ibClk           : in  sl;
     ibRst           : in  sl;
     ibMaster        : out AxiStreamMasterType;
     ibSlave         : in  AxiStreamSlaveType;
     linkUp          : out sl;
     rxErr           : out sl;
     --
     obClk           : in  sl;
     obRst           : in  sl;
     obTrig          : in  XpmPartitionDataType;
     obTrigValid     : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiUsSimApp;

architecture top_level_app of DtiUsSimApp is

  type StateType is (S_IDLE, S_READTAG, S_READOUT, S_PAYLOAD);
  
  type RegType is record
    state    : StateType;
    status   : DtiUsAppStatusType;
    delay    : slv(31 downto 0);
    target   : slv(31 downto 0);
    count    : slv(31 downto 0);
    payload  : slv(31 downto 0);
    scratch  : slv(31 downto 0);
    localts  : slv(63 downto 0);
    wordcnt  : slv(31 downto 0);
    tagRd    : sl;
    l1a      : sl;
    master   : AxiStreamMasterType;
    slave    : AxiStreamSlaveType;
  end record;
  
  constant REG_INIT_C : RegType := (
    state    => S_IDLE,
    status   => DTI_US_APP_STATUS_INIT_C,
    delay    => toSlv(400,32),
    target   => toSlv( 0,32),
    count    => toSlv( 0,32),
    payload  => toSlv(32,32),
    scratch  => x"DEADBEEF",
    localts  => (others=>'0'),
    wordcnt  => (others=>'0'),
    tagRd    => '0',
    l1a      => '0',
    master   => AXI_STREAM_MASTER_INIT_C,
    slave    => AXI_STREAM_SLAVE_INIT_C );
  
  signal r   : RegType := REG_INIT_C;
  signal rin : RegType;

  constant AXIS_CONFIG_C : AxiStreamConfigType := (
    TSTRB_EN_C    => false,
    TDATA_BYTES_C => 8,
    TID_BITS_C    => 5,
    TDEST_BITS_C  => 1,
    TKEEP_MODE_C  => TKEEP_COMP_C,
    TUSER_BITS_C  => 2,
    TUSER_MODE_C  => TUSER_FIRST_LAST_C );
  
  signal amcIbMaster, amcObMaster : AxiStreamMasterType;
  signal amcIbSlave , amcObSlave  : AxiStreamSlaveType;

  signal l0a, l0S, l1S, l1aS : sl;
  signal obTrigV  : slv(47 downto 0);
  signal obTrigVS : slv(47 downto 0);
  signal obTrigS : XpmPartitionDataType;
  signal obTrigSValid : sl;
  
  signal tagdout : slv(4 downto 0);
  signal tgtdout : slv(31 downto 0);
  signal tsdout  : slv(63 downto 0);
  signal tagValid : sl;
  signal tgtValid : sl;

  component ila_0
    port ( clk    : sl;
           probe0 : slv(255 downto 0) );
  end component;

  signal iobSlave  : AxiStreamSlaveType;
  signal iibMaster : AxiStreamMasterType;
  signal r_state   : slv(1 downto 0);
  
begin

  linkUp  <= '1';
  obSlave <= iobSlave;

  l0a     <= obTrigValid and obTrig.l0a;

  GEN_DEBUG : if DEBUG_G generate
    U_ILA_OB : ila_0
      port map ( clk                 => obClk,
                 probe0(0)           => obMaster.tValid, 
                 probe0(1)           => obMaster.tLast,
                 probe0(2)           => iobSlave.tReady,
                 probe0( 6 downto 3) => obMaster.tDest( 3 downto 0),
                 probe0(70 downto 7) => obMaster.tData(63 downto 0),
                 probe0(255 downto 71) => (others=>'0') );
    U_ILA_IB : ila_0
      port map ( clk                 => ibClk,
                 probe0(0)           => iibMaster.tValid, 
                 probe0(1)           => iibMaster.tLast,
                 probe0(2)           => ibSlave.tReady,
                 probe0( 6 downto 3) => iibMaster.tDest( 3 downto 0),
                 probe0(70 downto 7) => iibMaster.tData(63 downto 0),
                 probe0(255 downto 71) => (others=>'0') );

    r_state <= "00" when r.state = S_IDLE else
               "01" when r.state = S_READTAG else
               "10" when r.state = S_READOUT else
               "11";
    
    U_ILA_AMC : ila_0
      port map ( clk                 => amcClk,
                 probe0(0)           => amcIbMaster.tValid, 
                 probe0(1)           => amcIbMaster.tLast,
                 probe0(2)           => amcIbSlave.tReady,
                 probe0( 6 downto 3) => amcIbMaster.tDest( 3 downto 0),
                 probe0(70 downto 7) => amcIbMaster.tData(63 downto 0),
                 probe0(72 downto 71) => r_state,
                 probe0(73)          => amcObMaster.tValid, 
                 probe0(74)          => amcObMaster.tLast,
                 probe0(75)          => amcObSlave.tReady,
                 probe0(79 downto 76) => amcObMaster.tDest( 3 downto 0),
                 probe0(143 downto 80) => amcObMaster.tData(63 downto 0),
                 probe0(255 downto 144) => (others=>'0') );
  end generate;

  ibMaster <= iibMaster;
  
  U_IbFifo : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => AXIS_CONFIG_C,
                  MASTER_AXI_CONFIG_G => US_IB_CONFIG_C )
    port map ( sAxisClk    => amcClk,
               sAxisRst    => amcRst,
               sAxisMaster => amcIbMaster,
               sAxisSlave  => amcIbSlave,
               mAxisClk    => ibClk,
               mAxisRst    => ibRst,
               mAxisMaster => iibMaster,
               mAxisSlave  => ibSlave );

  U_ObFifo : entity work.AxiStreamFifo
    generic map ( SLAVE_AXI_CONFIG_G  => US_OB_CONFIG_C,
                  MASTER_AXI_CONFIG_G => AXIS_CONFIG_C )
    port map ( sAxisClk    => obClk,
               sAxisRst    => obRst,
               sAxisMaster => obMaster,
               sAxisSlave  => iobSlave,
               mAxisClk    => amcClk,
               mAxisRst    => amcRst,
               mAxisMaster => amcObMaster,
               mAxisSlave  => amcObSlave );

  U_TgtFifo : entity work.FifoSync
    generic map ( FWFT_EN_G    => true,
                  DATA_WIDTH_G => 32,
                  ADDR_WIDTH_G => 5 )
    port map ( rst    => fifoRst,
               clk    => amcClk,
               wr_en  => l0S,
               din    => r.target,
               rd_en  => r.tagRd,
               dout   => tgtdout,
               valid  => tgtValid );

  U_TagFifo : entity work.FifoAsync
    generic map ( FWFT_EN_G    => true,
                  DATA_WIDTH_G => 5,
                  ADDR_WIDTH_G => 5 )
    port map ( rst    => fifoRst,
               wr_clk => obClk,
               wr_en  => l0a,
               din    => obTrig.l0tag,
               rd_clk => amcClk,
               rd_en  => r.tagRd,
               dout   => tagdout,
               valid  => tagValid );

  U_TsFifo : entity work.FifoSync
    generic map ( FWFT_EN_G    => true,
                  DATA_WIDTH_G => 64,
                  ADDR_WIDTH_G => 5 )
    port map ( rst    => fifoRst,
               clk    => amcClk,
               wr_en  => l0S,
               rd_en  => r.tagRd,
               din    => r.localts,
               dout   => tsdout );

  obTrigV <= toSlv(obTrig);
  U_ObTrigS : entity work.SynchronizerFifo
    generic map ( DATA_WIDTH_G => 48,
                  ADDR_WIDTH_G => 4 )
    port map ( rst    => fifoRst,
               wr_clk => obClk,
               wr_en  => obTrigValid,
               din    => obTrigV,
               rd_clk => amcClk,
               valid  => obTrigSValid,
               dout   => obTrigVS );
  obTrigS <= toPartitionWord(obTrigVS);
  l0S     <= obTrigSValid and obTrigS.l0a;
  l1S     <= obTrigSValid and obTrigS.l1e;
  l1aS    <= obTrigS.l1a;

  --
  --  Parse amcOb stream for register transactions or obTrig
  --
  comb : process ( fifoRst, r, amcObMaster, amcObSlave, amcIbSlave, l1S, l1aS,
                   tagValid, tagdout, tgtValid, tgtdout, tsdout ) is
    variable v   : RegType;
    variable reg : RegTransactionType;
  begin
    v := r;

    v.tagRd   := '0';
    v.localts := r.localts+1;
    v.slave.tReady := '1';
    v.count   := r.count+1;
    v.target  := r.count+r.delay;
    
    if amcObMaster.tValid = '1' and amcObSlave.tReady = '1' then
      v.status.obReceived := r.status.obReceived+1;
    end if;

    if v.master.tValid = '1' and amcIbSlave.tReady = '1' then
      v.status.obSent := r.status.obSent+1;
    end if;
    
    if amcIbSlave.tReady='1' then
      v.master.tValid := '0';
    end if;
    
    reg := toRegTransType(amcObMaster.tData(63 downto 0));
    
    case r.state is
      when S_IDLE =>
        if v.master.tValid='0' and amcObMaster.tValid='1' then -- register transaction
          v.master.tValid := '1';
          v.master.tLast  := '1';
          v.master.tData  := amcObMaster.tData;
          v.master.tDest(0) := '1';
          ssiSetUserSof (AXIS_CONFIG_C,v.master,'1');
          ssiSetUserEofe(AXIS_CONFIG_C,v.master,'0');
          if reg.rnw='1' then
            case conv_integer(reg.address) is
              when      0 => v.master.tData(63 downto 32) := SERIAL_ID_G;
              when      4 => v.master.tData(63 downto 32) := r.payload;
              when      8 => v.master.tData(63 downto 32) := r.scratch;
              when     12 => v.master.tData(63 downto 32) := r.delay;
              when others => v.master.tData(63 downto 32) := x"DEADBEEF";
            end case;
          else
            case conv_integer(reg.address) is
              when      4 => v.payload := amcObMaster.tData(63 downto 32);
              when      8 => v.scratch := amcObMaster.tData(63 downto 32);
              when     12 => v.delay   := amcObMaster.tData(63 downto 32);
              when others => null;
            end case;
          end if;
        end if;

        if l1S='1' then  -- readout
          if l1aS='0' then
            v.tagRd := '1';
          else
            v.wordcnt := (others=>'0');
            v.state := S_READOUT;
          end if;
        end if;

      when S_READOUT =>
        if v.master.tValid='0' and tagValid='1' and tgtValid='1' and tgtdout=r.count then
          v.slave .tReady := '0';
          v.tagRd := '1';
          v.master.tId(4 downto 0) := tagdout;
          v.master.tDest(0) := '0';
          v.master.tValid := '1';
          v.master.tLast  := '0';
          ssiSetUserSof (AXIS_CONFIG_C,v.master,'1');
          v.master.tData(63 downto 0) := tsdout;
          v.wordcnt       := r.wordcnt+1;
          v.state         := S_PAYLOAD;
        end if;

      when S_PAYLOAD =>
        if v.master.tValid='0' then
          v.slave .tReady := '0';
          v.master.tValid := '1';
          v.master.tLast  := '0';
          v.master.tData(63 downto 0) := r.localts(31 downto 0) & r.wordcnt;
          v.wordcnt       := r.wordcnt+1;
          v.state         := S_PAYLOAD;
          if r.wordcnt = r.payload then
            ssiSetUserEofe(AXIS_CONFIG_C,v.master,'0');
            v.master.tLast := '1';
            v.state        := S_IDLE;
          end if;
        end if;
        
      when others =>
        null;
    end case;

    if fifoRst = '1' then
      v := REG_INIT_C;
    end if;
    
    rin <= v;

    amcIbMaster <= r.master;
    amcObSlave  <= r.slave;
    status      <= r.status;
    
  end process;
            
  seq : process (amcClk) is
  begin
    if rising_edge(amcClk) then
      r <= rin;
    end if;
  end process;
  
end top_level_app;
