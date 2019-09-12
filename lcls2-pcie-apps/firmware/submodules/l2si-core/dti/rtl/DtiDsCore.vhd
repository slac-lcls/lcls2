-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiDsCore.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2018-07-26
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
use work.DtiPkg.all;

library unisim;
use unisim.vcomponents.all;

entity DtiDsCore is
   generic (
      TPD_G               : time                := 1 ns;
      DEBUG_G             : boolean             := false );
   port (
     --  Core Interface
     clear           : in  sl := '0';
     update          : in  sl := '1';
--     config          : in  DtiUsLinkConfigType,
     remLinkID       : in  slv(31 downto 0);
     status          : out DtiDsLinkStatusType;
     --
     eventClk        : in  sl;
     eventRst        : in  sl;
     eventMasters    : in  AxiStreamMasterArray(MaxUsLinks-1 downto 0);
     eventSlaves     : out AxiStreamSlaveArray (MaxUsLinks-1 downto 0);
     fullOut         : out sl;
     --  App Interface
     linkUp          : in  sl;
     rxErrs          : in  slv(31 downto 0);
     fullIn          : in  sl;
     --
     obClk           : out sl;
     obMaster        : out AxiStreamMasterType;
     obSlave         : in  AxiStreamSlaveType );
end DtiDsCore;

architecture rtl of DtiDsCore is

  signal srxFull : sl;
  signal supdate : sl;
  signal sclear  : sl;
  
  type RegType is record
    status  : DtiDsLinkStatusType;
    statusO : DtiDsLinkStatusType;
  end record;
  constant REG_INIT_C : RegType := (
    status  => DTI_DS_LINK_STATUS_INIT_C,
    statusO => DTI_DS_LINK_STATUS_INIT_C );
  
  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;

  signal tMaster : AxiStreamMasterType;

  component ila_0
    port ( clk  : in sl;
           probe0 : in slv(255 downto 0) );
  end component;
  
begin

  status.linkUp    <= linkUp;
  status.remLinkID <= remLinkID;
  status.rxFull    <= r.status.rxFull;
  status.rxErrs    <= rxErrs;
  status.obSent    <= r.status.obSent;

  fullOut  <= fullIn;
  obClk    <= eventClk;
  obMaster <= tMaster;

  U_SRxFull : entity work.Synchronizer
    port map ( clk     => eventClk,
               dataIn  => fullIn,
               dataOut => srxFull );

  U_SUpdate : entity work.synchronizer
    port map ( clk     => eventClk,
               dataIn  => update,
               dataOut => supdate );
  
  U_SClear : entity work.synchronizer
    port map ( clk     => eventClk,
               dataIn  => clear,
               dataOut => sclear );
  
  U_Mux : entity work.AxiStreamMux
    generic map ( NUM_SLAVES_G => MaxUsLinks )
    port map ( axisClk      => eventClk,
               axisRst      => eventRst,
               sAxisMasters => eventMasters,
               sAxisSlaves  => eventSlaves,
               mAxisMaster  => tMaster,
               mAxisSlave   => obSlave );

  comb : process ( r, eventRst, clear, supdate, srxFull, tMaster, obSlave ) is
    variable v : RegType;
  begin
    v := r;

    if supdate = '1' then
      if srxFull = '1' then
        v.status.rxFull := r.status.rxFull+1;
      end if;

      if tMaster.tValid='1' and obSlave.tReady='1' then
        v.status.obSent := r.status.obSent+1;
      end if;
    end if;
    
    if eventRst = '1' or clear = '1' then
      v := REG_INIT_C;
    end if;
    
    r_in <= v;
  end process;

  seq : process (eventClk) is
  begin
    if rising_edge(eventClk) then
      r <= r_in;
    end if;
  end process;
  
end rtl;
