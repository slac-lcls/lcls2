-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiDsSimApp.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-07-26
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

entity DtiDsSimApp is
   generic (
      TPD_G               : time                := 1 ns );
   port (
     amcClk          : in  sl;
     amcRst          : in  sl;
     --amcRxP          : in  sl;
     --amcRxN          : in  sl;
     --amcTxP          : out sl;
     --amcTxN          : out sl;
     --  App Interface
     ibRst           : in  sl;
     linkUp          : out sl;
     rxErrs          : out slv(31 downto 0);
     full            : out sl;
     --
     obClk           : in  sl;
     obMaster        : in  AxiStreamMasterType;
     obSlave         : out AxiStreamSlaveType );
end DtiDsSimApp;

architecture rtl of DtiDsSimApp is

begin

  linkUp  <= '1';
  rxErrs  <= (others=>'0');
  full    <= '0';
  obSlave <= AXI_STREAM_SLAVE_FORCE_C;
  
end rtl;
