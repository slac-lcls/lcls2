-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiAppPkg.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-03-25
-- Last update: 2017-01-16
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: Programmable configuration and status fields
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Dti Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Dti Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;

package DtiAppPkg is

  type DtiLinkConfigType is record
    enable    : sl;
    partition : slv(3 downto 0);
    tfDelay   : slv(6 downto 0);
  end record;

  constant DTI_LINK_CONFIG_INIT_C : DtiLinkConfigType := (
    enable    => '0',
    partition => (others=>'0'),
    tfDelay   => (others=>'0') );

  type DtiLinkConfigArray is array(natural range<>) of DtiLinkConfigType;
  
  type DtiConfigType is record
    link      : DtiLinkConfigArray     ( 0 downto 0);
  end record;

  constant DTI_CONFIG_INIT_C : DtiConfigType := (
    link      => (others=>DTI_LINK_CONFIG_INIT_C) );
  
  type DtiStatusType is record
    status : sl;
  end record;

  constant DTI_STATUS_INIT_C : DtiStatusType := (
    status => '0' );
  
end package DtiAppPkg;

package body DtiAppPkg is

end package body DtiAppPkg;
