-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiSimPkg.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-03-25
-- Last update: 2017-04-08
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
use work.AxiStreamPkg.all;

package DtiSimPkg is
  
  type RegTransactionType is record
    rnw     : sl;
    address : slv(30 downto 0);
    data    : slv(31 downto 0);
  end record;

  constant REG_TRANSACTION_INIT_C : RegTransactionType := (
    rnw     => '1',
    address => (others=>'0'),
    data    => (others=>'0') );
  
  function toSlv         (reg : RegTransactionType) return slv;
  function toRegTransType(vector : slv) return RegTransactionType;

end package DtiSimPkg;

package body DtiSimPkg is

  function toSlv (reg : RegTransactionType) return slv
  is
    variable vector : slv(63 downto 0) := (others=>'0');
    variable i      : integer          := 0;
  begin
    assignSlv(i, vector, reg.rnw);
    assignSlv(i, vector, reg.address);
    assignSlv(i, vector, reg.data);
    return vector;
  end function;

  function toRegTransType(vector : slv) return RegTransactionType
  is
    variable reg : RegTransactionType  := REG_TRANSACTION_INIT_C;
    variable i   : integer             := 0;
  begin
    assignRecord(i, vector, reg.rnw);
    assignRecord(i, vector, reg.address);
    assignRecord(i, vector, reg.data);
    return reg;
  end function;
    
end package body DtiSimPkg;
