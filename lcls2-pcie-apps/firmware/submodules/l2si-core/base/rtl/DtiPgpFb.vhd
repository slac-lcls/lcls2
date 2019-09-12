-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiPgpFb.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-11-20
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

library unisim;
use unisim.vcomponents.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.AxiLitePkg.all;
use work.DtiPkg.all;
use work.Pgp2bPkg.all;

entity DtiPgpFb is
   port (
     pgpClk          : in  sl;
     pgpRst          : in  sl;
     pgpRxOut        : in  Pgp2bRxOutType;
     rxAlmostFull    : out sl;
     txAlmostFull    : out sl );
end DtiPgpFb;

architecture rtl of DtiPgpFb is

  -- Tx Opcodes
  constant NONE_AF_OPCODE   : slv(7 downto 0) := x"00";
  constant RX_AF_OPCODE     : slv(7 downto 0) := x"01";   -- receive queue almost full
  constant TX_AF_OPCODE     : slv(7 downto 0) := x"02";   -- transmit queue almost full
  constant BOTH_AF_OPCODE   : slv(7 downto 0) := x"03";   -- both queues almost full

  type RegType is record
    rx_full        : sl;
    rx_almost_full : sl;
    tx_almost_full : sl;
    tmo            : slv(11 downto 0);
    opCodeFound    : sl;
  end record;

  constant REG_INIT_C : RegType := (
    rx_full        => '0',
    rx_almost_full => '1',
    tx_almost_full => '1',
    tmo            => (others=>'0'),
    opCodeFound    => '0' );

  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;
  
begin

  comb : process ( r, pgpRst, pgpRxOut ) is
    variable v : RegType;
  begin
    v := r;

    v.tmo := r.tmo + 1;
    
    if pgpRxOut.opCodeEn = '1' then
      v.opCodeFound := '1';
      case (pgpRxOut.opCode) is
        when NONE_AF_OPCODE =>
          v.rx_almost_full := '0';
          v.tx_almost_full := '0';
        when RX_AF_OPCODE => 
          v.rx_almost_full := '1';
          v.tx_almost_full := '0';
        when TX_AF_OPCODE => 
          v.rx_almost_full := '0';
          v.tx_almost_full := '1';
        when BOTH_AF_OPCODE => 
          v.rx_almost_full := '1';
          v.tx_almost_full := '1';
        when others =>
          null;
      end case;
    end if;

    v.rx_full := pgpRxOut.remPause(0);

    if r.tmo = 0 then
      if r.opCodeFound = '0' then
        v.rx_almost_full := '1';
        v.tx_almost_full := '1';
      end if;
      v.opCodeFound := '0';
    end if;
    
    if pgpRst = '1' then
      v := REG_INIT_C;
    end if;
    
    r_in <= v;

    rxAlmostFull <= r.rx_almost_full or r.rx_full;
    txAlmostFull <= r.tx_almost_full;
  end process;

  seq : process (pgpClk) is
  begin
    if rising_edge(pgpClk) then
      r <= r_in;
    end if;
  end process;
  
end rtl;
