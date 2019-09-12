-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiCPllArray.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-04-08
-- Last update: 2017-11-15
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Ethernet Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Ethernet Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

use work.StdRtlPkg.all;
use work.DtiPkg.all;

library unisim;
use unisim.vcomponents.all;

entity DtiCPllArray is
   generic (
      TPD_G             : time            := 1 ns;
      REF_CLK_FREQ_G    : real            := 156.25E+6;  -- Support 156.25MHz or 312.5MHz   
      CPLL_REFCLK_SEL_G : slv(2 downto 0) := "001");
   port (
      -- MGT Clock Port (156.25 MHz or 312.5 MHz)
      amcClkP       : in  sl;
      amcClkN       : in  sl;
      amcTxP        : out slv(6 downto 0);
      amcTxN        : out slv(6 downto 0);
      amcRxP        : in  slv(6 downto 0);
      amcRxN        : in  slv(6 downto 0);
      -- channel ports
      chanClk       : out slv(6 downto 0);
      chanRefClk    : out slv(6 downto 0);
      chanTxP       : in  slv(6 downto 0);
      chanTxN       : in  slv(6 downto 0);
      chanRxP       : out slv(6 downto 0);
      chanRxN       : out slv(6 downto 0) );
end DtiCPllArray;

architecture mapping of DtiCPllArray is

  constant DIV_C : slv(2 downto 0) := ite((REF_CLK_FREQ_G = 156.25E+6), "000", "001");

  signal amcRefClk     : sl;
  signal amcRefClkCopy : sl;
  signal amcCoreClk    : sl;
  
begin

    IBUFDS_GTE3_Inst : IBUFDS_GTE3
      generic map (
        REFCLK_EN_TX_PATH  => '0',
        REFCLK_HROW_CK_SEL => "00",    -- 2'b00: ODIV2 = O
        REFCLK_ICNTL_RX    => "00")
      port map (
        I     => amcClkP,
        IB    => amcClkN,
        CEB   => '0',
        ODIV2 => amcRefClkCopy,
        O     => amcRefClk);  

    BUFG_GT_Inst : BUFG_GT
      port map (
        I       => amcRefClkCopy,
        CE      => '1',
        CEMASK  => '1',
        CLR     => '0',
        CLRMASK => '1',
        DIV     => DIV_C,
        O       => amcCoreClk);

  --
  --  The AMC SFP channels are reordered - the mapping to MGT quads is non-trivial
  --    amcTx/Rx indexed by MGT
  --    iamcTx/Rx indexed by SFP
  --
  reorder_p : process (amcRxP,amcRxN,chanTxP,chanTxN,
                       amcCoreClk,amcRefClk) is
  begin
    for j in 0 to 3 loop
      amcTxP    (j)   <= chanTxP(j+2);
      amcTxN    (j)   <= chanTxN(j+2);
      chanRxP   (j+2) <= amcRxP(j);
      chanRxN   (j+2) <= amcRxN(j);
      chanClk   (j+2) <= amcCoreClk;
      chanRefClk(j+2) <= amcRefClk;
    end loop;
    for j in 4 to 5 loop
      amcTxP    (j)   <= chanTxP(j-4);
      amcTxN    (j)   <= chanTxN(j-4);
      chanRxP   (j-4) <= amcRxP(j);
      chanRxN   (j-4) <= amcRxN(j);
      chanClk   (j-4) <= amcCoreClk;
      chanRefClk(j-4) <= amcRefClk;
    end loop;
    for j in 6 to 6 loop
      amcTxP    (j) <= chanTxP(j);
      amcTxN    (j) <= chanTxN(j);
      chanRxP   (j) <= amcRxP(j);
      chanRxN   (j) <= amcRxN(j);
      chanClk   (j) <= amcCoreClk;
      chanRefClk(j) <= amcRefClk;
    end loop;
  end process;
    
end mapping;
