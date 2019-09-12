-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : DtiPgp3QuadPllArray.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-04-08
-- Last update: 2018-02-17
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

entity DtiPgp3QuadPllArray is
   generic (
      TPD_G             : time            := 1 ns;
      REF_CLK_FREQ_G    : real            := 156.25E+6;  -- Support 156.25MHz or 312.5MHz   
      QPLL_REFCLK_SEL_G : slv(2 downto 0) := "001");
   port (
      axilClk       : in  sl;
      axilRst       : in  sl;
      -- MGT Clock Port (156.25 MHz or 312.5 MHz)
      amcClkP       : in  sl;
      amcClkN       : in  sl;
      amcTxP        : out slv(6 downto 0);
      amcTxN        : out slv(6 downto 0);
      amcRxP        : in  slv(6 downto 0);
      amcRxN        : in  slv(6 downto 0);
      -- channel ports
      chanPllRst    : in  Slv2Array(6 downto 0);
      chanTxP       : in  slv(6 downto 0);
      chanTxN       : in  slv(6 downto 0);
      chanRxP       : out slv(6 downto 0);
      chanRxN       : out slv(6 downto 0);
      chanQuad      : out QuadArray(6 downto 0) );
end DtiPgp3QuadPllArray;

architecture mapping of DtiPgp3QuadPllArray is

  constant DIV_C : slv(2 downto 0) := ite((REF_CLK_FREQ_G = 156.25E+6), "000", "001");

  signal amcRefClk     : sl;
  signal amcRefClkCopy : sl;
  signal amcCoreClk    : sl;
  signal amcQuad       : QuadArray(1 downto 0);
  signal qpllclk       : Slv2Array(7 downto 0) := (others=>"00");
  signal qpllrefclk    : Slv2Array(7 downto 0) := (others=>"00");
  signal qplllock      : Slv2Array(7 downto 0) := (others=>"00");
  signal qpllrst       : Slv2Array(7 downto 0) := (others=>"00");
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

    GEN_TENGIGCLK : for i in 0 to 1 generate

      U_QPLL : entity work.Pgp3GthUsQpll
        generic map ( TPD_G      => TPD_G,
                      EN_DRP_G   => false )
        port map (
          stableClk  => axilClk,
          stableRst  => axilRst,
          --
          pgpRefClk  => amcRefClk,
          qpllLock   => qplllock  (4*i+3 downto 4*i),
          qpllClk    => qpllclk   (4*i+3 downto 4*i),
          qpllRefClk => qpllrefclk(4*i+3 downto 4*i),
          qpllRst    => qpllrst   (4*i+3 downto 4*i) );
      
      amcQuad(i).coreClk       <= amcCoreClk;
      amcQuad(i).refClk        <= amcRefClk;
      amcQuad(i).qplloutclk    <= qpllclk   (4*i);
      amcQuad(i).qplloutrefclk <= qpllrefclk(4*i);
      amcQuad(i).qplllock      <= qplllock  (4*i);
    end generate;
  --
  --  The AMC SFP channels are reordered - the mapping to MGT quads is non-trivial
  --    amcTx/Rx indexed by MGT
  --    iamcTx/Rx indexed by SFP
  --
  reorder_p : process (amcRxP,amcRxN,chanTxP,chanTxN,
                       amcCoreClk,amcRefClk,chanPllRst,amcQuad) is
  begin
    for j in 0 to 3 loop
      amcTxP    (j)   <= chanTxP(j+2);
      amcTxN    (j)   <= chanTxN(j+2);
      chanRxP   (j+2) <= amcRxP(j);
      chanRxN   (j+2) <= amcRxN(j);
      qpllrst   (j)   <= chanPllRst(j+2);
      chanQuad  (j+2) <= amcQuad(0);
    end loop;
    for j in 4 to 5 loop
      amcTxP    (j)   <= chanTxP(j-4);
      amcTxN    (j)   <= chanTxN(j-4);
      chanRxP   (j-4) <= amcRxP(j);
      chanRxN   (j-4) <= amcRxN(j);
      qpllrst   (j-4) <= chanPllRst(j-4);
      chanQuad  (j-4) <= amcQuad(1);
    end loop;
    for j in 6 to 6 loop
      amcTxP    (j) <= chanTxP(j);
      amcTxN    (j) <= chanTxN(j);
      chanRxP   (j) <= amcRxP(j);
      chanRxN   (j) <= amcRxN(j);
      qpllrst   (j) <= chanPllRst(j);
      chanQuad  (j) <= amcQuad(1);
    end loop;
  end process;

end mapping;
