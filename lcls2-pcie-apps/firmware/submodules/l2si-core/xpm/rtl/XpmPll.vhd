------------------------------------------------------------------------------
-- This file is part of 'LCLS2 DAQ Software'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 DAQ Software', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
------------------------------------------------------------------------------
library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;
use work.StdRtlPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;

entity XpmPll is
  port ( config      : in    XpmPllConfigType;
         status      : out   XpmPllStatusType;
         frqTbl      : inout sl;
         frqSel      : inout slv(3 downto 0);
         bwSel       : inout slv(1 downto 0);
         inc         : out   sl;
         dec         : out   sl;
         sfOut       : inout slv(1 downto 0);
         rate        : inout slv(1 downto 0);
         bypass      : out   sl;
         pllRst      : out   sl;
         lol         : in    sl;
         los         : in    sl );
end XpmPll;

architecture rtl of XpmPll is

begin

    U_FRQTB : OBUFT
      port map ( O  => frqTbl,
                 I  => config.frqTbl(0),
                 T  => config.frqTbl(1));
    U_FRQSEL3 : OBUFT
      port map ( O  => frqSel(3),
                 I  => config.frqSel(6),
                 T  => config.frqSel(7) );
    U_FRQSEL2 : OBUFT
      port map ( O  => frqSel(2),
                 I  => config.frqSel(4),
                 T  => config.frqSel(5) );
    U_FRQSEL1 : OBUFT
      port map ( O  => frqSel(1),
                 I  => config.frqSel(2),
                 T  => config.frqSel(3) );
    U_FRQSEL0 : OBUFT
      port map ( O  => frqSel(0),
                 I  => config.frqSel(0),
                 T  => config.frqSel(1) );
    U_BWSEL1 : OBUFT
      port map ( O  => bwSel(1),
                 I  => config.bwsel(2),
                 T  => config.bwsel(3) );
    U_BWSEL0 : OBUFT
      port map ( O  => bwSel(0),
                 I  => config.bwsel(0),
                 T  => config.bwsel(1) );
    U_INC : OBUF
      port map ( O  => inc,
                 I  => config.inc );
    U_DEC : OBUF
      port map ( O  => dec,
                 I  => config.dec );
    U_RST : OBUF
      port map ( O  => pllRst,
                 I  => config.rstn );
    U_BYPASS : OBUF
      port map ( O  => bypass,
                 I  => config.bypass );
    U_SFOUT1 : OBUFT
      port map ( O  => sfOut(1),
                 I  => config.sfOut(2),
                 T  => config.sfOut(3) );
    U_SFOUT0 : OBUFT
      port map ( O  => sfOut(0),
                 I  => config.sfOut(0),
                 T  => config.sfOut(1) );
    U_RATE1 : OBUFT
      port map ( O  => rate(1),
                 I  => config.rate(2),
                 T  => config.rate(3) );
    U_RATE0 : OBUFT
      port map ( O  => rate(0),
                 I  => config.rate(0),
                 T  => config.rate(1) );
    U_LOL : IBUF
      port map ( O  => status.lol,
                 I  => lol );
    U_LOS : IBUF
      port map ( O  => status.los,
                 I  => los );

end rtl;
