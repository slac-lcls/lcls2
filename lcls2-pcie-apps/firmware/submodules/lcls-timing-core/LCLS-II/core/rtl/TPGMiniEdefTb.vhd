-------------------------------------------------------------------------------
-- Title      : TPGMiniEdefTb
-------------------------------------------------------------------------------
-- File       : TPGMiniEdefTb.vhd
-- Author     : Till Straumann <strauman@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2018-03-08
-- Last update: 2018-03-08
-- Platform   :
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description:
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the
-- top-level directory of this distribution and at:
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
-- No part of 'LCLS2 Timing Core', including this file,
-- may be copied, modified, propagated, or distributed except according to
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

use work.StdRtlPkg.all;
use work.TPGPkg.all;
use work.TimingPkg.all;
use work.TextUtilPkg.all;
use work.TPGMiniEdefPkg.all;

entity TPGMiniEdefTb is
end entity TPGMiniEdefTb;

architecture TPGMiniEdefTbImpl of TPGMiniEdefTb is

	signal clk  : sl      := '0';
	signal rst  : sl      := '1';
	signal don  : boolean := false;

	signal cnt  : natural := 0;

	signal patt : TimingDataBuffType;
	signal bsen : sl;

	signal cfg  : TPGConfigType := TPG_CONFIG_INIT_C;

	signal slot : slv(2 downto 0);

	signal evts : slv(69 downto 0);

	signal edefActv, edefAvgDone, edefAllDone, edefInit : sl;

	signal edefConfigInp : TPGMiniEdefConfigType := TPG_MINI_EDEF_CONFIG_INIT_C;

begin

	slot <= patt.dmod(3*32+29+2 downto 3*32 + 29);

	P_CLK : process is
	begin
		clk <= not clk;
		if not don then
           wait for 50 ns;
        else
           wait;
        end if;
	end process P_CLK;

	P_CNT: process(clk) is
		variable ncnt : natural;
	begin
		if ( rising_edge(clk) ) then
			ncnt := cnt + 1;

			case ncnt is
				when 1      => print(" TS 720 BASER INIT ACTV AVGD ALLD");
				when 4      => rst <= '0';

				when 40     =>
					edefConfigInp.wrEn <= '1';
					edefConfigInp.navg <= slv(to_unsigned(2, edefConfigInp.navg'length));
					edefConfigInp.nsmp <= slv(to_unsigned(1, edefConfigInp.nsmp'length));
					edefConfigInp.rate <= slv(to_unsigned(1, edefConfigInp.rate'length));
					edefConfigInp.slot <= slv(to_unsigned(2, edefConfigInp.slot'length));
				when 41     =>
                    edefConfigInp.wrEn <= '0';
				when 1600   => don <= true;
				when others =>
			end case;

			cnt <= ncnt;
		end if;
		
	end process P_CNT;

	P_REP : process (clk) is
	begin
		if ( rising_edge(clk) and bsen = '1' and rst = '0' ) then
			print( str( slot )
			      & " " & "  " & str( patt.dmod(15) )
                  & " " & str(patt.dmod(32*4+20+4 downto 32*4+20))
                  & " " & str(patt.edefInit(3 downto 0))
                  & " " & str(patt.dmod(32*4+3 downto 32*4))
                  & " " & str(patt.edefAvgDn(3 downto 0))
                  & " " & str(patt.edefMinor(23 downto 20))
			);
		end if;
	end process P_REP;

	U_DUT_STREAM : entity work.TPGMiniStream
		generic map (
			TPD_G      => 1 ns,
			AC_PERIOD  => 2
		)
		port map (
			config     => cfg,
			edefConfig => edefConfigInp,
			txClk      => clk,
			txRst      => rst,
			txRdy      => '1',
			simData    => patt,
			simStrobe  => bsen,
			simEvents  => evts
		);


end architecture TPGMiniEdefTbImpl;
