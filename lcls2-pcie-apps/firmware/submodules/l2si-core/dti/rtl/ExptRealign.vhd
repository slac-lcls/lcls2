-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : ExptRealign.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-07-10
-- Last update: 2017-10-06
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
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
use work.TimingPkg.all;
use work.XpmPkg.all;

library unisim;
use unisim.vcomponents.all;

entity ExptRealign is
   generic (
      TPD_G               : time                := 1 ns;
      TF_DELAY_G          : slv(6 downto 0)     := toSlv(100,7) );
   port (
     rst             : in  sl;
     clk             : in  sl;
     timingI_strobe  : in  sl;
     timingI_pid     : in  slv(63 downto 0);
     timingI_time    : in  slv(63 downto 0);
     exptBusI        : in  ExptBusType;
     timingO_strobe  : out sl;
     timingO_pid     : out slv(63 downto 0);
     timingO_time    : out slv(63 downto 0);
     exptBusO        : out ExptBusType );
end ExptRealign;

architecture rtl of ExptRealign is

  type RegType is record
    rden   : sl;
    wren   : sl;
    rdaddr : Slv7Array(NPartitions downto 0);
    pdelay : Slv7Array(NPartitions-1 downto 0);
  end record;

  constant REG_INIT_C : RegType := (
    rden   => '0',
    wren   => '0',
    rdaddr => (others=>(others=>'0')),
    pdelay => (others=>(others=>'0')) );

  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;
    
  constant EXPT_INIT_C : slv(47 downto 0) := x"000000008000";

begin

  U_Ram : entity work.SimpleDualPortRam
    generic map ( DATA_WIDTH_G => 129,
                  ADDR_WIDTH_G => 7 )
    port map ( clka   => clk,
               ena    => '1',
               wea    => timingI_strobe,
               addra  => timingI_pid(6 downto 0),
               dina( 63 downto  0) => timingI_pid,
               dina(127 downto 64) => timingI_time,
               dina(128)           => exptBusI.valid,
               clkb   => clk,
               enb    => r.rden,
               addrb  => r.rdaddr(NPartitions),
               doutb( 63 downto  0) => timingO_pid,
               doutb(127 downto 64) => timingO_time,
               doutb(128)           => exptBusO.valid );
  timingO_strobe <= r.rden;
  
  GEN_PART : for i in 0 to NPartitions-1 generate
    U_Ram : entity work.SimpleDualPortRam
    generic map ( DATA_WIDTH_G => 48,
                  ADDR_WIDTH_G => 7,
                  INIT_G       => EXPT_INIT_C)
    port map ( clka   => clk,
               ena    => '1',
               wea    => timingI_strobe,
               addra  => timingI_pid(6 downto 0),
               dina   => exptBusI.message.partitionWord(i),
               clkb   => clk,
               enb    => '1',
               addrb  => r.rdaddr(i),
               doutb  => exptBusO.message.partitionWord(i) );
  end generate;
  exptBusO.message.partitionAddr <= exptBusI.message.partitionAddr;
  
  comb : process( r, rst, timingI_strobe, timingI_pid, exptBusI ) is
    variable v    : RegType;
    variable pvec : slv(47 downto 0); 
  begin
    v := r;

    v.rden      := '0';
    v.wren      := '0';
    
    if timingI_strobe = '1' then
      v.rden   := '1';
      v.rdaddr(NPartitions) := timingI_pid(6 downto 0) - TF_DELAY_G;
      for ip in 0 to NPartitions-1 loop
        v.rdaddr(ip) := timingI_pid(6 downto 0) - TF_DELAY_G + r.pdelay(ip);
        pvec := exptBusI.message.partitionWord(ip);
        if (exptBusI.valid = '1' and
            pvec(15) = '0' and
            toPartitionMsg(pvec).hdr = MSG_DELAY_PWORD) then
          v.pdelay(ip) := toPartitionMsg(pvec).payload(6 downto 0);
        end if;
      end loop;
    end if;

    if rst = '1' then
      v := REG_INIT_C;
    end if;

    r_in <= v;
  end process;
  
  seq : process (clk) is
  begin
    if rising_edge(clk) then
      r <= r_in;
    end if;
  end process;

end rtl;
