-------------------------------------------------------------------------------
-- File       : AxiStreamBytePackerTbTx.vhd
-- Company    : SLAC National Accelerator Laboratory
-------------------------------------------------------------------------------
-- Description:
-- AxiStream data packer tester, tx module
-------------------------------------------------------------------------------
-- This file is part of 'SLAC Firmware Standard Library'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'SLAC Firmware Standard Library', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.numeric_std.all;
use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;

entity AxiStreamCameraRealSim is
   generic (
      TPD_G           : time                := 1 ns;
      BYTE_SIZE_C     : positive            := 1;
      FRAMES_PER_BYTE : positive            := 6;
      AXIS_CONFIG_G   : AxiStreamConfigType := AXI_STREAM_CONFIG_INIT_C);
   port (
      -- System clock and reset
      axiClk       : in  sl;
      axiRst       : in  sl;
      -- Outbound frame
      mAxisMaster  : out AxiStreamMasterType);
end AxiStreamCameraRealSim;

architecture rtl of AxiStreamCameraRealSim is

   type RegType is record
      byteCount    : natural;
      frameCount   : natural;
      sleepCount   : natural;
      master       : AxiStreamMasterType;
   end record RegType;

   constant REG_INIT_C : RegType := (
      byteCount    => 0,
      frameCount   => 0,
      sleepCount   => 0,
      master       => AXI_STREAM_MASTER_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   comb : process (r, axiRst ) is
      variable v : RegType;
      variable frameCounter_slv : slv(8 downto 0);
   begin
      v := r;

      if v.sleepCount=0 then

            v.master := AXI_STREAM_MASTER_INIT_C;
            v.master.tKeep  := (others=>'0');
            v.master.tValid := '1';

          
            v.master.tData(7 downto 0)   := toSlv(128/((v.frameCount mod 11)*(v.frameCount mod 11)+1 ),8);
                  
         
            v.master.tKeep(BYTE_SIZE_C-1 downto 0) := (others=>'1');
            v.byteCount := v.byteCount + 1;


            --for i in 0 to BYTE_SIZE_C-1 loop
            --   v.master.tData(i*8+7 downto i*8) := toSlv(v.byteCount,8);
            --   v.master.tKeep(i) := '1';
            --   v.byteCount := v.byteCount + 1;
            --end loop;

            if v.byteCount = (FRAMES_PER_BYTE+1)*BYTE_SIZE_C then
               v.master.tLast := '1';
               v.sleepCount   := 4;
               v.byteCount    := 0;
               v.frameCount   := v.frameCount + 1;
            end if;

            -- Reset
            if (axiRst = '1') then
               v := REG_INIT_C;
            end if;
      else
            v.master.tValid     := '0';
            v.master.tLast      := '0';
            v.sleepCount        := v.sleepCount-1;

      end if;

      rin <= v;

      mAxisMaster <= r.master;

   end process;

   seq : process (axiClk) is
   begin  
      if (rising_edge(axiClk)) then
         r <= rin;
      end if;
   end process;

end architecture rtl;
