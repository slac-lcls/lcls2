-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : EvrV2ChannelReg.vhd
-- Author     : Matt Weaver <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2016-01-04
-- Last update: 2019-03-14
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
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TimingPkg.all;
use work.EvrV2Pkg.all;

entity EvrV2ChannelReg is
   generic (
      TPD_G        : time    := 1 ns;
      NCHANNELS_G  : integer := 1;
      EVR_CARD_G   : boolean := false;  -- false = packs registers in tight 256B for small BAR0 applications, true = groups registers in 4kB boundary to "virtualize" the channels allowing separate processes to memory map the register space for their dedicated channels.
      DMA_ENABLE_G : boolean := false);
   port (
      -- AXI-Lite and IRQ Interface
      axiClk          : in  sl;
      axiRst          : in  sl;
      axilWriteMaster : in  AxiLiteWriteMasterType;
      axilWriteSlave  : out AxiLiteWriteSlaveType;
      axilReadMaster  : in  AxiLiteReadMasterType;
      axilReadSlave   : out AxiLiteReadSlaveType;
      -- configuration
      channelConfig   : out EvrV2ChannelConfigArray(NCHANNELS_G-1 downto 0);
      -- status
      eventCount      : in  Slv32Array(NCHANNELS_G-1 downto 0));
end EvrV2ChannelReg;

architecture rtl of EvrV2ChannelReg is

   constant STRIDE_C : positive := ite(EVR_CARD_G, 17, 12);
   constant GRP_C    : positive := ite(EVR_CARD_G, 4096, 256);

   type RegType is record
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
      channelConfig  : EvrV2ChannelConfigArray(NCHANNELS_G-1 downto 0);
   end record;
   constant REG_INIT_C : RegType := (
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C,
      channelConfig  => (others => EVRV2_CHANNEL_CONFIG_INIT_C));

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   comb : process (axiRst, axilReadMaster, axilWriteMaster, eventCount, r)
      variable v      : RegType;
      variable axilEp : AxiLiteEndpointType;
      variable i      : natural;
   begin
      -- Latch the current value
      v := r;

      -- Determine the transaction type
      axiSlaveWaitTxn(axilEp, axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave);

      -- Loop through the channels
      for i in 0 to NCHANNELS_G-1 loop

         axiSlaveRegister (axilEp, toSlv(i*GRP_C+ 0, STRIDE_C), 0, v.channelConfig(i).enabled);
         axiSlaveRegister (axilEp, toSlv(i*GRP_C+ 4, STRIDE_C), 0, v.channelConfig(i).rateSel);
         axiSlaveRegister (axilEp, toSlv(i*GRP_C+ 4, STRIDE_C), 13, v.channelConfig(i).destSel);
         axiSlaveRegisterR(axilEp, toSlv(i*GRP_C+ 8, STRIDE_C), 0, eventCount(i));

         if DMA_ENABLE_G then
            axiSlaveRegister(axilEp, toSlv(i*GRP_C+ 0, STRIDE_C), 1, v.channelConfig(i).bsaEnabled);
            axiSlaveRegister(axilEp, toSlv(i*GRP_C+ 0, STRIDE_C), 2, v.channelConfig(i).dmaEnabled);
            axiSlaveRegister(axilEp, toSlv(i*GRP_C+12, STRIDE_C), 0, v.channelConfig(i).bsaActiveDelay);
            axiSlaveRegister(axilEp, toSlv(i*GRP_C+12, STRIDE_C), 20, v.channelConfig(i).bsaActiveSetup);
            axiSlaveRegister(axilEp, toSlv(i*GRP_C+16, STRIDE_C), 0, v.channelConfig(i).bsaActiveWidth);
         end if;

      end loop;

      -- Close the transaction
      axiSlaveDefault(axilEp, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_OK_C);

      -- Outputs 
      axilReadSlave  <= r.axilReadSlave;
      axilWriteSlave <= r.axilWriteSlave;
      channelConfig  <= r.channelConfig;

      -- Reset
      if (axiRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

   end process comb;

   seq : process (axiClk) is
   begin
      if (rising_edge(axiClk)) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end architecture rtl;
