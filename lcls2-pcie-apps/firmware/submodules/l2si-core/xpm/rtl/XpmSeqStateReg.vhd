-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmSeqStateReg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-25
-- Last update: 2018-03-24
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
use ieee.numeric_std.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TPGPkg.all;
use work.XpmSeqPkg.all;

entity XpmSeqStateReg is
   generic (
      TPD_G            : time            := 1 ns;
      USE_WSTRB_G      : boolean         := false;
      AXI_ERROR_RESP_G : slv(1 downto 0) := AXI_RESP_OK_C);      
   port (
      -- AXI-Lite Interface
      axiReadMaster  : in  AxiLiteReadMasterType;
      axiReadSlave   : out AxiLiteReadSlaveType;
      axiWriteMaster : in  AxiLiteWriteMasterType;
      axiWriteSlave  : out AxiLiteWriteSlaveType;
      -- EVR Interface      
      status         : in  XpmSeqStatusType;
      config         : out XpmSeqConfigType;
      -- Clock and Reset
      axiClk         : in  sl;
      axiRst         : in  sl);
end XpmSeqStateReg;

architecture rtl of XpmSeqStateReg is

   type RegType is record
     config            : XpmSeqConfigType;
     seqState          : SequencerState;
     axiReadSlave      : AxiLiteReadSlaveType;
     axiWriteSlave     : AxiLiteWriteSlaveType;
   end record RegType;
   
   constant REG_INIT_C : RegType := (
     config            => XPM_SEQ_CONFIG_INIT_C,
     seqState          => SEQUENCER_STATE_INIT_C,
     axiReadSlave      => AXI_LITE_READ_SLAVE_INIT_C,
     axiWriteSlave     => AXI_LITE_WRITE_SLAVE_INIT_C);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   -------------------------------
   -- Configuration Register
   -------------------------------  
   comb : process (axiReadMaster, axiRst, axiWriteMaster, r, status) is
      variable v            : RegType;
      variable ep           : AxiLiteEndpointType;
   begin
      -- Latch the current value
      v := r;
      v.config.SeqRestart := (others=>'0');
      
      axiSlaveWaitTxn(ep, axiWriteMaster, axiReadMaster, v.axiWriteSlave, v.axiReadSlave);

      axiSlaveRegisterR(ep, toSlv(0,12), 0, toSlv(SEQADDRLEN ,4));
      axiSlaveRegisterR(ep, toSlv(0,12),16, toSlv(MAXEXPSEQDEPTH,8));
      axiSlaveRegisterR(ep, toSlv(0,12),24, toSlv(XPMSEQDEPTH,8));
      axiSlaveRegister (ep, toSlv(4,12), 0, v.config.seqEnable);
      axiSlaveRegister (ep, toSlv(8,12), 0, v.config.seqRestart);
      
      for i in 0 to XPMSEQDEPTH-1 loop
        if std_match(axiReadMaster.araddr(6 downto 4),toSlv(i,3)) then
          v.seqState := status.seqState(i);
        end if;

        axiSlaveRegisterR(ep, toSlv(i*16+128,12), 0, status.countRequest(i));
        axiSlaveRegisterR(ep, toSlv(i*16+132,12), 0, status.countInvalid(i));
        axiSlaveRegisterR(ep, toSlv(i*16+136,12), 0, slv(status.seqState(i).index));
        for j in 0 to 3 loop
          axiSlaveRegisterR(ep, toSlv(i*16+140,12), j*4, r.seqState.count(j));
        end loop;
      end loop;

      axiSlaveDefault(ep, v.axiWriteSlave, v.axiReadSlave );

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs
      axiWriteSlave   <= r.axiWriteSlave;
      axiReadSlave    <= r.axiReadSlave;

      config          <= r.config;
   end process comb;

   seq : process (axiClk) is
   begin
      if rising_edge(axiClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end rtl;
