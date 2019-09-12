-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmSeqMemReg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-25
-- Last update: 2018-03-24
-- Platform   : 
-- Standard   : VHDL'93/02
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS2 Timing Firmware'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS2 Timing Firmware', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TPGPkg.all;
use work.XpmSeqPkg.all;

entity XpmSeqMemReg is
   generic (
      TPD_G            : time            := 1 ns;
      USE_WSTRB_G      : boolean         := false;
      ADDR_BITS_G      : natural         := 14;
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
end XpmSeqMemReg;

architecture rtl of XpmSeqMemReg is

   type RegType is record
                     config            : XpmSeqConfigType;
                     seqRd             : sl;
                     seqRdSeq          : slv(3 downto 0);
                     seqState          : SequencerState;
                     axiReadSlave      : AxiLiteReadSlaveType;
                     axiWriteSlave     : AxiLiteWriteSlaveType;
                     axiRdEn           : slv(1 downto 0);
   end record RegType;
   
   constant REG_INIT_C : RegType := (
     config            => XPM_SEQ_CONFIG_INIT_C,
     seqRd             => '0',
     seqRdSeq          => (others=>'0'),
     seqState          => SEQUENCER_STATE_INIT_C,
     axiReadSlave      => AXI_LITE_READ_SLAVE_INIT_C,
     axiWriteSlave     => AXI_LITE_WRITE_SLAVE_INIT_C,
     axiRdEn           => (others=>'0') );

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

begin

   -------------------------------
   -- Configuration Register
   -------------------------------  
   comb : process (axiReadMaster, axiRst, axiWriteMaster, r, status) is
      variable v            : RegType;
      variable axiStatus    : AxiLiteStatusType;
      variable axiWriteResp : slv(1 downto 0);
      variable axiReadResp  : slv(1 downto 0);
      variable rdPntr       : natural;
      variable wrPntr       : natural;
      variable iseq         : natural;
      variable ichn         : natural;
      variable regWrData    : slv(31 downto 0);
      variable tmpRdData    : slv(31 downto 0);
      variable regAddr      : slv(31 downto 2);
   begin
      -- Latch the current value
      v := r;

      -- Calculate the address pointers
      wrPntr := conv_integer(axiWriteMaster.awaddr(ADDR_BITS_G-1 downto 2));
      rdPntr := conv_integer(axiReadMaster .araddr(ADDR_BITS_G-1 downto 2));
      
      -- Reset strobing signals
      v.config.seqWrEn       := (others=>'0');
      v.seqRd                := '0';
      v.seqRdSeq             := (others=>'0');
      v.axiRdEn              := (others=>'0');
      
      -----------------------------
      -- AXI-Lite Write Logic
      -----------------------------      

      axiSlaveWaitWriteTxn(axiWriteMaster,v.axiWriteSlave,axiStatus.writeEnable);

      if (axiStatus.writeEnable = '1') then
        -- Check for alignment
        if axiWriteMaster.awaddr(1 downto 0) = "00" then
          -- Update external data/address buses
          v.config.seqWrData    := axiWriteMaster.wdata;
          -- Address is aligned
          regAddr   := axiWriteMaster.awaddr(regAddr'range);
          regWrData := axiWriteMaster.wdata;
          axiWriteResp          := AXI_RESP_OK_C;
          case wrPntr is
            when 0 to XPMSEQDEPTH*2048-1 =>
              iseq := conv_integer(regAddr(ADDR_BITS_G-1 downto SEQADDRLEN+2));
              v.config.seqWrEn(iseq) := '1';
              v.config.seqAddr       := SeqAddrType(regAddr(SEQADDRLEN+1 downto 2));
            when others  =>  axiWriteResp := AXI_ERROR_RESP_G;
          end case;
          axiSlaveWriteResponse(v.axiWriteSlave, axiWriteResp);
        else                            -- if axiWriteMaster.awaddr(1 downto 0) = "00"
          axiSlaveWriteResponse(v.axiWriteSlave, AXI_ERROR_RESP_G);
        end if;
      end if;
      
      -----------------------------
      -- AXI-Lite Read Logic
      -----------------------------      

      axiSlaveWaitReadTxn(axiReadMaster,v.axiReadSlave,axiStatus.readEnable);

      if (axiStatus.readEnable = '1') then
        -- Reset the bus
        v.axiReadSlave.rdata := (others => '0');
        regAddr   := axiReadMaster.araddr(regAddr'range);
        tmpRdData := (others=>'0');
        -- Check for alignment
        if axiReadMaster.araddr(1 downto 0) = "00" then
          -- Update external data/address buses
          v.config.seqAddr := SeqAddrType(regAddr(SEQADDRLEN+1 downto 2));
          -- Address is aligned
          axiReadResp           := AXI_RESP_OK_C;
          -- BRAM 2 cycles read delay
          v.axiRdEn             := r.axiRdEn(0) & '1';
          -- Check if BRAM is valid
          if r.axiRdEn(1) = '1' then
            
            -- Decode the read address
            case rdPntr is
              when 0 to XPMSEQDEPTH*2048-1 =>
                iseq       := conv_integer(regAddr(ADDR_BITS_G-1 downto SEQADDRLEN+2));
                v.seqRd    := '1';
                v.seqRdSeq := std_logic_vector(conv_unsigned(iseq,v.seqRdSeq'length));
              when others     => tmpRdData := x"DEAD" & regAddr(15 downto 2) & "00";
            end case;
            v.axiReadSlave.rdata := tmpRdData;
            -- Send AXI response
            axiSlaveReadResponse(v.axiReadSlave, axiReadResp);
          end if;
        else
          axiSlaveReadResponse(v.axiReadSlave, AXI_ERROR_RESP_G);
        end if;
      end if;
      
      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs
      axiWriteSlave   <= r.axiWriteSlave;
      axiReadSlave    <= r.axiReadSlave;
      if r.seqRd='1' then
        axiReadSlave.rdata  <= status.seqRdData(conv_integer(r.seqRdSeq));
      end if;

      config          <= r.config;
      config.seqAddr  <= v.config.seqAddr;
   end process comb;

   seq : process (axiClk) is
   begin
      if rising_edge(axiClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;

end rtl;
