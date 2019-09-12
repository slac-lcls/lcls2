-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : XpmSeqJumpReg.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-09-25
-- Last update: 2018-03-23
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
use work.TPGPkg.all;
use work.XpmSeqPkg.all;

entity XpmSeqJumpReg is
   generic (
      TPD_G            : time            := 1 ns;
      USE_WSTRB_G      : boolean         := false;
      ADDR_BITS_G      : natural         := 8;
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
end XpmSeqJumpReg;

architecture rtl of XpmSeqJumpReg is

   type RegType is record
                     config            : XpmSeqConfigType;
                     axiReadSlave      : AxiLiteReadSlaveType;
                     axiWriteSlave     : AxiLiteWriteSlaveType;
   end record RegType;
   
   constant REG_INIT_C : RegType := (
     config            => XPM_SEQ_CONFIG_INIT_C,
     axiReadSlave      => AXI_LITE_READ_SLAVE_INIT_C,
     axiWriteSlave     => AXI_LITE_WRITE_SLAVE_INIT_C );

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
      
      -- Determine the transaction type

      -----------------------------
      -- AXI-Lite Write Logic
      -----------------------------      
      axiSlaveWaitWriteTxn(axiWriteMaster,v.axiWriteSlave,axiStatus.writeEnable);

      if (axiStatus.writeEnable = '1') then
        -- Check for alignment
        if axiWriteMaster.awaddr(1 downto 0) = "00" then
          -- Address is aligned
            regAddr   := axiWriteMaster.awaddr(regAddr'range);
            regWrData := axiWriteMaster.wdata;
            axiWriteResp          := AXI_RESP_OK_C;
            case wrPntr is
              when 0 to XPMSEQDEPTH*16-1 =>
                iseq := conv_integer(regAddr(ADDR_BITS_G-1 downto 6));
                ichn := conv_integer(regAddr( 5 downto 2));
                if (iseq >= Allow'right and iseq <= Allow'left) then
                  case ichn is
                    when 14     => v.config.seqJumpConfig(iseq).bcsClass  := regWrData(15 downto 12);
                                   v.config.seqJumpConfig(iseq).bcsJump   := SeqAddrType(regWrData(SeqAddrType'range));
                    when 15     => v.config.seqJumpConfig(iseq).syncSel   := regWrData(31 downto 16);
                                   v.config.seqJumpConfig(iseq).syncClass := regWrData(15 downto 12);
                                   v.config.seqJumpConfig(iseq).syncJump := SeqAddrType(regWrData(SeqAddrType'range));
                    when others =>
                      v.config.seqJumpConfig(iseq).mpsClass(ichn) := regWrData(15 downto 12);
                      v.config.seqJumpConfig(iseq).mpsJump (ichn) := SeqAddrType(regWrData(SeqAddrType'range));
                  end case;
                elsif (ichn=15) then
                  v.config.seqJumpConfig(iseq).syncSel  := regWrData(31 downto 16);
                  v.config.seqJumpConfig(iseq).syncClass:= regWrData(15 downto 12);
                  v.config.seqJumpConfig(iseq).syncJump := SeqAddrType(regWrData(SeqAddrType'range));
                end if;
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
          -- Address is aligned
          axiReadResp           := AXI_RESP_OK_C;
          -- Decode the read address
          case rdPntr is
            when 0 to XPMSEQDEPTH*16-1 =>
              iseq := conv_integer(regAddr(ADDR_BITS_G-1 downto 6));
              ichn := conv_integer(regAddr(5 downto 2));
              if (iseq >= Allow'right and iseq <= Allow'left) then
                case ichn is
                  when 14     => tmpRdData(15 downto 12)      := r.config.seqJumpConfig(iseq).bcsClass;
                                 tmpRdData(SeqAddrType'range) := slv(r.config.seqJumpConfig(iseq).bcsJump);
                  when 15     => tmpRdData(31 downto 16)      := r.config.seqJumpConfig(iseq).syncSel;
                                 tmpRdData(15 downto 12)      := r.config.seqJumpConfig(iseq).syncClass;
                                 tmpRdData(SeqAddrType'range) := slv(r.config.seqJumpConfig(iseq).syncJump);
                  when others => tmpRdData(15 downto 12)      := r.config.seqJumpConfig(iseq).mpsClass(ichn);
                                 tmpRdData(SeqAddrType'range) := slv(r.config.seqJumpConfig(iseq).mpsJump(ichn));
                end case;
              elsif (ichn=15) then
                tmpRdData(31 downto 16)      := r.config.seqJumpConfig(iseq).syncSel;
                tmpRdData(15 downto 12)      := r.config.seqJumpConfig(iseq).syncClass;
                tmpRdData(SeqAddrType'range) := slv(r.config.seqJumpConfig(iseq).syncJump);
              end if;
            when others     => tmpRdData := x"DEAD" & regAddr(15 downto 2) & "00";
          end case;
          v.axiReadSlave.rdata := tmpRdData;
          -- Send AXI response
          axiSlaveReadResponse(v.axiReadSlave, axiReadResp);
        else
          axiSlaveReadResponse(v.axiReadSlave, AXI_ERROR_RESP_G);
        end if;
      end if;
      
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
