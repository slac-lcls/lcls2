-------------------------------------------------------------------------------
-- File       : EvrV1CoreIsrCtrl.vhd
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2017-03-03
-- Last update: 2018-02-12
-------------------------------------------------------------------------------
-- Description: 
-------------------------------------------------------------------------------
-- This file is part of 'LCLS1 Timing Core'.
-- It is subject to the license terms in the LICENSE.txt file found in the 
-- top-level directory of this distribution and at: 
--    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
-- No part of 'LCLS1 Timing Core', including this file, 
-- may be copied, modified, propagated, or distributed except according to 
-- the terms contained in the LICENSE.txt file.
-------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_arith.all;

use work.StdRtlPkg.all;
use work.AxiStreamPkg.all;
use work.SsiPkg.all;
use work.AxiLitePkg.all;
use work.AxiLiteMasterPkg.all;

entity EvrV1CoreIsrCtrl is
   generic (
      TPD_G                 : time                := 1 ns;
      DEFAULT_ISR_SEL_G     : sl                  := '1';  -- '1' = SW, '0' = FW
      TDEST_IRQ_MSG_G       : slv(7 downto 0)     := x"FF";
      TDEST_DATABUF_MSG_G   : slv(7 downto 0)     := x"F5";
      TDEST_PULSE_MSG_G     : slv(7 downto 0)     := x"F4";
      TDEST_EVENT_MSG_G     : slv(7 downto 0)     := x"F3";
      TDEST_HEARTBEAT_MSG_G : slv(7 downto 0)     := x"F2";
      TDEST_FIFOFULL_MSG_G  : slv(7 downto 0)     := x"F1";
      TDEST_VIOLATION_MSG_G : slv(7 downto 0)     := x"F0";
      BRAM_EN_G             : boolean             := true;
      FIFO_ADDR_WIDTH_G     : positive            := 9;
      REM_BASE_ADDR_G       : slv(31 downto 0)    := (others => '0');
      AXIS_CONFIG_G         : AxiStreamConfigType := ssiAxiStreamConfig(4));
   port (
      -- AXI-Lite and 
      axilClk          : in  sl;
      axilRst          : in  sl;
      axilReadMaster   : in  AxiLiteReadMasterType;
      axilReadSlave    : out AxiLiteReadSlaveType;
      axilWriteMaster  : in  AxiLiteWriteMasterType;
      axilWriteSlave   : out AxiLiteWriteSlaveType;
      mAxilReadMaster  : out AxiLiteReadMasterType;
      mAxilReadSlave   : in  AxiLiteReadSlaveType;
      mAxilWriteMaster : out AxiLiteWriteMasterType;
      mAxilWriteSlave  : in  AxiLiteWriteSlaveType;
      mAxisMaster      : out AxiStreamMasterType;
      mAxisSlave       : in  AxiStreamSlaveType;
      -- IRQ Interface
      irqActive        : out sl;
      irqEnable        : in  sl;
      irqReq           : in  sl;
      -- EVR Interface
      evrClk           : in  sl;
      evrRst           : in  sl;
      gtLinkUp         : in  sl;
      gtRxData         : in  slv(15 downto 0);
      gtRxDataK        : in  slv(1 downto 0);
      gtRxDispErr      : in  slv(1 downto 0);
      gtRxDecErr       : in  slv(1 downto 0);
      rxLinkUp         : out sl;
      rxError          : out sl;
      rxData           : out slv(15 downto 0);
      rxDataK          : out slv(1 downto 0));
end EvrV1CoreIsrCtrl;

architecture rtl of EvrV1CoreIsrCtrl is

   constant AXIS_CONFIG_C : AxiStreamConfigType := ssiAxiStreamConfig(4);

   constant WR_C : sl := '0';           -- Write operation
   constant RD_C : sl := '1';           -- Read operation

   constant CTRL_ADDR_C           : slv(31 downto 0) := (REM_BASE_ADDR_G + x"004");
   constant IRQ_FLAG_ADDR_C       : slv(31 downto 0) := (REM_BASE_ADDR_G + x"008");
   constant DBUF_CTRL_ADDR_C      : slv(31 downto 0) := (REM_BASE_ADDR_G + x"020");
   constant FIFO_SECONDS_ADDR_C   : slv(31 downto 0) := (REM_BASE_ADDR_G + x"070");
   constant FIFO_TIMESTAMP_ADDR_C : slv(31 downto 0) := (REM_BASE_ADDR_G + x"074");
   constant FIFO_EVENT_ADDR_C     : slv(31 downto 0) := (REM_BASE_ADDR_G + x"078");
   constant DATA_BUF_ADDR_C       : slv(31 downto 0) := (REM_BASE_ADDR_G + x"800");

   constant IRQFLAG_DATABUF_C   : natural := 5;
   constant IRQFLAG_PULSE_C     : natural := 4;
   constant IRQFLAG_EVENT_C     : natural := 3;
   constant IRQFLAG_HEARTBEAT_C : natural := 2;
   constant IRQFLAG_FIFOFULL_C  : natural := 1;
   constant IRQFLAG_VIOLATION_C : natural := 0;

   constant FIFO_EVENT_LIMIT_C : positive := 256;

   type GtRegType is record
      rxLinkUp : sl;
      rxData   : slv(15 downto 0);
      rxDataK  : slv(1 downto 0);
      cnt      : slv(23 downto 0);
   end record GtRegType;
   constant GT_REG_INIT_C : GtRegType := (
      rxLinkUp => '0',
      rxData   => (others => '0'),
      rxDataK  => (others => '0'),
      cnt      => (others => '0'));

   signal gtR   : GtRegType := GT_REG_INIT_C;
   signal gtRin : GtRegType;

   type StateType is (
      IDLE_S,
      SEND_IRQ_MSG_S,
      SW_ISR_S,
      IRQ_FLAG_S,
      IRQ_CLR_REQ_S,
      IRQ_CLR_ACK_S,
      FW_ISR_S,
      IRQ_DBUF0_S,
      IRQ_DBUF1_S,
      IRQ_DBUF2_S,
      IRQ_DBUF3_S,
      IRQ_DBUF4_S,
      IRQ_DBUF5_S,
      IRQ_DBUF6_S,
      IRQ_DBUF7_S,
      IRQ_DBUF8_S,
      IRQ_DBUF9_S,
      IRQ_DBUF10_S,
      IRQ_EVENT0_S,
      IRQ_EVENT1_S,
      IRQ_EVENT2_S,
      IRQ_EVENT3_S,
      IRQ_EVENT4_S,
      IRQ_EVENT5_S,
      IRQ_EVENT6_S,
      IRQ_EVENT7_S,
      IRQ_FIFOFULL0_S,
      IRQ_FIFOFULL1_S,
      IRQ_FIFOFULL2_S,
      IRQ_FIFOFULL3_S,
      FW_ISR_RTN_S);

   type RegType is record
      isrSelect      : sl;
      cnt            : slv(9 downto 0);
      dbufSize       : slv(9 downto 0);
      irqCnt         : slv(31 downto 0);
      irqflags       : slv(31 downto 0);
      isrCnt         : slv(3 downto 0);
      irqActive      : sl;
      axilReadSlave  : AxiLiteReadSlaveType;
      axilWriteSlave : AxiLiteWriteSlaveType;
      txMaster       : AxiStreamMasterType;
      req            : AxiLiteMasterReqType;
      state          : StateType;
   end record RegType;
   constant REG_INIT_C : RegType := (
      isrSelect      => DEFAULT_ISR_SEL_G,
      cnt            => (others => '0'),
      dbufSize       => (others => '0'),
      irqCnt         => (others => '1'),-- preset such that 1st IRQ event is irqCnt=0x0
      irqflags       => (others => '0'),
      isrCnt         => (others => '0'),
      irqActive      => '0',
      axilReadSlave  => AXI_LITE_READ_SLAVE_INIT_C,
      axilWriteSlave => AXI_LITE_WRITE_SLAVE_INIT_C,
      txMaster       => AXI_STREAM_MASTER_INIT_C,
      req            => AXI_LITE_MASTER_REQ_INIT_C,
      state          => IDLE_S);

   signal r   : RegType := REG_INIT_C;
   signal rin : RegType;

   signal txSlave : AxiStreamSlaveType;
   signal ack     : AxiLiteMasterAckType;

begin

   gtComb : process (evrRst, gtLinkUp, gtR, gtRxData, gtRxDataK, gtRxDecErr,
                     gtRxDispErr) is
      variable v         : GtRegType;
      variable dataValid : sl;
   begin
      -- Latch the current value
      v := gtR;

      -- Check if valid data 
      dataValid := not (uOr(gtRxDispErr) or uOr(gtRxDecErr));

      -- Check the counter
      if gtR.cnt = x"FFFFFF" then
         v.rxLinkUp := '1';
         v.rxData   := gtRxData;
         v.rxDataK  := gtRxDataK;
      else
         -- Increment the counter
         v.cnt := gtR.cnt + 1;
      end if;

      -- Reset
      if (evrRst = '1') or (gtLinkUp = '0') or (dataValid = '0') then
         v := GT_REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      gtRin <= v;

      -- Outputs 
      rxLinkUp <= gtR.rxLinkUp;
      rxError  <= not(dataValid) and gtR.rxLinkUp;
      rxData   <= gtR.rxData;
      rxDataK  <= gtR.rxDataK;

   end process gtComb;

   gtSeq : process (evrClk, evrRst) is
   begin
      if rising_edge(evrClk) then
         gtR <= gtRin after TPD_G;
      end if;
   end process gtSeq;

   comb : process (axilReadMaster, axilRst, axilWriteMaster, irqEnable, irqReq,
                   r, txSlave, ack) is
      variable v      : RegType;
      variable axilEp : AxiLiteEndpointType;
   begin
      -- Latch the current value
      v := r;

      -- Reset the flags
      if txSlave.tReady = '1' then
         v.txMaster.tValid := '0';
         v.txMaster.tLast  := '0';
         v.txMaster.tUser  := (others => '0');
         v.txMaster.tKeep  := x"000F";  -- --32-bit interface
      end if;

      ------------------------------      
      -- Slave AXI-Lite Transactions
      ------------------------------      
      -- Determine the transaction type
      axiSlaveWaitTxn(axilEp, axilWriteMaster, axilReadMaster, v.axilWriteSlave, v.axilReadSlave);

      -- Register Transactions
      axiSlaveRegister(axilEp, x"0", 0, v.isrSelect);

      -- Close out the transaction
      axiSlaveDefault(axilEp, v.axilWriteSlave, v.axilReadSlave, AXI_RESP_DECERR_C);

      -- State Machine
      case r.state is
         ----------------------------------------------------------------------
         when IDLE_S =>
            -- Check for interrupt request
            if (irqReq = '1') and (irqEnable = '1') then
               -- Set the flag
               v.irqActive := '1';
               -- Increment the counter
               v.irqCnt    := r.irqCnt + 1;
               -- Next state
               v.state     := SEND_IRQ_MSG_S;
            end if;
            -- Check if IRQ has been serviced (only used by FW ISR)
            if (irqReq = '0') or (irqEnable = '0') then
               -- Reset the flag
               v.irqActive := '0';
            end if;
         ----------------------------------------------------------------------
         when SEND_IRQ_MSG_S =>
            -- Check if ready to move data
            if (v.txMaster.tValid = '0') then
               -- Send the IRQ message
               v.txMaster.tValid             := r.isrSelect;
               v.txMaster.tLast              := '1';
               v.txMaster.tData(31 downto 0) := r.irqCnt;
               v.txMaster.tDest              := TDEST_IRQ_MSG_G;
               ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
               -- Check for software controlled ISR
               if (r.isrSelect = '1') then
                  -- Next state
                  v.state := SW_ISR_S;
               else
                  -- Get the IRQ flags
                  v.req.request := '1';
                  v.req.rnw     := RD_C;
                  v.req.address := IRQ_FLAG_ADDR_C;
                  -- Next state
                  v.state       := IRQ_FLAG_S;
               end if;
            end if;
         ----------------------------------------------------------------------
         when SW_ISR_S =>
            -- Check if IRQ has been serviced by software
            if (irqReq = '0') or (irqEnable = '0') then
               -- Reset the flag
               v.irqActive := '0';
               -- Next state
               v.state     := IDLE_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_FLAG_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Save the read data
               v.irqflags    := ack.rdData;
               -- Next state
               v.state       := IRQ_CLR_REQ_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_CLR_REQ_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Clear everything except FIFOFULL, DATABUF and EVENT
               v.req.request                    := '1';
               v.req.rnw                        := WR_C;
               v.req.address                    := IRQ_FLAG_ADDR_C;
               v.req.wrData                     := r.irqflags;
               v.req.wrData(IRQFLAG_FIFOFULL_C) := '0';
               v.req.wrData(IRQFLAG_DATABUF_C)  := '0';
               v.req.wrData(IRQFLAG_EVENT_C)    := '0';
               -- Next state
               v.state                          := IRQ_CLR_ACK_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_CLR_ACK_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := FW_ISR_S;
            end if;
         ----------------------------------------------------------------------
         when FW_ISR_S =>
            -- AXI-Lite transaction handshaking and if ready to move data
            if (ack.done = '0') and (v.txMaster.tValid = '0') then
               -- Increment counter
               v.isrCnt := r.isrCnt + 1;
               -- Check for Data buffer interrupt and sending message
               if (r.isrCnt = 0) and (r.irqflags(IRQFLAG_DATABUF_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_DATABUF_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
                  -- Get the DBUF CTRL register
                  v.req.request                 := '1';
                  v.req.rnw                     := RD_C;
                  v.req.address                 := DBUF_CTRL_ADDR_C;
                  -- Next state
                  v.state                       := IRQ_DBUF0_S;
               elsif (r.isrCnt = 1) and (r.irqflags(IRQFLAG_EVENT_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_EVENT_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
                  -- Get the event code
                  v.req.request                 := '1';
                  v.req.rnw                     := RD_C;
                  v.req.address                 := FIFO_EVENT_ADDR_C;
                  -- Next state
                  v.state                       := IRQ_EVENT0_S;
               elsif (r.isrCnt = 2) and (r.irqflags(IRQFLAG_FIFOFULL_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tLast              := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_FIFOFULL_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
                  -- Get the control word
                  v.req.request                 := '1';
                  v.req.rnw                     := RD_C;
                  v.req.address                 := CTRL_ADDR_C;
                  -- Next state
                  v.state                       := IRQ_FIFOFULL0_S;
               elsif (r.isrCnt = 3) and (r.irqflags(IRQFLAG_PULSE_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tLast              := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_PULSE_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
               elsif (r.isrCnt = 4) and (r.irqflags(IRQFLAG_HEARTBEAT_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tLast              := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_HEARTBEAT_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
               elsif (r.isrCnt = 5) and (r.irqflags(IRQFLAG_VIOLATION_C) = '1') then
                  -- Send the IRQ message
                  v.txMaster.tValid             := '1';
                  v.txMaster.tLast              := '1';
                  v.txMaster.tData(31 downto 0) := r.irqCnt;
                  v.txMaster.tDest              := TDEST_VIOLATION_MSG_G;
                  ssiSetUserSof(AXIS_CONFIG_C, v.txMaster, '1');
               else
                  -- Reset the counter
                  v.isrCnt := (others => '0');
                  -- Next state
                  v.state  := IDLE_S;
               end if;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF0_S =>
            -- AXI-Lite transaction handshaking and if ready to move data            
            if (ack.done = '1') and (v.txMaster.tValid = '0') then
               -- Reset the flag
               v.req.request                 := '0';
               -- Send the DBUF CTRL register
               v.txMaster.tValid             := '1';
               v.txMaster.tData(31 downto 0) := ack.rdData;
               -- Save the buffer size (units of 32-bit words)
               v.dbufSize                    := ack.rdData(11 downto 2);
               -- Check if zero buffer or checksum error
               if (v.dbufSize = 0) or (ack.rdData(13) = '1') then
                  -- Terminate the AXIS frame
                  v.txMaster.tLast := '1';
                  -- Next state
                  v.state          := FW_ISR_S;
               else
                  -- Next state
                  v.state := IRQ_DBUF1_S;
               end if;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF1_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the DBUF CTRL register
               v.req.request := '1';
               v.req.rnw     := RD_C;
               -- Check for first buffer read transaction 
               if (r.cnt = 0) then
                  -- Initialize
                  v.req.address := DATA_BUF_ADDR_C;
               else
                  -- Increment by 32-bit word (4 bytes)
                  v.req.address := r.req.address+4;
               end if;
               -- Next state
               v.state := IRQ_DBUF2_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF2_S =>
            -- AXI-Lite transaction handshaking and if ready to move data            
            if (ack.done = '1') and (v.txMaster.tValid = '0') then
               -- Reset the flag
               v.req.request                 := '0';
               -- Send the data buffer
               v.txMaster.tValid             := '1';
               v.txMaster.tData(31 downto 0) := ack.rdData;
               -- Increment the counter
               v.cnt                         := r.cnt + 1;
               -- Check for counter
               if (r.cnt = (r.dbufSize-1)) then
                  -- Reset the counter
                  v.cnt            := (others => '0');
                  -- Terminate the AXIS frame
                  v.txMaster.tLast := '1';
                  -- Next state
                  v.state          := IRQ_DBUF3_S;
               else
                  -- Next state
                  v.state := IRQ_DBUF1_S;
               end if;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF3_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the DBUF CTRL register
               v.req.request := '1';
               v.req.rnw     := RD_C;
               v.req.address := DBUF_CTRL_ADDR_C;
               -- Next state
               v.state       := IRQ_DBUF4_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF4_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := IRQ_DBUF5_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF5_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Re-enable the DATABUF
               v.req.request    := '1';
               v.req.rnw        := WR_C;
               v.req.address    := DBUF_CTRL_ADDR_C;
               v.req.wrData     := ack.rdData;
               v.req.wrData(15) := '1';  -- C_EVR_DATABUF_LOAD             
               -- Next state
               v.state          := IRQ_DBUF6_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF6_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := IRQ_DBUF7_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF7_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Set the DATABUF IRQ clear bit
               v.req.request                   := '1';
               v.req.rnw                       := WR_C;
               v.req.address                   := IRQ_FLAG_ADDR_C;
               v.req.wrData                    := (others => '0');
               v.req.wrData(IRQFLAG_DATABUF_C) := '1';
               -- Next state
               v.state                         := IRQ_DBUF8_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF8_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := IRQ_DBUF9_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF9_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Reset the DATABUF IRQ clear bit
               v.req.request := '1';
               v.req.rnw     := WR_C;
               v.req.address := IRQ_FLAG_ADDR_C;
               v.req.wrData  := (others => '0');
               -- Next state
               v.state       := IRQ_DBUF10_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_DBUF10_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := FW_ISR_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT0_S =>
            -- AXI-Lite transaction handshaking and if ready to move data            
            if (ack.done = '1') and (v.txMaster.tValid = '0') then
               -- Reset the flag
               v.req.request                 := '0';
               -- Send the event code
               v.txMaster.tValid             := '1';
               v.txMaster.tData(31 downto 0) := ack.rdData;
               -- Next state
               v.state                       := IRQ_EVENT1_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT1_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the seconds
               v.req.request := '1';
               v.req.rnw     := RD_C;
               v.req.address := FIFO_SECONDS_ADDR_C;
               -- Next state
               v.state       := IRQ_EVENT2_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT2_S =>
            -- AXI-Lite transaction handshaking and if ready to move data            
            if (ack.done = '1') and (v.txMaster.tValid = '0') then
               -- Reset the flag
               v.req.request                 := '0';
               -- Send the seconds
               v.txMaster.tValid             := '1';
               v.txMaster.tData(31 downto 0) := ack.rdData;
               -- Next state
               v.state                       := IRQ_EVENT3_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT3_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the timestamp
               v.req.request := '1';
               v.req.rnw     := RD_C;
               v.req.address := FIFO_TIMESTAMP_ADDR_C;
               -- Next state
               v.state       := IRQ_EVENT4_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT4_S =>
            -- AXI-Lite transaction handshaking and if ready to move data            
            if (ack.done = '1') and (v.txMaster.tValid = '0') then
               -- Reset the flag
               v.req.request                 := '0';
               -- Save the value but don't see it (in case last value)
               v.txMaster.tData(31 downto 0) := ack.rdData;
               -- Next state
               v.state                       := IRQ_EVENT5_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT5_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the IRQ flags
               v.req.request := '1';
               v.req.rnw     := RD_C;
               v.req.address := IRQ_FLAG_ADDR_C;
               -- Next state
               v.state       := IRQ_EVENT6_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT6_S =>
            -- AXI-Lite transaction handshaking (AXIS already ready from  state = IRQ_EVENT4_S)          
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request     := '0';
               -- Send the timestamp
               v.txMaster.tValid := '1';
               -- Increment the counter
               v.cnt             := r.cnt + 1;
               -- Check if not IRQ bit or packet limit 
               if (ack.rdData(IRQFLAG_EVENT_C) = '0') or (r.cnt = (FIFO_EVENT_LIMIT_C-1)) then
                  -- Reset the counter
                  v.cnt            := (others => '0');
                  -- Terminate the AXIS frame
                  v.txMaster.tLast := '1';
                  -- Next state
                  v.state          := FW_ISR_S;
               else
                  -- Next state
                  v.state := IRQ_EVENT7_S;
               end if;
            end if;
         ----------------------------------------------------------------------
         when IRQ_EVENT7_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Get the event code
               v.req.request := '1';
               v.req.rnw     := RD_C;
               v.req.address := FIFO_EVENT_ADDR_C;
               -- Next state
               v.state       := IRQ_EVENT0_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_FIFOFULL0_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := IRQ_FIFOFULL1_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_FIFOFULL1_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Clear everything except FIFOFULL, DATABUF and EVENT
               v.req.request   := '1';
               v.req.rnw       := WR_C;
               v.req.address   := CTRL_ADDR_C;
               v.req.wrData    := ack.rdData;
               -- Set the FIFO reset bit
               v.req.wrData(3) := '1';
               -- Next state
               v.state         := IRQ_FIFOFULL2_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_FIFOFULL2_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := IRQ_FIFOFULL3_S;
            end if;
         ----------------------------------------------------------------------
         when IRQ_FIFOFULL3_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '0') then
               -- Clear the FIFOFULL IRQ bit
               v.req.request                    := '1';
               v.req.rnw                        := WR_C;
               v.req.address                    := IRQ_FLAG_ADDR_C;
               v.req.wrData                     := (others => '0');
               v.req.wrData(IRQFLAG_FIFOFULL_C) := '1';
               -- Next state
               v.state                          := FW_ISR_RTN_S;
            end if;
         ----------------------------------------------------------------------
         when FW_ISR_RTN_S =>
            -- AXI-Lite transaction handshaking
            if (ack.done = '1') then
               -- Reset the flag
               v.req.request := '0';
               -- Next state
               v.state       := FW_ISR_S;
            end if;
      ----------------------------------------------------------------------
      end case;

      -- Reset
      if (axilRst = '1') then
         v := REG_INIT_C;
      end if;

      -- Register the variable for next clock cycle
      rin <= v;

      -- Outputs 
      axilReadSlave  <= r.axilReadSlave;
      axilWriteSlave <= r.axilWriteSlave;

   end process comb;

   seq : process (axilClk) is
   begin
      if rising_edge(axilClk) then
         r <= rin after TPD_G;
      end if;
   end process seq;

   U_AxiLiteMaster : entity work.AxiLiteMaster
      generic map (
         TPD_G => TPD_G)
      port map (
         req             => r.req,
         ack             => ack,
         axilClk         => axilClk,
         axilRst         => axilRst,
         axilWriteMaster => mAxilWriteMaster,
         axilWriteSlave  => mAxilWriteSlave,
         axilReadMaster  => mAxilReadMaster,
         axilReadSlave   => mAxilReadSlave);

   TX_FIFO : entity work.AxiStreamFifoV2
      generic map (
         -- General Configurations
         TPD_G               => TPD_G,
         SLAVE_READY_EN_G    => true,
         VALID_THOLD_G       => 1,
         -- FIFO configurations
         BRAM_EN_G           => BRAM_EN_G,
         GEN_SYNC_FIFO_G     => true,
         FIFO_ADDR_WIDTH_G   => FIFO_ADDR_WIDTH_G,
         -- AXI Stream Port Configurations
         SLAVE_AXI_CONFIG_G  => AXIS_CONFIG_C,
         MASTER_AXI_CONFIG_G => AXIS_CONFIG_G)
      port map (
         -- Slave Port
         sAxisClk    => axilClk,
         sAxisRst    => axilRst,
         sAxisMaster => r.txMaster,
         sAxisSlave  => txSlave,
         -- Master Port
         mAxisClk    => axilClk,
         mAxisRst    => axilRst,
         mAxisMaster => mAxisMaster,
         mAxisSlave  => mAxisSlave);

end rtl;
