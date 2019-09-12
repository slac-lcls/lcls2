-------------------------------------------------------------------------------
-- Title      : 
-------------------------------------------------------------------------------
-- File       : TPGMiniCore.vhd
-- Author     : Matt Weaver  <weaver@slac.stanford.edu>
-- Company    : SLAC National Accelerator Laboratory
-- Created    : 2015-11-09
-- Last update: 2016-07-09
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

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.TPGPkg.all;
use work.TPGMiniEdefPkg.all;

entity TPGMiniCore is
   generic (
      TPD_G           : time    := 1 ns;
      NARRAYSBSA      : natural := 1);
   port (
     txClk           : in  sl;
     txRst           : in  sl;
     txRdy           : in  sl;
     txData          : out Slv16Array(1 downto 0);
     txDataK         : out Slv2Array (1 downto 0);
     txPolarity      : out sl;
     txResetO        : out sl;
     txLoopback      : out slv( 2 downto 0);
     txInhibit       : out sl;
     axiClk          : in  sl;
     axiRst          : in  sl;
     axiReadMaster   : in  AxiLiteReadMasterType;
     axiReadSlave    : out AxiLiteReadSlaveType;
     axiWriteMaster  : in  AxiLiteWriteMasterType;
     axiWriteSlave   : out AxiLiteWriteSlaveType );
end TPGMiniCore;

architecture rtl of TPGMiniCore is

   signal status : TPGStatusType;
   signal config : TPGConfigType;

   signal regClk            : sl;
   signal regRst            : sl;
   signal regReadMaster     : AxiLiteReadMasterType;
   signal regReadSlave      : AxiLiteReadSlaveType;
   signal regWriteMaster    : AxiLiteWriteMasterType;
   signal regWriteSlave     : AxiLiteWriteSlaveType;
   signal edefConfig        : TPGMiniEdefConfigType;

begin  -- rtl

   regClk     <= txClk;
   regRst     <= txRst;
   txPolarity <= config.txPolarity;
   
   U_AxiLiteAsync : entity work.AxiLiteAsync
      generic map (
         TPD_G => TPD_G)
      port map (
         -- Slave Port
         sAxiClk         => axiClk,
         sAxiClkRst      => axiRst,
         sAxiReadMaster  => axiReadMaster,
         sAxiReadSlave   => axiReadSlave,
         sAxiWriteMaster => axiWriteMaster,
         sAxiWriteSlave  => axiWriteSlave,
         -- Master Port
         mAxiClk         => regClk,
         mAxiClkRst      => regRst,
         mAxiReadMaster  => regReadMaster,
         mAxiReadSlave   => regReadSlave,
         mAxiWriteMaster => regWriteMaster,
         mAxiWriteSlave  => regWriteSlave);     
  
   TPGMiniReg_Inst : entity work.TPGMiniReg
      generic map (
         TPD_G       => TPD_G,
         NARRAYS_BSA => NARRAYSBSA)
      port map (
         axiClk         => regClk,
         axiRst         => regRst,
         axiReadMaster  => regReadMaster,
         axiReadSlave   => regReadSlave,
         axiWriteMaster => regWriteMaster,
         axiWriteSlave  => regWriteSlave,
         status         => status,
         config         => config,
         edefConfig     => edefConfig,
         txReset        => txResetO,
         txLoopback     => txLoopback,
         txInhibit      => txInhibit,
         irqActive      => '0',
         irqEnable      => open,
         irqReq         => open );

   TPGMini_Inst : entity work.TPGMini
      generic map (
         TPD_G          => TPD_G,
         NARRAYSBSA     => NARRAYSBSA )
      port map (
         -- Register Interface
         statusO        => status,
         configI        => config,
         -- TPG Interface
         txClk          => txClk,
         txRst          => txRst,
         txRdy          => txRdy,
         txData         => txData (1),
         txDataK        => txDataK(1) );

   TPGMiniStream_Inst : entity work.TPGMiniStream
      generic map (
         TPD_G          => TPD_G)   
      port map (
         -- Register Interface
         config         => config,
         edefConfig     => edefConfig,
         -- TPG Interface
         txClk          => txClk,
         txRst          => txRst,
         txRdy          => txRdy,
         txData         => txData (0),
         txDataK        => txDataK(0) );

end rtl;
