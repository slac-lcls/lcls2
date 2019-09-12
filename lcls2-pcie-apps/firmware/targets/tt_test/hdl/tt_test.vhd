library ieee;
use ieee.std_logic_1164.all;
--use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
use ieee.std_logic_signed.all;
use ieee.numeric_std.ALL;

use work.StdRtlPkg.all;
use work.AxiLitePkg.all;
use work.AxiStreamPkg.all;

library unisim;
use unisim.vcomponents.all;

entity tt_test is
     port (
      -- System Interface
      sysClk_i        : in    sl;
      sysRst          : in    sl;
      -- DMA Interfaces  (sysClk domain)
      dataOutData     : out   slv(127 downto 0);
      -- AXI-Lite Interface
      axilReadData    : out   slv(31 downto 0) );
end tt_test;

architecture top of tt_test is

  type RegType is record
    req    : AxiLiteReqType;
    master : AxiStreamMasterType;
    count  : slv(7 downto 0);
  end record;

  constant REG_INIT_C : RegType := (
    req    => AXI_LITE_REQ_INIT_C,
    master => AXI_STREAM_MASTER_INIT_C,
    count  => (others=>'0') );

  signal r    : RegType := REG_INIT_C;
  signal r_in : RegType;
  
  -- DMA Interfaces  (sysClk domain)
  signal dataInMaster    :    AxiStreamMasterType;
  signal dataInSlave     :   AxiStreamSlaveType;
  signal dataOutMaster   :   AxiStreamMasterType;
  signal dataOutSlave    :    AxiStreamSlaveType;
  -- AXI-Lite Interface
  signal axilReadMaster  :    AxiLiteReadMasterType;
  signal axilReadSlave   :   AxiLiteReadSlaveType;
  signal axilWriteMaster :    AxiLiteWriteMasterType;
  signal axilWriteSlave  : AxiLiteWriteSlaveType;

  signal sysClk : sl;
  
begin

  U_BUFG : BUFG
    port map ( I   => sysClk_i,
               O   => sysClk );
  
  dataOutData  <= dataOutMaster.tData(dataOutData'range);
  dataOutSlave <= AXI_STREAM_SLAVE_FORCE_C;
  axilReadData <= axilReadSlave.rdata;

  --U_DUT : entity work.FrameIIR
  --  port map (
  --      -- System Interface
  --    sysClk          => sysClk,
  --    sysRst          => sysRst,
      -- DMA Interfaces  (sysClk domain)
  --    dataInMaster    => dataInMaster,
  --    dataInSlave     => dataInSlave,     
  --    dataOutMaster   => dataOutMaster,   
  --    dataOutSlave    => dataOutSlave,    
      -- AXI-Lite Interface
  --    axilReadMaster  => axilReadMaster,  
  --    axilReadSlave   => axilReadSlave,   
  --    axilWriteMaster => axilWriteMaster, 
  --    axilWriteSlave  => axilWriteSlave 
  --    );


   U_FrameSubtractor : entity work.FrameSubtractor
      port map (
         -- System Clock and Reset
         sysClk           => sysClk,
         sysRst           => sysRst,
         -- DMA Interface (sysClk domain)
         dataInMaster     => dataInMaster,
         dataInSlave      => dataInSlave,
         dataOutMaster    => dataOutMaster,
         dataOutSlave     => dataOutSlave,
         -- Pedestal DMA Interfaces  (sysClk domain)
         pedestalInMaster =>  dataInMaster,
         pedestalInSlave  =>  dataInSlave,
         -- AXI-Lite Interface (sysClk domain)
         axilReadMaster  => axilReadMaster,
         axilReadSlave   => axilReadSlave,
         axilWriteMaster => axilWriteMaster ,
         axilWriteSlave  => axilWriteSlave);

  
  U_LiteMaster : entity work.AxiLiteMaster
    port map ( axilClk => sysClk,
               axilRst => sysRst,
               req     => r.req,
               ack     => open,
               axilWriteMaster => axilWriteMaster,
               axilWriteSlave  => axilWriteSlave ,
               axilReadMaster  => axilReadMaster,
               axilReadSlave   => axilReadSlave );


  comb : process ( r, sysRst ) is
    variable v : RegType;
  begin
    v := r;

    v.req.rnw     := '1';
    v.req.request := '0';
    if r.count = 0 then
      v.req.request := '1';
      v.req.address := r.req.address+4;
    end if;
    v.req.wrData := (others=>'0');

    v.master.tValid := '1';
    v.master.tLast  := '0';
    if r.count = 0 then
      v.master.tLast  := '1';
    end if;
    v.master.tData(31 downto 0) := r.master.tData(31 downto 0)+1;
    v.master.tKeep := GenTKeep(16);
    
    if sysRst = '1' then
      v := REG_INIT_C;
    end if;

    r_in <= v;
  end process comb;

  seq : process ( sysClk ) is
  begin
    if rising_edge(sysClk) then
      r <= r_in;
    end if;
  end process seq;

end top;
