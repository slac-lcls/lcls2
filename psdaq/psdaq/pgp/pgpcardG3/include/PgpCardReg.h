#ifndef PGPCARDREG_H
#define PGPCARDREG_H

#define NUMBER_OF_LANES 8
class PgpReg {
public:
    __u32 version;       // Software_Addr = 0x000,        Firmware_Addr(13 downto 2) = 0x000
    __u32 serNumLower;   // Software_Addr = 0x004,        Firmware_Addr(13 downto 2) = 0x001
    __u32 serNumUpper;   // Software_Addr = 0x008,        Firmware_Addr(13 downto 2) = 0x002
    __u32 scratch;       // Software_Addr = 0x00C,        Firmware_Addr(13 downto 2) = 0x003
    __u32 cardRstStat;   // Software_Addr = 0x010,        Firmware_Addr(13 downto 2) = 0x004
    __u32 irq;           // Software_Addr = 0x014,        Firmware_Addr(13 downto 2) = 0x005
    __u32 pgpRate;       // Software_Addr = 0x018,        Firmware_Addr(13 downto 2) = 0x006
    __u32 sysSpare0;     // Software_Addr = 0x01C,        Firmware_Addr(13 downto 2) = 0x007
    __u32 txOpCode;      // Software_Addr = 0x020,        Firmware_Addr(13 downto 2) = 0x008
    __u32 txLocPause;    // Software_Addr = 0x024,        Firmware_Addr(13 downto 2) = 0x009
    __u32 txLocOvrFlow;  // Software_Addr = 0x028,        Firmware_Addr(13 downto 2) = 0x00A
    __u32 pciStat[4];    // Software_Addr = 0x038:0x02C,  Firmware_Addr(13 downto 2) = 0x00E:0x00B
    __u32 sysSpare1;     // Software_Addr = 0x03C,        Firmware_Addr(13 downto 2) = 0x00F

    __u32 evrCardStat[3];// Software_Addr = 0x048:0x040,  Firmware_Addr(13 downto 2) = 0x012:0x010
    __u32 evrLinkErrorCount; // Software_Addr = 0x04C,    Firmware_Addr ????
    __u32 evrFiducial;   // Software_addr = 0x050,
    __u32 evrSpare0[11]; // Software_Addr = 0x07C:0x054,  Firmware_Addr(13 downto 2) = 0x01F:0x013

   __u32 pgpCardStat[2];// Software_Addr = 0x084:0x080,  Firmware_Addr(13 downto 2) = 0x021:0x020
    __u32 pgpSpare0[54]; // Software_Addr = 0x15C:0x088,  Firmware_Addr(13 downto 2) = 0x057:0x022

    __u32 fiducials[NUMBER_OF_LANES]; // Software_Addr = 0x17C:0x160,  Firmware_Addr(13 downto 2) = 0x05F:0x058
    __u32 runCode[NUMBER_OF_LANES];   // Software_Addr = 0x19C:0x180,  Firmware_Addr(13 downto 2) = 0x067:0x060
    __u32 acceptCode[NUMBER_OF_LANES];// Software_Addr = 0x1BC:0x1A0,  Firmware_Addr(13 downto 2) = 0x06F:0x068

    __u32 runDelay[NUMBER_OF_LANES];   // Software_Addr = 0x1DC:0x1C0,  Firmware_Addr(13 downto 2) = 0x077:0x070
    __u32 acceptDelay[NUMBER_OF_LANES];// Software_Addr = 0x1FC:0x1E0,  Firmware_Addr(13 downto 2) = 0x07F:0x078

    __u32 pgpLaneStat[NUMBER_OF_LANES];// Software_Addr = 0x21C:0x200,  Firmware_Addr(13 downto 2) = 0x087:0x080
    __u32 evrRunCodeCount[NUMBER_OF_LANES]; // Software_Addr = 0x23C:0x220, Firmware_Addr ????
    __u32 LutDropCnt[NUMBER_OF_LANES]; // Software_addr = ox25C:0x240, Firmware_addr ????
    __u32 AcceptCnt[NUMBER_OF_LANES]; // Software addr = 0x27C:0x260, Firmware_addr ????
    __u32 pgpSpare1[32]; // Software_Addr = 0x2FC:0x280,  Firmware_Addr(13 downto 2) = 0x0BF:0x088
    __u32 BuildStamp[64];// Software_Addr = 0x3FC:0x300,  Firmware_Addr(13 downto 2) = 0x0FF:0x0C0

    //PciRxDesc.vhd
    __u32 rxFree[NUMBER_OF_LANES];     // Software_Addr = 0x41C:0x400,  Firmware_Addr(13 downto 2) = 0x107:0x100
    __u32 rxSpare0[24];  // Software_Addr = 0x47C:0x420,  Firmware_Addr(13 downto 2) = 0x11F:0x108
    __u32 rxFreeStat[NUMBER_OF_LANES]; // Software_Addr = 0x49C:0x480,  Firmware_Addr(13 downto 2) = 0x127:0x120
    __u32 rxSpare1[24];  // Software_Addr = 0x4FC:0x4A0,  Firmware_Addr(13 downto 2) = 0x13F:0x128
    __u32 rxMaxFrame;    // Software_Addr = 0x500,        Firmware_Addr(13 downto 2) = 0x140
    __u32 rxCount;       // Software_Addr = 0x504,        Firmware_Addr(13 downto 2) = 0x141
    __u32 rxStatus;      // Software_Addr = 0x508,        Firmware_Addr(13 downto 2) = 0x142
    __u32 rxRead[2];     // Software_Addr = 0x510:0x50C,  Firmware_Addr(13 downto 2) = 0x144:0x143
    __u32 rxSpare2[187]; // Software_Addr = 0x77C:0x514,  Firmware_Addr(13 downto 2) = 0x1FF:0x145

    //PciTxDesc.vhd
    __u32 txWrA[8];      // Software_Addr = 0x81C:0x800,  Firmware_Addr(13 downto 2) = 0x207:0x200
    __u32 txSpare0[24];  // Software_Addr = 0x87C:0x820,  Firmware_Addr(13 downto 2) = 0x21F:0x208
    __u32 txWrB[8];      // Software_Addr = 0x89C:0x880,  Firmware_Addr(13 downto 2) = 0x227:0x220
    __u32 txSpare1[24];  // Software_Addr = 0x8FC:0x8A0,  Firmware_Addr(13 downto 2) = 0x23F:0x228
    __u32 txCount[8];    // Software_Addr = 0x900,        Firmware_Addr(13 downto 2) = 0x240
    __u32 txAFull;       // Software_Addr = 0x920,        Firmware_Addr(13 downto 2) = 0x248
    __u32 txControl;     // Software_Addr = 0x924,        Firmware_Addr(13 downto 2) = 0x249
                         // txClear[23:16], txEnable[7:0]
    /* __u32 txStat[2];     // Software_Addr = 0x904:0x900,  Firmware_Addr(13 downto 2) = 0x241:0x240 */
    /* __u32 txCount;       // Software_Addr = 0x908,        Firmware_Addr(13 downto 2) = 0x242 */
    /* __u32 txRead;        // Software_Addr = 0x90C,        Firmware_Addr(13 downto 2) = 0x243 */
};
#endif
