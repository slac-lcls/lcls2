#ifndef Pds_HSD_FlashController_hh
#define Pds_HSD_FlashController_hh

#include <stdint.h>
#include <stdio.h>

namespace Pds {
  namespace HSD {
    class FlashController {
    public:
      void write(FILE* f);
      void write(const uint32_t* p, unsigned nwords);
      int  read (const uint32_t* p, unsigned nwords);
    private:
      uint32_t _reserved0[3];
      uint32_t _destn;  // user=0, safe=0xff
      uint32_t _bytes_to_prog;
      uint32_t _reserved5;
      uint32_t _bytes_to_read;
      uint32_t _reserved7[9];
      uint32_t _command;
      //  b0  = flash_init/fifo_reset
      //  b15 = start_read (unnecessary)
      //  b17 = start_prog (unnecessary)
      uint32_t _reserved17[15];
      volatile uint32_t _data;
    };
  };
};

#endif


#if 0
Steps:

  rc = _4FM_OpenDevice(&ctx, argv[1], devno);
  rc = _4FM_ResetDevice(&ctx);
                rc = _4FM_GetInformation(&ctx, &_4FM_Info);
                if(((_4FM_Info.cpld_revision.hi == 15) && (_4FM_Info.cpld_revision.lo == 15)) ||
                  ((_4FM_Info.cpld_revision.hi == 0) && (_4FM_Info.cpld_revision.lo == 0)))
                {
                  /* try to set CPLD addressing type to the different flash */
                  rc = _4FM_WriteRegister(&ctx, 0x1B, 1);
                  rc = _4FM_ResetDevice(&ctx);
                  usleep(250000);
                  rc = _4FM_ResetDevice(&ctx);
                  usleep(250000);
                  rc = _4FM_GetInformation(&ctx, &_4FM_Info);
                  if(rc != _4FM_OK)
                        abort_with_vmessage(rc, "Unable to change flash type on device %d, upload canceled", devno);    

                  if(((_4FM_Info.cpld_revision.hi == 15) && (_4FM_Info.cpld_revision.lo == 15)) ||
                          ((_4FM_Info.cpld_revision.hi == 0) && (_4FM_Info.cpld_revision.lo == 0)))
                  {
                          abort_with_vmessage(rc, "Unable to change FLASH addressing type (B) on device %d", devno);

    rc = _4FM_UploadFirmware(&ctx, CREG_BASED, argv[argc-1], fpga_dest);
        _4FM_CloseDevice(&ctx);
  return EXIT_SUCCESS;


_4FM_error_t _4FM_CALL _4FM_UploadFirmware(_4FM_DeviceContext *ctx, enum firmupMode mode, const char *filename, enum ufTarget target)
                size = process_hexfile(filename, &p);
                        size = realign(&p, size, 256);
                        usleep(200000);
                                status = _4FM_SelectTargetUnsafe(ctx, fpga_dest);
                                usleep(2000);
                                        /* write the size */
                                        _4FM_WriteRegister(ctx, 9, size);
                                        status = _4FM_SendData(ctx, p, size);
                                        usleep(1000000);
        return status;




_4FM_error_t _4FM_CALL _4FM_ResetDevice(_4FM_DeviceContext *ctx)
  else if(!strcmp(ctx->lpType->name, "PC820"))
        return FDSP_OK;


_4FM_error_t _4FM_CALL _4FM_SelectTargetUnsafe(_4FM_DeviceContext *ctx, unsigned int target)
  if (_4FM_IoCtl_Wr(ctx, IOCTL_4FM_SELECT_TARGET, &target, sizeof(target)))


_4FM_error_t _4FM_CALL _4FM_SendData(_4FM_DeviceContext *ctx, const void *buffer, unsigned int count)
  /* compute the number of iterations required and the remainder */
  reqiter = count/optionstable[OPTION_DMA_BURST_SIZE];
  remainder = count - reqiter*optionstable[OPTION_DMA_BURST_SIZE];
  for(indexiter = 0; indexiter<reqiter; indexiter++) {
          rc = write(ctx->fd, buffer+indexiter*optionstable[OPTION_DMA_BURST_SIZE], optionstable[OPTION_DMA_BURST_SIZE]);
  if(remainder) {
          rc = write(ctx->fd, buffer+reqiter*optionstable[OPTION_DMA_BURST_SIZE], remainder);
  return FDSP_OK;


_4FM_IoCtl_Wr(ctx, IOCTL_4FM_SELECT_TARGET, &target, sizeof(target)))
            writel(uval, dev->bar[0].virt_address + SOURCE_DEST_REGISTER_OFFS);


#define SOURCE_DEST_REGISTER_OFFS       60ul
#endif

