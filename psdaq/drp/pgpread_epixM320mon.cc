#include <atomic>
#include <string>
#include <iostream>
#include <signal.h>
#include <cstdio>
#include <cmath>
#include "psdaq/aes-stream-drivers/AxisDriver.h"
#include <stdlib.h>
#include "drp.hh"
#include <unistd.h>
#include <getopt.h>

#define MAX_RET_CNT_C 1000
static int fd;
std::atomic<bool> terminate;

using namespace Drp;

struct __attribute__((packed)) AxiPacketHeader {
    uint32_t version:4;
    uint32_t crc_type:4;
    uint32_t tuser_first:8;
    uint32_t tdest:8;
    uint32_t tid:8;
    uint32_t seq:16;
    uint32_t reserved0:15;
    uint32_t sof:1;
};

struct __attribute__((packed)) AxiPacketFooter {
    uint32_t tuser_last:8;
    uint32_t eof:1;
    uint32_t reserved0:7;
    uint32_t last_byte_cnt:4;
    uint32_t reserved1:12;
    uint32_t crc;
};

class AxiPacket {
public:
    AxiPacket(void* buffer, size_t size) :
        size_(size),
        aph_(reinterpret_cast<AxiPacketHeader*>(buffer)),
        apf_(reinterpret_cast<AxiPacketFooter*>((char*)(buffer) + size - sizeof(AxiPacketFooter)))
    {}
    ~AxiPacket() = default;

    AxiPacketHeader* aph() const { return aph_; }
    AxiPacketFooter* apf() const { return apf_; }

    void* data() const { return aph_+1; }
    size_t size() const { return size_ - sizeof(AxiPacketHeader) - sizeof(AxiPacketFooter); }

    uint32_t version() const { return aph_->version; }
    uint32_t tdest() const { return aph_->tdest; }
    uint32_t crc_type() const { return aph_->crc_type; }
    uint32_t crc() const { return apf_-> crc; }
    uint32_t last_byte_cnt() const { return apf_->last_byte_cnt; }
    uint32_t seq() const { return aph_->seq; }
    uint32_t sof() const { return aph_->sof; }
    uint32_t eof() const { return apf_->eof; }
    uint32_t tid() const { return aph_->tid; }
    uint32_t tuser_first() const { return aph_->tuser_first; }
    uint32_t tuser_last() const { return apf_->tuser_last; }

private:
    size_t           size_;
    AxiPacketHeader* aph_;
    AxiPacketFooter* apf_;
};

class EpixMonStream {
public:
    uint64_t ticks() const {
        return (raw[0] & 0x0fffffff) * 16;
    }

    double timestamp() const {
        return ticks() / clock_freq;
    }

protected:
    double convert(int idx) const {
        if (idx < nelems) {
            return (2.5 * double(raw[idx] & 0xffffff)) / 16777216.;
        } else {
            return 0.0;
        } 
    }

    double therm(int idx) const {
        double volt = convert(idx);
        if (volt > 0.0) {
            return (1/((std::log((volt/0.0001992)/10000)/3750)+(1/298.15)))-273.15;
        } else {
            return 0.0;
        }
    }

    double humidity(int idx, bool alt=false) const {
        if (alt) {
            return convert(idx)*0.97647*47.646-7.2184;
        } else {
            return convert(idx)*0.97647*58.2-8.8225;
        }
    }

    double current(int idx) const {
        return (convert(idx)*3)*0.97647*2.24/0.8;
    }

    double voltage(int idx) const {
        return convert(idx)*3*0.97647;
    }

private:
    static constexpr int nelems = 9;
    static constexpr double clock_freq = 156.25e6;
    uint64_t raw[nelems];
};

class EpixMonStreamAsic : public EpixMonStream {
public:
    double carrier_therm() const {
        return therm(1);
    }

    double digital_therm() const {
        return therm(2);
    }

    double digital_humidity() const {
        return humidity(3);   
    }

    double misc_current_1() const {
        return current(4);
    }

    double asic_analog_current() const {
        return current(5);
    }

    double misc_voltage_1() const {
        return voltage(6);
    }

    double misc_voltage_2() const {
        return voltage(7);
    }

    double asic_analog_voltage() const {
        return voltage(8);
    }
};

class EpixMonStreamPcb : public EpixMonStream {
public:
    double pcb_humidity() const {
        return humidity(1, true);
    }

    double pcb_therm() const {
        return therm(2);
    }

    double pcb_v_3v3() const {
        return convert(3)*2.;
    }

    double pcb_v_1v8() const {
        return convert(4);
    }

    double pcb_ana_v() const {
        return convert(5)*3.;
    }

    double pcb_dig_v() const {
        return convert(6)*3.;
    }

    double pcb_dig_i() const {
        return convert(7)*10.;
    }

    double pcb_ana_i() const {
        return convert(8)*10.;
    }
};

static size_t calc_pad(const std::string& str, size_t max_pad) {
    return max_pad > str.length() ? max_pad - str.length() : 0;
}

unsigned dmaDest(unsigned lane, unsigned vc)
{
    return (lane<<PGP_MAX_LANES) | vc;
}

void int_handler(int dummy)
{
    terminate.store(true, std::memory_order_release);
    // dmaUnMapDma();
}

static void show_usage(const char* p)
{
    printf("Usage: %s -d <device file> [-c <virtChan>] [-l <laneMask>]\n",p);
    printf("       -v  verbose\n");
}

int main(int argc, char* argv[])
{
    int c, virtChan;

    virtChan = 0;
    uint8_t laneMask = (1 << PGP_MAX_LANES) - 1;
    std::string device;
    unsigned lverbose = 0;
    bool lusage = false;
    while((c = getopt(argc, argv, "c:d:l:vh?")) != EOF) {
        switch(c) {
            case 'd':
                device = optarg;
                break;
            case 'c':
                virtChan = atoi(optarg);
                break;
            case 'l':
                laneMask = std::stoul(optarg, nullptr, 16);
                break;
            case 'v':
                ++lverbose;
                break;
            default:
                lusage = true;
                break;
        }
    }

    if (lusage) {
        show_usage(argv[0]);
        return -1;
    }

    // Complain if all arguments weren't consummed
    if (optind < argc) {
        printf("Unrecognized argument:\n");
        while (optind < argc)
            printf("  %s ", argv[optind++]);
        printf("\n");
        show_usage(argv[0]);
        return 1;
    }

    terminate.store(false, std::memory_order_release);
    signal(SIGINT, int_handler);

    uint8_t mask[DMA_MASK_SIZE];
    dmaInitMaskBytes(mask);

    for (unsigned i=0; i<PGP_MAX_LANES; i++) {
        if (laneMask & (1 << i)) {
            if (virtChan<0) {
                for(unsigned j=0; j<4; j++) {
                    uint32_t dest = dmaDest(i, j);
                    printf("setting lane %u, dest 0x%x \n",i,dest);
                    dmaAddMaskBytes((uint8_t*)mask, dest);
                }
            }
            else {
                uint32_t dest = dmaDest(i, virtChan);
                printf("setting lane %u, dest 0x%x \n",i,dest);
                dmaAddMaskBytes((uint8_t*)mask, dest);
            }
        }
    }

    std::cout<<"device  "<<device<<'\n';
    fd = open(device.c_str(), O_RDWR);
    if (fd < 0) {
        std::cout<<"Error opening "<<device<<'\n';
        return -1;
    }

    uint32_t dmaCount, dmaSize;
    void** dmaBuffers = dmaMapDma(fd, &dmaCount, &dmaSize);
    if (dmaBuffers == NULL ) {
        printf("Failed to map dma buffers!\n");
        return -1;
    }
    printf("dmaCount %u  dmaSize %u\n", dmaCount, dmaSize);

    if (dmaSetMaskBytes(fd, mask)) {
        perror("dmaSetMaskBytes");
        printf("Failed to allocate lane/vc "
               "- does another process have %s open?\n", device.c_str());
        const unsigned* u = reinterpret_cast<const unsigned*>(mask);
        for(unsigned i=0; i<DMA_MASK_SIZE/4; i++)
            printf("%08x%c", u[i], (i%8)==7?'\n':' ');
        return -1;
    }

    std::vector<std::vector<std::string>> misc_names {
        { "DUAL LDO 0 I 1V8", "An V 6V", "DUAL LDO 0 V 1V8" },
        { "DUAL LDO 1 I 1V8", "VCCA V 3V0", "DUAL LDO 1 V 1V8" },
        { "Digital I 2V5", "Digital V 6V", "Digital V 6V" },
        { "DS_PLL I 2V5", "VCC V 2V7", "DS_PLL V 2V5" },
    };

    uint64_t nevents = 0L;
    int32_t dmaRet[MAX_RET_CNT_C];
    uint32_t dmaIndex[MAX_RET_CNT_C];
    uint32_t dmaDest[MAX_RET_CNT_C];
    while (1) {
        if (terminate.load(std::memory_order_acquire) == true) {
            close(fd);
            printf("closed\n");
            break;
        }

        int32_t ret = dmaReadBulkIndex(fd, MAX_RET_CNT_C, dmaRet, dmaIndex, NULL, NULL, dmaDest);
        for (int b=0; b < ret; b++) {
            uint32_t index = dmaIndex[b];
            uint32_t size = dmaRet[b];
            uint32_t dest = dmaDest[b] >> 8;
            uint32_t vc   = dmaDest[b] & 0xff;

            if (lverbose) {
              printf("\nSize %u B | Dest %u.%u | index %u\n", size, dest, vc, index);
            } else {
              printf("\n");
            }

            AxiPacket ap(dmaBuffers[index], size);
            AxiPacketHeader* aph = ap.aph();
            AxiPacketFooter* apf = ap.apf();
            if (lverbose) {
                printf("AxiPacketHeader: vers %u, crc_type %u, tuser_first %u, tdest %u, tid %u, seq %u, sof %u\n",
                       aph->version, aph->crc_type, aph->tuser_first, aph->tdest, aph->tid, aph->seq, aph->sof);
                printf("AxiPacketFooter: tuser_last %u, eof %u, last_byte_cnt %u, crc %u\n",
                       apf->tuser_last, apf->eof, apf->last_byte_cnt, apf->crc);
            }

            printf("Monitoring data for tdest %u:\n", aph->tdest);
            if (aph->tdest < 4) {
                size_t pad = 20;
                EpixMonStreamAsic* data = reinterpret_cast<EpixMonStreamAsic*>(ap.data());
                printf("Time -> ticks:       %lu\n", data->ticks());
                printf("Time -> seconds:     %f\n",  data->timestamp());
                printf("Carrier Therm:       %f\n", data->carrier_therm());
                printf("Digital Therm:       %f\n", data->digital_therm());
                printf("Digital Humidity:    %f\n", data->digital_humidity());
                printf("%s:%*s%f\n",
                        misc_names[aph->tdest][0].c_str(),
                        (int) calc_pad(misc_names[aph->tdest][0], pad),
                        "",
                        data->misc_current_1());
                printf("ASIC%d An I 2V5:      %f\n", aph->tdest, data->asic_analog_current());
                printf("%s:%*s%f\n",
                       misc_names[aph->tdest][1].c_str(),
                       (int) calc_pad(misc_names[aph->tdest][1], pad),
                       "",
                       data->misc_voltage_1());
                printf("%s:%*s%f\n",
                       misc_names[aph->tdest][2].c_str(),
                       (int) calc_pad(misc_names[aph->tdest][2], pad),
                       "",
                       data->misc_voltage_2());
                printf("ASIC%d An V 2V5:      %f\n", aph->tdest, data->asic_analog_voltage());
            } else {
                EpixMonStreamPcb* data = reinterpret_cast<EpixMonStreamPcb*>(ap.data());
                printf("Time -> ticks:       %lu\n", data->ticks());
                printf("Time -> seconds:     %f\n",  data->timestamp());
                printf("PCB Humidity:        %f\n",  data->pcb_humidity());
                printf("PCB Thermal:         %f\n",  data->pcb_therm());
                printf("PCB V 3V3:           %f\n",  data->pcb_v_3v3());
                printf("PCB V 1V8:           %f\n",  data->pcb_v_1v8());
                printf("PCB Ana V 6V:        %f\n",  data->pcb_ana_v());
                printf("PCB Dig V 6V:        %f\n",  data->pcb_dig_v());
                printf("PCB Dig I 6V:        %f\n",  data->pcb_dig_i());
                printf("PCB Ana I 6V:        %f\n",  data->pcb_ana_i());
            }

            ++nevents;
        }
	    if ( ret > 0 ) dmaRetIndexes(fd, ret, dmaIndex);
	    //sleep(0.1)
    }
    printf("finished: nEvents %lu\n", nevents);
}
