
#include "Worker.hh"
#include "xtcdata/xtc/Dgram.hh"
#include "TimingHeader.hh"
#include "xtcdata/xtc/Sequence.hh"
#include "xtcdata/xtc/TransitionId.hh"

using namespace XtcData;

bool check_pulse_id(PGPData* pgp_data)
{
    uint64_t pulse_id = 0;
    for (int l=0; l<8; l++) {
        if (pgp_data->buffer_mask  & (1 << l)) {
            Transition* event_header = reinterpret_cast<Transition*>(pgp_data->buffers[l].data);
            if (pulse_id == 0) {
                pulse_id = event_header->seq.pulseId().value();
            }
            else {
                if (pulse_id != event_header->seq.pulseId().value()) {
                    printf("Wrong pulse id! expected %lu but got %lu instea  d\n", pulse_id, event_header->seq.pulseId().value());
                    return false;
                }
            }
        }
    }
    return true;
}


void worker(Detector* det, PebbleQueue& worker_input_queue, PebbleQueue& worker_output_queue, int rank)
{
    while (true) {
        Pebble* pebble;
        if (!worker_input_queue.pop(pebble)) {
            break;
        }
        // get first set bit to find index of the first lane
        int index = __builtin_ffs(pebble->pgp_data->buffer_mask) - 1;
        Pds::TimingHeader* event_header = reinterpret_cast<Pds::TimingHeader*>(pebble->pgp_data->buffers[index].data);
        TransitionId::Value transition_id = event_header->seq.service();
        if (transition_id == XtcData::TransitionId::Configure) {
            printf("Worker %d saw configure transition\n", rank);
        }

        check_pulse_id(pebble->pgp_data);

        // uint16_t* rawdata = (uint16_t*)(event_header+1);
        // printf("data %u %u \n", rawdata[0], rawdata[1]);

        Dgram& dgram = *(Dgram*)pebble->fex_data();
        TypeId tid(TypeId::Parent, 0);
        dgram.xtc.contains = tid;
        dgram.xtc.damage = 0;
        dgram.xtc.extent = sizeof(Xtc);
        // Event
        if (transition_id == XtcData::TransitionId::L1Accept) {
            det->event(dgram, pebble->pgp_data);
        }
        // Configure
        else if (transition_id == XtcData::TransitionId::Configure) {
            det->configure(dgram, pebble->pgp_data);
        }

        worker_output_queue.push(pebble);
    }
}
