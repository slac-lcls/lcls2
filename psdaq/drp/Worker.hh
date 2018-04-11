#ifndef WORKERS_H
#define WORKERS_H

#include <thread>
#include <vector>

#include "drp.hh"
#include "Detector.hh"

void worker(Detector* det, PebbleQueue& worker_input_queue, PebbleQueue& worker_output_queue, int rank);

#endif // WORKERS_H
