import numpy as np
import subprocess,os


def flush_writer(obj):
    obj.flush()
    os.fsync(obj.fileno())


def chunk_test(params, loop_iter):
    with open('test_results/HF_1M_%i_chunking_write_3.txt' % params[1], 'ab') as hf16_wr,\
         open('test_results/VL_1M_%i_chunking_write_3.txt' % params[1], 'ab') as vl16_wr,\
         open('test_results/HF_1M_%i_chunking_read_3.txt' % params[1], 'ab') as hf16_re,\
         open('test_results/VL_1M_%i_chunking_read_3.txt' % params[1], 'ab') as vl16_re:

        for i in range(loop_iter):
            print(i)
            vlwr = subprocess.run("./VLWrite /nvme5n1/vldata.h5 %i %i %i" % (params[0], params[1], params[2]), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            hfwr = subprocess.run("./HFWrite /nvme5n1/hfdata.h5 %i %i %i" % (params[0], params[1], params[2]), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            subprocess.call('echo 3 | sudo tee /proc/sys/vm/drop_caches', shell=True)

            vlre = subprocess.run("./VLRead /nvme5n1/vldata.h5 %i %i" % (i, 2*params[1]), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            hfre = subprocess.run("./HFRead /nvme5n1/hfdata.h5 %i %i" % (i, 2*params[1]), shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            vl16_wr.write(vlwr.stdout)
            hf16_wr.write(hfwr.stdout)
            vl16_re.write(vlre.stdout)
            hf16_re.write(hfre.stdout)

            for obj in [vl16_wr, hf16_wr, vl16_re, hf16_re]:
                flush_writer(obj)


loop_iter = 32

# params = [10000,16,1000000]
# chunk_test(params, loop_iter)

params = [32000,16,1000000]
chunk_test(params, loop_iter)
