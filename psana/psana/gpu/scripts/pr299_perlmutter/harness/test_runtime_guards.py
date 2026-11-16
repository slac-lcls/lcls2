import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import perlmutter_screen as screen
from perlmutter_support import runtime_provenance

MPI_PATH='/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3/lib/libmpi_gnu_123.so.12.0.0'
CONDA=Path('/global/homes/m/monarin/.conda-envs/psana-build').resolve()
KVIKIO=str(CONDA/'lib/python3.9/site-packages/kvikio/_lib/libkvikio.cpython-39-x86_64-linux-gnu.so')
CUDART=str(CONDA/'lib/python3.9/site-packages/kvikio_cu12.libs/libcudart-16e8815a.so.12.5.82')
CUDA='/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9'

class RuntimeGuardTests(unittest.TestCase):
    def observe(self, paths, gpu=True, compat=True):
        mpi=SimpleNamespace(MPI=SimpleNamespace(Get_library_version=lambda:'CRAY MPICH 9.0.1'))
        defaults=SimpleNamespace(compat_mode=lambda:compat)
        kvikio=SimpleNamespace(__version__='24.08.02',__file__=str(CONDA/'lib/python3.9/site-packages/kvikio/__init__.py'),defaults=defaults)
        cupy=SimpleNamespace(__version__='13.6.0',__file__=str(CONDA/'lib/python3.9/site-packages/cupy/__init__.py'),cuda=SimpleNamespace(runtime=SimpleNamespace(runtimeGetVersion=lambda:12090,driverGetVersion=lambda:13000)))
        lines='\n'.join('000-001 r--p 0 00:00 0 '+p for p in paths)
        with patch.object(Path,'read_text',return_value=lines), patch.dict(sys.modules,{'mpi4py':mpi,'kvikio':kvikio,'kvikio.defaults':defaults}):
            return runtime_provenance(cupy if gpu else None)
    def test_cpu_mapping_only(self):
        value=self.observe([MPI_PATH],gpu=False)
        self.assertFalse(value['gpu_initialized']); self.assertNotIn('cupy_version',value)
    def test_gpu_expected_mapping(self):
        value=self.observe([MPI_PATH,KVIKIO,CUDART])
        self.assertTrue(value['compat_mode']);self.assertEqual(value['cuda_runtime_version'],12090)
    def test_reject_other_mpi(self):
        with self.assertRaisesRegex(RuntimeError,'MPI mappings'):self.observe([MPI_PATH.replace('9.0.1','9.1.0')],gpu=False)
    def test_reject_new_kvikio_lib(self):
        with self.assertRaisesRegex(RuntimeError,'KvikIO library'):self.observe([MPI_PATH,KVIKIO,CUDART,'/unexpected/libkvikio.so'])
    def test_reject_unexpected_cudart(self):
        with self.assertRaisesRegex(RuntimeError,'cudart library'):self.observe([MPI_PATH,KVIKIO,CUDART,'/unexpected/libcudart.so'])
    def test_reject_unexpected_nvrtc(self):
        with self.assertRaisesRegex(RuntimeError,'NVRTC library'):self.observe([MPI_PATH,KVIKIO,CUDART,'/unexpected/libnvrtc.so'])
    def test_reject_gds(self):
        with self.assertRaisesRegex(RuntimeError,'compatibility mode'):self.observe([MPI_PATH,KVIKIO,CUDART],compat=False)
    def test_missing_kvikio(self):
        with self.assertRaisesRegex(RuntimeError,'not mapped'):self.observe([MPI_PATH,CUDART])
    def test_cuda_module_root_accepted(self):
        value=self.observe([MPI_PATH,KVIKIO,CUDA+'/lib64/libcudart.so',CUDA+'/lib64/libnvrtc.so'])
        self.assertTrue(value['compat_mode'])

class AllocationGuardTests(unittest.TestCase):
    def check(self,limit,account='lcls_g',qos='gpu_premium'):
        env=dict(SLURM_JOB_NUM_NODES='1',SLURM_NTASKS='4',SLURM_CPUS_PER_TASK='8',SLURM_JOB_ID='1')
        with patch.dict('os.environ',env),patch.object(screen.subprocess,'check_output',return_value=json.dumps(dict(jobs=[dict(time_limit=limit,account=account,qos=qos)]))):
            return screen.allocation_identity()
    def test_slurm_number_and_integer_forms(self):
        for limit in (30,dict(number=30,set=True)):
            with self.subTest(limit=limit):self.assertIn('slurm_job',self.check(limit))
    def test_reject_wrong_allocation(self):
        for options in [dict(limit=31),dict(limit=0),dict(limit=30,account='wrong'),dict(limit=30,qos='regular')]:
            with self.subTest(options=options),self.assertRaises(RuntimeError):self.check(**options)

if __name__=='__main__':unittest.main()
