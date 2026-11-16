import ctypes
import json
import os
import re
import subprocess
import sys
import uuid


ENVIRONMENT_KEYS = ('SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'CUDA_VISIBLE_DEVICES')
GPU_UUID = re.compile(r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}')


def validate_environment(environment):
    job = environment.get('SLURM_JOB_GPUS')
    step = environment.get('SLURM_STEP_GPUS')
    assignment = step or job
    visibility = environment.get('CUDA_VISIBLE_DEVICES')
    single = lambda value: bool(value) and bool(re.fullmatch(r'[0-9]+', value) or GPU_UUID.fullmatch(value))
    if not environment.get('SLURM_JOB_ID') or not single(assignment) or not single(visibility):
        raise RuntimeError('Slurm must assign exactly one full GPU with explicit CUDA visibility')
    if job:
        devices = job.split(',')
        if len(set(devices)) != len(devices) or not all(single(value) for value in devices):
            raise RuntimeError('Slurm must assign exactly one full GPU per step with a valid job allocation')
        if step and step.isdigit() and all(value.isdigit() for value in devices) and step not in devices:
            raise RuntimeError('Step GPU differs from parent job allocation')


def verify_step_gpu(expected_uuid):
    environment = {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}
    if not environment.get('SLURM_STEP_GPUS'):
        raise RuntimeError('BD requires an explicit single-GPU Slurm step')
    validate_environment(environment)
    identity = cuda_identity()
    if identity['uuid'] != expected_uuid:
        raise RuntimeError('Slurm step GPU UUID differs from selected benchmark GPU')
    return dict(environment=environment, cuda=identity)


def cuda_identity():
    driver = ctypes.CDLL('libcuda.so.1')

    def call(name, types, *arguments):
        function = getattr(driver, name)
        function.argtypes = types
        function.restype = ctypes.c_int
        status = function(*arguments)
        if status:
            raise RuntimeError(f'{name} returned CUDA error {status}')

    call('cuInit', [ctypes.c_uint], 0)
    count = ctypes.c_int()
    call('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)], ctypes.byref(count))
    if count.value != 1:
        raise RuntimeError(f'expected exactly one CUDA-visible GPU; got {count.value}')
    device = ctypes.c_int()
    call('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.byref(device), 0)
    identifier = (ctypes.c_ubyte * 16)()
    call('cuDeviceGetUuid_v2', [ctypes.c_void_p, ctypes.c_int], identifier, device)
    bus = ctypes.create_string_buffer(32)
    call('cuDeviceGetPCIBusId', [ctypes.c_void_p, ctypes.c_int, ctypes.c_int], bus, len(bus), device)
    return dict(count=count.value, ordinal=0, uuid='GPU-' + str(uuid.UUID(bytes=bytes(identifier))),
                pci_bus_id=bus.value.decode())


def pci_identity(value):
    match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not match:
        raise RuntimeError(f'invalid PCI bus identity: {value!r}')
    return tuple(int(component, 16) for component in match.groups())


def discover(record):
    environment = record['environment']
    validate_environment(environment)
    identity = cuda_identity()
    record['cuda'] = identity
    gpu_uuid = identity['uuid']
    if identity['count'] != 1 or not GPU_UUID.fullmatch(gpu_uuid):
        raise RuntimeError('expected exactly one full CUDA-visible GPU UUID')
    for key in ENVIRONMENT_KEYS[1:]:
        values = (environment.get(key, '') or '').split(',')
        uuids = [value for value in values if value.startswith('GPU-')]
        if uuids and gpu_uuid not in uuids:
            raise RuntimeError(f'{key} UUID differs from CUDA-visible GPU')
    command = ['nvidia-smi', '-i', gpu_uuid,
               '--query-gpu=uuid,pci.bus_id,name,memory.total,mig.mode.current',
               '--format=csv,noheader,nounits']
    query = dict(command=command)
    record['nvidia_smi'] = query
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except subprocess.TimeoutExpired as error:
        query.update(timeout=True, stdout=as_text(error.stdout), stderr=as_text(error.stderr))
        raise RuntimeError('nvidia-smi UUID lookup timed out') from error
    query.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    if result.returncode:
        raise RuntimeError(f'nvidia-smi UUID lookup failed ({result.returncode})')
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1 or len(lines[0].split(',')) != 5:
        raise RuntimeError('expected one complete nvidia-smi identity row')
    observed_uuid, bus, name, memory, mig = [value.strip() for value in lines[0].split(',')]
    if observed_uuid != gpu_uuid or pci_identity(bus) != pci_identity(identity['pci_bus_id']):
        raise RuntimeError('CUDA and nvidia-smi GPU identities differ')
    if 'A100' not in name or not (39000 < float(memory) < 83000) or mig != 'Disabled':
        raise RuntimeError('expected full Perlmutter A100 with MIG disabled')
    record.update(gpu_uuid=gpu_uuid, gpu_lines=lines)


def as_text(value):
    return value.decode(errors='replace') if isinstance(value, bytes) else (value or '')


def main():
    record = dict(host=os.uname().nodename, environment={key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
                  complete=False)
    try:
        discover(record)
        record['complete'] = True
    except Exception as error:
        record['error'] = f'{type(error).__name__}: {error}'
    print(json.dumps(record), flush=True)
    return 0 if record['complete'] else 1


if __name__ == '__main__':
    sys.exit(main())
