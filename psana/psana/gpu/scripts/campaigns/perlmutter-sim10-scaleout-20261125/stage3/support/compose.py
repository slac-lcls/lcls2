"""Compose a new Stage 3 bundle from byte-pinned Stage 2 sources."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent


def replace(files,name,old,new):
    text=files[name].decode()
    if text.count(old)!=1:raise ValueError((name,old,text.count(old)))
    files[name]=text.replace(old,new).encode()


def compose():
    manifest=json.loads((HERE/'evidence/base-bundle.json').read_text())
    files={n:(HERE/'base'/n).read_bytes() for n in manifest['files']}
    for n,digest in manifest['files'].items():
        if hashlib.sha256(files[n]).hexdigest()!=digest:raise ValueError('Base changed '+n)
    for name in ('coordinator.py','gate.py','analyze.py','on_allocation.sh'):
        files.pop(name,None)
    support=files['support.py'].decode();begin=support.index('def affinity():');end=support.index('def runtime_identity(')
    files['support.py']=(support[:begin]+'from topology import affinity\nfrom data_contract import input_info, verify_inputs\n\n'+support[end:]).encode()
    replace(files,'clean_exec.py',"'JF_GPU_UUID')","'JF_GPU_UUID','SIM10_INPUTS','SIM10_PLACEMENT')")
    replace(files,'runtime.sh','export SIM10_PR_SOURCE="$sim10_source"',
            '[[ "$SIM10_RUNTIME" == C ]] || exit 2\nexport SIM10_INPUTS="$attempt/inputs.json"\nexport SIM10_PR_SOURCE="$sim10_source"')
    # Runtime/check_tests probes also use the new explicitly qualified input.
    replace(files,'runtime_check.py'," info=input_info();verify_inputs(info)",
            " if len(sys.argv)<3 or sys.argv[2]!='--runtime-only':\n  info=input_info();verify_inputs(info)")
    replace(files,'runtime_check.py',"if role not in ('P','C'):","if role != 'C':")
    replace(files,'runner.py',"    require(arguments.bds in (1, 4) and arguments.batch == {1: 50, 4: 200}[arguments.bds], 'Topology/batch')",
            "    nodes=arguments.bds//4\n    require(nodes in (1,2,4) and arguments.bds==4*nodes and arguments.batch==200, 'Topology/batch')")
    replace(files,'runner.py',"    require(active_gpus == arguments.bds or (arguments.stage == 'S' and arguments.bds == 4 and active_gpus == 1), 'Sharing only for staging bridge')",
            "    require(active_gpus == arguments.bds and len(placement['hosts'])==nodes, 'Four GPUs per node')")
    replace(files,'runner.py',"    require(set(placement['rank_uuids'][2:]) <= set(granted), 'GPU outside allocation')",
            "    local=[r for r,h in enumerate(placement['rank_hosts']) if h==platform.node()]\n    require(rank in local, 'Actual rank host')\n    local_uuids=[placement['rank_uuids'][r] for r in local if r>=2]\n    require(set(local_uuids)==set(granted) and len(local_uuids)==4, 'Local granted GPU set')\n    node_leader=min(local)==rank")
    replace(files,'runner.py',"    host_memory = dict(startup=snapshot(), demand=demand(arguments.bds, 4, 16))",
            "    host_memory = dict(startup=snapshot(), demand=demand(4, 4, 16))")
    replace(files,'runner.py',"    require(info['media'] == 'lustre', 'Qualified count2 Lustre')\n    if rank == 0:\n        verify_inputs(info)",
            "    require(info['media'] == 'lustre', 'Qualified run4 Lustre')\n    reference.SNAPSHOT=Path(info['calibration_snapshot'])\n    if node_leader:\n        verify_inputs(info)")
    replace(files,'runner.py',"        require(arguments.events == (4000 if arguments.kind == 'pilot' else 100000), 'Frozen event count')",
            "        require(arguments.events == 100000*nodes and arguments.kind in ('full','anchor'), 'Frozen event count')")
    replace(files,'runner.py',"        require(arguments.events == {1: 201, 4: 1003}[arguments.bds], 'Bounded tail fixture')",
            "        require(arguments.events == 1003*nodes, 'Bounded tail fixture')")
    replace(files,'runner.py',"    if rank == 0:\n        leaves = info.get('observed_leaves', [])",
            "    from node_observer import network, start_memory, stop_memory\n    watches=start_memory(local_uuids) if node_leader else []\n    if node_leader:\n        leaves = info.get('observed_leaves', [])")
    replace(files,'runner.py',"                                     Path(str(arguments.output) + '.cache.json'))",
            "                                     Path(str(arguments.output) + '.cache.'+platform.node()+'.json'))")
    replace(files,'runner.py',"    begin = time.perf_counter()\n    delivered, errors_seen, cpu_reads, io = [], 0, [0], None",
            "    net_before=network() if node_leader else None\n    comm.Barrier()\n    begin = time.perf_counter()\n    delivery_ticks=[]\n    delivered, errors_seen, cpu_reads, io = [], 0, [0], None")
    replace(files,'runner.py',"                    delivered.append(int(event.timestamp))",
            "                    delivered.append(int(event.timestamp))\n                    if len(delivered)%200==0:delivery_ticks.append([time.perf_counter()-begin,len(delivered)])")
    replace(files,'runner.py',"    if rank == 0:\n        observation.update(seconds=seconds, after=counters(info.get('observed_leaves', [])))",
            "    net_after=network() if node_leader else None\n    if node_leader:\n        observation.update(seconds=seconds, after=counters(info.get('observed_leaves', [])))")
    replace(files,'runner.py',"    net_after=network() if node_leader else None",
            "    net_after=network() if node_leader else None\n    delivered_by_rank=comm.allgather(len(delivered))\n    node_delivered=sum(n for r,n in enumerate(delivered_by_rank) if r>=2 and placement['rank_hosts'][r]==platform.node())")
    replace(files,'runner.py',"evidence(observation, 'lustre', EVENT_BYTES * arguments.events, seconds)",
            "evidence(observation, 'lustre', EVENT_BYTES * node_delivered, seconds)")
    replace(files,'runner.py',"    validate_host_memory(host_memory, arguments.bds, 4, 16)",
            "    validate_host_memory(host_memory, 4, 4, 16)\n    memory_receipts=stop_memory(watches) if node_leader else []")
    replace(files,'runner.py',"    row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,",
            "    row = dict(rank=rank, host=platform.node(), device=device, affinity=chosen, timestamps=delivered,\n               delivery_ticks=delivery_ticks, node_observation=observation if node_leader else None,\n               node_memory=memory_receipts, network_before=net_before, network_after=net_after,")
    replace(files,'runner.py',"        require(len({entry['device']['pci'] for entry in rows[2:]}) == active_gpus, 'Physical placement collision')",
            "        from topology import audit_placement\n        node_events=audit_placement(rows,placement)\n        require(len({(entry['host'],entry['device']['pci']) for entry in rows[2:]}) == active_gpus, 'Physical placement collision')")
    replace(files,'runner.py',"                      rows=rows, observation=observation,",
            "                      rows=rows, observation=observation, node_events=node_events,\n                      node_observations={r['host']:r['node_observation'] for r in rows if r['node_observation'] is not None},")
    replace(files,'runner.py',"native_caveat='Stage 2 selected pinned runtime; see runtime-check and qualification receipts'",
            "native_caveat='Stage 3 pinned PR331 candidate; clock sampled every 200 delivered events'")
    replace(files,'runner.py',"choices=(1, 4), required=True","choices=(4, 8, 16), required=True")
    # Preserve independent per-rank/range/cube checks, extending only host/cache scope.
    replace(files,'audit.py',"    require(len({r['host'] for r in rows})==1, 'One node')",
            "    from topology import audit_placement\n    counts=audit_placement(rows,gpu_plan)\n    require(len(counts)==case['nodes'] and counts==report['node_events'], 'Node event distribution')")
    replace(files,'audit.py',"    require(len({r['device']['pci'] for r in rows[2:]})==case['gpus'], 'Physical GPU count')",
            "    require(len({(r['host'],r['device']['pci']) for r in rows[2:]})==case['gpus'], 'Physical GPU count')\n    require(set(report['node_observations'])==set(gpu_plan['hosts']),'Every node cache observation')\n    from memory_watch import validate_memory\n    for host in gpu_plan['hosts']:\n        leaders=[r for r in rows if r['host']==host and r['node_observation'] is not None]\n        require(len(leaders)==1,'One cache observer per node')\n        leader=leaders[0]\n        require(leader['node_observation']==report['node_observations'][host],'Node observation binding')\n        observation_local=leader['node_observation']\n        scans_local=validate_cache_gate(observation_local['cache_gate'],info['files'])\n        require(observation_local['conditioning']==scans_local[0]['files'] and observation_local['initial_cache']==scans_local[1]['files'],'Local cache receipt binding')\n        require(observation_local['conditioning_seconds']==observation_local['cache_gate']['elapsed_seconds'],'Local cache timing binding')\n        require({m['uuid'] for m in leader['node_memory']}=={r['device']['mask'] for r in rows[2:] if r['host']==host} and len(leader['node_memory'])==4,'Four distinct assigned memory watches per node')\n        for receipt in leader['node_memory']:validate_memory(receipt)")
    for name in ('design.py','topology.py','probe.py','data_contract.py','node_observer.py',
                 'coordinator.py','launch_steps.py','gate.py','on_allocation.sh','device_node.py','runtime_node.py','analyze.py'):
        files[name]=(HERE/name).read_bytes()
    return files
