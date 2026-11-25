"""Bounded host integrity checks. Does not read the 313 GiB payload or submit."""
import json
import subprocess
import sys
import time
from pathlib import Path
from support import ROOT, SOURCE, PREFIX, PR_SHA, PACKAGE, input_info, verify_inputs, sha, require, write

def main(output):
    provenance = json.loads((ROOT/'build-provenance.json').read_text())
    require(provenance['commit'] == PR_SHA and provenance['prefix'] == str(PREFIX), 'Build provenance pin')
    require(subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip() == PR_SHA,
            'Source revision changed')
    require(not subprocess.check_output(['git', '-C', str(SOURCE), 'status', '--porcelain'], text=True), 'Dirty PR source')
    counts = {}
    for section, root in [('installed', PREFIX), ('tests', SOURCE)]:
        for relative, digest in provenance[section].items():
            require(sha(root/relative) == digest, 'Build file changed: '+relative)
        counts[section] = len(provenance[section])
    require(sha(ROOT/'build.log') == provenance['build_log_sha256'], 'Build log identity')
    info = input_info()
    verify_inputs(info)
    qualification = json.loads((ROOT/'qualify01/results/audit.json').read_text())
    require(qualification['state'] == 'pass' and qualification['job'] == '61359629', 'Qualification receipt')
    from task_runtime import kvikio_identity
    # KvikIO and CuPy identities supplement the prior compute qualification.
    import importlib.util
    cupy_root = Path(importlib.util.find_spec('cupy').origin).parent
    cupy_files = sorted(p for p in cupy_root.rglob('*') if p.is_file() and p.suffix in ('.py', '.so'))
    result = dict(status='pass', pr_sha=PR_SHA, counts=counts, inputs=info,
                  kvikio_identity=kvikio_identity(), cupy_identity={str(p):sha(p) for p in cupy_files},
                  qualification=qualification, checked_unix=time.time(),
                  pins={str(p):sha(p) for p in [ROOT/'build-provenance.json', ROOT/'inputs.json',
                        ROOT/'qualify01/results/audit.json', PACKAGE/'PACKAGE.json']},
                  limitations=['Current source/install hashes and payload stat/layout only; no GPU execution in this check.',
                               'Original full hashes/traversal and count2 actual-open qualification are reused.'])
    write(output, result)
    print(json.dumps(dict(status='pass', counts=counts, input_files=len(info['files']),
                          output=str(output))))

if __name__ == '__main__':
    main(Path(sys.argv[1]))
