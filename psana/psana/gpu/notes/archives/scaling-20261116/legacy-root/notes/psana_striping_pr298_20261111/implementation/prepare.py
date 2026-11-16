"""Create small immutable views; payload files are reused without edits."""
import json,shutil,subprocess
from pathlib import Path
from support import ROOT,SOURCE,PREFIX,SHA,COUNTS,require,sha,identity,write

def main(repo):
    require(subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()==SHA,'PR SHA')
    require(not subprocess.check_output(['git','-C',str(SOURCE),'diff','--name-only'],text=True),'Dirty PR source')
    staging=json.loads((repo/'notes/sim10_striping_20261110/staging/native/complete.json').read_text())
    transfer=json.loads((repo/'notes/sim10_globus_transfer_20261109/transfer-manifest.json').read_text())
    original=Path(transfer['destination_root']);manifest=json.loads((original/'generation.json').read_text())
    stamps=[r['timestamp'] for r in manifest['plan'] if r['service']==12];require(len(stamps)==100000,'100k fixture')
    refs={Path(r['destination_path']).name:r for r in transfer['files']}
    (ROOT/'views').mkdir(exist_ok=False);sets=[]
    for g in staging['sets']:
        count=g['count'];v=ROOT/'views'/('count-%02d'%count);v.mkdir();(v/'smalldata').mkdir()
        rows=[]
        layouts={x['path']:x['layout']['raw'] for x in g['copies']}
        for r in g['verified']:
            p=Path(r['path']);actual=identity(p)
            require(all(actual[k]==value for k,value in r['identity'].items()),'Changed payload')
            require(refs[p.name]['sha256']==r['sha256'],'Accepted hash mismatch')
            raw=subprocess.check_output(['lfs','getstripe',str(p)],text=True);require(raw==layouts[str(p)],'Layout mismatch')
            (v/p.name).symlink_to(p)
            smdname=p.name.replace('.xtc2','.smd.xtc2');sm=refs[smdname]
            dst=v/'smalldata'/smdname;shutil.copyfile(sm['destination_path'],dst);require(sha(dst)==sm['sha256'],'SMD hash');dst.chmod(0o444)
            rows.append({'path':str(p),'view':str(v/p.name),'bytes':r['bytes'],'sha256':r['sha256'],'identity':actual,'layout_raw':raw,'smd':str(dst),'smd_sha256':sm['sha256']})
        shutil.copyfile(original/'generation.json',v/'generation.json');require(sha(v/'generation.json')==refs['generation.json']['sha256'],'Manifest hash');(v/'generation.json').chmod(0o444)
        row={'count':count,'directory':str(v),'files':rows};write(v/'subset-manifest.json',row);(v/'subset-manifest.json').chmod(0o444)
        (v/'smalldata').chmod(0o555);v.chmod(0o555);sets.append(row)
    write(ROOT/'inputs.json',{'source_sha':SHA,'exp':'mfx101555026','run':10,'timestamps':stamps,'sets':sets,'manifest':str(ROOT/'views/count-02/generation.json'),'transfer_sha':sha(repo/'notes/sim10_globus_transfer_20261109/transfer-manifest.json'),'staging_sha':sha(repo/'notes/sim10_striping_20261110/staging/native/complete.json')})
    installed={str(p.relative_to(PREFIX)):sha(p) for p in sorted(PREFIX.rglob('*')) if p.is_file() and p.suffix in ('.py','.so')}
    require(installed,'Empty installation')
    tests={str(p.relative_to(SOURCE)):sha(p) for p in sorted((SOURCE/'psana/psana/tests/gpu').rglob('*.py'))}
    write(ROOT/'build-provenance.json',{'commit':SHA,'prefix':str(PREFIX),'installed':installed,'tests':tests,'build_log_sha256':sha(ROOT/'build.log')})
    print('Prepared six verified metadata views and pinned',len(installed),'installed modules')
if __name__=='__main__':
    import sys;main(Path(sys.argv[1]).resolve())
