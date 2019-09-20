import subprocess
import shutil

def setup_input_files(tmp_path, n_files=2, slow_update_freq=4, n_motor_steps=1, n_events_per_step=10, gen_run2=True):
    xtc_dir = tmp_path / '.tmp'
    xtc_dir.mkdir()
    smd_dir = xtc_dir / 'smalldata'
    smd_dir.mkdir()

    for i in range(n_files):
        # Only segments 0,1 has "counting" timestamps for event-building
        filename = 'data-r0001-s%s.xtc2'%(str(i).zfill(2))
        smd_filename = 'data-r0001-s%s.smd.xtc2'%(str(i).zfill(2))
        sfile = str(xtc_dir / filename)
        subprocess.call(['xtcwriter','-f',sfile,'-t','-n',str(n_events_per_step),'-s',str(i*2),'-e',str(slow_update_freq),'-m',str(n_motor_steps)])
        subprocess.call(['smdwriter','-f',sfile,'-o',str(smd_dir / smd_filename)])
        
        # Copy small data for run no. 2 (test reading small data w/o big data)
        if gen_run2:
            shutil.copy(sfile,str(smd_dir / smd_filename.replace('r0001','r0002')))

if __name__ == "__main__":
    import pathlib
    setup_input_files(pathlib.Path('.'))
