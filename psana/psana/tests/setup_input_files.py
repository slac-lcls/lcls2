import subprocess

def setup_input_files(tmp_path, n_files=2, slow_update_freq=4, n_motor_steps=1, n_events_per_step=10, gen_run2=True):
    xtc_dir = tmp_path / '.tmp'
    xtc_dir.mkdir()
    smd_dir = xtc_dir / 'smalldata'
    smd_dir.mkdir()
    expcode = 'xpptut15'
    runnum = 14

    for i in range(n_files):
        # Only segments 0,1 has "counting" timestamps for event-building
        filename = f"{expcode}-r{str(runnum).zfill(4)}-s{str(i).zfill(3)}-c000.xtc2"
        smd_filename = f"{expcode}-r{str(runnum).zfill(4)}-s{str(i).zfill(3)}-c000.smd.xtc2"
        sfile = str(xtc_dir / filename)
        subprocess.call(['xtcwriter','-f',sfile,'-t','-n',str(n_events_per_step),'-s',str(i*2),'-e',str(slow_update_freq),'-m',str(n_motor_steps)])
        subprocess.call(['smdwriter','-f',sfile,'-o',str(smd_dir / smd_filename)])
    # breakpoint()
    return xtc_dir

if __name__ == "__main__":
    import pathlib
    setup_input_files(pathlib.Path('.'))
