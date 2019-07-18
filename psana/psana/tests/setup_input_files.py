import subprocess
import shutil

def setup_input_files(tmp_path):
    xtc_dir = tmp_path / '.tmp'
    xtc_dir.mkdir()
    smd_dir = xtc_dir / 'smalldata'
    smd_dir.mkdir()

    # segments 0,1 and "counting" timestamps for event-building
    s01file = str(xtc_dir / 'data-r0001-s00.xtc2')
    subprocess.call(['xtcwriter','-f',s01file,'-t','-n','10']) # ask for 10 events with every 4 events, one SlowUpdate inserted.
    subprocess.call(['smdwriter','-f',s01file,'-o',str(smd_dir / 'data-r0001-s00.smd.xtc2')])

    # segments 2,3
    s23file = str(xtc_dir / 'data-r0001-s01.xtc2')
    subprocess.call(['xtcwriter','-f',s23file,'-t','-s','2','-n','10'])
    subprocess.call(['smdwriter','-f',s23file,'-o',str(smd_dir / 'data-r0001-s01.smd.xtc2')])

    shutil.copy(s01file,str(smd_dir / 'data-r0002-s00.smd.xtc2'))
    shutil.copy(s23file,str(smd_dir / 'data-r0002-s01.smd.xtc2'))
