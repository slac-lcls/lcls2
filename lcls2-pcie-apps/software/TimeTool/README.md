### slaclab/lcls2-pcie-apps/software/TimeTool

<!--- ######################################################## -->

# Dependences 

1) Software: Rogue@[v3.5](https://github.com/slaclab/rogue/releases/tag/v3.5.0)

2) DMA Driver: aes-stream-drivers@[v5.4.0](https://github.com/slaclab/aes-stream-drivers/releases/tag/v5.4.0)

3) FEB board Firmware Image: [ClinkFebPgp2b_1ch-0x00000030](https://github.com/slaclab/cameralink-gateway/blob/master/firmware/targets/ClinkFebPgp2b_1ch/images/ClinkFebPgp2b_1ch-0x00000030-20190422155523-ruckman-31260360.mcs)

4) Firmware Submodules:
```
 f59a00e4294803fa3c11a5e33da7c41b7c1241d4 axi-pcie-core (v2.1.5)
 1df89143fe43f2ce942dc8a4f51fd0f9d9a1418a clink-gateway-fw-lib (v1.0.4)
 949a2e14cda1e2c4fbdd43676ca9210296719105 l2si-core (v1.0.1)
 8bb7837112a5b71d65bac7bc9ae57eacc71aaa35 lcls-timing-core (v1.12.6)
 4870f98eacd26a543b5ec7c2ef64cfbade0d2ed6 lcls2-pgp-fw-lib (v1.1.1)
 b447a00864e68c39ed2e7b93f5b00869adf91369 ruckus (v1.7.6)
 29f5472999a85c5943fce0124652019a8e4bf660 surf (v1.9.9)
```

<!--- ######################################################## -->

# How to install the Rogue With Anaconda

> https://slaclab.github.io/rogue/installing/anaconda.html

<!--- ######################################################## -->

# How to reprogram the FEB firmware via Rogue software

1) Setup the rogue environment
```
$ cd lcls2-pcie-apps/software/TimeTool
$ source setup_env_template.sh
```

2) Run the FEB firmware update script:
```
$ python scripts/updateFeb.py --lane <PGP_LANE> --mcs <PATH_TO_MCS>
```
where <PGP_LANE> is the PGP lane index (range from 0 to 3)
and <PATH_TO_MCS> is the path to the firmware .MCS file


<!--- ######################################################## -->

# How to reprogram the PCIe firmware via Rogue software

1) Setup the rogue environment
```
$ cd cd lcls2-pcie-apps/software/TimeTool
$ source setup_env_template.sh
```

2) Run the PCIe firmware update script:
```
$ python scripts/updatePcieFpga.py --path ../../firmware/targets/TimeToolKcu1500/images/
```

3) Reboot the computer
```
sudo reboot
```

<!--- ######################################################## -->

# How to run the Rogue PyQT GUI

1) Setup the rogue environment
```
$ cd cd lcls2-pcie-apps/software/TimeTool
$ source setup_env_template.sh
```

2) Lauch the GUI:
```
$ python scripts/gui.py
```

<!--- ######################################################## -->

# How to run the Rogue PyQT GUI with VCS firmware simulator

1) Start up two terminal

2) In the 1st terminal, launch the VCS simulation
```
$ source lcls2-pcie-apps/firmware/setup_env_slac.sh
$ cd lcls2-pcie-apps/firmware/targets/TimeToolKcu1500/
$ make vcs
$ cd ../../build/TimeToolKcu1500/TimeToolKcu1500_project.sim/sim_1/behav/
$ source setup_env.sh
$ ./sim_vcs_mx.sh
$ ./simv -gui &
```

3) When the VCS GUI pops up, start the simulation run

4) In the 2nd terminal, launch the PyQT GUI in simulation mode
```
$ cd cd lcls2-pcie-apps/software/TimeTool
$ source setup_env_template.sh
$ python scripts/gui.py --dev sim
```

<!--- ######################################################## -->
