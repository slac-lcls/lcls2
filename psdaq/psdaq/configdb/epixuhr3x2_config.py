import copy  # deepcopy
import fcntl
import json
import logging
import os
import socket
import time
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import numpy.typing as npt

from psdaq.utils import enable_epix_uhr3x2
import epixuhr_3x2_readout_testing as epixUhrDev
from .epixuhr3x2 import EpixUHR3x2_Manager
#import surf.protocols.batcher as batcher

from psdaq.cas.xpm_utils import timTxId
from psdaq.configdb.barrier import Barrier
from psdaq.configdb.det_config import *
from psdaq.configdb.get_config import get_config
from psdaq.configdb.scan_utils import *
from psdaq.configdb.typed_json import cdict

base = None
pv = None

chan = None
group = None
origcfg = None

segids = None
seglist = [0, 1]
asics = None

# Timing delay scans can be limited by this

LOGGER_NAME: str = "ePixUHR3x2"
logger: logging.Logger = logging.Logger(LOGGER_NAME)
logger.setLevel(logging.INFO)


xpmpv_global = None
barrier_global: Barrier = Barrier()


# Used to determine if cofiguration has changed
def _dict_compare(d1, d2, path):
    for k in d1.keys():
        if k in d2.keys():
            if isinstance(d1[k], dict):
                _dict_compare(d1[k], d2[k], path + "." + k)
            elif d1[k] != d2[k]:
                print(f"key[{k}] d1[{d1[k]}] != d2[{d2[k]}]")
        else:
            print(f"key[{k}] not in d1")
    for k in d2.keys():
        if k not in d1.keys():
            print(f"key[{k}] not in d2")


# Sanitize the json for json2xtc by removing offensive characters
def sanitize_config(src):
    dst = {}
    for k, v in src.items():
        if isinstance(v, dict):
            v = sanitize_config(v)
        dst[k.replace("[", "").replace("]", "").replace("(", "").replace(")", "")] = v
    return dst


def supervisor_info(json_msg):
    nworker: int = 0
    supervisor: Optional[bool] = None
    mypid: int = os.getpid()
    myhostname: str = socket.gethostname()
    for drp in json_msg['body']['drp'].values():
        proc_info = drp['proc_info']
        host: str = proc_info['host']
        pid: int = proc_info['pid']
        if host == myhostname and drp['active']:
            if supervisor is None:
                # We are supervisor if our pid is the first entry
                supervisor = pid == mypid
            else:
                # only count workers for second and subsequent entries on this host
                nworker += 1
    return supervisor, nworker


class ReadoutSystemDefn(TypedDict):
    """The multi-panel ePixUHR setup is split into readout systems as defined here.

    Per system (analgous to a panel, or single C1100 for data) you provide:
    - name: Must be of format ROS[\d] -- Using the config C1100 lane makes sense
    - cfgLane: The lane on the config C1100 used by this panel
    - dataDev0: The device file ("/dev/datadev_1a") etc.
    - dataDev1: The second device file if bifurcation is on.
    """
    name: str
    dataDev0: str
    dataDev1: str


def construct_devs_dict(
    dataDev0: str,
    dataDev1: Optional[str],
    rosNum: int = 0,
) -> ReadoutSystemDefn:
    """Construct the panel dictionary in appropriate format for the root object.

    Args:
        dataDev0 (str): The device file for data.

        dataDev1 (str | None): The second device file if using bifurcation.

        rosNum (int): The readout system identifier.

    Returns:
        devReadoutSystem (ReadoutSystemDefn): The device configuration dict suitable for
            the root object.
    """
    devReadoutSystem: ReadoutSystemDefn = {
        "name": f"ROS[{rosNum}]",
        "dataDev0": dataDev0,
        "dataDev1": dataDev1 if dataDev1 else "",
    }

    return devReadoutSystem


def mask_to_lane(lanemask: int) -> int:
    """Convert a lane mask to a lane.

    Args:
        lanemask (int): The lane mask (e.g. 0x1).

    Returns:
        lane (int): The lane.
    """
    return (lanemask & -lanemask).bit_length() - 1


#
#  Initialize the rogue accessor
#
def epixuhr3x2_init(
    arg,
    dev="/dev/datadev_1a",
    lanemask=0x1,
    xpmpv=None,
    timebase="186M",
    verbosity=0
):
    """Initialize the rogue device accessor objects.

    In contrast to other detectors, lanemask is used to specify the lane of the
    *configuration* C1100 rather than the data C1100. All lanes are always used on
    the data C1100.
    """
    global base
    global pv

    global gainMapSelection
    global gainValSelection

    # used to store gain configuration
    gainMapSelection = np.zeros((6, 32256), dtype=np.uint8)
    gainValSelection = np.zeros(6, dtype=np.uint8)

    logger.info("epixuhr3x2_init")

    rosNum: int = 0
    devReadoutSystem: ReadoutSystemDefn = construct_devs_dict(
        dataDev0=dev,
        dataDev1=None,
        # rosNum=rosNum, # Will need to update eventually for multi-panel
    )

    # ... gets passed around via this horrible global variable strategy ...
    base = {}
    #  Connect to the camera and the PCIe card
    detectorRoot = epixUhrDev.Root(
        # This specifies datadev for config and data C1100's
        ReadoutSystems = [devReadoutSystem], # List, as you could hvae multiple in 1 proc
        # dualPcie specifies using bifurcated bus - we're not
        dualPcie       = False,
        defaultFile    = "",        # No config to load
        # Emulator doesn't have all registers - emuMode prevents erroneous r/w in root
        emuMode        = True,
        justCtrl       = True,      # If False, data doesn't make it to DAQ
        loadPllCsv     = False,     # No config to load
        standAloneMode = False,     # Use LCLSII timing
        # Whether to have automated register polling - must disable for multi-panel
        pollEn         = False,
        # Read register values initially - must disable for multi-panel
        initRead       = False,
        otherViewers   = False,     # Initialize other viewers/modules
        promProg       = False,     # Not programming...
        loadCfgAtStart = False,     # Not loading config yamls
        numOfAsics     = 6,         # How many asics per panel
        numOfAdcmons   = 0,
        numOfPscope    = 0,
    )
    detectorRoot.__enter__()

    base["cam"] = detectorRoot
    # The manager object encapsulates access to the rogue object and simplifies
    # most routines. Many registers can be read and written with properties.
    # Methods will do bulk configuration via many writes.
    emulator: bool = False
    manager: EpixUHR3x2_Manager = EpixUHR3x2_Manager(
        root=detectorRoot,
        nasics=6,
        readout_system_num=rosNum,
        logger_name=LOGGER_NAME,
        emulator=emulator,
    )
    base["manager"] = manager

    manager.init_board()
    manager.initialize_timing(timebase=timebase)

    # Firmware info reads can hang if the bus is locked or PGP is down.
    # We will try to read them, but not crash if it fails.
    try:
        firmwareVersion: int = manager.c1100_firmware_version
        buildDate: str = manager.c1100_build_date
        gitHashShort: str = manager.c1100_build_hash

        logging.info(f"firmwareVersion [{firmwareVersion:x}]")
        logging.info(f"buildDate       [{buildDate}]")
        logging.info(f"gitHashShort    [{gitHashShort}]")
    except Exception as e:
        logging.error(f"Failed to read firmware info: {e}. Possible bus lockup!")

    #  store previously applied configuration
    base["cfg"] = None
    base["timebase"] = timebase

    # PLL check
    # ConfigLclsTimingV2
    # StopRun
    # enableClockDependencies
    # waveform control enable
    # trigger reg load
    # waveform reg load
    # fn init asic

    return base


#
#  Set the PGP lane
#
def epixuhr3x2_init_feb(slane=None, schan=None):
    global lane
    global chan

    if slane is not None:
        lane = int(slane)
    if schan is not None:
        chan = int(schan)


#
#  Set the local timing ID and fetch the remote timing ID
#
def epixuhr3x2_connectionInfo(base, alloc_json_str):
    global lane
    global chan

    alloc_json: Dict[str, Any] = json.loads(alloc_json_str)

    supervisor, nworker = supervisor_info(json_msg=alloc_json)
    barrier_global.init(supervisor=supervisor, nworker=nworker)

    logger = logging.Logger(LOGGER_NAME)
    txId: int = timTxId("epixuhr3x2")

    manager = base["manager"]
    timebase = base.get("timebase", "186M")

    if xpmpv_global is not None:
        ...
    else:
        # Property assignments write the registers
        manager.TxId = txId
        rxId: int = manager.RxId
        if rxId == 0xFFFFFFFF:
            logger.error(f"rxId invalid after timing configuration! RxId: {rxId:x}")
            # We don't raise here to allow workers to at least try to sync
        else:
            logger.info(f"Found rxId: {rxId:x}")
        barrier_global.wait()

    # Ensure everyone is stopped and has fresh register values
    manager.Stop()

    # After initial setup, all processes (supervisor and workers) perform ReadAll
    # to sync their local shadow memories with the hardware state
    manager.ReadAll()

    rxId = manager.RxId
    logger.info(f"Using TxId: {txId:x}. Reminder -- RxId: {rxId:x}")

    connect_info = {}
    connect_info["paddr"] = rxId
    connect_info["serno"] = manager.SerNo

    # Include a short serial number + detector type for S/N configdb lookup
    # Add a zero for simplicity -- everything expects segment numbers. So we
    # just make a `0` segment
    det_type: str = "epixuhr3x2"
    connect_info["short_sn_id"] = f"{det_type}_{manager.ShortSerNo}_0"

    print(
        "**** ePixUHR3x2 Panel Short Serial Number Identifier:",
        connect_info["short_sn_id"],
        "****"
    )

    return connect_info


#
#  Translate the 'user' components of the cfg dictionary into 'expert' settings
#  The cfg dictionary may be partial (scanning), so the origcfg dictionary is
#  reference for the full set.
#
def user_to_expert(base, cfg, fullConfig=False):
    global origcfg
    global group
    global lane
    ...
    return base
#
#  Apply the cfg dictionary settings
#
def config_expert(base, cfg, writeCalibRegs=True, secondPass=False):
    global asics  # Need to maintain this across configuration updates
    global gainMapSelection
    global gainValSelection
    ...

    return base

def get_active_asics(asic_mask: int) -> List[int]:
    """Convert an asic bit mask to a 1-indexed position active list.

    E.g. A value of 63 = 0b111111 indicating asics [1, 2, 3, 4, 5, 6] are active.
    """
    result: List[int] = []
    pos: int = 1

    n: int = asic_mask
    while n:
        if n & 1:
            result.append(pos)
        n >>= 1
        pos += 1

    return result

#
#  Called on Configure
#
def epixuhr3x2_config(base, connect_str, cfgtype, detname, detsegm, rog):
    """
        cfg = get_config(connect_str, cfgtype, detname, detsegm)
    manager = base["manager"]
    manager.Stop()
    manager.configure(cfg)
    manager.Start()

    # ... handle segment return values as before ...
    return result
    """
    global origcfg
    global group
    global segids
    global asics
    global gainMapSelection
    global gainValSelection
    group = rog

    #
    #  Retrieve the full configuration from the configDB
    #
    cfg = get_config(connect_str, cfgtype, detname, detsegm)
    origcfg = cfg

    manager = base["manager"]
    manager.Stop()

    # Determine which asics are active
    asic_mask: int = cfg["user"]["asic_enable"]

    # The asic_mask disables analog power to disabled asics
    manager.power_on(asic_mask=asic_mask)
    time.sleep(1)

    # Convert bit-mask to list of asics used elsewhere.
    # This will also disable the data paths for the disabled asics
    asics = get_active_asics(asic_mask=asic_mask)

    manager.setup_bypasses_for_disabled_asics(asic_mask=asic_mask)

    # Enable passThru, i.e., disable the gain expansion
    for i in range(0, 6):
        manager.DataFpga.DataInterpreter.DataGainMultiplier[i].enable.set(True)
        manager.DataFpga.DataInterpreter.DataGainMultiplier[i].passThru.set(0x1)

    # Setup BoardCtrl3x2Readout Registers - this also sets the debug timing outputs
    board_ctrl: Dict[str, Any] = cfg["expert"]["FebFpga"]["App"]["BoardCtrl3x2Readout"]
    # manager.setup_board_control_registers(board_ctrl=board_ctrl)
    manager.setup_debug_timing_out(board_ctrl=board_ctrl)

    # Setup trigger register config - tell detector to run off LCLS2 triggers
    # Configure delays and what not
    # Setup waveform
    manager.init_waveform_control()

    # Setup each Asic
    manager.set_running_asics(asics=asics, app_cfg=cfg["expert"]["FebFpga"]["App"])

    manager.reset_asic_gt(asics=asics, emulator=manager.emulator)

    time.sleep(1)

    # Configure any gain settings
    if cfg["user"]["Gain"]["SetSameGain4All"]:
        if cfg["user"]["Gain"]["UsePixelMap"]:
            sel: int = cfg["user"]["Gain"]["PixelBitMapSel"]
            gain_map_keys: List[str] = list(cfg["expert"]["pixelBitMaps"].keys())
            gain_map_key: str = gain_map_keys[sel]
            gain_map_list: List[int] = cfg["expert"]["pixelBitMaps"][gain_map_key]
            gain_map: npt.NDArray[np.uint8] = np.array(gain_map_list, dtype=np.uint8).reshape((168, 192))
            manager.set_pixel_gain_map(asics=asics, map_data=gain_map)

            for i in range(6):
                # Want to only record it for the "enabled" asics?
                gainMapSelection[i] = gain_map.reshape(32256)
        else:
            new_gain: int = int(cfg["user"]["Gain"]["SetGainValue"])
            manager.set_pixel_gain(asics=asics, gain_value=new_gain)

            # Want to only record it for the "enabled" asics?
            gainValSelection[:] = new_gain

        # Need to check for the pixel gain maps... and also handle not SetSameGain4All
    time.sleep(1)

    # Configure any charge injection
    VINJ_DAC: Dict[str, Any] = cfg["user"]["FebFpga"]["App"]["VINJ_DAC"]
    manager.set_charge_injection(
        enable=bool(VINJ_DAC["enable"]),
        single_val=VINJ_DAC.get("dacSingleValue"),
        asics=asics,
        start=VINJ_DAC.get("dacStartValue"),
        stop=VINJ_DAC.get("dacStopValue"),
        step=VINJ_DAC.get("dacStepValue"),
        level=cfg["user"]["Gain"]["SetGainValue"],
        skip_x=VINJ_DAC.get("SKIP_X"),
        skip_y=VINJ_DAC.get("SKIP_Y"),
    )

    time.sleep(1)

    manager.running_trigger_registers(
        rog=rog,
        start_ns=cfg["user"]["start_ns"],
        trig_cfg=cfg["expert"]["FebFpga"]["App"]["TimingRx"]["TriggerEventManager"]
    )

    manager.reset_counters()
    manager.Start()

    #barrier_global.wait()

#    manager.start_auto_trigger()

    cfg[":types:"]["use_serial_db"] = "boolEnum"
    cfg["use_serial_db"] = 0
    origcfg = cfg
    topname = cfg["detName:RO"].split("_")

    segcfg = {}
    segids = {}

    #  Rename the complete config detector
    segcfg[0] = cfg.copy()
    segcfg[0]["detName:RO"] = "_".join(topname[:-1]) + "hw_" + topname[-1]

    #  Construct the ID
    # digitalId = 0
    # pwrCommId = 0
    # carrierId = 0

    # detId = "%010d-%010d-%010d-%010d" % (firmwareVersion, carrierId, digitalId, pwrCommId)
    #detId = "%010d-%010d-%010d-%010d" % (0, 0, 0, 0)
    detId: str = manager.SerNo

    segids[0] = detId
    top = cdict()

    top.setAlg("config", [0, 1, 0])
    top.setInfo(
        detType="epixuhr3x2",
        detName="_".join(topname[:-1]),
        detSegm=int(topname[-1]),
        detId=detId,
        doc="No comment",
    )

    top.set(
        f"gainCSVAsic", gainMapSelection.tolist(), "UINT8"
    )  # only the rows which have readable pixels
    top.set(f"gainAsic", gainValSelection.tolist(), "UINT8")

    segcfg[1] = top.typed_json()

    result = []
    for i in seglist:
        logging.debug("json seg {}  detname {}".format(i, segcfg[i]["detName:RO"]))
        result.append(json.dumps(sanitize_config(segcfg[i])))

    base["cfg"] = copy.deepcopy(cfg)
    base["result"] = copy.deepcopy(result)
    logging.info("created gain values in XTC file")
    return result


def epixuhr3x2_unconfig(base):
    logging.info("epixuhr3x2_unconfig")

    manager = base["manager"]

    if barrier_global.supervisor:
        manager.Stop()

    barrier_global.wait()
    return base


#
#  Build the set of all configuration parameters that will change
#  in response to the scan parameters
#
def epixuhr3x2_scan_keys(update):
    logging.debug("epixuhr3x2_scan_keys")
    global origcfg
    global base
    global gainValSelection
    manager = base["manager"]
    upd_keys = json.loads(update)

    cfg = {}
    copy_reconfig_keys(cfg, origcfg, upd_keys)

    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, origcfg, key)
        copy_config_entry(cfg[":types:"], origcfg[":types:"], key)
    topname = cfg["detName:RO"].split("_")
    segcfg = {0: cfg.copy()}
    segcfg[0]["detName:RO"] = "_".join(topname[:-1]) + "hw_" + topname[-1]
    top = cdict()
    top.setAlg("config", [1, 1, 0])
    top.setInfo(
        detType="epixuhr3x2",
        detName="_".join(topname[:-1]),
        detSegm=int(topname[-1]),
        detId=manager.SerNo,
        doc="Scan Key"
    )
    top.set("gainAsic", gainValSelection.tolist(), "UINT8")
    segcfg[1] = top.typed_json()
    result = [json.dumps(sanitize_config(segcfg[i])) for i in range(len(segcfg))]
    return result


#
#  Return the set of configuration updates for a scan step
#
def epixuhr3x2_update(update):
    """
        global base
    manager = base["manager"]
    manager.Stop()
    # 'update' contains the scan step changes (e.g. {"user.Gain": 1})
    manager.configure({'user': json.loads(update)})
    manager.Start()

    return result
    """
    logging.debug("epixuhr3x2_update")
    global origcfg
    global base
    global group # Readout group set on configure
    global asics
    global gainValSelection
    global gainMapSelection

    manager = base["manager"]
    manager.Stop()

    # Extract the step-specific configuration changes
    upd_dict = json.loads(update)
    cfg = {}
    update_config_entry(cfg, origcfg, upd_dict)

    # Check for the start_ns for timing scans
    start_ns = None

    # Direct gain update logic: check for 'SetGainValue' in the update
    # Check for charge injection values as well
    new_gain = None

    skip_x = 0
    skip_y = 0
    dac_start = None
    dac_stop = None
    dac_step = None
    inj_en = None
    for k, v in upd_dict.items():
        if "user.FebFpga.App.VINJ_DAC.SKIP_X" in k:
            skip_x = int(v)

        if "user.FebFpga.App.VINJ_DAC.SKIP_Y" in k:
            skip_y = int(v)

        if "user.Gain.SetGainValue" in k:
            new_gain = int(v)

        if "FebFpga.App.VINJ_DAC.dacStartValue" in k:
            dac_start = int(v)

        if "FebFpga.App.VINJ_DAC.dacStopValue" in k:
            dac_stop = int(v)

        if "FebFpga.App.VINJ_DAC.dacStepValue" in k:
            dac_step = int(v)

        if "expert.FebFpga.App.WaveformControl.InjEn" in k:
            inj_en = int(v)

        if "user.start_ns" in k:
            start_ns = int(v)

    if new_gain is None:
        user = upd_dict.get('user', {})
        gain_cfg = user.get('Gain', {})
        if 'SetGainValue' in gain_cfg:
            new_gain = int(gain_cfg['SetGainValue'])
    if new_gain is not None:
        # Apply specifically to all 6 ASICs
        active_asics = asics
        manager.set_pixel_gain(active_asics, new_gain)
        gainValSelection[:] = new_gain
        logging.info(f"Scan update: Applied set_pixel_gain({new_gain}) to ASICs {active_asics}")

    if any(val is not None for val in [inj_en, dac_start, dac_stop, dac_step]):
        active_asics = asics
        manager.set_charge_injection(
            enable=True,
            single_val=None,
            asics=active_asics,
            start=dac_start,
            stop=dac_stop,
            step=dac_step,
            level=new_gain,
            skip_x=skip_x,
            skip_y=skip_y,
        )

    if start_ns is not None:
        # Pass empty dict to not reconfigure event codes etc.
        # DAQ Trigger
        _ = manager.setup_daq_trigger(rog=group, start_ns=start_ns, trig_cfg={})

        # Run Trigger
        _ = manager.setup_run_trigger(rog=group, start_ns=start_ns, trig_cfg={})

    manager.reset_counters()
    manager.Start()
    # Document mandatory fields for XTC translation
    for key in ("detType:RO", "detName:RO", "detId:RO", "doc:RO", "alg:RO"):
        copy_config_entry(cfg, origcfg, key)
        copy_config_entry(cfg[":types:"], origcfg[":types:"], key)

    topname = cfg["detName:RO"].split("_")
    segcfg = {0: cfg.copy()}
    segcfg[0]["detName:RO"] = "_".join(topname[:-1]) + "hw_" + topname[-1]
    # Documentation segment for calibration step tracking
    top = cdict()
    top.setAlg("config", [1, 1, 0])
    top.setInfo(
        detType="epixuhr3x2",
        detName="_".join(topname[:-1]),
        detSegm=int(topname[-1]),
        detId=manager.SerNo,
        doc="Scan Step"
    )
    top.set("gainAsic", gainValSelection.tolist(), "UINT8")
    segcfg[1] = top.typed_json()
    result = [json.dumps(sanitize_config(segcfg[i])) for i in [0, 1]]
    return result


def epixuhr3x2_enable(base):
    manager = base["manager"]
    logging.info("epixuhr3x2_enable")

    #epixuhr3x2_external_trigger(base)
    #_start(base)


def epixuhr3x2_disable(base):
    logging.info("epixuhr3x2_disable")
    # Prevents transitions going through: epixuhr3x2_internal_trigger(base)


def _stop(base):
    logging.info("_stop")
    detectorRoot = base["cam"]
    detectorRoot.StopRun()
    time.sleep(0.1)  #  let last triggers pass through


def _start(base):
    logging.info("_start")
    detectorRoot = base["cam"]
    detectorRoot.FebFpga.App.SetTimingTrigger()

    manager = base["manager"]
    # Turn on the triggering
    manager.write_and_check(
        detectorRoot.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[0].MasterEnable,
        True,
    )
    manager.write_and_check(
        detectorRoot.FebFpga.App.TimingRx.TriggerEventManager.TriggerEventBuffer[1].MasterEnable,
        True,
    )

    detectorRoot.StartTimingRun()

