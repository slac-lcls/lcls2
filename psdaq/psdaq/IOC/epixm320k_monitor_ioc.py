import time
import numpy as np
from typing import Any, List, Dict

# rougue imports
from psdaq.utils import enable_epix_hr_m_320k
import epix_hr_leap_common as leapCommon
import ePix320kM
import rogue
import rogue.hardware.axi
import rogue.protocols.srp
import pyrogue
# caproto imports
import caproto as ca
from caproto.server import (
        PvpropertyEnum,
        pvproperty,
        template_arg_parser,
        run
)
# shared monitoring IOC code
from psdaq.IOC.epix_monitor_base import (
        EpixMonitoringIOCBase,
        MonitorAxiPacket,
        add_common_args,
        counter_pv,
        current_pv,
        humidity_pv,
        setup_logging,
        string_pv,
        temp_pv,
        voltage_pv,
)


class EpixM320kMonitorUtils:
    """
    Namespace functions that are local functions in a constructor or lambdas in rogue so we can't use them directly.
    """
    @staticmethod
    def convert(raw: int) -> float:
        return 2.5 * (raw & 0xffffff) / 16777216.

    @staticmethod
    def getThermal(raw: int) -> float:
        volt = EpixM320kMonitorUtils.convert(raw)
        if volt > 0.0:
            return (1/((np.log((volt/0.0000791)/10000)/3750)+(1/298.15)))-273.15
        else:
            return 0.0

    @staticmethod
    def getHumidity(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 0.97647 * 58.2 - 8.8225

    @staticmethod
    def getVoltage(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 3 * 0.97647

    @staticmethod
    def getCurrent(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 3. * 0.97647 * 2.24 / 0.8;

    @staticmethod
    def getPcbHumidity(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 0.97647 * 47.646 - 7.2184

    @staticmethod
    def getPcbVoltage3v3(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 2.

    @staticmethod
    def getPcbVoltage1v8(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw)

    @staticmethod
    def getPcbVoltage(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 3.

    @staticmethod
    def getPcbCurrent(raw: int) -> float:
        return EpixM320kMonitorUtils.convert(raw) * 10.


CHANNEL_DEFS: List[Dict[int, Any]] = [
    {
        0: dict(
            name="Carrier Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp1",
        ),
        1: dict(
            name="Digital Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp2",
        ),
        2: dict(
            name="Digital Humidity",
            unit="%",
            conv=EpixM320kMonitorUtils.getHumidity,
            pv_signal="humidity",
        ),
        3: dict(
            name="DUAL LDO 0 I 1V8",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="dual_ldo0_curr",
        ),
        4: dict(
            name="ASIC0 An I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="asic0_ana_curr",
        ),
        5: dict(
            name="An V 6V",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="ana_6v0_volt",
        ),
        6: dict(
            name="DUAL LDO 0 V 1V8",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="dual_ldo0_volt",
        ),
        7: dict(
            name="ASIC0 An V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="asic0_ana_volt",
        ),
    },
    {
        0: dict(
            name="Carrier Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp1",
        ),
        1: dict(
            name="Digital Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp2",
        ),
        2: dict(
            name="Digital Humidity",
            unit="%",
            conv=EpixM320kMonitorUtils.getHumidity,
            pv_signal="humidity",
        ),
        3: dict(
            name="DUAL LDO 1 I 1V8",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="dual_ldo1_curr",
        ),
        4: dict(
            name="ASIC1 An I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="asic1_ana_curr",
        ),
        5: dict(
            name="VCCA V 3V0",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="vcca_3v0_volt",
        ),
        6: dict(
            name="DUAL LDO 1 V 1V8",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="dual_ldo1_volt",
        ),
        7: dict(
            name="ASIC1 An V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="asic1_ana_volt",
        ),
    },
    {
        0: dict(
            name="Carrier Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp1",
        ),
        1: dict(
            name="Digital Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp2",
        ),
        2: dict(
            name="Digital Humidity",
            unit="%",
            conv=EpixM320kMonitorUtils.getHumidity,
            pv_signal="humidity",
        ),
        3: dict(
            name="Digital I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="dig_2v5_curr",
        ),
        4: dict(
            name="ASIC2 An I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="asic2_ana_curr",
        ),
        5: dict(
            name="Digital V 6V",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="dig_6v0_volt",
        ),
        6: dict(
            name="Digital V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="dig_2v5_volt",
        ),
        7: dict(
            name="ASIC2 An V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="asic2_ana_volt",
        ),
    },
    {
        0: dict(
            name="Carrier Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp1",
        ),
        1: dict(
            name="Digital Therm",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="temp2",
        ),
        2: dict(
            name="Digital Humidity",
            unit="%",
            conv=EpixM320kMonitorUtils.getHumidity,
            pv_signal="humidity",
        ),
        3: dict(
            name="DS_PLL I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="dspll_2v5_curr",
        ),
        4: dict(
            name="ASIC3 An I 2V5",
            unit="A",
            conv=EpixM320kMonitorUtils.getCurrent,
            pv_signal="asic3_ana_curr",
        ),
        5: dict(
            name="VCC V 2V7",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="vcc_2v7_volt",
        ),
        6: dict(
            name="DS_PLL V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="dspll_2v5_volt",
        ),
        7: dict(
            name="ASIC3 An V 2V5",
            unit="V",
            conv=EpixM320kMonitorUtils.getVoltage,
            pv_signal="asic3_ana_volt",
        ),
    },
    {
        0: dict(
            name="PCB Humidity",
            unit="%",
            conv=EpixM320kMonitorUtils.getPcbHumidity,
            pv_signal="pcb_humidity",
        ),
        1: dict(
            name="PCB Thermal",
            unit="°C",
            conv=EpixM320kMonitorUtils.getThermal,
            pv_signal="pcb_temp",
        ),
        2: dict(
            name="PCB V 3V3",
            unit="V",
            conv=EpixM320kMonitorUtils.getPcbVoltage3v3,
            pv_signal="pcb_3v3_volt",
        ),
        3: dict(
            name="PCB V 1V8",
            unit="V",
            conv=EpixM320kMonitorUtils.getPcbVoltage1v8,
            pv_signal="pcb_1v8_volt",
        ),
        4: dict(
            name="PCB An V 6V",
            unit="V",
            conv=EpixM320kMonitorUtils.getPcbVoltage,
            pv_signal="pcb_ana_volt",
        ),
        5: dict(
            name="PCB Dig V 6V",
            unit="V",
            conv=EpixM320kMonitorUtils.getPcbVoltage,
            pv_signal="pcb_dig_volt",
        ),
        6: dict(
            name="PCB Dig I 6V",
            unit="A",
            conv=EpixM320kMonitorUtils.getPcbCurrent,
            pv_signal="pcb_dig_curr",
        ),
        7: dict(
            name="PCB An I 6V",
            unit="A",
            conv=EpixM320kMonitorUtils.getPcbCurrent,
            pv_signal="pcb_ana_curr",
        ),
    }
]

class EpixM320kMonitorPacket(MonitorAxiPacket):
    """
    One ePixM320k slow-ADC monitor stream packet.

    Format (68 bytes = 9 × uint64 little-endian):
      word[ 0]        : packet counter (raw uint64)
      word[ 1]        : channel 0
      ...
      word[ 8]        : channel 7

    See CHANNEL_DEFS above for the channel mapping.  Values are uint64;
    channels defs depend on the tdest of the axi packet.
    """

    STRUCT_FMT = "<9Q"
    HEADER_WORDS = 1
    N_CHANNELS = 8
    CHANNEL_DEFS = CHANNEL_DEFS


class EpixM320kBoard(pyrogue.Root):
    def __init__(self, dev, lane, vc):
        super().__init__(name='ePixBoard', pollEn=False, initRead=False)
        dma = rogue.hardware.axi.AxiStreamDma(dev, lane << 8 | vc, True)
        srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(dma, srp)
        self.addInterface(dma)
        self.addInterface(srp)
        self.add(leapCommon.Core(
            offset   = 0x0000_0000,
            memBase  = srp,
            promProg = False,
            pgpLaneVc= [1,1,1,1,0,3,1,3]))
        self.add(ePix320kM.App(
            offset   = 0x8000_0000,
            memBase  = srp,
            sim      = False,
            enabled  = True))

    @staticmethod
    def configure(dev, lane, vc, flag, mon_period, trig_period, queue):
        data = {'logs': {'info': [], 'warn': [], 'error': []}}
        try:
            with EpixM320kBoard(dev, lane, vc) as root:
                # read firmware info
                fw = root.Core.AxiVersion.FpgaVersion.get()
                data["firmware_version"] = '0x%08x' % fw
                githash = root.Core.AxiVersion.GitHash.get()
                data["firmware_githash"] = '%040x' % githash
                bldstr = root.Core.AxiVersion.BuildStamp.get()
                data["firmware_bldstr"] = bldstr
                # configure the monitoring registers
                root.App.Adcs.DigSlowADC.enableADC.set(flag)
                data['logs']['info'].append(f"set App.Adcs.DigSlowADC.enableADC to {flag}")
                root.App.Adcs.DigSlowADC.autoTrig_us.set(mon_period)
                data['logs']['info'].append(f"set App.Adcs.DigSlowADC.autoTrig_us to {mon_period}")
                root.App.Adcs.PCBSlowADC.enableADC.set(flag)
                data['logs']['info'].append(f"set App.Adcs.PCBSlowADC.enableADC to {flag}") 
                root.App.Adcs.PCBSlowADC.autoTrig_us.set(mon_period)
                data['logs']['info'].append(f"set App.Adcs.PCBSlowADC.autoTrig_us to {mon_period}")
        except Exception as exc:
            data['logs']['error'].append(f"exception encountering during configuration: {exc}")
        finally:
            # send the firmware info back
            queue.put(data)


class EpixM320kMonitoringIOC(EpixMonitoringIOCBase):
    """
    EPICS IOC publishing the ePixQuad environmental monitor readings.
    """
    board_cls = EpixM320kBoard
    packet_cls = EpixM320kMonitorPacket
    monrateconv = 1000000
    
    # sensor temps, with the reading below which the sensor is considered invalid
    stemp_channels = {
            ("temp1", -273.15),
            ("temp2", -273.15),
    }
    # electronics temps, with the range outside which the reading is invalid
    etemp_channels = {
            ("pcb_temp", 0.01, 200),
    }

    @property
    def monitor_setting(self):
        """
        Monitor rate (Hz) converted to period (us)
        """
        return int(self.monrateconv / self.set_monitor_rate.value)

    def check_temps(self, data):
        """
        Check that at least on the of the sensor or electronics temps are valid
        """
        stemp = False
        etemp = False

        # loop over the sensor temps to find if at least one is valid
        for channame, lowlim in self.stemp_channels:
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim:
                    stemp = True
                    break

        # loop over the elec temps to find if at least one is valid
        for channame, lowlim, highlim in self.etemp_channels:
            if hasattr(self, channame):
                chan = getattr(self, channame)
                if chan.value > lowlim and chan.value < highlim:
                    etemp = True
                    break

        return stemp, etemp

    async def update_extra(self, data):
        """
        Publish the derived sensor/electronics temperature validity flags.
        """
        stemp, etemp = self.check_temps(data)
        await self.stemp_ok.write(value=stemp)
        await self.etemp_ok.write(value=etemp)

    temp1 = temp_pv(name="TEMP1",
                    alarm_group="temp1",
                    doc="Carrier Therm")
    temp2 = temp_pv(name="TEMP2",
                    alarm_group="temp2",
                    doc="Digital Therm")
    humidity = humidity_pv(name="HUMIDITY",
                           alarm_group="humidity",
                           doc="Digital Humidity")
    dual_ldo0_curr = current_pv(name="DUAL_LDO0_CURR",
                                alarm_group="dual_ldo0_curr",
                                doc="DUAL LDO 0 I 1V8")
    dual_ldo1_curr = current_pv(name="DUAL_LDO1_CURR",
                                alarm_group="dual_ldo1_curr",
                                doc="DUAL LDO 1 I 1V8")
    asic0_ana_curr = current_pv(name="ASIC0_ANA_CURR",
                                alarm_group="asic0_ana_curr",
                                doc="ASIC0 An I 2V5")
    asic1_ana_curr = current_pv(name="ASIC1_ANA_CURR",
                                alarm_group="asic1_ana_curr",
                                doc="ASIC1 An I 2V5")
    asic2_ana_curr = current_pv(name="ASIC2_ANA_CURR",
                                alarm_group="asic2_ana_curr",
                                doc="ASIC2 An I 2V5")
    asic3_ana_curr = current_pv(name="ASIC3_ANA_CURR",
                                alarm_group="asic3_ana_curr",
                                doc="ASIC3 An I 2V5")
    dig_2v5_curr = current_pv(name="DIG_2V5_CURR",
                              alarm_group="dig_2v5_curr",
                              doc="Digital I 2V5")
    dspll_2v5_curr = current_pv(name="DSPLL_2V5_CURR",
                                alarm_group="dspll_2v5_curr",
                                doc="DS_PLL I 2V5")
    ana_6v0_volt = voltage_pv(name="ANA_6V0_VOLT",
                              alarm_group="ana_6v0_volt",
                              doc="An V 6V")
    vcca_3v0_volt = voltage_pv(name="VCCA_3V0_VOLT",
                               alarm_group="vcca_3v0_volt",
                               doc="VCCA V 3V0")
    dig_6v0_volt = voltage_pv(name="DIG_6V0_VOLT",
                              alarm_group="dig_6v0_volt",
                              doc="Digital V 6V")
    vcc_2v7_volt = voltage_pv(name="VCC_2V7_VOLT",
                              alarm_group="vcc_2v7_volt",
                              doc="VCC V 2V7")
    dual_ldo0_volt = voltage_pv(name="DUAL_LDO0_VOLT",
                                alarm_group="dual_ldo0_volt",
                                doc="DUAL LDO 0 V 1V8")
    dual_ldo1_volt = voltage_pv(name="DUAL_LDO1_VOLT",
                                alarm_group="dual_ldo1_volt",
                                doc="DUAL LDO 1 V 1V8")
    dig_2v5_volt = voltage_pv(name="DIG_2V5_VOLT",
                              alarm_group="dig_2v5_volt",
                              doc="Digital V 2V5")
    dspll_2v5_volt = voltage_pv(name="DSPLL_2V5_VOLT",
                                alarm_group="dspll_2v5_volt",
                                doc="DS_PLL V 2V5")
    asic0_ana_volt = voltage_pv(name="ASIC0_ANA_VOLT",
                                alarm_group="asic0_ana_volt",
                                doc="ASIC0 An V 2V5")
    asic1_ana_volt = voltage_pv(name="ASIC1_ANA_VOLT",
                                alarm_group="asic1_ana_volt",
                                doc="ASIC1 An V 2V5")
    asic2_ana_volt = voltage_pv(name="ASIC2_ANA_VOLT",
                                alarm_group="asic2_ana_volt",
                                doc="ASIC2 An V 2V5")
    asic3_ana_volt = voltage_pv(name="ASIC3_ANA_VOLT",
                                alarm_group="asic3_ana_volt",
                                doc="ASIC3 An V 2V5")
    pcb_humidity = humidity_pv(name="PCB_HUMIDITY",
                               alarm_group="pcb_humidity",
                               doc="PCB Humidity")
    pcb_temp = temp_pv(name="PCB_TEMP",
                       alarm_group="pcb_temp",
                       doc="PCB Thermal")
    pcb_3v3_volt = voltage_pv(name="PCB_3V3_VOLT",
                              alarm_group="pcb_3v3_volt",
                              doc="PCB V 3V3")
    pcb_1v8_volt = voltage_pv(name="PCB_1V8_VOLT",
                              alarm_group="pcb_1v8_volt",
                              doc="PCB V 1V8")
    pcb_ana_volt = voltage_pv(name="PCB_ANA_VOLT",
                              alarm_group="pcb_ana_volt",
                              doc="PCB An V 6V")
    pcb_dig_volt = voltage_pv(name="PCB_DIG_VOLT",
                              alarm_group="pcb_dig_volt",
                              doc="PCB Dig V 6V")
    pcb_dig_curr = current_pv(name="PCB_DIG_CURR",
                              alarm_group="pcb_dig_curr",
                              doc="PCB Dig I 6V")
    pcb_ana_curr = current_pv(name="PCB_ANA_CURR",
                              alarm_group="pcb_ana_curr",
                              doc="PCB An I 6V")
    stemp_ok = counter_pv(name="STEMP_OK",
                          alarm_group="stemp_ok",
                          doc="epixm sensor temp OK")
    etemp_ok = counter_pv(name="ETEMP_OK",
                          alarm_group="etemp_ok",
                          doc="epixm electronics temp OK")

def main():
    # Parse standard EPICS IOC command-line options
    parser, split_args = template_arg_parser(
        default_prefix="DRP:EPIX:CMP055:",
        desc="Read ePixM320k environmental monitor packets via rogue and publish via caproto IOC"
    )
    add_common_args(parser, lane=6, vc=0, regvc=0)
    parser.add_argument(
        "--reglane",
        default=5,
        type=int,
        help="PGP lane number for register access (default: 5)",
    )

    args = parser.parse_args()
    ioc_options, run_options = split_args(args)

    setup_logging(args)

    # Start the server
    ioc = EpixM320kMonitoringIOC(dev=args.dev, lane=args.lane, vc=args.vc, reglane=args.reglane, regvc=args.regvc, **ioc_options)
    run(ioc.pvdb, **run_options, startup_hook=ioc.__ainit__)



if __name__ == '__main__':
    main()
