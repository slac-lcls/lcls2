#-----------------------------------------------------------------------------
# Description:
# PyRogue Gthe3Channel
#-----------------------------------------------------------------------------
# This file is part of the 'SLAC Firmware Standard Library'. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the 'SLAC Firmware Standard Library', including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr

import math
import sys

import time

import matplotlib.pyplot as plt
import numpy as np

class EyeGth(pr.Device):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ##############################
        # Variables
        ##############################
        self.add(pr.RemoteVariable(
            name         = "RX_DATA_WIDTH",
            offset       =  0x03 << 2,
            bitSize      =  4,
            bitOffset    =  5,
            mode         = "RW",
            # enum         = {
                # 0 : '-',
                # 2 : '16',
                # 3 : '20',
                # 4 : '32',
                # 5 : '40',
                # 6 : '64',
                # 7 : '80',
                # 8 : '128',
                # 9 : '160'},
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_PRESCALE",
            offset       =  0xF0,
            bitSize      =  5,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_EYE_SCAN_EN",
            offset       =  0xF1,
            bitSize      =  1,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RX_EYESCAN_VS_NEG_DIR",
            offset       =  0x25D,
            bitSize      =  1,
            bitOffset    =  2,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RX_EYESCAN_VS_UT_SIGN",
            offset       =  0x25D,
            bitSize      =  1,
            bitOffset    =  1,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RX_EYESCAN_VS_CODE",
            offset       =  0x25C,
            bitSize      =  7,
            bitOffset    =  2,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "RX_EYESCAN_VS_RANGE",
            offset       =  0x25C,
            bitSize      =  2,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_CLK_PHASE_SEL",
            offset       =  0x251,
            bitSize      =  1,
            bitOffset    =  3,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_PMA_CFG",
            offset       =  0x144,
            bitSize      =  10,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_HORZ_OFFSET",
            offset       =  0x13C,
            bitSize      =  12,
            bitOffset    =  4,
            mode         = "RW",
        ))

        self.addRemoteVariables(
            name         = "ES_SDATA_MASK",
            offset       =  0x124,
            bitSize      =  16,
            mode         = "RW",
            number       =  5,
            stride       =  4,
        )

        self.addRemoteVariables(
            name         = "ES_QUAL_MASK",
            offset       =  0x110,
            bitSize      =  16,
            mode         = "RW",
            number       =  5,
            stride       =  4,
        )

        self.addRemoteVariables(
            name         = "ES_QUALIFIER",
            offset       =  0xFC,
            bitSize      =  16,
            mode         = "RW",
            number       =  5,
            stride       =  4,
        )

        self.add(pr.RemoteVariable(
            name         = "ES_CONTROL",
            offset       =  0xF1,
            bitSize      =  6,
            bitOffset    =  2,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_ERRDET_EN",
            offset       =  0xF1,
            bitSize      =  1,
            bitOffset    =  1,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_CONTROL_STATUS",
            offset       =  0x54c,
            bitSize      =  4,
            bitOffset    =  0,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_SAMPLE_COUNT",
            offset       =  0x548,
            bitSize      =  16,
            bitOffset    =  1,
            mode         = "RW",
        ))

        self.add(pr.RemoteVariable(
            name         = "ES_ERROR_COUNT",
            offset       =  0x544,
            bitSize      =  16,
            bitOffset    =  1,
            mode         = "RW",
        ))

    def bathtubPlot(self, fname=None):
        bers = self.bathtub()
        extrapolated = self.extrapolate(bers)
        plt.clf()
        plt.close()
        plt.figure()

        plt.plot(np.array(bers), color='blue', label='measurement')
        plt.plot(np.array(extrapolated), color='blue', label='Extrapolation (random jitter)', linestyle='dashed')

        plt.axvline(x=54, color="green", label="CDR Mask: 0.15UI ({})".format(54), linewidth=1)

        plt.plot(54, extrapolated[54], marker="x", markersize=4, markeredgecolor="red")
        plt.text(44, extrapolated[54], 'BER: {:.2e}'.format(extrapolated[54]), ha="center")

        plt.legend()
        plt.yscale("log")
       
        if fname:
            plt.savefig(fname)
        else:
            plt.show()
            
        return extrapolated[54]

    def eyePlot(self, target = 1e-4):
        eye = self.getEye(target)
        plt.clf()
        plt.close()
        plt.figure()

        ypoints = np.array(eye)
        xpoints = np.array(list(range(-64,64)) + list(reversed(range(-64,64))))
        plt.plot(xpoints, ypoints, color='blue', label='BER: {:.2e}'.format(target))
        plt.fill_between([-10, 0, 10, 0, -10], [0, 30, 0, -30, 0], color='red', label='CDR Mask')

        plt.legend()
        plt.show()

    def extrapolate(self, bathtub):
        log = np.array([
            np.log(bathtub[-4]),
            np.log(bathtub[-3]),
            np.log(bathtub[-2]),
            np.log(bathtub[-1]),
        ])

        x = np.array([
            len(bathtub)-4,
            len(bathtub)-3,
            len(bathtub)-2,
            len(bathtub)-1,
        ])

        p = np.poly1d(np.polyfit(x, log, 1))
        
        extrapolated = []
        for i in range(64):
            extrapolated.append(1 if np.exp(p(i)) > 1 else np.exp(p(i)))

        return extrapolated


    def bathtub(self):
        y = 0
        bers = []

        for pos in range(-64, 0):
            ber = self.getBERFast(1e-9, pos, y)
            if ber != 0:
                bers.append(ber)
            else:
                break

            #print("[{}] {}".format(64-pos, ber))

        return bers

    def getEye(self, target = 1e-4):
        start = time.time()

        positiveY = []
        negativeY = []

        print(self.RX_DATA_WIDTH.get())

        ber = -1
        y = 0

        for pos in range(-64, 64):
            #print("[{}%] Eye measurement in progress          \r".format(round((float(64.0+pos)/256.0)*100)), end='')
            prev = ber

            currY = []
            while True:

                if y in currY:
                    positiveY.append(y)
                    break

                currY.append(y)
                ber = self.getBER(target, pos, y)

                if ber > target:
                    y -= 2 if pos < 0 else 10
                else:
                    y += 2 if pos > 0 else 10

                if y < 0:
                    y = 0
                    positiveY.append(0)
                    break

                if y > 127:
                    y = 127
                    positiveY.append(127)
                    break

                if prev <= target and ber > target:
                    positiveY.append(y+2)
                    break

                if prev <= target and ber < target:
                    positiveY.append(y-2)
                    break

        ber = -1
        y = 0

        for pos in range(-64, 64):
            print("[{}%] Eye measurement in progress          \r".format(round((float(128.0+64.0+pos)/256.0)*100)), end='')
            prev = ber

            currY = []
            while True:

                if y in currY:
                    negativeY.append(y)
                    break

                currY.append(y)
                ber = self.getBER(target, pos, y)

                if ber > target:
                    y += 1 if pos < 0 else 10
                else:
                    y -= 1 if pos > 0 else 10

                if y < -127:
                    y = -127
                    negativeY.append(-127)
                    break

                if y > 0:
                    y = 0
                    negativeY.append(0)
                    break

                if prev <= target and ber > target:
                    negativeY.append(y-2)
                    break

                if prev <= target and ber < target:
                    negativeY.append(y+2)
                    break

        print("Eyepos generated in {}".format(time.time()-start))

        ret = positiveY
        negativeY.reverse()
        ret.extend(negativeY)

        return ret

    def getBERsample(self, prescale, x, y):

        #self.ES_EYE_SCAN_EN.set(0x00)
        #self.ES_ERRDET_EN.set(0x00)

        ## This requires a PMA reset
        self.ES_EYE_SCAN_EN.set(0x01)
        self.ES_ERRDET_EN.set(0x01)

        self.ES_PRESCALE.set(prescale)

        self.ES_SDATA_MASK[0].set(0xffff)
        self.ES_SDATA_MASK[1].set(0x000f)
        self.ES_SDATA_MASK[2].set(0xff00)
        self.ES_SDATA_MASK[3].set(0xffff)
        self.ES_SDATA_MASK[4].set(0xffff)

        for i in range(len(self.ES_QUAL_MASK)):
            self.ES_QUAL_MASK[i].set(0xffff)
        
        self.RX_EYESCAN_VS_RANGE.set(0x00)
        self.ES_CONTROL.set(0x00)

        if y > 0:
            self.RX_EYESCAN_VS_NEG_DIR.set(0x00)
            self.RX_EYESCAN_VS_UT_SIGN.set(0x01)
            self.RX_EYESCAN_VS_CODE.set(y)
        else:
            self.RX_EYESCAN_VS_NEG_DIR.set(0x00)
            self.RX_EYESCAN_VS_UT_SIGN.set(0x00)
            self.RX_EYESCAN_VS_CODE.set(-1*y)

        if x < 0:
            tmp = 0xFFF & x
            self.ES_HORZ_OFFSET.set(tmp)
        else:
            self.ES_HORZ_OFFSET.set(x)

        #Wait for status being WAIT
        ts0 = time.perf_counter()
        status = self.ES_CONTROL_STATUS.get()
        np = 0
        while (status & 0x01) != 1:
            time.sleep(0.1)
            status = self.ES_CONTROL_STATUS.get()
            np += 1
        #print('Wait for WAIT status: {}(1) {}'.format(status,np))

        self.ES_CONTROL.set(0x01)

        #Wait for status being RESET
        status = self.ES_CONTROL_STATUS.get()
        np = 0
        ts1 = time.perf_counter()
        while (status & 0x01) != 1:
            time.sleep(0.1)
            status = self.ES_CONTROL_STATUS.get()
            np += 1
        #print("Wait for ready status: {}(5) {}".format(status,np))

        ts2 = time.perf_counter()
        errCount = self.ES_ERROR_COUNT.get()
        sampleCount = self.ES_SAMPLE_COUNT.get()
        bitCount = 20*math.pow(2, 1+prescale)*sampleCount

        #print(f'getBERsample {errCount} {bitCount} {sampleCount} {ts0} {ts1} {ts2}')
        return (errCount,bitCount)

    def getBER(self, berTarget, x, y):
        bitCntTarget = 1/berTarget

        prescale = 0
        while True:
            if prescale == 32:
                raise Exception('BER to high')

            if 20*math.pow(2, 1+prescale)*32768 < bitCntTarget*2:
                prescale += 1

            else:
                break
        
        (errCount,bitCount) = self.getBERsample(prescale, x, y)

        print('[{}/{}] Err = {}, Bit = {}'.format(x,y, errCount, bitCount))

        if bitCount == 0:
            return 1

        #Sample count to bit = 20*2^(1+prescale)*sampleCount
        ber = float(errCount)/float(bitCount)

        return ber

    def getBERFast(self, berTarget, x, y):
        bitCntTarget = 1/berTarget

        errSum = 0
        bitSum = 0
        ts1 = time.perf_counter()
        prescale = 0
        while True:
            prescale += 1

            if prescale == 32:
                raise Exception('BER to high')

            (errCount,bitCount) = self.getBERsample(prescale, x, y)

            errSum += errCount
            bitSum += bitCount
            
            if errSum > 1000 or bitSum > bitCntTarget:
                break

            errNum = 1 if errSum == 0 else math.sqrt(errSum)
            if errNum/bitSum < 0.1*berTarget:
                break

        ts2 = time.perf_counter()
        print('[{}/{}] Err = {}, Bit = {}, T = {}'.format(x,y, errSum, bitSum, ts2-ts1))

        if bitSum == 0:
            return 1

        #Sample count to bit = 20*2^(1+prescale)*sampleCount
        ber = float(errSum)/float(bitSum)

        return ber

