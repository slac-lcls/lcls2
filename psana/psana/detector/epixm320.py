import os
import numpy as np
from amitypes import Array2d, Array3d
import psana.detector.epix_base as eb
import logging
from psana.detector.detector_impl import DetectorImpl
logger = logging.getLogger(__name__)

#############################################
# Descramble class
#############################################
class DataDebug():

    def __init__(self, name):

        self.name = name
        self.framePixelRow = 192
        self.framePixelColumn = 384
        pixelsPerLanesRows = 48
        pixelsPerLanesColumns = 64
        numOfBanks = 24
        bankHeight = pixelsPerLanesRows
        bankWidth = pixelsPerLanesColumns

        imageSize = self.framePixelColumn * self.framePixelRow

        self.lookupTableCol = np.zeros(imageSize, dtype=int)
        self.lookupTableRow = np.zeros(imageSize, dtype=int)

        # based on descrambling pattern described here figure out the location of the pixel based on its index in raw data
        # https://confluence.slac.stanford.edu/download/attachments/392826236/image-2023-8-9_16-6-42.png?version=1&modificationDate=1691622403000&api=v2
        descarambledImg = np.zeros((numOfBanks, bankHeight,bankWidth), dtype=int)
        for row in range(bankHeight) :
            for col in range (bankWidth) :
                for bank in range (numOfBanks) :
                    #                                  (even cols w/ offset       +  row offset       + increment every two cols)   * fill one pixel / bank + bank increment
                    descarambledImg[bank, row, col] = (((col+1) % 2) * 1536       +   32 * row        + int(col / 2))               * numOfBanks            + bank


        # reorder banks from
        # 18    19    20    21    22    23
        # 12    13    14    15    16    17
        #  6     7     8     9    10    11
        #  0     1     2     3     4     5
        #
        #                To
        #  3     7    11    15    19    23         <= Quadrant[3] 48 x 64 x 6
        #  2     6    10    14    18    22         <= Quadrant[2] 48 x 64 x 6
        #  1     5     9    13    17    21         <= Quadrant[1] 48 x 64 x 6
        #  0     4     8    12    16    20         <= Quadrant[0] 48 x 64 x 6

        quadrant = [bytearray(),bytearray(),bytearray(),bytearray()]
        for i in range(4):
            quadrant[i] = np.concatenate((descarambledImg[0+i],
                                        descarambledImg[4+i],
                                        descarambledImg[8+i],
                                        descarambledImg[12+i],
                                        descarambledImg[16+i],
                                        descarambledImg[20+i]),1)

        descarambledImg = np.concatenate((quadrant[0], quadrant[1]),0)
        descarambledImg = np.concatenate((descarambledImg, quadrant[2]),0)
        descarambledImg = np.concatenate((descarambledImg, quadrant[3]),0)

        # Work around ASIC/firmware bug: first and last row of each bank are exchanged
        # Create lookup table where each row points to the next
        hardwareBugWorkAroundRowLUT = np.zeros((self.framePixelRow))
        for index in range (self.framePixelRow) :
            hardwareBugWorkAroundRowLUT[index] = index + 1
        # handle bank/lane roll over cases
        hardwareBugWorkAroundRowLUT[47] = 0
        hardwareBugWorkAroundRowLUT[95] = 48
        hardwareBugWorkAroundRowLUT[143] = 96
        hardwareBugWorkAroundRowLUT[191] = 144

        # reverse pixel original index to new row and column to generate lookup tables
        for row in range (self.framePixelRow) :
            for col in range (self.framePixelColumn):
                index = descarambledImg[row,col]
                self.lookupTableRow[index] = hardwareBugWorkAroundRowLUT[row]
                self.lookupTableCol[index] = col

        # reshape column and row lookup table
        self.lookupTableCol = np.reshape(self.lookupTableCol, (self.framePixelRow, self.framePixelColumn))
        self.lookupTableRow = np.reshape(self.lookupTableRow, (self.framePixelRow, self.framePixelColumn))


    def descramble(self, rawData):
        current_frame_temp = np.zeros((self.framePixelRow, self.framePixelColumn), dtype=int)
        """performs the EpixMv2 image descrambling (simply applying lookup table) """
        if rawData.shape==(384,192):
            #imgDesc = np.frombuffer(rawData[24:73752],dtype='uint16').reshape(192, 384)
            imgDesc = np.transpose(rawData)
        else:
            print("{}: descramble error".format(self.name))
            print('rawData length {}'.format(len(rawData)))
            imgDesc = np.zeros((192,384), dtype='uint16')

        # apply lookup table
        current_frame_temp[self.lookupTableRow, self.lookupTableCol] = imgDesc
        # returns final image
        #return np.bitwise_and(current_frame_temp, self.PixelBitMask.get())
        return current_frame_temp


# make an empty detector interface for Matt's hardware
# configuration object so that config_dump works - cpo
class epixm320hw_config_0_0_0(DetectorImpl):
    def __init__(self, *args, **kwargs):
        super(epixm320hw_config_0_0_0, self).__init__(*args)

class epixm320_raw_0_0_0(eb.epix_base):
    def __init__(self, *args, **kwargs):
        logger.debug('epixm320_raw_0_0_0.__init__')
        eb.epix_base.__init__(self, *args, **kwargs)
        self._seg_geo = eb.sgs.Create(segname='EPIXM320:V1')
        self._data_bit_mask = eb.M14 # for epixhr2x2 data on 2023-10-30 Dionisio - HR has 14 data bits.
        self._data_gain_bit = eb.B15
        self._gain_bit_shift = 10
        self._gains_def = (41.0, 13.7, 0.512) # epixhr2x2 ADU/keV H:M:L = 1 : 1/3 : 1/80
        self._path_geo_default = 'pscalib/geometry/data/geometry-def-epixm320.data'
        self._dataDebug = None

    def _array(self, evt) -> Array2d:
        f = None
        segs = self._segments(evt)
        if segs is None:
            pass
        else:
            nx = segs[0].raw.shape[1]
            ny = segs[0].raw.shape[0]
            f = segs[0].raw & self._data_bit_mask # 0x7fff
        return f

    def raw(self, evt) -> Array3d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None

        if self._dataDebug is None:
            self._dataDebug = DataDebug('epixm320_raw_0_0_0')

        return np.stack([self._dataDebug.descramble(segs[0].raw)])

    def image(self, evt, **kwargs) -> Array2d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None

        return self.raw(evt)[0]

    def rawImg(self, evt, **kwargs) -> Array2d:
        if evt is None: return None
        segs = self._segments(evt)    # dict = {seg_index: seg_obj}
        if segs is None: return None

        return np.transpose(segs[0].raw)

