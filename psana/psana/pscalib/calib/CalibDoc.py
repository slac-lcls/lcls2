
"""
Usage ::

    # Import
    from psana.pscalib.calib.CalibDoc import CalibDoc
    cdoc = CalibDoc(doc)
"""

import psana.pscalib.calib.MDBUtils as mu

class CalibDoc():
    """see LCLS1 module PSCalib/src/CalibFileFinder.py
    """
    rnum_max = 9999

    def __init__(self, doc):
        self.doc = doc

        begin = doc['run']
        end   = doc['run_end']
        self.tsec_id, self.tstamp_id = mu.sec_and_ts_from_id(doc['_id'])
        self.valid = False

        self.begin = int(begin)
        if self.begin>self.rnum_max:
            self.set_invalid('WARNING! INVALID run "%s" - begin value is too big' % str(begin))
            return
        else:
            return

        self.end = None
        if str(end).isdigit():
            self.end = int(end)
            if self.end>self.rnum_max:
                self.set_invalid('WARNING! INVALID run "%d" - end value is too big' % end)
                return
        elif end == 'end':
            self.end = self.rnum_max
        else:
            self.set_invalid('WARNING! INVALID run end value "%s" - is not recognized' % str(end))
            return

        self.valid = True


    def set_invalid(self, msg):
        logger.warning(msg)
        self.valid = False

    def info_calibdoc(self, fmt='begin:%4d  end:%4d  tsec_id:%d  tstamp_id:%s'):
        return fmt % (self.begin, self.end, self.tsec_id, self.tstamp_id)

    def cmp_tsec_id(self, other):
        if   self.tsec_id < other.tsec_id: return -1
        elif self.tsec_id > other.tsec_id: return  1
        else: return 0

    def __cmp__(self, other):
        if   self.begin < other.begin: return -1
        elif self.begin > other.begin: return  1
        else:
            if   self.end > other.end: return -1
            elif self.end < other.end: return  1
            else: return self.cmp_tsec_id(other)
            #else: return 0

    def __eq__(self, other):
        return self.__cmp__(other) == 0

    def __ne__(self, other):
        return self.__cmp__(other) != 0

    def __lt__(self, other):
        return self.__cmp__(other) < 0

    def __le__(self, other):
        return self.__cmp__(other) <= 0

    def __gt__(self, other):
        return self.__cmp__(other) > 0

    def __ge__(self, other):
        return self.__cmp__(other) >= 0

# EOF
