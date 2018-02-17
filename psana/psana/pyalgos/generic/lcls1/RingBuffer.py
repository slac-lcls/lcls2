#------------------------------
"""
RingBuffer - access to cycled list of records
Created: 2017-05-19
Author : Mikhail Dubrovin

Usage ::
    from expmon.RingBuffer import RingBuffer

    o = RingBuffer(size=1000)
    o.print_buffer()
    rec=(a,b,c)
    o.save_record(rec)
    rec  = o.record_next()
    rec  = o.record_last()
    recs = o.buffer()
    recs = o.records_last(nrecs=100) # nrecs <= size
    recs = o.records_new()

    # example of access from other modules
    from expmon.EMConfigParameters import cp
    cp.ringbuffer1 = o
"""

#------------------------------

#import sys
#from time import time

#------------------------------

class RingBuffer() :
    """Supports round-robbin buffer for data records.
    """
    def __init__(self, size=1000) :
        """ Reserves cycled buffer of specified size.
        """
        self._name = self.__class__.__name__
        self.bsize  = size
        self.bsize1 = size - 1
        self.iw = -1
        self.ir = -1
        self.iw_incremented = False # for records_new()
        self.buf = [None] * size


    def print_buffer(self) :
        """ Prints entire buffer content.
        """ 
        for i,r in enumerate(self.buf) :
            print str(r),
            if i>0 and not(i%10) : print ''


    def buffer(self) :
        """ Returns buffer
        """
        return self.buf


    def save_record(self, rec) :
        """ Saves data record in the cycled buffer.
        """
        self.iw += 1
        self.iw_incremented = True
        if self.iw > self.bsize1 : self.iw = 0
        self.buf[self.iw] = rec


    def record_next(self) :
        """ Returns the the next data record from the cycled buffer.
        """
        if self.iw == -1 : return []
        self.ir += 1
        if self.ir > self.bsize1 : self.ir = 0
        return self.buf[self.ir]


    def record_last(self) : 
        """ Returns the last written record.
        """
        if self.iw == -1 : return None
        return self.buf[self.iw]


    def records_last(self, nrecs=10) :
        """ Returns list of last nrecs data records. 
            If nrecs > buffer nrr, full buffer is retirned
        """
        if self.iw == -1 : return []
        nrr = nrecs if nrecs <= self.bsize else self.bsize 
        if self.ir == -1 : self.ir = 0
        ir0 = self.iw - nrr + 1
        bufret = self.buf[ir0:self.iw+1] if ir0>=0 else\
                 self.buf[ir0+self.bsize:] + self.buf[:self.iw+1]
        self.ir = self.iw
        return bufret
        #return [self.get_record_next() for n in range(nrecs)]


    def records_new(self) :
        """ Returns list of records since last call. 
        """
        if self.iw == -1 : return []
        if not self.iw_incremented : return []
        self.iw_incremented = False
        if self.ir == self.iw : return self.records_last(nrecs=self.bsize)
        if self.ir == -1 : self.ir = 0
        bufret = self.buf[self.ir+1:self.iw+1] if self.ir < self.iw else\
                 self.buf[self.ir+1:] + self.buf[:self.iw+1]
        self.ir = self.iw
        return bufret

#------------------------------

if __name__ == "__main__" :

    #from expmon.EMConfigParameters import cp
    o = RingBuffer(size=10)
    for i in range(16) : o.save_record(i)
    o.print_buffer()

    print '\no.records_new():', o.records_new()
    print 'o.record_last():', o.record_last()

    print 'o.records_last(3):', o.records_last(nrecs=3)
    print 'o.records_last(8):', o.records_last(nrecs=8)
    print 'o.records_last(12) at size=10: ', o.records_last(nrecs=12)

#------------------------------
