import numpy as np
from psana.detector.detector_impl import DetectorImpl
from amitypes import Array1d

class tt_raw_2_0_0(DetectorImpl):
    def __init__(self, *args):
        super().__init__(*args)
        self._add_fields()

    def _info(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        return segments[0]

# leftover from when we wrote out the EventBatcher format.
# still used at the moment in the self-tests
class ttdet_ttalg_0_0_1(DetectorImpl):
    def __init__(self, *args):
        super(ttdet_ttalg_0_0_1, self).__init__(*args)

    def _image(self,evt):
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all timetool data comes from one segment
        return segments[0].data

    def _header(self,evt):                           #take out.  scientists don't care about this. 
        # check for missing data
        segments = self._segments(evt)
        if segments is None: return None
        # seems reasonable to assume that all timetool data comes from one segment
        return segments[0].data[:16]
    
    def parsed_frame(self,evt):
        pFrame = timeToolParser()
        p = self._segments(evt)[0].data.tobytes()
        pFrame._parseData(p)

        return pFrame

    def image(self, evt):
        parsed_frame_object = self.parsed_frame(evt)
        return parsed_frame_object.prescaled_frame

    #def edge_value(self,evt):              #speak with damiani for how to send back this data to ami
    #def edge_uncertainty(self,evt):        #needs _(underscore) to hide things AMI doesn't need to see.
    #

class eventBuilderParser():
    def __init__(self):        
        return

    def _frames_to_position(self,frame_bytearray,position):
        return int('0b'+'{0:08b}'.format(frame_bytearray[position+1])+'{0:08b}'.format(frame_bytearray[position]),2)
    
    def _checkForSubframes(self):
        self._sub_is_fullframe = [False]*len(self._frame_list)
        
        for i in range(len(self._frame_list)):
            
            if self._main_header[:2] == self._frame_list[i][:self._HEADER_WIDTH ][:2]:
                self._sub_is_fullframe[i] = True
        
        return
    
    def _resolveSubFrames(self):
        self._checkForSubframes()
        self._sub_frames = [False]*len(self._frame_list)
        for i in range(len(self._frame_list)):
            if self._sub_is_fullframe[i]:
                self._sub_frames[i] =  eventBuilderParser()
                #print("length = ",len(self._frame_list[i]))
                self._sub_frames[i]._parseArray(self._frame_list[i])
                self._frame_list[i] = False
        return
    
    def print_info(self):
        for i in self.__dict__:
            if i != "_frame_list" and type(self.__dict__[i]) is not bytearray:
                print(i," = ",self.__dict__[i])
                
        for i in range(len(self._sub_is_fullframe)):
            if(self._sub_is_fullframe[i]):
                print("\nsubframe = ",i)
                self._sub_frames[i].print_info()
                
        return
        
    def _parseArray(self,frame_bytearray:bytearray):
        self._frame_bytes     = len(frame_bytearray)
        self._main_header     = frame_bytearray[0:16]

        self._version                = self._main_header[0] & int('00001111', 2)
        self._axi_stream_bit_width   = 8*2**((self._main_header[0] >> 4) + 1)
        self.sequence_count         = self._main_header[1]
        self._HEADER_WIDTH           = int(self._axi_stream_bit_width/8)
        
        
        self._frame_sizes_reversed         = [self._frames_to_position(frame_bytearray,-16)]
        self._frame_positions_reversed     = [[self._frame_bytes-16-self._frame_sizes_reversed[0],self._frame_bytes-16]] #[start, and]
        self._frame_list                   = [frame_bytearray[self._frame_positions_reversed[-1][0]:self._frame_positions_reversed[-1][1]]]
        
        self._tdest                        = [frame_bytearray[-12]]
        
        
        parsed_frame_size = sum(self._frame_sizes_reversed) +(len(self._frame_sizes_reversed)+1)*self._HEADER_WIDTH
        #print("parsing")
        while len(frame_bytearray)>=(parsed_frame_size+self._HEADER_WIDTH):
            #print(len(frame_bytearray))
            self._frame_sizes_reversed.append(self._frames_to_position(frame_bytearray,self._frame_positions_reversed[-1][0]-16))
            
            self._frame_positions_reversed.append([self._frame_positions_reversed[-1][0]-16-self._frame_sizes_reversed[-1],self._frame_positions_reversed[-1][0]-16]) #[start, and]            
            
            self._frame_list.append(frame_bytearray[self._frame_positions_reversed[-1][0]:self._frame_positions_reversed[-1][1]])
            self._tdest.append(frame_bytearray[self._frame_positions_reversed[-1][1]+4])
            
            
          
            parsed_frame_size = sum(self._frame_sizes_reversed) +(len(self._frame_sizes_reversed)+1)*self._HEADER_WIDTH
        
        #self._sub_is_fullframe = [False]*len(self._frame_list)
        self._resolveSubFrames()
        
        return
    
class timeToolParser(eventBuilderParser):
    def _parseData(self,frame_bytearray:bytearray):
        self._parseArray(frame_bytearray)
        
        
        try:
            _timing_bus_idx           = [i[0] for i in enumerate(self._tdest) if i[1]==0][0] #timing bus always has _tdest of 0
            self._timing_bus          = self._frame_list[_timing_bus_idx]  #frame_bytearray[16:32]
        except IndexError:
            self._timing_bus = None
        
        sub_frame_idx            = [i[0] for i in enumerate(self._tdest) if i[1]==1][0]
        edge_pos_idx             = [i[0] for i in enumerate(self._sub_frames[sub_frame_idx]._tdest) if i[1]==0][0]
        self.edge_position       = self._sub_frames[sub_frame_idx]._frame_list[edge_pos_idx][0] + self._sub_frames[sub_frame_idx]._frame_list[edge_pos_idx][1]*256
        
        try:
            bkg_frame_idx            = [i[0] for i in enumerate(self._sub_frames[sub_frame_idx]._tdest) if i[1]==1][0]
            self.background_frame    = self._sub_frames[sub_frame_idx]._frame_list[bkg_frame_idx]
        except IndexError:    
            self.background_frame    = None

        try:
            prescaled_frame_idx      = [i[0] for i in enumerate(self._tdest) if i[1]==2][0]
            #self.prescaled_frame    = self._frame_list[prescaled_frame_idx] 
            self.prescaled_frame     = np.frombuffer(self._frame_list[prescaled_frame_idx],dtype = np.int8)
            
        except IndexError:
            self.prescaled_frame     = None
            
        
        
        
    
        return
