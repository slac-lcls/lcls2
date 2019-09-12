import numpy as np

class eventBuilderParser():
    def __init__(self):        
        return

    def frames_to_position(self,frame_bytearray,position):
        return int('0b'+'{0:08b}'.format(frame_bytearray[position+1])+'{0:08b}'.format(frame_bytearray[position]),2)
    
    def _checkForSubframes(self):
        self.sub_is_fullframe = [False]*len(self.frame_list)
        
        for i in range(len(self.frame_list)):
            
            if self.main_header[:2] == self.frame_list[i][:self.HEADER_WIDTH ][:2]:
                self.sub_is_fullframe[i] = True
        
        return
    
    def _resolveSubFrames(self):
        self._checkForSubframes()
        self.sub_frames = [False]*len(self.frame_list)
        for i in range(len(self.frame_list)):
            if self.sub_is_fullframe[i]:
                self.sub_frames[i] =  eventBuilderParser()
                #print("length = ",len(self.frame_list[i]))
                self.sub_frames[i].parseArray(self.frame_list[i])
                self.frame_list[i] = False
        return
    
    def print_info(self):
        for i in self.__dict__:
            if i != "frame_list" and type(self.__dict__[i]) is not bytearray:
                print(i," = ",self.__dict__[i])
                
        for i in range(len(self.sub_is_fullframe)):
            if(self.sub_is_fullframe[i]):
                print("\nsubframe = ",i)
                self.sub_frames[i].print_info()
                
        return
        
    def parseArray(self,frame_bytearray:bytearray):
        self.frame_bytes     = len(frame_bytearray)
        self.main_header     = frame_bytearray[0:16]

        self.version                = self.main_header[0] & int('00001111', 2)
        self.axi_stream_bit_width   = 8*2**((self.main_header[0] >> 4) + 1)
        self.sequence_count         = self.main_header[1]
        self.HEADER_WIDTH           = int(self.axi_stream_bit_width/8)
        
        
        self.frame_sizes_reversed         = [self.frames_to_position(frame_bytearray,-16)]
        self.frame_positions_reversed     = [[self.frame_bytes-16-self.frame_sizes_reversed[0],self.frame_bytes-16]] #[start, and]
        self.frame_list                   = [frame_bytearray[self.frame_positions_reversed[-1][0]:self.frame_positions_reversed[-1][1]]]
        
        self.tdest                        = [frame_bytearray[-12]]
        
        
        parsed_frame_size = sum(self.frame_sizes_reversed) +(len(self.frame_sizes_reversed)+1)*self.HEADER_WIDTH
        #print("parsing")
        while(len(frame_bytearray)>parsed_frame_size):
            #print(len(frame_bytearray))
            self.frame_sizes_reversed.append(self.frames_to_position(frame_bytearray,self.frame_positions_reversed[-1][0]-16))
            
            self.frame_positions_reversed.append([self.frame_positions_reversed[-1][0]-16-self.frame_sizes_reversed[-1],self.frame_positions_reversed[-1][0]-16]) #[start, and]            
            
            self.frame_list.append(frame_bytearray[self.frame_positions_reversed[-1][0]:self.frame_positions_reversed[-1][1]])
            self.tdest.append(frame_bytearray[self.frame_positions_reversed[-1][1]+4])
            
            
          
            parsed_frame_size = sum(self.frame_sizes_reversed) +(len(self.frame_sizes_reversed)+1)*self.HEADER_WIDTH
        
        #self.sub_is_fullframe = [False]*len(self.frame_list)
        self._resolveSubFrames()
        
        return
    
class timeToolParser(eventBuilderParser):
    def parseData(self,frame_bytearray:bytearray):
        self.parseArray(frame_bytearray)
        
        
        try:
            timing_bus_idx           = [i[0] for i in enumerate(self.tdest) if i[1]==0][0] #timing bus always has tdest of 0
            self.timing_bus          = self.frame_list[timing_bus_idx]  #frame_bytearray[16:32]
        except IndexError:
            self.timing_bus = None
        
        sub_frame_idx            = [i[0] for i in enumerate(self.tdest) if i[1]==1][0]
        edge_pos_idx             = [i[0] for i in enumerate(self.sub_frames[sub_frame_idx].tdest) if i[1]==0][0]
        self.edge_position       = self.sub_frames[sub_frame_idx].frame_list[edge_pos_idx][0] + self.sub_frames[sub_frame_idx].frame_list[edge_pos_idx][1]*256
        
        try:
            bkg_frame_idx            = [i[0] for i in enumerate(self.sub_frames[sub_frame_idx].tdest) if i[1]==1][0]
            self.background_frame    = self.sub_frames[sub_frame_idx].frame_list[bkg_frame_idx]
        except IndexError:    
            self.background_frame    = None

        try:
            prescaled_frame_idx      = [i[0] for i in enumerate(self.tdest) if i[1]==2][0]
            self.prescaled_frame     = self.frame_list[prescaled_frame_idx] 
        except IndexError:
            self.prescaled_frame     = None
            
        
        
        
    
        return
