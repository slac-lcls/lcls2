//////////////////////////////////////////////////////////////////////////////
// This file is part of 'LCLS1 Timing Core'.
// It is subject to the license terms in the LICENSE.txt file found in the 
// top-level directory of this distribution and at: 
//    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
// No part of 'LCLS1 Timing Core', including this file, 
// may be copied, modified, propagated, or distributed except according to 
// the terms contained in the LICENSE.txt file.
//////////////////////////////////////////////////////////////////////////////
`timescale 1ns/100ps
module EvrV1TimeofDayReceiver(Clock, Reset, EventStream, TimeStamp, timeDebug, secondsShift);
   input Clock;
   input Reset;
   input   [7:0] EventStream;
   output [63:0] TimeStamp;
   output [36:0] timeDebug;
   output [31:0] secondsShift;

   reg [31:0] Seconds;
   reg [4:0]  Position;
   reg [63:0] TimeStamp;
   reg [63:0] TimeStampDly1;
   reg [63:0] TimeStampDly0;
   
   // the time of day is updated by a serial stream of 0 events (0x70) and 1 events
   // (0x71). This code implements a pointer into the time of day register and writes
   // the data into that position. On receibt of the latch event (0x7d) the data is
   // moved to the output register and the pointer is cleared. The offset is cleared
   // on event 0x7d then incremented on the input clock edge.
   always @ (posedge Clock)
   begin
      if (Reset || (EventStream == 8'h7d))  Position <= 5'd0;
      else if ((EventStream == 8'h70) || (EventStream == 8'h71)) Position <= (Position + 1);
      else Position <= Position;
   end
   
   always @ (posedge Clock)
   begin
      if (Reset) Seconds <= 32'd0;      
      else if (EventStream == 8'h70) Seconds[31-Position] <= 1'b0;
      else if (EventStream == 8'h71) Seconds[31-Position] <= 1'b1;
      else Seconds <= Seconds;
   end
   
   always @ (posedge Clock)
   begin
      if (Reset) 
         begin
            TimeStampDly0 <= 64'd0;
            TimeStampDly1 <= 64'd0;
            TimeStamp <= 64'd0;
         end
      else 
         begin
            TimeStampDly1 <= TimeStampDly0;
            TimeStamp     <= TimeStampDly1;
            if (EventStream == 8'h7d) TimeStampDly0 <= {Seconds, 32'b0};
            else  TimeStampDly0 <= (TimeStampDly0 + 1);
         end
   end
   
   assign timeDebug = {Position, Seconds};
   assign secondsShift = Seconds;

endmodule
