//////////////////////////////////////////////////////////////////////////////
// This file is part of 'LCLS1 Timing Core'.
// It is subject to the license terms in the LICENSE.txt file found in the 
// top-level directory of this distribution and at: 
//    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
// No part of 'LCLS1 Timing Core', including this file, 
// may be copied, modified, propagated, or distributed except according to 
// the terms contained in the LICENSE.txt file.
//////////////////////////////////////////////////////////////////////////////
`timescale 1ns / 1ps
module EvrV1TimeStampGenerator(Clock, Reset, TimeStamp);
    input 		Clock;
    input 		Reset;
    output 	[63:0] 	TimeStamp;


	reg [31:0] Seconds;
	reg [31:0] Count;
        reg [63:0] TimeStamp;
	
	always @ (posedge Clock)
	begin
		if (Reset) Count <= 32'b0;
		else if (Count == 32'b0) Count <= 32'd330555;
		else Count <= Count-1;
	end
	
	always @ (posedge Clock)
	begin
		if (Reset) Seconds <= 32'd0;		
		else if (Count == 32'b0) Seconds <= Seconds+1;
		else Seconds <= Seconds;
	end
	
	always @ (posedge Clock)
	begin
		if (Reset) TimeStamp <= 64'd0;
		else if (Count == 32'b0) TimeStamp <= {Seconds, 32'b0};
		else  TimeStamp <= (TimeStamp + 1);
	end

endmodule
