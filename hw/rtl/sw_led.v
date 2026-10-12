// Basys 3 bring-up test: each LED follows its switch.
// Passing this test confirms Vivado -> bitstream -> board programming works.
module sw_led (
    input  wire [15:0] sw,
    output wire [15:0] led
);
    assign led = sw;
endmodule
