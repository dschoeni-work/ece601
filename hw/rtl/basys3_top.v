// Basys 3 top level: 100 MHz clock, UART over the PROG/UART micro-USB port,
// result on the rightmost 7-segment digit and the LEDs.
//   btnC (center button)  reset
//   LED 15                CPU trap (should stay off)
`timescale 1ns / 1ps
`default_nettype none

module basys3_top (
    input  wire        clk,     // 100 MHz
    input  wire        btnC,
    input  wire        RsRx,    // from the PC
    output wire        RsTx,    // to the PC
    output wire [15:0] led,
    output reg  [6:0]  seg,     // active low, seg[0] = CA ... seg[6] = CG
    output wire        dp,
    output reg  [3:0]  an       // active low, an[0] = rightmost digit
);
    // Reset: held for 16 cycles after configuration, and while btnC is pressed.
    reg [3:0] por = 4'd0;
    reg [1:0] btn_sync = 2'b00;
    always @(posedge clk) begin
        btn_sync <= {btn_sync[0], btnC};
        if (por != 4'hf) por <= por + 1'b1;
    end
    wire resetn = (por == 4'hf) && !btn_sync[1];

    wire [15:0] leds;
    wire [19:0] seg_value;
    wire        trap;

    soc #(
        .MEM_INIT     ("firmware.hex"),
        .CLKS_PER_BIT (868)              // 115200 baud
    ) u_soc (
        .clk       (clk),
        .resetn    (resetn),
        .uart_rx   (RsRx),
        .uart_tx   (RsTx),
        .leds      (leds),
        .seg_value (seg_value),
        .trap      (trap)
    );

    assign led = {trap, leds[14:0]};
    assign dp  = 1'b1;

    // 7-segment: show one digit at a time, switching every 2^16 cycles (~1.5 kHz).
    reg  [17:0] refresh = 18'd0;
    always @(posedge clk) refresh <= refresh + 1'b1;
    wire [1:0] sel    = refresh[17:16];
    wire [3:0] nibble = seg_value[sel * 4 +: 4];

    always @(*) begin
        an      = 4'b1111;
        an[sel] = ~seg_value[16 + sel];
        case (nibble)
            4'h0: seg = 7'b1000000;
            4'h1: seg = 7'b1111001;
            4'h2: seg = 7'b0100100;
            4'h3: seg = 7'b0110000;
            4'h4: seg = 7'b0011001;
            4'h5: seg = 7'b0010010;
            4'h6: seg = 7'b0000010;
            4'h7: seg = 7'b1111000;
            4'h8: seg = 7'b0000000;
            4'h9: seg = 7'b0010000;
            4'ha: seg = 7'b0001000;
            4'hb: seg = 7'b0000011;
            4'hc: seg = 7'b1000110;
            4'hd: seg = 7'b0100001;
            4'he: seg = 7'b0000110;
            default: seg = 7'b0001110;
        endcase
    end
endmodule

`default_nettype wire
