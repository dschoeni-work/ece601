// UART, 8 data bits, no parity, 1 stop bit (8N1), LSB first.
// CLKS_PER_BIT = clock frequency / baud rate (100 MHz / 115200 = 868).
`timescale 1ns / 1ps
`default_nettype none

module uart_tx #(
    parameter integer CLKS_PER_BIT = 868
) (
    input  wire       clk,
    input  wire       resetn,
    input  wire       start,      // one-cycle pulse: send `data` (ignored while busy)
    input  wire [7:0] data,
    output wire       busy,
    output reg        tx
);
    localparam integer      CW       = $clog2(CLKS_PER_BIT + 1);
    localparam [CW-1:0]     BIT_LAST = CLKS_PER_BIT - 1;

    reg [8:0]    shift;  // bits still to send after the current one: data (LSB first), stop
    reg [3:0]    left;
    reg [CW-1:0] cnt;
    reg          busy_r;
    assign busy = busy_r;

    always @(posedge clk) begin
        if (!resetn) begin
            tx     <= 1'b1;
            busy_r <= 1'b0;
        end else if (!busy_r) begin
            if (start) begin
                tx     <= 1'b0;            // start bit
                shift  <= {1'b1, data};
                left   <= 4'd9;
                cnt    <= BIT_LAST;
                busy_r <= 1'b1;
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else if (left != 0) begin
            tx    <= shift[0];
            shift <= {1'b1, shift[8:1]};
            left  <= left - 1'b1;
            cnt   <= BIT_LAST;
        end else begin
            busy_r <= 1'b0;                // stop bit has been held for a full bit time
        end
    end
endmodule

module uart_rx #(
    parameter integer CLKS_PER_BIT = 868
) (
    input  wire       clk,
    input  wire       resetn,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        strobe      // one-cycle pulse when `data` holds a new byte
);
    localparam integer      CW       = $clog2(CLKS_PER_BIT + 1);
    localparam [CW-1:0]     BIT_LAST = CLKS_PER_BIT - 1;
    localparam [CW-1:0]     BIT_HALF = CLKS_PER_BIT / 2;

    reg          rx1, rx2;        // synchronizer: rx comes from outside the clock domain
    reg          active;
    reg [3:0]    idx;             // 0 = start bit, 1..8 = data bits, 9 = stop bit
    reg [CW-1:0] cnt;
    reg [7:0]    shift;

    always @(posedge clk) begin
        rx1    <= rx;
        rx2    <= rx1;
        strobe <= 1'b0;
        if (!resetn) begin
            rx1    <= 1'b1;
            rx2    <= 1'b1;
            active <= 1'b0;
        end else if (!active) begin
            if (!rx2) begin                // falling edge: a start bit begins
                active <= 1'b1;
                idx    <= 4'd0;
                cnt    <= BIT_HALF;         // sample in the middle of each bit
            end
        end else if (cnt != 0) begin
            cnt <= cnt - 1'b1;
        end else begin
            cnt <= BIT_LAST;
            if (idx == 0) begin
                if (rx2) active <= 1'b0;   // glitch, not a real start bit
                else     idx    <= 4'd1;
            end else if (idx <= 8) begin
                shift <= {rx2, shift[7:1]};
                idx   <= idx + 1'b1;
            end else begin
                active <= 1'b0;
                if (rx2) begin             // valid stop bit
                    data   <= shift;
                    strobe <= 1'b1;
                end
            end
        end
    end
endmodule

`default_nettype wire
