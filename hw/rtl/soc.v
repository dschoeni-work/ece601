// SoC for phase 1: PicoRV32 (RV32IM) + 128 KB RAM + UART + LED and 7-segment registers.
//
// Memory map
//   0x0000_0000 - 0x0001_FFFF  RAM: program, weights, stack (loaded from MEM_INIT)
//   0x1000_0000  UART data     write: send a byte   read: received byte (clears "rx ready")
//   0x1000_0004  UART status   bit 0 = tx busy, bit 1 = rx byte ready
//   0x1000_0008  LEDs          bits 15:0
//   0x1000_000C  7-segment     bits 15:0 = four hex digits, bits 19:16 = which digits are on
//
// Every access takes two cycles: the request, then mem_ready with the data.
`timescale 1ns / 1ps
`default_nettype none

module soc #(
    parameter integer MEM_WORDS    = 32768,          // 128 KB
    parameter         MEM_INIT     = "firmware.hex",
    parameter integer CLKS_PER_BIT = 868             // 100 MHz / 115200 baud
) (
    input  wire        clk,
    input  wire        resetn,
    input  wire        uart_rx,
    output wire        uart_tx,
    output reg  [15:0] leds,
    output reg  [19:0] seg_value,
    output wire        trap
);
    localparam integer AW = $clog2(MEM_WORDS);

    // ------------------------------------------------------------------ CPU
    wire        mem_valid, mem_instr;
    reg         mem_ready;
    wire [31:0] mem_addr, mem_wdata, mem_rdata;
    wire [3:0]  mem_wstrb;

    picorv32 #(
        .ENABLE_COUNTERS   (1),      // rdcycle, used to time inference
        .ENABLE_COUNTERS64 (0),
        .BARREL_SHIFTER    (1),      // 1-cycle shifts (the MLP shifts every pixel)
        .ENABLE_FAST_MUL   (1),      // hardware multiplier (DSP slices): the fair CPU baseline
        .ENABLE_DIV        (0),
        .COMPRESSED_ISA    (0),
        .ENABLE_IRQ        (0),
        .PROGADDR_RESET    (32'h0000_0000),
        .STACKADDR         (MEM_WORDS * 4)
    ) cpu (
        .clk          (clk),
        .resetn       (resetn),
        .trap         (trap),
        .mem_valid    (mem_valid),
        .mem_instr    (mem_instr),
        .mem_ready    (mem_ready),
        .mem_addr     (mem_addr),
        .mem_wdata    (mem_wdata),
        .mem_wstrb    (mem_wstrb),
        .mem_rdata    (mem_rdata),
        .mem_la_read  (),
        .mem_la_write (),
        .mem_la_addr  (),
        .mem_la_wdata (),
        .mem_la_wstrb (),
        .pcpi_valid   (),
        .pcpi_insn    (),
        .pcpi_rs1     (),
        .pcpi_rs2     (),
        .pcpi_wr      (1'b0),
        .pcpi_rd      (32'b0),
        .pcpi_wait    (1'b0),
        .pcpi_ready   (1'b0),
        .irq          (32'b0),
        .eoi          (),
        .trace_valid  (),
        .trace_data   ()
    );

    wire          request = mem_valid && !mem_ready;
    wire          ram_sel = mem_addr[31:28] == 4'h0;
    wire [AW-1:0] word    = mem_addr[AW+1:2];

    // ------------------------------------------------------------------ RAM
    // One port with byte-write enables, so Vivado maps it to block RAM.
    reg [31:0] ram [0:MEM_WORDS-1];
    reg [31:0] ram_rdata;
    initial $readmemh(MEM_INIT, ram);

    always @(posedge clk) begin
        if (request && ram_sel) begin
            ram_rdata <= ram[word];
            if (mem_wstrb[0]) ram[word][ 7: 0] <= mem_wdata[ 7: 0];
            if (mem_wstrb[1]) ram[word][15: 8] <= mem_wdata[15: 8];
            if (mem_wstrb[2]) ram[word][23:16] <= mem_wdata[23:16];
            if (mem_wstrb[3]) ram[word][31:24] <= mem_wdata[31:24];
        end
    end

    // ------------------------------------------------------------- UART
    reg        tx_start;
    reg  [7:0] tx_data;
    wire       tx_busy;
    wire [7:0] rx_data;
    wire       rx_strobe;
    reg  [7:0] rx_buf;
    reg        rx_ready;

    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .resetn(resetn), .start(tx_start), .data(tx_data), .busy(tx_busy), .tx(uart_tx));
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .resetn(resetn), .rx(uart_rx), .data(rx_data), .strobe(rx_strobe));

    // -------------------------------------------------- I/O registers + ready
    reg        ram_sel_q;
    reg [31:0] io_rdata;
    assign mem_rdata = ram_sel_q ? ram_rdata : io_rdata;

    always @(posedge clk) begin
        mem_ready <= 1'b0;
        tx_start  <= 1'b0;
        if (!resetn) begin
            rx_ready  <= 1'b0;
            leds      <= 16'b0;
            seg_value <= 20'b0;
        end else begin
            if (request) begin
                mem_ready <= 1'b1;
                ram_sel_q <= ram_sel;
                io_rdata  <= 32'b0;
                if (!ram_sel) begin
                    case (mem_addr[3:2])
                        2'd0: if (|mem_wstrb) begin
                                  tx_data  <= mem_wdata[7:0];
                                  tx_start <= 1'b1;
                              end else begin
                                  io_rdata <= {24'b0, rx_buf};
                                  rx_ready <= 1'b0;
                              end
                        2'd1: io_rdata <= {30'b0, rx_ready, tx_busy | tx_start};
                        2'd2: if (|mem_wstrb) leds      <= mem_wdata[15:0];
                              else            io_rdata  <= {16'b0, leds};
                        2'd3: if (|mem_wstrb) seg_value <= mem_wdata[19:0];
                              else            io_rdata  <= {12'b0, seg_value};
                    endcase
                end
            end
            // Last, so a byte arriving in the same cycle as a read is not lost.
            if (rx_strobe) begin
                rx_buf   <= rx_data;
                rx_ready <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
