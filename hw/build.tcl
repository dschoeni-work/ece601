# Build the bitstream for the RISC-V SoC (Vivado non-project mode).
#
# In the Vivado Tcl Shell, or the Tcl Console with no project open
# (use forward slashes in Windows paths):
#     cd C:/path/to/Ec601_mouseRISCV/hw
#     source build.tcl                      ;# Basys 3 (default)
#
# For the Nexys A7-100T, set the board first:
#     set board nexys_a7
#     source build.tcl
#
# Needs hw/firmware.mem, the firmware compiled on the Mac:
#     make -C firmware && cp firmware/build/firmware.hex hw/firmware.mem
#
# Output in hw/build/: <board>_soc.bit, <board>_timing.txt, <board>_utilization.txt

set hw   [file normalize [file dirname [info script]]]
set out  [file join $hw build]
set mem  [file join $hw firmware.mem]

if {![info exists board]} { set board basys3 }
switch -- $board {
    basys3   { set part xc7a35tcpg236-1;  set top basys3_top;   set xdc basys3_soc.xdc }
    nexys_a7 { set part xc7a100tcsg324-1; set top nexys_a7_top; set xdc nexys_a7_soc.xdc }
    default  { error "Unknown board '$board'. Use: set board basys3   or   set board nexys_a7" }
}
puts "Building for $board ($part)"

if {![file exists $mem]} {
    error "hw/firmware.mem not found. On the Mac run:\n  make -C firmware && cp firmware/build/firmware.hex hw/firmware.mem\nthen commit it and pull it here."
}
file mkdir $out

# If synthesis cannot open firmware.mem it only warns and fills the RAM with
# zeros, which gives a CPU that runs nothing. Make that an error instead.
set_msg_config -id {Synth 8-4445} -new_severity ERROR

read_verilog [list \
    [file join $hw rtl picorv32.v] \
    [file join $hw rtl uart.v] \
    [file join $hw rtl soc.v] \
    [file join $hw rtl $top.v]]
read_xdc [file join $hw constraints $xdc]
read_mem $mem    ;# lets $readmemh("firmware.mem") in the top level find the file
cd $hw           ;# and so does a relative path, as a fallback

synth_design -top $top -part $part
opt_design
place_design
route_design

report_utilization    -file [file join $out ${board}_utilization.txt]
report_timing_summary -file [file join $out ${board}_timing.txt]
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
write_bitstream -force [file join $out ${board}_soc.bit]

puts ""
puts "Bitstream: [file join $out ${board}_soc.bit]"
if {$wns >= 0} {
    puts "Timing met at 100 MHz (worst slack $wns ns)."
} else {
    puts "WARNING: timing NOT met at 100 MHz (worst slack $wns ns). See [file join $out ${board}_timing.txt]"
}
puts "Next: plug in the board and run   source program.tcl"
