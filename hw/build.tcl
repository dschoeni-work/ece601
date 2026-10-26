# Build the Basys 3 bitstream for the RISC-V SoC (Vivado non-project mode).
#
# In the Vivado Tcl Shell, or the Tcl Console at the bottom of the Vivado window
# (use forward slashes in Windows paths):
#     cd C:/path/to/Ec601_mouseRISCV/hw
#     source build.tcl
#
# Needs hw/firmware.mem, the firmware compiled on the Mac:
#     make -C firmware && cp firmware/build/firmware.hex hw/firmware.mem
#
# Output in hw/build/: basys3_soc.bit, timing.txt, utilization.txt

set hw   [file normalize [file dirname [info script]]]
set out  [file join $hw build]
set part xc7a35tcpg236-1
set mem  [file join $hw firmware.mem]

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
    [file join $hw rtl basys3_top.v]]
read_xdc [file join $hw constraints basys3_soc.xdc]
read_mem $mem    ;# lets $readmemh("firmware.mem") in basys3_top.v find the file
cd $hw           ;# and so does a relative path, as a fallback

synth_design -top basys3_top -part $part
opt_design
place_design
route_design

report_utilization    -file [file join $out utilization.txt]
report_timing_summary -file [file join $out timing.txt]
set wns [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
write_bitstream -force [file join $out basys3_soc.bit]

puts ""
puts "Bitstream: [file join $out basys3_soc.bit]"
if {$wns >= 0} {
    puts "Timing met at 100 MHz (worst slack $wns ns)."
} else {
    puts "WARNING: timing NOT met at 100 MHz (worst slack $wns ns). See [file join $out timing.txt]"
}
puts "Next: plug in the board and run   source program.tcl"
