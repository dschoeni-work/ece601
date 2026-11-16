# Program the board with hw/build/<board>_soc.bit over the micro-USB cable.
#
# In the Vivado Tcl Shell or Tcl Console, after build.tcl:
#     source program.tcl                    ;# Basys 3 (default)
#
# For the Nexys A7-100T:
#     set board nexys_a7
#     source program.tcl
#
# The FPGA forgets its configuration when the board loses power (unplugging the
# cable or the power switch), so run this again after every power-up.

set hw  [file normalize [file dirname [info script]]]
if {![info exists board]} { set board basys3 }
switch -- $board {
    basys3   { set device xc7a35t*;  set name "Basys 3" }
    nexys_a7 { set device xc7a100t*; set name "Nexys A7-100T" }
    default  { error "Unknown board '$board'. Use: set board basys3   or   set board nexys_a7" }
}
set bit [file join $hw build ${board}_soc.bit]
if {![file exists $bit]} {
    error "$bit not found. Run build.tcl first."
}

open_hw_manager
connect_hw_server
open_hw_target
set dev [lindex [get_hw_devices $device] 0]
if {$dev eq ""} {
    error "No $name found. Is the cable plugged in and the power switch on?"
}
current_hw_device $dev
set_property PROGRAM.FILE $bit $dev
program_hw_devices $dev
close_hw_target
disconnect_hw_server
close_hw_manager

puts ""
puts "Programmed $dev. The RISC-V CPU is running; LED 15 should stay off."
puts "Next: python host/board_test.py"
