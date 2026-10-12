# Simulate the SoC with Verilator: run MNIST test images through the RISC-V
# firmware, check the results against Python, and count cycles per inference.
#
#   make run          20 images
#   make run N=200    more images

N   ?= 20
CPB ?= 16            # UART clocks per bit in simulation (the board uses 868 = 115200 baud)

HEX := $(abspath ../firmware/build/firmware.hex)
RTL := ../hw/rtl/soc.v ../hw/rtl/uart.v ../hw/rtl/picorv32.v

obj_dir/Vsoc: $(RTL) sim_main.cpp
	verilator --cc --exe --build -j 0 -O3 --top-module soc \
	    -Wno-fatal -Wno-lint -Wno-style -Wno-MULTIDRIVEN -Wno-TIMESCALEMOD \
	    -GMEM_INIT='"$(HEX)"' -GCLKS_PER_BIT=$(CPB) -CFLAGS -DCLKS_PER_BIT=$(CPB) -CFLAGS -std=c++17 \
	    $(RTL) sim_main.cpp

$(HEX):
	$(MAKE) -C ../firmware

run: $(HEX) obj_dir/Vsoc
	./obj_dir/Vsoc $(N) ../firmware/test_data

clean:
	rm -rf obj_dir

.PHONY: run clean
