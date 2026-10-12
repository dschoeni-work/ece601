# Build the RISC-V firmware (build/firmware.hex) and the laptop C-vs-Python test.
#
#   make           firmware for the PicoRV32 SoC
#   make test      build and run test_mlp on this computer

# Homebrew names the cross compiler riscv64-elf-gcc; Ubuntu riscv64-unknown-elf-gcc.
CROSS   ?= $(if $(shell command -v riscv64-elf-gcc 2>/dev/null),riscv64-elf-,riscv64-unknown-elf-)
CC       = $(CROSS)gcc
OBJCOPY  = $(CROSS)objcopy
SIZE     = $(CROSS)size
PYTHON  ?= python3
MEM_WORDS = 32768

CFLAGS = -march=rv32im_zicsr -mabi=ilp32 -O2 -ffreestanding -fno-builtin -nostdlib \
         -fno-asynchronous-unwind-tables -Wall -Wextra

SRC = start.S main.c mlp.c libc.c

all: build/firmware.hex

build/firmware.elf: $(SRC) mlp.h $(wildcard mlp_weights.h) link.ld
	@test -f mlp_weights.h || (echo "mlp_weights.h missing: run model/train_quantize.py first"; false)
	@mkdir -p build
	$(CC) $(CFLAGS) -T link.ld -Wl,-Map,build/firmware.map -Wl,--no-warn-rwx-segments -o $@ $(SRC)
	$(SIZE) $@

build/firmware.bin: build/firmware.elf
	$(OBJCOPY) -O binary $< $@

build/firmware.hex: build/firmware.bin makehex.py
	$(PYTHON) makehex.py $< $(MEM_WORDS) > $@

test: test_mlp
	./test_mlp

test_mlp: test_mlp.c mlp.c mlp.h $(wildcard mlp_weights.h)
	gcc -O2 -o $@ test_mlp.c mlp.c

clean:
	rm -rf build test_mlp

.PHONY: all test clean
