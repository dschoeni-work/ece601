#!/usr/bin/env python3
"""firmware.bin -> firmware.hex: one 32-bit little-endian word per line, for
$readmemh in hw/rtl/soc.v (simulation and Vivado both read this file).

Usage: python3 makehex.py firmware.bin MEM_WORDS > firmware.hex
"""
import sys

data = open(sys.argv[1], "rb").read()
mem_words = int(sys.argv[2])
data += b"\0" * (-len(data) % 4)
if len(data) // 4 > mem_words:
    sys.exit(f"firmware is {len(data)} bytes, RAM holds only {mem_words * 4}")
for i in range(0, len(data), 4):
    print(f"{int.from_bytes(data[i:i + 4], 'little'):08x}")
