"""Talk to the RISC-V SoC on the Basys 3 over its USB-UART (115200 baud, 8N1).

Protocol (see firmware/main.c; numbers little-endian):
  PC -> board  'P'                     ping
  board -> PC  'K'
  PC -> board  'I' + 784 bytes         28x28 image, row-major, 0 = background
  board -> PC  'R', digit, cycles (uint32), 10 logits (int32)    46 bytes

Needs pyserial:  pip install pyserial

    from board import Board
    with Board() as b:              # finds the Basys 3 by itself
        r = b.infer(img_28x28)      # r.digit, r.cycles, r.logits, r.compute_ms, r.round_trip_ms
"""
import struct
import time
from dataclasses import dataclass

import numpy as np

try:
    import serial
    from serial.tools import list_ports
except ImportError:  # reported when a Board is opened
    serial = None

BAUD = 115200
CLOCK_HZ = 100_000_000      # Basys 3 clock; the firmware counts cycles of it
REPLY_BYTES = 46
FTDI_VID = 0x0403           # the Basys 3's USB chip (FTDI FT2232)


class BoardError(RuntimeError):
    pass


@dataclass
class Result:
    digit: int
    cycles: int
    logits: np.ndarray       # int64, shape (10,)
    round_trip_ms: float     # measured on this computer: UART both ways + inference

    @property
    def compute_ms(self):
        """Inference time on the board, from its own cycle counter."""
        return self.cycles / CLOCK_HZ * 1000.0


def candidate_ports():
    """Serial ports that may be the Basys 3, most likely first."""
    if serial is None:
        return []
    ftdi = [p for p in list_ports.comports() if p.vid == FTDI_VID]
    # The FT2232 has two channels: A is JTAG (used by Vivado), B is the UART.
    # Where both show up as serial ports, B is the one whose name ends in B or 1.
    ftdi.sort(key=lambda p: not (p.device.endswith(("B", "1")) or (p.location or "").endswith(".1")))
    return [p.device for p in ftdi]


def describe_ports():
    if serial is None:
        return "pyserial is not installed (pip install pyserial)."
    ports = list(list_ports.comports())
    if not ports:
        return "No serial ports found. Is the board plugged in and switched on?"
    return "\n".join(f"  {p.device}  {p.description}" for p in ports)


class Board:
    def __init__(self, port=None, timeout=2.0):
        if serial is None:
            raise BoardError("pyserial is not installed. Run: pip install pyserial")
        tried = []
        for dev in [port] if port else candidate_ports():
            try:
                ser = serial.Serial(dev, BAUD, timeout=timeout)
            except (serial.SerialException, OSError) as e:
                tried.append(f"{dev}: {e}")
                continue
            self.ser, self.port = ser, dev
            if self.ping():
                return
            ser.close()
            tried.append(f"{dev}: no reply to ping")
        detail = "\n".join("  " + t for t in tried) or "  No Basys 3 serial port found. Serial ports:\n" + describe_ports()
        raise BoardError("Could not reach the board. Is it programmed (hw/program.tcl), "
                         "plugged in and switched on?\n" + detail)

    def ping(self, attempts=3):
        timeout, self.ser.timeout = self.ser.timeout, 0.5
        try:
            for _ in range(attempts):
                self.ser.reset_input_buffer()
                self.ser.write(b"P")
                if self.ser.read(1) == b"K":
                    return True
            return False
        finally:
            self.ser.timeout = timeout

    def infer(self, img_u8):
        img = np.asarray(img_u8, dtype=np.uint8).reshape(784)
        self.ser.reset_input_buffer()
        t0 = time.perf_counter()
        self.ser.write(b"I" + img.tobytes())
        reply = self.ser.read(REPLY_BYTES)
        round_trip_ms = (time.perf_counter() - t0) * 1000.0
        if len(reply) != REPLY_BYTES or reply[0] != ord("R"):
            raise BoardError(f"Bad reply from the board ({len(reply)} bytes). Press the center "
                             "button (reset) or reprogram it, then try again.")
        cycles = struct.unpack_from("<I", reply, 2)[0]
        logits = np.array(struct.unpack_from("<10i", reply, 6), dtype=np.int64)
        return Result(reply[1], cycles, logits, round_trip_ms)

    def close(self):
        self.ser.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
