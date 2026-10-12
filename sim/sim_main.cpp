// Verilator testbench for hw/rtl/soc.v running firmware/build/firmware.hex.
//
// It plays the role of the laptop: sends MNIST test images to the simulated
// board over UART with the same protocol as the real board (see
// firmware/main.c), then checks every logit against the Python integer model
// (firmware/test_data/test_ref.bin) and reports the clock cycles per inference.
//
// Usage: ./obj_dir/Vsoc [number_of_images] [test_data_dir]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

#include "Vsoc.h"
#include "verilated.h"

#ifndef CLKS_PER_BIT
#define CLKS_PER_BIT 16
#endif

static const int IMG = 784;
static const double CLOCK_MHZ = 100.0;  // Basys 3

class Bench {
  public:
    Bench() : top(new Vsoc) {
        top->clk = 0;
        top->resetn = 0;
        top->uart_rx = 1;
        top->eval();
        for (int i = 0; i < 20; i++) tick();
        top->resetn = 1;
    }
    ~Bench() {
        top->final();
        delete top;
    }

    void send(uint8_t b) { to_soc.push_back(b); }

    // Run until `n` bytes have arrived from the SoC or the cycle limit is hit.
    bool receive(size_t n, std::vector<uint8_t> &out, uint64_t limit) {
        uint64_t end = cycle + limit;
        while (from_soc.size() < n && cycle < end) {
            tick();
            if (top->trap) {
                std::fprintf(stderr, "CPU trapped at cycle %llu\n", (unsigned long long)cycle);
                return false;
            }
        }
        if (from_soc.size() < n) return false;
        out.assign(from_soc.begin(), from_soc.begin() + n);
        from_soc.erase(from_soc.begin(), from_soc.begin() + n);
        return true;
    }

  private:
    Vsoc *top;
    uint64_t cycle = 0;
    std::deque<uint8_t> to_soc, from_soc;

    // UART into the SoC: one 10-bit frame (start, 8 data bits, stop) at a time.
    int drv_bit = -1, drv_cnt = 0;
    uint16_t drv_frame = 0;
    // UART out of the SoC.
    int dec_state = 0, dec_cnt = 0, dec_bit = 0;
    uint8_t dec_byte = 0;

    void tick() {
        if (drv_bit < 0 && !to_soc.empty()) {
            drv_frame = (uint16_t)((1u << 9) | (uint16_t(to_soc.front()) << 1));
            to_soc.pop_front();
            drv_bit = 0;
            drv_cnt = CLKS_PER_BIT;
        }
        top->uart_rx = drv_bit < 0 ? 1 : (drv_frame >> drv_bit) & 1;
        if (drv_bit >= 0 && --drv_cnt == 0) {
            drv_cnt = CLKS_PER_BIT;
            if (++drv_bit == 10) drv_bit = -1;
        }

        top->clk = 0;
        top->eval();
        top->clk = 1;
        top->eval();
        cycle++;

        int tx = top->uart_tx;
        switch (dec_state) {
        case 0:  // idle, wait for a start bit
            if (!tx) {
                dec_state = 1;
                dec_cnt = CLKS_PER_BIT / 2;
            }
            break;
        case 1:  // middle of the start bit
            if (--dec_cnt == 0) {
                dec_state = tx ? 0 : 2;
                dec_cnt = CLKS_PER_BIT;
                dec_bit = 0;
                dec_byte = 0;
            }
            break;
        case 2:  // data bits
            if (--dec_cnt == 0) {
                dec_byte |= uint8_t(tx << dec_bit);
                dec_cnt = CLKS_PER_BIT;
                if (++dec_bit == 8) dec_state = 3;
            }
            break;
        case 3:  // stop bit
            if (--dec_cnt == 0) {
                if (tx) from_soc.push_back(dec_byte);
                dec_state = 0;
            }
            break;
        }
    }
};

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s (run model/train_quantize.py first)\n", path.c_str());
        std::exit(2);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static int32_t le32(const uint8_t *p) {
    return int32_t(uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    int n = argc > 1 ? std::atoi(argv[1]) : 20;
    std::string dir = argc > 2 ? argv[2] : "../firmware/test_data";
    std::vector<uint8_t> images = read_file(dir + "/test_images.bin");
    std::vector<uint8_t> labels = read_file(dir + "/test_labels.bin");
    std::vector<uint8_t> ref = read_file(dir + "/test_ref.bin");
    if (n > (int)labels.size()) n = (int)labels.size();

    Bench b;
    std::vector<uint8_t> r;

    b.send('P');
    if (!b.receive(1, r, 200000) || r[0] != 'K') {
        std::printf("FAIL: no reply to ping (is firmware.hex loaded?)\n");
        return 1;
    }
    std::printf("Ping OK. Running %d MNIST test images on the simulated SoC...\n", n);

    int mismatches = 0, correct = 0;
    uint64_t sum = 0;
    uint32_t lo = UINT32_MAX, hi = 0;
    for (int i = 0; i < n; i++) {
        b.send('I');
        for (int k = 0; k < IMG; k++) b.send(images[size_t(i) * IMG + k]);
        if (!b.receive(46, r, 50000000ull) || r[0] != 'R') {
            std::printf("FAIL: no result for image %d\n", i);
            return 1;
        }
        int digit = r[1];
        uint32_t cycles = uint32_t(le32(&r[2]));
        bool same = true;
        for (int k = 0; k < 10; k++)
            same &= le32(&r[6 + 4 * k]) == le32(&ref[(size_t(i) * 10 + k) * 4]);
        mismatches += !same;
        correct += digit == labels[i];
        sum += cycles;
        lo = cycles < lo ? cycles : lo;
        hi = cycles > hi ? cycles : hi;
        if (i < 5 || !same)
            std::printf("  image %d: label %d, board says %d, %u cycles, logits %s Python\n", i, labels[i],
                        digit, cycles, same ? "match" : "DIFFER from");
    }

    double avg = double(sum) / n;
    std::printf("\nLogits identical to Python: %d / %d images\n", n - mismatches, n);
    std::printf("Correct digits:             %d / %d\n", correct, n);
    std::printf("Cycles per inference:       %.0f (min %u, max %u)\n", avg, lo, hi);
    std::printf("Time per inference:         %.2f ms at %.0f MHz\n", avg / (CLOCK_MHZ * 1000.0), CLOCK_MHZ);
    std::printf("%s\n", mismatches ? "FAIL" : "PASS: the RISC-V SoC matches Python bit for bit");
    return mismatches ? 1 : 0;
}
