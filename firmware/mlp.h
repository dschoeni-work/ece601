/* Integer-only MNIST MLP inference (784 -> MLP_HIDDEN -> 10).
 *
 * Freestanding: uses only <stdint.h>, no libc, no floating point, so the same
 * file compiles for the laptop (test_mlp.c) and for the RISC-V CPU on the FPGA.
 */
#ifndef MLP_H
#define MLP_H

#include <stdint.h>

#define MLP_IN     784
#define MLP_HIDDEN 64   /* must match --hidden used in train_quantize.py */
#define MLP_OUT    10

/* img:    28x28 grayscale image, row-major, 0 = background, 255 = ink.
 * logits: optional output (may be NULL), the 10 raw int32 class scores.
 * Returns the predicted digit 0..9. */
int mlp_infer(const uint8_t img[MLP_IN], int32_t logits[MLP_OUT]);

#endif /* MLP_H */
