/* Integer-only MNIST MLP inference. Must stay bit-exact with model/qmlp.py. */
#include "mlp.h"
#include "mlp_weights.h"

#if MLP_WEIGHTS_IN != MLP_IN || MLP_WEIGHTS_HIDDEN != MLP_HIDDEN || MLP_WEIGHTS_OUT != MLP_OUT
#error "mlp_weights.h does not match mlp.h: update MLP_HIDDEN or rerun train_quantize.py"
#endif

int mlp_infer(const uint8_t img[MLP_IN], int32_t logits[MLP_OUT])
{
    int8_t h[MLP_HIDDEN];

    /* Layer 1: 784 -> HIDDEN, then requantize to 0..127 (the clamp at 0 is the ReLU). */
    for (int j = 0; j < MLP_HIDDEN; j++) {
        const int8_t *w = &MLP_W1[j * MLP_IN];
        int32_t acc = MLP_B1[j];
        for (int i = 0; i < MLP_IN; i++)
            acc += (int32_t)(img[i] >> 1) * w[i];

        /* Fixed-point multiply by M1 * 2^-SHIFT1 with round-half-up.
         * For negative acc the result is <= 0 whichever way >> rounds, and the
         * clamp below turns it into 0, so the result is the same on any compiler. */
        int64_t s = ((int64_t)acc * MLP_M1 + ((int64_t)1 << (MLP_SHIFT1 - 1))) >> MLP_SHIFT1;
        if (s < 0)   s = 0;
        if (s > 127) s = 127;
        h[j] = (int8_t)s;
    }

    /* Layer 2: HIDDEN -> 10 logits, argmax (first maximum wins, like numpy). */
    int best = 0;
    int32_t best_v = 0;
    for (int k = 0; k < MLP_OUT; k++) {
        const int8_t *w = &MLP_W2[k * MLP_HIDDEN];
        int32_t acc = MLP_B2[k];
        for (int j = 0; j < MLP_HIDDEN; j++)
            acc += (int32_t)h[j] * w[j];
        if (logits)
            logits[k] = acc;
        if (k == 0 || acc > best_v) {
            best_v = acc;
            best = k;
        }
    }
    return best;
}
