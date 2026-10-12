/* Laptop test: run mlp.c on the 10,000 MNIST test images and compare every
 * logit with the Python integer model (test_data/test_ref.bin).
 *
 * Build and run (in this folder):
 *     gcc -O2 -o test_mlp test_mlp.c mlp.c
 *     ./test_mlp              (Windows: test_mlp.exe)
 * Exit code 0 = C matches Python on every image.
 */
#include <stdio.h>
#include <stdlib.h>
#include "mlp.h"

#define N_TEST 10000

static unsigned char *read_file(const char *dir, const char *name, size_t expect)
{
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s. Run model/train_quantize.py first.\n", path);
        exit(2);
    }
    unsigned char *buf = malloc(expect);
    size_t got = buf ? fread(buf, 1, expect, f) : 0;
    fclose(f);
    if (got != expect) {
        fprintf(stderr, "%s: expected %zu bytes, got %zu\n", path, expect, got);
        exit(2);
    }
    return buf;
}

static int32_t read_le32(const unsigned char *p)
{
    return (int32_t)((uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24);
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "test_data";
    unsigned char *images = read_file(dir, "test_images.bin", (size_t)N_TEST * MLP_IN);
    unsigned char *labels = read_file(dir, "test_labels.bin", N_TEST);
    unsigned char *ref    = read_file(dir, "test_ref.bin", (size_t)N_TEST * MLP_OUT * 4);

    int bad_images = 0, correct = 0;
    for (int n = 0; n < N_TEST; n++) {
        int32_t logits[MLP_OUT];
        int pred = mlp_infer(images + (size_t)n * MLP_IN, logits);
        int bad = 0;
        for (int k = 0; k < MLP_OUT; k++) {
            int32_t r = read_le32(ref + ((size_t)n * MLP_OUT + k) * 4);
            if (logits[k] != r) {
                if (bad_images < 5)
                    printf("  image %d class %d: C %ld, Python %ld\n", n, k, (long)logits[k], (long)r);
                bad = 1;
            }
        }
        bad_images += bad;
        correct += (pred == labels[n]);
    }

    printf("C vs Python: %d / %d images identical\n", N_TEST - bad_images, N_TEST);
    printf("C accuracy:  %.2f%%\n", correct * 100.0 / N_TEST);
    printf(bad_images ? "FAIL: C and Python differ\n" : "PASS: C matches Python bit for bit\n");

    free(images);
    free(labels);
    free(ref);
    return bad_images ? 1 : 0;
}
