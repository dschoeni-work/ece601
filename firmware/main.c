/* Firmware for the PicoRV32 SoC: receive an image over UART, run the MLP,
 * send back the answer and how many clock cycles inference took.
 *
 * Protocol (115200 baud, 8N1; all numbers little-endian):
 *   PC -> board  'P'                  ping
 *   board -> PC  'K'
 *   PC -> board  'I' + 784 bytes      28x28 image, row-major, 0 = background
 *   board -> PC  'R', digit (1 byte), cycles (uint32), logits (10 x int32)   = 46 bytes
 *
 * The digit also appears on the rightmost 7-segment digit and as LED <digit>.
 */
#include <stdint.h>
#include "mlp.h"

#define UART_DATA (*(volatile uint32_t *)0x10000000u)
#define UART_STAT (*(volatile uint32_t *)0x10000004u)
#define LEDS      (*(volatile uint32_t *)0x10000008u)
#define SEG       (*(volatile uint32_t *)0x1000000Cu)

#define TX_BUSY  1u
#define RX_READY 2u

static inline uint32_t rdcycle(void)
{
    uint32_t c;
    __asm__ volatile("rdcycle %0" : "=r"(c));
    return c;
}

static void uart_put(uint8_t c)
{
    while (UART_STAT & TX_BUSY)
        ;
    UART_DATA = c;
}

static uint8_t uart_get(void)
{
    while (!(UART_STAT & RX_READY))
        ;
    return (uint8_t)UART_DATA;
}

static void uart_put32(uint32_t v)
{
    for (int i = 0; i < 4; i++)
        uart_put((uint8_t)(v >> (8 * i)));
}

static uint8_t image[MLP_IN];

int main(void)
{
    SEG = 0;        /* display blank until the first result */
    LEDS = 0;

    for (;;) {
        uint8_t cmd = uart_get();
        if (cmd == 'P') {
            uart_put('K');
            continue;
        }
        if (cmd != 'I')
            continue;

        for (int i = 0; i < MLP_IN; i++)
            image[i] = uart_get();

        int32_t logits[MLP_OUT];
        uint32_t t0 = rdcycle();
        int digit = mlp_infer(image, logits);
        uint32_t cycles = rdcycle() - t0;

        SEG = (1u << 16) | (uint32_t)digit;   /* rightmost digit on */
        LEDS = 1u << digit;

        uart_put('R');
        uart_put((uint8_t)digit);
        uart_put32(cycles);
        for (int k = 0; k < MLP_OUT; k++)
            uart_put32((uint32_t)logits[k]);
    }
}
