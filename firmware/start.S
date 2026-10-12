/* Reset entry point: the CPU starts executing at address 0. */
    .section .text.start
    .global _start
_start:
    la   sp, _stack_top        /* stack grows down from the end of RAM */

    la   t0, _bss_start        /* zero .bss (C globals without an initializer) */
    la   t1, _bss_end
1:  bgeu t0, t1, 2f
    sw   zero, 0(t0)
    addi t0, t0, 4
    j    1b

2:  call main
3:  j    3b                    /* main never returns */
