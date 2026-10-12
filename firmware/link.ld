/* Everything (code, weights, data, stack) lives in the 128 KB on-chip RAM.
 * The RAM is preloaded with the whole image, so .data needs no copying. */
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x00000000, LENGTH = 128K
}

SECTIONS
{
    .text : {
        *(.text.start)
        *(.text .text.*)
    } > RAM

    .rodata : ALIGN(4) {
        *(.rodata .rodata.* .srodata .srodata.*)
    } > RAM

    .data : ALIGN(4) {
        *(.data .data.* .sdata .sdata.*)
    } > RAM

    .bss (NOLOAD) : ALIGN(4) {
        _bss_start = .;
        *(.bss .bss.* .sbss .sbss.* COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM

    _stack_top = ORIGIN(RAM) + LENGTH(RAM);

    /DISCARD/ : { *(.comment) *(.eh_frame) *(.note*) }
}
