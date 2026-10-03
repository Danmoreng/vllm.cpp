EXL3 ESIMD kernels from 0xSero/exl3xpu, commit
`c59d9442aba8610188837e37724600f1517d7335`, MIT license (see LICENSE).

`exl3_esimd.h` is an unchanged copy of `csrc/exl3_esimd.h`, SHA-256
`aabdb13eddbcf7387dac2716b26e1005259a0a658d4b7d8b0e6b338499d0fdc6`.
Native VT launch/allocation wrappers live in `src/vt/xpu/xpu_exl3_smallm.cpp`;
they do not link Torch or use its tensor/stream wrappers.
