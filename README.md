# numkit/ops

High-performance numerical operations, SIMD math kernels, multidimensional FFT, BLAS, and parallel execution engine for NumKit.

Part of the **NumKit** modular runtime architecture (Layer 0.5).

## Features

- **Element-wise arithmetic & fused operations**: Vectorized element-wise kernels with Google Highway SIMD acceleration and portable fallbacks.
- **Multidimensional FFT**: Fast Stockham, Radix-2, and Radix-4 algorithms.
- **BLAS 1/2/3**: Matrix multiplication (GEMM) and vector operations.
- **Filtering & Convolutions**: Time-domain 1-D/2-D convolutions, Direct-Form II transposed IIR filter engines.
- **Array Utilities**: Multidimensional broadcasting (`broadcastDims`), slicing, reductions, and NaN-aware reductions.
- **Parallel Computing**: Lightweight thread pool and `parallel_for` scheduler.
- **Random Number Generation**: Reproducible Mersenne Twister (MT19937) matching MATLAB distributions.

## Dependencies

- C++17 compatible compiler
- [numkit/value](https://github.com/numkit/value) (Layer 0 core substrate)
- [Google Highway](https://github.com/google/highway) (Optional, for SIMD acceleration)

## License

[0BSD](LICENSE)
