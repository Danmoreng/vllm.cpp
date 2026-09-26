ID: ISSUE-LOCAL-01M37TQF1JN58Q7MCY4ZEJJ8VR
Title: ROCm 7.2 hipblas API change: #ifndef HIPBLAS_COMPUTE_32F guard is broken (enum value, not macro)
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-23
Updated: 2026-09-23
Closed: 2026-09-23

## Problem

On ROCm 7.2, HIPBLAS_COMPUTE_32F is an enum value in hipblasComputeType_t, not a preprocessor macro. The #ifndef HIPBLAS_COMPUTE_32F guard in src/vt/rocm/rocm_matmul_hipblaslt.hip:19 evaluates to true (macro undefined), so the code enters the ROCm <6 compat branch, which typedefs hipblasDatatype_t hipblasComputeType_t — but hipblasDatatype_t does not exist on ROCm 7.2 (replaced by hipDataType). This causes compile errors: unknown type hipblasDatatype_t, VT_BLAS_DT static_cast failure, and hipblasGemmEx argument type mismatch. Fix: replace #ifndef HIPBLAS_COMPUTE_32F with #if HIP_VERSION_MAJOR < 6, which correctly detects the ROCm version boundary.

## Resolution

Fixed on `row/BACKEND-ROCM-HIPBLAS-ROCm7`. The `#ifndef HIPBLAS_COMPUTE_32F`
guard is replaced with `#if HIP_VERSION_MAJOR < 6`, which correctly detects
the ROCm version boundary. The same fix is applied to `hip_shfl_compat.h`,
where the `#ifndef` guards for `__shfl_down_sync` / `__shfl_sync` are now
gated behind `#if HIP_VERSION_MAJOR < 6` because on ROCm >=6 these are native
inline templates, and a `#ifndef` check cannot detect a function definition.
