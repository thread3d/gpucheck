// Shaders.swift — Metal Shading Language sources compiled at runtime.
//
// Shaders are grouped so that a feature which fails to compile on a given GPU
// (for example simdgroup matrix on an older AMD part) only disables its own
// group instead of aborting the whole suite.

import Foundation
import Metal

struct ShaderGroup {
    let key: String
    let source: String
    let functions: [String]
    var fastMath: Bool = false
    /// Optional groups cover features a GPU may legitimately lack (FP64, the
    /// matrix coprocessor); failing to compile them is a finding, not an error.
    var optional: Bool = false
}

// MARK: - Core group

let coreShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void k_saxpy(device const float *x [[buffer(0)]],
                    device float *y [[buffer(1)]],
                    constant float &a [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    y[gid] = a * x[gid] + y[gid];
}

kernel void k_pattern(device float *dst [[buffer(0)]],
                      constant uint &seed [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    uint h = (gid ^ seed) * 2654435761u;
    dst[gid] = float(h & 0xFFFFu) * (1.0f / 65536.0f);
}

kernel void k_copy(device const float4 *src [[buffer(0)]],
                   device float4 *dst [[buffer(1)]],
                   constant uint &count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]],
                   uint nthreads [[threads_per_grid]]) {
    for (uint i = gid; i < count; i += nthreads) { dst[i] = src[i]; }
}

kernel void k_fill(device float4 *dst [[buffer(0)]],
                   constant float4 &v [[buffer(1)]],
                   constant uint &count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]],
                   uint nthreads [[threads_per_grid]]) {
    for (uint i = gid; i < count; i += nthreads) { dst[i] = v; }
}

kernel void k_read(device const float4 *src [[buffer(0)]],
                   device float *out [[buffer(1)]],
                   constant uint &count [[buffer(2)]],
                   uint gid [[thread_position_in_grid]],
                   uint nthreads [[threads_per_grid]]) {
    float4 acc = float4(0.0f);
    for (uint i = gid; i < count; i += nthreads) { acc += src[i]; }
    out[gid] = acc.x + acc.y + acc.z + acc.w;
}

// 4 independent FMA chains x float4 = 16 FMAs (32 FLOP) per loop iteration.
kernel void k_fma32(device float4 *out [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    float4 a = float4(1.0000001f, 0.9999999f, 1.0000002f, 0.9999998f);
    float4 b = float4(1.0e-7f, 2.0e-7f, 3.0e-7f, 4.0e-7f);
    float base = float(gid & 1023u) * 1.0e-3f;
    float4 c0 = float4(base, base + 0.1f, base + 0.2f, base + 0.3f);
    float4 c1 = c0 + 0.5f, c2 = c0 + 0.7f, c3 = c0 + 0.9f;
    for (uint i = 0; i < iters; ++i) {
        c0 = fma(a, c0, b);
        c1 = fma(a, c1, b);
        c2 = fma(a, c2, b);
        c3 = fma(a, c3, b);
    }
    out[gid] = c0 + c1 + c2 + c3;
}

// mul + add + xor + shift on int4 = 16 integer ops per loop iteration.
kernel void k_iops32(device int4 *out [[buffer(0)]],
                     constant uint &iters [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
    int4 a = int4(3, 5, 7, 9);
    int4 b = int4(11, 13, 17, 19);
    int4 c = int4(int(gid & 1023u));
    for (uint i = 0; i < iters; ++i) {
        c = c * a + b;
        c = c ^ (c >> 7);
    }
    out[gid] = c;
}

kernel void k_sinf(device float4 *out [[buffer(0)]],
                   constant uint &iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    float4 c = float4(gid & 255u) * (1.0f / 256.0f);
    for (uint i = 0; i < iters; ++i) { c = sin(c); }
    out[gid] = c;
}

kernel void k_sqrtf(device float4 *out [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    float4 c = float4(gid & 255u) * (1.0f / 256.0f) + 0.5f;
    for (uint i = 0; i < iters; ++i) { c = sqrt(c); }
    out[gid] = c;
}

// Threadgroup memory + barrier tree reduction. Validates shared memory and
// synchronisation correctness against a host-side reference sum.
kernel void k_reduce(device const float *in [[buffer(0)]],
                     device float *out [[buffer(1)]],
                     uint gid [[thread_position_in_grid]],
                     uint lid [[thread_position_in_threadgroup]],
                     uint tgSize [[threads_per_threadgroup]]) {
    threadgroup float scratch[1024];
    scratch[lid] = in[gid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tgSize >> 1; s > 0; s >>= 1) {
        if (lid < s) { scratch[lid] += scratch[lid + s]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) { out[gid / tgSize] = scratch[0]; }
}

kernel void k_atomic_hist(device atomic_uint *hist [[buffer(0)]],
                          device const uint *keys [[buffer(1)]],
                          constant uint &bins [[buffer(2)]],
                          uint gid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(&hist[keys[gid] % bins], 1u, memory_order_relaxed);
}

// Used by the GPU timestamp-counter test.
kernel void k_spin(device float *out [[buffer(0)]],
                   constant uint &iters [[buffer(1)]],
                   uint gid [[thread_position_in_grid]]) {
    float c = float(gid & 255u) * 1.0e-3f;
    for (uint i = 0; i < iters; ++i) { c = fma(1.000001f, c, 1.0e-7f); }
    out[gid] = c;
}
"""

// MARK: - Isolated feature groups

let fp16ShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void k_fma16(device half4 *out [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    half4 a = half4(1.0001h, 0.9999h, 1.0002h, 0.9998h);
    half4 b = half4(1.0e-4h, 2.0e-4h, 3.0e-4h, 4.0e-4h);
    half base = half(gid & 1023u) * 1.0e-3h;
    half4 c0 = half4(base);
    half4 c1 = c0 + 0.1h, c2 = c0 + 0.2h, c3 = c0 + 0.3h;
    for (uint i = 0; i < iters; ++i) {
        c0 = fma(a, c0, b);
        c1 = fma(a, c1, b);
        c2 = fma(a, c2, b);
        c3 = fma(a, c3, b);
    }
    out[gid] = c0 + c1 + c2 + c3;
}

kernel void k_half_scale(device const half *in [[buffer(0)]],
                         device half *out [[buffer(1)]],
                         constant half &scale [[buffer(2)]],
                         uint gid [[thread_position_in_grid]]) {
    out[gid] = in[gid] * scale;
}
"""

// FP64 lives in its own group: Metal on some GPUs (including AMD parts on
// macOS) rejects the `double` type outright, and an isolated group keeps that
// failure from disabling every other compute shader.
// Math accuracy lives in its own group so that a single function missing from
// this GPU's standard library cannot disable the compute benchmarks.
let mathShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void k_math_accuracy(device const float *in [[buffer(0)]],
                            device float *out [[buffer(1)]],
                            constant uint &fn [[buffer(2)]],
                            constant uint &count [[buffer(3)]],
                            uint gid [[thread_position_in_grid]],
                            uint nthreads [[threads_per_grid]]) {
    for (uint i = gid; i < count; i += nthreads) {
        float x = in[i];
        float r = 0.0f;
        switch (fn) {
            case 0: r = exp(x); break;
            case 1: r = log(x); break;
            case 2: r = sin(x); break;
            case 3: r = cos(x); break;
            case 4: r = sqrt(x); break;
            case 5: r = rsqrt(x); break;
            case 6: r = pow(x, 1.5f); break;
            case 7: r = tanh(x); break;
            case 8: r = atan(x); break;
            case 9: r = exp2(x); break;
            case 10: r = log2(x); break;
            case 11: r = tan(x); break;
            case 12: r = asin(x); break;
            case 13: r = sinh(x); break;
            case 14: r = cosh(x); break;
            default: r = 0.0f; break;
        }
        out[i] = r;
    }
}
"""

let fp64ShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void k_fma64(device double *out [[buffer(0)]],
                    constant uint &iters [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
    double a = 1.0000000001;
    double b = 1.0e-12;
    double base = double(gid & 1023u) * 1.0e-6;
    double c0 = base, c1 = base + 0.1, c2 = base + 0.2, c3 = base + 0.3;
    for (uint i = 0; i < iters; ++i) {
        c0 = fma(a, c0, b);
        c1 = fma(a, c1, b);
        c2 = fma(a, c2, b);
        c3 = fma(a, c3, b);
    }
    out[gid] = c0 + c1 + c2 + c3;
}
"""

let textureShaderSource = """
#include <metal_stdlib>
using namespace metal;

kernel void k_tex_read(device float4 *out [[buffer(0)]],
                       texture2d<float, access::read> tex [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    out[gid.y * tex.get_width() + gid.x] = tex.read(gid);
}

kernel void k_tex_sample(device float4 *out [[buffer(0)]],
                         texture2d<float> tex [[texture(0)]],
                         sampler s [[sampler(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    float2 uv = (float2(gid) + 0.5f) / float2(tex.get_width(), tex.get_height());
    out[gid.y * tex.get_width() + gid.x] = tex.sample(s, uv);
}

// Samples at caller-supplied texture coordinates so filtering and addressing
// modes can be verified against an exact CPU reference.
kernel void k_tex_sample_uv(device float4 *out [[buffer(0)]],
                            texture2d<float> tex [[texture(0)]],
                            sampler s [[sampler(0)]],
                            device const float2 *uvs [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    out[gid] = tex.sample(s, uvs[gid]);
}

kernel void k_tex_sample_batch(device float4 *out [[buffer(0)]],
                               texture2d<float> tex [[texture(0)]],
                               sampler s [[sampler(0)]],
                               constant uint &iters [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
    float fg = float(gid);
    float2 uv = float2(fract(sin(fg) * 43758.5453f), fract(cos(fg) * 12345.6789f));
    float4 acc = float4(0.0f);
    for (uint i = 0; i < iters; ++i) {
        uv = fract(uv + 0.017f);
        acc += tex.sample(s, uv);
    }
    out[gid] = acc;
}
"""

let simdMatrixShaderSource = """
#include <metal_stdlib>
using namespace metal;

// simdgroup_store has no overload taking a threadgroup or device pointer on
// this toolchain, so the per-thread elements are read directly instead.
kernel void k_simdmat(device float *out [[buffer(0)]],
                      uint lid [[thread_position_in_threadgroup]]) {
    simdgroup_matrix<float, 8, 8> a(2.0f);
    simdgroup_matrix<float, 8, 8> b(3.0f);
    simdgroup_matrix<float, 8, 8> c(0.0f);
    simdgroup_multiply_accumulate(c, a, b, c);
    auto e = c.thread_elements();
    out[lid * 2] = e[0];
    out[lid * 2 + 1] = e[1];
}
"""

// Rasterisation path: verifies the graphics pipeline independently of compute.
let renderShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct VOut {
    float4 pos [[position]];
};

// Oversized triangle that covers the entire clip volume.
vertex VOut v_full(uint vid [[vertex_id]]) {
    float2 pts[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    VOut o;
    o.pos = float4(pts[vid], 0.0, 1.0);
    return o;
}

// Right half of the screen with a diagonal edge, to make MSAA coverage
// produce partially covered pixels along the hypotenuse.
vertex VOut v_half(uint vid [[vertex_id]]) {
    float2 pts[3] = { float2(0.0, -1.0), float2(1.0, -1.0), float2(0.0, 1.0) };
    VOut o;
    o.pos = float4(pts[vid], 0.0, 1.0);
    return o;
}

fragment float4 f_const(VOut in [[stage_in]],
                        constant float4 &color [[buffer(0)]]) {
    return color;
}
"""

// MARK: - Group registry

let shaderGroups: [ShaderGroup] = [
    ShaderGroup(key: "core", source: coreShaderSource, functions: [
        "k_saxpy", "k_pattern", "k_copy", "k_fill", "k_read", "k_fma32",
        "k_iops32", "k_sinf", "k_sqrtf", "k_reduce", "k_atomic_hist", "k_spin",
    ]),
    ShaderGroup(key: "math", source: mathShaderSource, functions: ["k_math_accuracy"]),
    ShaderGroup(key: "math_fast", source: mathShaderSource, functions: ["k_math_accuracy"], fastMath: true),
    ShaderGroup(key: "fp16", source: fp16ShaderSource, functions: ["k_fma16", "k_half_scale"]),
    ShaderGroup(key: "fp64", source: fp64ShaderSource, functions: ["k_fma64"], optional: true),
    ShaderGroup(key: "texture", source: textureShaderSource, functions: ["k_tex_read", "k_tex_sample", "k_tex_sample_uv", "k_tex_sample_batch"]),
    ShaderGroup(key: "simdmat", source: simdMatrixShaderSource, functions: ["k_simdmat"], optional: true),
    ShaderGroup(key: "render", source: renderShaderSource, functions: ["v_full", "v_half", "f_const"]),
]
