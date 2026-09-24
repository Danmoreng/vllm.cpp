#include "xpu_common.h"
#include "xpu_kernels.h"
#include "vt/sample_common.h"
#include <limits>
namespace vt::xpu {
namespace {
constexpr float NegInf = -std::numeric_limits<float>::infinity();
constexpr size_t SamplingLanes = 128;

// Row reductions stay on the GPU even for the 248,320-token vocabulary.
// Mode 0: softmax, 1: log-softmax, 2: in-place min-p masking.
template<int Mode>
void Normalize(Queue& q, Tensor& out, const Tensor& logits, const Tensor* min_p = nullptr) {
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  if (!rows || !vocab) return;
  const auto* src = static_cast<const float*>(logits.data);
  auto* dst = static_cast<float*>(out.data);
  const auto* mp = min_p ? static_cast<const float*>(min_p->data) : nullptr;
  NativeQueue(q).parallel_for(sycl::nd_range<1>(rows * SamplingLanes, SamplingLanes), [=](sycl::nd_item<1> item) {
    const int64_t row = item.get_group(0), lane = item.get_local_id(0);
    if constexpr (Mode == 2) { if (mp[row] <= 0) return; }
    float maximum = NegInf;
    for (int64_t j = lane; j < vocab; j += SamplingLanes) maximum = sycl::fmax(maximum, src[row * vocab + j]);
    maximum = sycl::reduce_over_group(item.get_group(), maximum, sycl::maximum<float>());
    float sum = 0;
    for (int64_t j = lane; j < vocab; j += SamplingLanes) sum += sycl::exp(src[row * vocab + j] - maximum);
    sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
    const float lse = maximum + sycl::log(sum);
    for (int64_t j = lane; j < vocab; j += SamplingLanes) {
      const auto index = row * vocab + j;
      if constexpr (Mode == 0) dst[index] = sycl::exp(src[index] - maximum) / sum;
      if constexpr (Mode == 1) dst[index] = src[index] - lse;
      if constexpr (Mode == 2)
        if (sycl::exp(src[index] - maximum) / sum < mp[row] * (1.0f / sum)) dst[index] = NegInf;
    }
  });
}
template<bool Bias>
void SparseMask(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols, const Tensor* biases) {
  const auto count = rows.Numel(), n = logits.shape[0], v = logits.shape[1];
  if (!count) return;
  const auto* rr = static_cast<const int32_t*>(rows.data);
  const auto* cc = static_cast<const int32_t*>(cols.data);
  const auto* bb = biases ? static_cast<const float*>(biases->data) : nullptr;
  auto* values = static_cast<float*>(logits.data);
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < count; ++i) if (rr[i] < 0 || rr[i] >= n || cc[i] < 0 || cc[i] >= v) return false;
    return true;
  }, "XPU sampling sparse index outside logits");
  NativeQueue(q).parallel_for(sycl::range<1>(count), [=](sycl::id<1> item) {
    const int64_t i = item[0], index = int64_t(rr[i]) * v + cc[i];
    // Preserve the reference's addition order for repeated bias coordinates;
    // duplicate masks also have just one writer, without atomics or races.
    for (int64_t j = 0; j < i; ++j) if (rr[j] == rr[i] && cc[j] == cc[i]) return;
    if constexpr (Bias) {
      float value = values[index];
      for (int64_t j = i; j < count; ++j) if (rr[j] == rr[i] && cc[j] == cc[i]) value += bb[j];
      values[index] = value;
    } else values[index] = NegInf;
  });
}
}
void ApplyTemperatureKernel(Queue& q, Tensor& logits, const Tensor& temp, bool all_random) {
  TraceOpTensors(OpId::kApplyTemperature, q, {&logits, &temp});
  if (!logits.Numel()) return;
  const auto vocab = logits.shape[1];
  auto* values = static_cast<float*>(logits.data);
  const auto* temps = static_cast<const float*>(temp.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> i) {
    float t = temps[i[0] / vocab];
    if (!all_random && t < kSamplingEps) t = 1;
    values[i] /= t;
  });
}
void ComputeProbsKernel(Queue& q, Tensor& probs, const Tensor& logits) {
  TraceOpTensors(OpId::kComputeProbs, q, {&probs, &logits});
  WithOutput(q, probs, {&logits}, [&](Tensor& target) { Normalize<0>(q, target, logits); });
}
void ComputeLogprobsKernel(Queue& q, Tensor& probs, const Tensor& logits) {
  TraceOpTensors(OpId::kComputeLogprobs, q, {&probs, &logits});
  WithOutput(q, probs, {&logits}, [&](Tensor& target) { Normalize<1>(q, target, logits); });
}
void ApplyMinPKernel(Queue& q, Tensor& logits, const Tensor& min_p) {
  TraceOpTensors(OpId::kApplyMinP, q, {&logits, &min_p});
  Normalize<2>(q, logits, logits, &min_p);
}
void ApplyPenaltiesKernel(Queue& q, Tensor& logits, const Tensor& prompt_mask,
                           const Tensor& counts, const Tensor& output_mask,
                           const Tensor& frequency, const Tensor& presence, const Tensor& repetition) {
  TraceOpTensors(OpId::kApplyPenalties, q, {&logits, &prompt_mask, &counts, &output_mask, &frequency, &presence, &repetition});
  if (!logits.Numel()) return;
  const auto vocab = logits.shape[1];
  auto* values = static_cast<float*>(logits.data);
  const auto* pm = static_cast<const int8_t*>(prompt_mask.data);
  const auto* om = static_cast<const int8_t*>(output_mask.data);
  const auto* oc = static_cast<const int32_t*>(counts.data);
  const auto* freq = static_cast<const float*>(frequency.data);
  const auto* pres = static_cast<const float*>(presence.data);
  const auto* rep = static_cast<const float*>(repetition.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> item) {
    const auto i = item[0], row = i / vocab;
    float value = values[i];
    if (pm[i] || om[i]) value = value > 0 ? value / rep[row] : value * rep[row];
    value -= freq[row] * float(oc[i]); value -= pres[row] * float(om[i]);
    values[i] = value;
  });
}
void ApplyLogitBiasKernel(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols, const Tensor& biases) {
  TraceOpTensors(OpId::kApplyLogitBias, q, {&logits, &rows, &cols, &biases});
  SparseMask<true>(q, logits, rows, cols, &biases);
}
void ApplyTokenMaskKernel(Queue& q, Tensor& logits, const Tensor& rows, const Tensor& cols) {
  TraceOpTensors(OpId::kApplyTokenMask, q, {&logits, &rows, &cols});
  SparseMask<false>(q, logits, rows, cols, nullptr);
}
void ApplyAllowedTokenIdsKernel(Queue& q, Tensor& logits, const Tensor& mask) {
  TraceOpTensors(OpId::kApplyAllowedTokenIds, q, {&logits, &mask});
  if (!logits.Numel()) return;
  auto* values = static_cast<float*>(logits.data);
  const auto* m = static_cast<const int8_t*>(mask.data);
  NativeQueue(q).parallel_for(sycl::range<1>(logits.Numel()), [=](sycl::id<1> i) { if (m[i]) values[i] = NegInf; });
}
void ApplyTopKTopPKernel(Queue& q, Tensor& logits, const Tensor* k, const Tensor* p) {
  TraceOpTensors(OpId::kApplyTopKTopP, q, {&logits, k, p});
  const int64_t rows = logits.shape[0], vocab = logits.shape[1];
  if (!rows || !vocab) return;
  // Process at most four rows at once, with a fixed 16 MiB context workspace.
  // Merge permutations instead of moving logits: masking only starts after
  // all ranks and cumulative probabilities have been computed.
  constexpr int64_t Tile = 128, Workspace = 16 * 1024 * 1024;
  const int64_t tiles = (vocab + Tile - 1) / Tile;
  const int64_t row_bytes = (2 * vocab + tiles + 2) * sizeof(int32_t);
  VT_CHECK(row_bytes <= Workspace, "XPU top-k/top-p vocabulary exceeds bounded workspace");
  const int64_t batch = std::min(int64_t{4}, Workspace / row_bytes);
  auto* values = static_cast<float*>(logits.data);
  const auto* ks = k ? static_cast<const int32_t*>(k->data) : nullptr;
  const auto* ps = p ? static_cast<const float*>(p->data) : nullptr;
  const bool available = WithSamplingWorkspace(q, Workspace, [&](void* storage) {
    for (int64_t start = 0; start < rows; start += batch) {
      const int64_t n = std::min(batch, rows - start);
      auto* first = static_cast<int32_t*>(storage);
      auto* second = first + n * vocab;
      auto* totals = reinterpret_cast<float*>(second + n * vocab);
      auto* stats = totals + n * tiles;
      auto* data = values + start * vocab;
      NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> i) { first[i] = i[0] % vocab; });
      // Stable bottom-up merge by binary rank. Each item has exactly one
      // destination, including equal logits (lowest original index first).
      for (int64_t width = 1; width < vocab; width *= 2) {
        const auto* src = first; auto* dst = second;
        NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> item) {
          const int64_t row = item[0] / vocab, col = item[0] % vocab;
          const int64_t base = (col / (2 * width)) * 2 * width;
          const int64_t middle = sycl::min(base + width, vocab), end = sycl::min(base + 2 * width, vocab);
          const bool right = col >= middle;
          const int64_t other = right ? base : middle, stop = right ? middle : end;
          const int32_t index = src[row * vocab + col];
          const float value = data[row * vocab + index];
          int64_t lo = other, hi = stop;
          while (lo < hi) {
            const auto mid = (lo + hi) / 2;
            const int32_t candidate = src[row * vocab + mid];
            const float c = data[row * vocab + candidate];
            const bool equal = c == value || (sycl::isnan(c) && sycl::isnan(value));
            if ((sycl::isnan(value) && !sycl::isnan(c)) || c < value || (equal && candidate < index)) lo = mid + 1; else hi = mid;
          }
          const auto rank = base + col - (right ? middle : base) + lo - other;
          dst[row * vocab + rank] = index;
        });
        std::swap(first, second);
      }
      const auto* order = first;
      // The unused permutation buffer becomes per-tile probability prefixes.
      auto* cdf = reinterpret_cast<float*>(second);
      NativeQueue(q).parallel_for(sycl::nd_range<1>(n * Tile, Tile), [=](sycl::nd_item<1> item) {
        const auto row = item.get_group(0), lane = item.get_local_id(0);
        const int64_t keep = ks ? ks[start + row] : vocab;
        const float threshold = keep >= 1 && keep < vocab ? data[row * vocab + order[row * vocab + vocab - keep]] : NegInf;
        const float maximum = data[row * vocab + order[row * vocab + vocab - 1]];
        float sum = 0;
        for (int64_t j = lane; j < vocab; j += Tile) {
          const float value = data[row * vocab + order[row * vocab + j]];
          sum += value >= threshold && value != NegInf ? sycl::exp(value - maximum) : 0;
        }
        sum = sycl::reduce_over_group(item.get_group(), sum, sycl::plus<float>());
        if (!lane) { stats[row * 2] = threshold; stats[row * 2 + 1] = sum; }
      });
      if (ps) {
        NativeQueue(q).parallel_for(sycl::nd_range<1>(n * tiles * Tile, Tile), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0) / tiles, tile = item.get_group(0) % tiles, lane = item.get_local_id(0);
          const int64_t j = tile * Tile + lane;
          const float maximum = data[row * vocab + order[row * vocab + vocab - 1]];
          const float value = j < vocab ? data[row * vocab + order[row * vocab + j]] : NegInf;
          const float probability = value >= stats[row * 2] && value != NegInf ? sycl::exp(value - maximum) / stats[row * 2 + 1] : 0;
          const float prefix = sycl::inclusive_scan_over_group(item.get_group(), probability, sycl::plus<float>());
          if (j < vocab) cdf[row * vocab + j] = prefix;
          if (lane == Tile - 1) totals[row * tiles + tile] = prefix;
        });
        NativeQueue(q).parallel_for(sycl::nd_range<1>(n * Tile, Tile), [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0), lane = item.get_local_id(0);
          float carry = 0;
          for (int64_t tile = 0; tile < tiles; tile += Tile) {
            const float value = tile + lane < tiles ? totals[row * tiles + tile + lane] : 0;
            const float prefix = sycl::exclusive_scan_over_group(item.get_group(), value, sycl::plus<float>());
            if (tile + lane < tiles) totals[row * tiles + tile + lane] = carry + prefix;
            carry += sycl::reduce_over_group(item.get_group(), value, sycl::plus<float>());
          }
        });
      }
      NativeQueue(q).parallel_for(sycl::range<1>(n * vocab), [=](sycl::id<1> item) {
        const int64_t row = item[0] / vocab, j = item[0] % vocab, index = row * vocab + order[item];
        const bool topk = data[index] < stats[row * 2];
        const bool topp = ps && j != vocab - 1 && cdf[item] + totals[row * tiles + j / Tile] <= 1 - ps[start + row];
        if (topk || topp) data[index] = NegInf;
      });
    }
  });
  VT_CHECK(available, "XPU top-k/top-p cannot reserve its 16 MiB sampling workspace");
}
void RandomSampleKernel(Queue& q, Tensor& token_ids, const Tensor& probs, const Tensor& seeds) {
  TraceOpTensors(OpId::kRandomSample, q, {&token_ids, &probs, &seeds});
  const auto rows = probs.shape[0], vocab = probs.shape[1];
  if (!rows) return;
  VT_CHECK(vocab > 0, "XPU random sampling requires a nonempty vocabulary");
  const auto* pp = static_cast<const float*>(probs.data);
  const auto* sp = static_cast<const int64_t*>(seeds.data);
  auto* ids = static_cast<int64_t*>(token_ids.data);
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> scores(SamplingLanes, h);
    sycl::local_accessor<int64_t> indices(SamplingLanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * SamplingLanes, SamplingLanes), [=](sycl::nd_item<1> item) {
      const int64_t row = item.get_group(0), lane = item.get_local_id(0);
      const auto key = sample::SplitMix64(uint64_t(sp[row]) + 0x9E3779B97F4A7C15ULL * uint64_t(row));
      float best = NegInf; int64_t index = sample::kArgSentinel;
      for (int64_t j = lane; j < vocab; j += SamplingLanes) {
        const uint64_t u = (sample::SplitMix64(key + uint64_t(j)) >> 11) + 1;
        // Same integer draws as VT's reference, with native F32 transcendental
        // math on Xe2. log1p preserves the tail near U=1 without requiring FP64.
        const float noise = u > (1ULL << 52) ? -sycl::log1p(-float((1ULL << 53) - u) * 0x1p-53f)
                                            : -sycl::log(float(u) * 0x1p-53f);
        const float score = pp[row * vocab + j] / noise;
        sample::ArgReduce(best, index, score, j);
      }
      scores[lane] = best; indices[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (int step = SamplingLanes / 2; step; step /= 2) {
        if (lane < step) {
          float v = scores[lane]; int64_t i = indices[lane];
          sample::ArgReduce(v, i, scores[lane + step], indices[lane + step]);
          scores[lane] = v; indices[lane] = i;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (!lane) ids[row] = indices[0] == sample::kArgSentinel ? 0 : indices[0];
    });
  });
}
void GreedyArgmaxKernel(Queue& q, Tensor& token_ids, const Tensor& logits) {
  TraceOpTensors(OpId::kGreedyArgmax, q, {&token_ids, &logits});
  if (logits.shape[0] == 0) return;
  const auto rows = static_cast<size_t>(logits.shape[0]);
  const auto vocab = logits.shape[1];
  const auto* values = static_cast<const float*>(logits.data);
  auto* ids = static_cast<int64_t*>(token_ids.data);
  constexpr size_t lanes = 128;
  NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float> best_values(lanes, h);
    sycl::local_accessor<int64_t> best_ids(lanes, h);
    h.parallel_for(sycl::nd_range<1>(rows * lanes, lanes), [=](sycl::nd_item<1> item) {
      const auto row = item.get_group(0), lane = item.get_local_id(0);
      // Match VT's strict '>' scan: NaN at index 0 wins; later NaNs are ignored.
      if (sycl::isnan(values[row * vocab])) { if (lane == 0) ids[row] = 0; return; }
      float value = -std::numeric_limits<float>::infinity();
      int64_t index = std::numeric_limits<int64_t>::max();
      for (int64_t i = lane; i < vocab; i += lanes) {
        const float v = values[row * vocab + i];
        if (v > value || (v == value && i < index)) { value = v; index = i; }
      }
      best_values[lane] = value; best_ids[lane] = index;
      item.barrier(sycl::access::fence_space::local_space);
      for (size_t step = lanes / 2; step; step /= 2) {
        if (lane < step) {
          const float v = best_values[lane + step]; const auto i = best_ids[lane + step];
          if (v > best_values[lane] || (v == best_values[lane] && i < best_ids[lane])) {
            best_values[lane] = v; best_ids[lane] = i;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (lane == 0) ids[row] = best_ids[0];
    });
  });
}
}  // namespace vt::xpu
