#include "xpu_test_helpers.h"
#include <limits>
#include "vt/xpu.h"
#include "vllm/v1/sample/sampler.h"

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
void Compare(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 3e-5f) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] == b[i] || (std::isnan(a[i]) && std::isnan(b[i]))) continue;
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance * (1 + std::abs(b[i]))) {
      CAPTURE(i);
      CAPTURE(a[i]);
      CAPTURE(b[i]);
      FAIL("sampling reference mismatch");
    }
  }
}
}
TEST_CASE("XPU sampling: temperatures, probabilities, logprobs and min-p") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int vocab : {1, 257, 248320}) {
    Buffer a(cpu.q, DType::kF32, {4, vocab}), b(gpu.q, DType::kF32, {4, vocab});
    Buffer t(cpu.q, DType::kF32, {4}), u(gpu.q, DType::kF32, {4});
    auto values = xpu_test::Values(4 * vocab, 3, 0.5f);
    if (vocab > 1) values[1] = -std::numeric_limits<float>::infinity();
    a.put(values); b.put(values); t.put({0, 0.7f, 1.2f, 1}); u.put({0, 0.7f, 1.2f, 1});
    vt::ApplyTemperature(cpu.q, a.tensor, t.tensor, false); vt::ApplyTemperature(gpu.q, b.tensor, u.tensor, false);
    Compare(b.floats(), a.floats());
    Buffer p(cpu.q, DType::kF32, {4, vocab}), r(gpu.q, DType::kF32, {4, vocab});
    vt::ComputeProbs(cpu.q, p.tensor, a.tensor); vt::ComputeProbs(gpu.q, r.tensor, b.tensor);
    Compare(r.floats(), p.floats());
    auto probabilities = r.floats();
    for (int row = 0; row < 4; ++row) {
      double sum = 0; for (int j = 0; j < vocab; ++j) sum += probabilities[row * vocab + j];
      CHECK(std::abs(sum - 1) < 2e-5);
    }
    vt::ComputeLogprobs(cpu.q, p.tensor, a.tensor); vt::ComputeLogprobs(gpu.q, r.tensor, b.tensor);
    Compare(r.floats(), p.floats(), 3e-4f); // CPU serial sum rounds across the full vocabulary.
    vt::ComputeLogprobs(gpu.q, b.tensor, b.tensor); Compare(b.floats(), r.floats(), 1e-6f);
    a.put(values); b.put(values); t.put({0, .1f, .8f, 1}); u.put({0, .1f, .8f, 1});
    vt::ApplyMinP(cpu.q, a.tensor, t.tensor); vt::ApplyMinP(gpu.q, b.tensor, u.tensor);
    Compare(b.floats(), a.floats(), 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: penalties, duplicate sparse biases, token and allowed masks") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  auto run = [](vt::Queue& q) {
    constexpr int N = 4, V = 33, M = 5;
    Buffer logits(q, DType::kF32, {N, V}), pm(q, DType::kI8, {N, V}), om(q, DType::kI8, {N, V}), counts(q, DType::kI32, {N, V});
    Buffer freq(q, DType::kF32, {N}), pres(q, DType::kF32, {N}), rep(q, DType::kF32, {N});
    logits.put(xpu_test::Values(N * V, 2));
    std::vector<int8_t> prompt(N * V), output(N * V); std::vector<int32_t> c(N * V);
    for (int i = 0; i < N * V; ++i) { prompt[i] = i % 3 == 0; output[i] = i % 4 == 0; c[i] = output[i] * (i % 7 + 1); }
    pm.upload(prompt.data()); om.upload(output.data()); counts.upload(c.data());
    freq.put({0, .3f, -.7f, 1}); pres.put({1, 0, .3f, -.1f}); rep.put({1, 1.2f, .8f, 2});
    vt::ApplyPenalties(q, logits.tensor, pm.tensor, counts.tensor, om.tensor, freq.tensor, pres.tensor, rep.tensor);
    Buffer rows(q, DType::kI32, {M}), cols(q, DType::kI32, {M}), bias(q, DType::kF32, {M});
    int32_t rr[] = {0, 1, 0, 3, 0}, cc[] = {3, 2, 3, 32, 3};
    rows.upload(rr); cols.upload(cc); bias.put({1e10f, .2f, -1e10f, .3f, .25f});
    vt::ApplyLogitBias(q, logits.tensor, rows.tensor, cols.tensor, bias.tensor);
    auto result = logits.floats();
    vt::ApplyTokenMask(q, logits.tensor, rows.tensor, cols.tensor);
    vt::ApplyAllowedTokenIds(q, logits.tensor, pm.tensor);
    const auto masked = logits.floats(); result.insert(result.end(), masked.begin(), masked.end());
    return result;
  };
  Compare(run(gpu.q), run(cpu.q), 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: invalid sparse indices do not partially mutate logits") {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {1, 7}), rows(gpu.q, DType::kI32, {2}), cols(gpu.q, DType::kI32, {2}), bias(gpu.q, DType::kF32, {2});
  const auto original = xpu_test::Values(7); logits.put(original);
  const int32_t rr[] = {0, 1}, cc[] = {2, 2}; rows.upload(rr); cols.upload(cc); bias.put({1, 2});
  CHECK_THROWS_AS(vt::ApplyLogitBias(gpu.q, logits.tensor, rows.tensor, cols.tensor, bias.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
  CHECK_THROWS_AS(vt::ApplyTokenMask(gpu.q, logits.tensor, rows.tensor, cols.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
}
TEST_CASE("XPU sampling: stable top-k/top-p, ties and uneven vocabularies") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  for (int vocab : {1, 7, 257, 1025}) for (int mode : {0, 1, 2}) {
    CAPTURE(vocab);
    CAPTURE(mode);
    Buffer a(cpu.q, DType::kF32, {4, vocab}), b(gpu.q, DType::kF32, {4, vocab});
    Buffer kc(cpu.q, DType::kI32, {4}), kg(gpu.q, DType::kI32, {4});
    Buffer pc(cpu.q, DType::kF32, {4}), pg(gpu.q, DType::kF32, {4});
    const int32_t ks[] = {1, 3, vocab, 0}; kc.upload(ks); kg.upload(ks);
    pc.put({.05f, .37f, .83f, 1}); pg.put({.05f, .37f, .83f, 1});
    const auto values = xpu_test::Values(4 * vocab, 12, .2f); a.put(values); b.put(values);
    vt::ApplyTopKTopP(cpu.q, a.tensor, mode == 1 ? nullptr : &kc.tensor, mode == 0 ? nullptr : &pc.tensor);
    vt::ApplyTopKTopP(gpu.q, b.tensor, mode == 1 ? nullptr : &kg.tensor, mode == 0 ? nullptr : &pg.tensor);
    Compare(b.floats(), a.floats(), 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  CHECK(vt::xpu::GetMemoryInfo().sampling_workspace_bytes == 16 * 1024 * 1024);
}
TEST_CASE("XPU sampling: nucleus probability mass at the real vocabulary size") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int V = 248320, N = 4;
  Buffer logits(gpu.q, DType::kF32, {N, V}), top_p(gpu.q, DType::kF32, {N});
  // Unequal, strictly monotone rows make the nucleus boundary unambiguous.
  std::vector<float> values(N * V); const float ps[] = {.2f, .7f, .95f, 1};
  for (int row = 0; row < N; ++row) for (int j = 0; j < V; ++j) values[row * V + j] = float(j) / V * (row + 1);
  logits.put(values); top_p.upload(ps);
  vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &top_p.tensor);
  const auto masked = logits.floats();
  for (int row = 0; row < N; ++row) {
    double total = 0, kept = 0, first = 0;
    for (int j = 0; j < V; ++j) {
      const double probability = std::exp(double(values[row * V + j]) - (row + 1));
      total += probability;
      if (std::isfinite(masked[row * V + j])) { kept += probability; if (!first) first = probability; }
    }
    CHECK(kept / total >= ps[row] - 2e-6);
    CHECK((kept - first) / total <= ps[row] + 2e-6);
  }
}
TEST_CASE("XPU sampling: seeded exponential race and statistical distribution") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  constexpr int N = 8192, V = 4;
  Buffer pc(cpu.q, DType::kF32, {N, V}), pg(gpu.q, DType::kF32, {N, V});
  Buffer sc(cpu.q, DType::kI64, {N}), sg(gpu.q, DType::kI64, {N});
  Buffer ic(cpu.q, DType::kI64, {N}), ig(gpu.q, DType::kI64, {N});
  std::vector<float> probs(N * V); std::vector<int64_t> seeds(N);
  for (int i = 0; i < N; ++i) { seeds[i] = 977 + i; for (int j = 0; j < V; ++j) probs[i * V + j] = .1f * (j + 1); }
  pc.put(probs); pg.put(probs); sc.upload(seeds.data()); sg.upload(seeds.data());
  vt::RandomSample(cpu.q, ic.tensor, pc.tensor, sc.tensor); vt::RandomSample(gpu.q, ig.tensor, pg.tensor, sg.tensor);
  const auto first = ig.download(); xpu_test::SameBytes(first, ic.download());
  vt::RandomSample(gpu.q, ig.tensor, pg.tensor, sg.tensor); xpu_test::SameBytes(first, ig.download());
  int counts[V] = {};
  for (int i = 0; i < N; ++i) { int64_t token; std::memcpy(&token, first.data() + i * sizeof(token), sizeof(token)); REQUIRE(token >= 0); REQUIRE(token < V); ++counts[token]; }
  double chi_squared = 0;
  for (int j = 0; j < V; ++j) { const double expected = N * .1 * (j + 1); chi_squared += (counts[j] - expected) * (counts[j] - expected) / expected; }
  CHECK(chi_squared < 24); // 3 degrees of freedom, fixed seed corpus.
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: top-k and top-p draws follow the truncated distribution") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int N = 6144, V = 8;
  const float weights[V] = {.01f, .02f, .04f, .08f, .12f, .18f, .24f, .31f};
  Buffer logits(gpu.q, DType::kF32, {N, V}), probs(gpu.q, DType::kF32, {N, V});
  Buffer ks(gpu.q, DType::kI32, {N}), ps(gpu.q, DType::kF32, {N});
  Buffer seeds(gpu.q, DType::kI64, {N}), ids(gpu.q, DType::kI64, {N});
  std::vector<float> values(N * V), p(N);
  std::vector<int32_t> k(N); std::vector<int64_t> seed(N);
  for (int i = 0; i < N; ++i) {
    // k-only and p-only retain {5,6,7}; combined filtering retains {6,7}.
    k[i] = i % 3 == 0 ? 3 : (i % 3 == 1 ? V : 4);
    p[i] = i % 3 == 0 ? 1.f : .6f; seed[i] = 419 + i;
    for (int j = 0; j < V; ++j) values[i * V + j] = std::log(weights[j]);
  }
  logits.put(values); ks.upload(k.data()); ps.put(p); seeds.upload(seed.data());
  vt::ApplyTopKTopP(gpu.q, logits.tensor, &ks.tensor, &ps.tensor);
  vt::ComputeProbs(gpu.q, probs.tensor, logits.tensor);
  vt::RandomSample(gpu.q, ids.tensor, probs.tensor, seeds.tensor);
  const auto bytes = ids.download();
  int counts[3][V] = {};
  for (int i = 0; i < N; ++i) {
    int64_t token; std::memcpy(&token, bytes.data() + i * sizeof(token), sizeof(token));
    REQUIRE(token >= (i % 3 == 2 ? 6 : 5)); REQUIRE(token < V); ++counts[i % 3][token];
  }
  for (int mode = 0; mode < 3; ++mode) {
    const int first = mode == 2 ? 6 : 5;
    const double sum = mode == 2 ? .55 : .73;
    double chi_squared = 0;
    for (int j = first; j < V; ++j) {
      const double expected = (N / 3) * weights[j] / sum;
      chi_squared += std::pow(counts[mode][j] - expected, 2) / expected;
    }
    CAPTURE(mode); CHECK(chi_squared < 24); // fixed seeds, at most 2 degrees of freedom
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: zero requests are a no-op") {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {1, 17}), params(gpu.q, DType::kF32, {1});
  Buffer ids(gpu.q, DType::kI64, {1}), seeds(gpu.q, DType::kI64, {1});
  // VT constructs positive allocations; empty batch views narrow their shape.
  for (auto* b : {&logits, &params, &ids, &seeds}) b->tensor.shape[0] = 0;
  vt::ApplyTemperature(gpu.q, logits.tensor, params.tensor, true);
  vt::ComputeProbs(gpu.q, logits.tensor, logits.tensor);
  vt::ComputeLogprobs(gpu.q, logits.tensor, logits.tensor);
  vt::ApplyMinP(gpu.q, logits.tensor, params.tensor);
  vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &params.tensor);
  vt::RandomSample(gpu.q, ids.tensor, logits.tensor, seeds.tensor);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: complete mixed batch and seeded row reordering") {
  Queue cpu(vt::DeviceType::kCPU), gpu(vt::DeviceType::kXPU);
  vllm::v1::Sampler cs, gs;
  auto run = [](vt::Queue& q, vllm::v1::Sampler& sampler, std::vector<int> order) {
    constexpr int V = 257;
    Buffer logits(q, DType::kF32, {int64_t(order.size()), V});
    auto base = xpu_test::Values(4 * V, 19, .2f); std::vector<float> values;
    vllm::v1::SamplingMetadata sm; sm.all_greedy = false; sm.all_random = false;
    sm.temperature.emplace(); sm.top_k.emplace(); sm.top_p.emplace();
    sm.no_penalties = false; sm.prompt_token_ids.emplace();
    for (size_t r = 0; r < order.size(); ++r) {
      const int id = order[r]; values.insert(values.end(), base.begin() + id * V, base.begin() + (id + 1) * V);
      sm.temperature->push_back(id == 0 ? 0 : .7f); sm.top_k->push_back(20); sm.top_p->push_back(.87f);
      sm.generators[int(r)] = 719 + id; sm.output_token_positions.push_back(id + 3);
      sm.frequency_penalties.push_back(.2f); sm.presence_penalties.push_back(.1f); sm.repetition_penalties.push_back(1.1f);
      sm.prompt_token_ids->push_back({2, 3, 4}); sm.output_token_ids.push_back({5, 5, 7});
      sm.min_p.push_back(.1f); sm.logit_bias[int(r)][8] = 1.2f;
      sm.bad_words_token_ids[int(r)] = {{9}};
    }
    logits.put(values);
    return sampler.forward(q, logits.tensor, sm).sampled_token_ids;
  };
  const auto reference = run(cpu.q, cs, {0, 1, 2, 3}), actual = run(gpu.q, gs, {0, 1, 2, 3});
  CHECK(actual == reference);
  const auto reordered = run(gpu.q, gs, {3, 0, 2, 1});
  CHECK(reordered[0] == actual[3]); CHECK(reordered[1] == actual[0]);
  CHECK(reordered[2] == actual[2]); CHECK(reordered[3] == actual[1]);
  CHECK(vt::GetReferenceTierHits() == 0);
}
TEST_CASE("XPU sampling: workspace admission fails without modifying logits"
          * doctest::skip(!std::getenv("VT_B70_LOW_MEMORY_TEST"))) {
  Queue gpu(vt::DeviceType::kXPU);
  Buffer logits(gpu.q, DType::kF32, {4, 257}), p(gpu.q, DType::kF32, {4});
  const auto original = xpu_test::Values(4 * 257); logits.put(original); p.put({.9f, .9f, .9f, .9f});
  CHECK_THROWS_AS(vt::ApplyTopKTopP(gpu.q, logits.tensor, nullptr, &p.tensor), std::runtime_error);
  CHECK(logits.floats() == original);
  CHECK(vt::xpu::GetMemoryInfo().sampling_workspace_bytes == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}
