#include <doctest/doctest.h>
#include "vt/gdn_fp16_plan.h"
#include <vector>

TEST_CASE("GDN FP16 C1 layout: exact lengths tail alignment and stable 4K budget") {
  for (int tokens : {1, 3, 127, 128, 129, 256, 4096}) {
    const auto p = vt::PlanGdnFp16C1(tokens);
    CHECK(p.tokens == tokens);
    CHECK(p.capacity == tokens + 63);
    CHECK(p.q_offset == 0);
    CHECK(p.k_offset == size_t(p.capacity) * 16 * 128 * 2);
    CHECK(p.v_offset == 2 * p.k_offset);
    CHECK(p.a_matrix_offset == p.v_offset + size_t(p.capacity) * 48 * 128 * 2);
    CHECK(p.w_offset == p.a_matrix_offset + size_t(p.capacity) * 48 * 64 * 2);
    CHECK(p.u_offset == p.w_offset + size_t(p.capacity) * 48 * 128 * 2);
    CHECK(p.raw_a_offset == p.u_offset + size_t(p.capacity) * 48 * 128 * 2);
    CHECK(p.beta_offset == p.raw_a_offset + size_t(p.capacity) * 48 * 4);
    CHECK(p.bias_offset == p.beta_offset + size_t(p.capacity) * 48 * 4);
    CHECK(p.index_offset == p.bias_offset + 128);
    CHECK(p.initial_offset == p.index_offset + 64);
    CHECK(p.bytes == size_t(p.capacity) * 51584 + 256);
    CHECK(p.bytes <= vt::kGdnFp16ReservationBytes);
    for (size_t offset : {p.q_offset, p.k_offset, p.v_offset, p.a_matrix_offset,
                         p.w_offset, p.u_offset, p.raw_a_offset, p.beta_offset,
                         p.bias_offset, p.index_offset, p.initial_offset, p.bytes})
      CHECK(offset % 64 == 0);
  }
  CHECK(vt::PlanGdnFp16C1(4096).bytes == 214538112);
  CHECK(vt::kGdnFp16ReservationBytes == 230686720);
  CHECK_THROWS_AS(vt::PlanGdnFp16C1(0), std::invalid_argument);
  CHECK_THROWS_AS(vt::PlanGdnFp16C1(4097), std::invalid_argument);
}
