#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"

namespace gptq4_model_bench {

inline vllm::v1::CommonAttentionMetadata AttentionMetadata(
    int query_len, int context, int block_size = 128) {
  if (query_len <= 0 || context < 0 || block_size <= 0)
    throw std::invalid_argument("invalid GPTQ4 benchmark attention length");
  vllm::v1::CommonAttentionMetadata meta;
  meta.num_reqs = 1;
  meta.num_actual_tokens = query_len;
  meta.query_start_loc = {0, query_len};
  meta.query_start_loc_cpu = meta.query_start_loc;
  meta.seq_lens = {context + query_len};
  meta.seq_lens_cpu = meta.seq_lens;
  meta.max_query_len = query_len;
  meta.max_seq_len = context + query_len;
  meta.block_table_num_cols =
      (meta.max_seq_len + block_size - 1) / block_size;
  meta.block_table_tensor.resize(meta.block_table_num_cols);
  for (int block = 0; block < meta.block_table_num_cols; ++block)
    meta.block_table_tensor[block] = block;
  for (int token = 0; token < query_len; ++token)
    meta.slot_mapping.push_back(context + token);
  meta.causal = true;
  return meta;
}

inline vllm::v1::GDNAttentionMetadata GdnMetadata(int query_len,
                                                   bool initial) {
  vllm::v1::GDNAttentionMetadata meta;
  meta.num_actual_tokens = query_len;
  meta.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  meta.non_spec_query_start_loc = std::vector<int32_t>{0, query_len};
  if (initial) {
    meta.num_decodes = 1;
    meta.num_decode_tokens = query_len;
  } else {
    meta.num_prefills = 1;
    meta.num_prefill_tokens = query_len;
    meta.has_initial_state = std::vector<uint8_t>{0};
    meta.prefill_query_start_loc = std::vector<int32_t>{0, query_len};
    meta.prefill_state_indices = std::vector<int32_t>{0};
    meta.prefill_has_initial_state = std::vector<uint8_t>{0};
    const auto conv = vllm::v1::ComputeCausalConv1dMetadata(
        *meta.non_spec_query_start_loc);
    meta.batch_ptr = conv.batch_ptr;
    meta.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }
  return meta;
}

}  // namespace gptq4_model_bench
