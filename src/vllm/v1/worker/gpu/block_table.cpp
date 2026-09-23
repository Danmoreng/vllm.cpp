// Ported from: vllm/v1/worker/block_table.py @ e24d1b24
// See include/vllm/v1/worker/gpu/block_table.h for the scope + the
// stale-brief / host-array-for-device-tensor deviations.

#include "vllm/v1/worker/gpu/block_table.h"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

#include "vllm/v1/kv_cache_interface.h"

namespace vllm::v1 {

namespace {

// Upstream vllm/v1/attention/backends/utils.py PAD_SLOT_ID.
constexpr int64_t kPadSlotId = -1;

// Ceiling division for non-negative operands (upstream vllm.utils cdiv).
int cdiv(int a, int b) { return (a + b - 1) / b; }

}  // namespace

int get_block_table_width(int max_num_blocks, int block_size,
                          std::optional<int> kernel_block_size,
                          std::optional<int> token_alignment) {
  // block_table.py:29-49.
  const int kbs = kernel_block_size.has_value() ? *kernel_block_size : block_size;
  if (kbs <= 0 || block_size % kbs != 0) {
    throw std::invalid_argument("kernel_block_size " + std::to_string(kbs) +
                                " must divide block_size " +
                                std::to_string(block_size) + " evenly");
  }
  if (token_alignment.has_value()) {
    if (*token_alignment <= 0) {
      throw std::invalid_argument("token_alignment must be positive");
    }
    const int block_alignment =
        *token_alignment / std::gcd(*token_alignment, block_size);
    max_num_blocks = cdiv(max_num_blocks, block_alignment) * block_alignment;
  }
  return max_num_blocks * block_size / kbs;
}

BlockTableGeometry block_table_geometry(const KVCacheConfig& kv_cache_config,
                                        int max_model_len) {
  // gpu_model_runner.py:7340-7360. The runner has no encoder-decoder path, so
  // upstream's max(max_model_len, max_encoder_len) is max_model_len, and it
  // keeps every group (no ENCODER_ONLY_ATTENTION group reaches this runner).
  BlockTableGeometry g;
  for (const KVCacheGroupSpec& group : kv_cache_config.kv_cache_groups) {
    const KVCacheSpec& spec = *group.kv_cache_spec;
    g.block_sizes.push_back(spec.block_size);
    g.slot_mapping_modes.push_back(spec.kind() == KVCacheSpecKind::kMamba
                                       ? SlotMappingMode::kNone
                                       : SlotMappingMode::kTokenToKvSlot);
    g.max_num_blocks.push_back(spec.max_num_blocks_per_req(max_model_len));
  }
  return g;
}

BlockTable::BlockTable(int block_size, int max_num_reqs,
                       int max_num_blocks_per_req, int max_num_batched_tokens,
                       int kernel_block_size, int cp_kv_cache_interleave_size)
    : max_num_reqs(max_num_reqs),
      max_num_batched_tokens(max_num_batched_tokens),
      cp_kv_cache_interleave_size_(cp_kv_cache_interleave_size),
      // No dcp/pcp process groups at T0 -> world size 1, rank 0.
      total_cp_world_size_(1),
      total_cp_rank_(0) {
  if (kernel_block_size == block_size) {
    // Standard case: allocation and computation use the same block size.
    this->block_size = block_size;
    this->blocks_per_kv_block = 1;
    this->use_hybrid_blocks = false;
  } else {
    // Hybrid case: allocation block size differs from kernel block size.
    if (block_size % kernel_block_size != 0) {
      throw std::invalid_argument(
          "kernel_block_size must divide block_size evenly");
    }
    this->block_size = kernel_block_size;
    this->blocks_per_kv_block = block_size / kernel_block_size;
    this->use_hybrid_blocks = true;
  }

  this->max_num_blocks_per_req = max_num_blocks_per_req * this->blocks_per_kv_block;

  const size_t table_size =
      static_cast<size_t>(this->max_num_reqs) * this->max_num_blocks_per_req;
  block_table_cpu_.assign(table_size, 0);
  block_table_device_.assign(table_size, 0);
  num_blocks_per_row.assign(static_cast<size_t>(this->max_num_reqs), 0);
  slot_mapping_.assign(static_cast<size_t>(this->max_num_batched_tokens), 0);
}

std::vector<int> BlockTable::map_to_kernel_blocks(
    const std::vector<int>& block_ids, int blocks_per_kv_block) {
  if (blocks_per_kv_block == 1) {
    return block_ids;
  }
  std::vector<int> kernel_block_ids;
  kernel_block_ids.reserve(block_ids.size() *
                           static_cast<size_t>(blocks_per_kv_block));
  for (int b : block_ids) {
    for (int k = 0; k < blocks_per_kv_block; ++k) {
      kernel_block_ids.push_back(b * blocks_per_kv_block + k);
    }
  }
  return kernel_block_ids;
}

void BlockTable::append_row(const std::vector<int>& block_ids, int row_idx) {
  if (block_ids.empty()) {
    return;
  }
  const std::vector<int>& ids =
      use_hybrid_blocks ? map_to_kernel_blocks(block_ids, blocks_per_kv_block)
                        : block_ids;
  const int num_blocks = static_cast<int>(ids.size());
  const int start = num_blocks_per_row[static_cast<size_t>(row_idx)];
  // gpu/block_table.py:125-131: refuse the write before any state changes.
  const int end = start + num_blocks;
  if (end > max_num_blocks_per_req) {
    throw std::runtime_error(
        "Block table write for request " + std::to_string(row_idx) +
        ", group " + std::to_string(group_index) +
        " exceeds row capacity (" + std::to_string(end) + " > " +
        std::to_string(max_num_blocks_per_req) + ")");
  }
  num_blocks_per_row[static_cast<size_t>(row_idx)] = end;
  const size_t base = static_cast<size_t>(row_idx) * max_num_blocks_per_req;
  for (int i = 0; i < num_blocks; ++i) {
    block_table_cpu_[base + start + i] = static_cast<int32_t>(ids[i]);
  }
}

void BlockTable::add_row(const std::vector<int>& block_ids, int row_idx) {
  num_blocks_per_row[static_cast<size_t>(row_idx)] = 0;
  append_row(block_ids, row_idx);
}

void BlockTable::clear_row(int row_idx) {
  const int num_blocks = num_blocks_per_row[static_cast<size_t>(row_idx)];
  const size_t base = static_cast<size_t>(row_idx) * max_num_blocks_per_req;
  for (int i = 0; i < num_blocks; ++i) {
    block_table_cpu_[base + i] = 0;
  }
  num_blocks_per_row[static_cast<size_t>(row_idx)] = 0;
}

void BlockTable::move_row(int src, int tgt) {
  const int num_blocks = num_blocks_per_row[static_cast<size_t>(src)];
  const size_t src_base = static_cast<size_t>(src) * max_num_blocks_per_req;
  const size_t tgt_base = static_cast<size_t>(tgt) * max_num_blocks_per_req;
  for (int i = 0; i < num_blocks; ++i) {
    block_table_cpu_[tgt_base + i] = block_table_cpu_[src_base + i];
  }
  num_blocks_per_row[static_cast<size_t>(tgt)] = num_blocks;
}

void BlockTable::swap_row(int src, int tgt) {
  std::swap(num_blocks_per_row[static_cast<size_t>(src)],
            num_blocks_per_row[static_cast<size_t>(tgt)]);
  const size_t src_base = static_cast<size_t>(src) * max_num_blocks_per_req;
  const size_t tgt_base = static_cast<size_t>(tgt) * max_num_blocks_per_req;
  for (int i = 0; i < max_num_blocks_per_req; ++i) {
    std::swap(block_table_cpu_[src_base + i], block_table_cpu_[tgt_base + i]);
  }
}

void BlockTable::compute_slot_mapping(int num_reqs,
                                      const std::vector<int32_t>& query_start_loc,
                                      const std::vector<int64_t>& positions) {
  const int64_t virtual_block_size =
      static_cast<int64_t>(block_size) * total_cp_world_size_;
  // Reads the "device" buffer (upstream reads block_table.gpu) -> requires a
  // prior commit_block_table.
  for (int req_idx = 0; req_idx < num_reqs; ++req_idx) {
    const int64_t start_idx = query_start_loc[static_cast<size_t>(req_idx)];
    const int64_t end_idx = query_start_loc[static_cast<size_t>(req_idx) + 1];
    const size_t row_offset =
        static_cast<size_t>(req_idx) * max_num_blocks_per_req;
    for (int64_t i = start_idx; i < end_idx; ++i) {
      const int64_t pos = positions[static_cast<size_t>(i)];
      const int64_t block_index = pos / virtual_block_size;
      const int64_t block_number =
          block_table_device_[row_offset + static_cast<size_t>(block_index)];
      const int64_t virtual_block_offset = pos - block_index * virtual_block_size;
      const bool is_local =
          (virtual_block_offset / cp_kv_cache_interleave_size_) %
              total_cp_world_size_ ==
          total_cp_rank_;
      const int64_t local_block_offset =
          (virtual_block_offset /
           (static_cast<int64_t>(total_cp_world_size_) *
            cp_kv_cache_interleave_size_)) *
              cp_kv_cache_interleave_size_ +
          (virtual_block_offset % cp_kv_cache_interleave_size_);
      int64_t slot_id = block_number * block_size + local_block_offset;
      if (!is_local) {
        slot_id = kPadSlotId;
      }
      slot_mapping_[static_cast<size_t>(i)] = slot_id;
    }
  }
  // TAIL-PAD DEVIATION (rescan-lost-lanes-2026-07-16 §1 item c): upstream's
  // Triton kernel pads slot_mapping[num_tokens:max_num_batched_tokens] to
  // PAD_SLOT_ID because vLLM's captured decode graph reads the persistent
  // slot_mapping buffer's padded tail. Our port instead builds the padded
  // decode-graph inputs explicitly (qwen3_5.cpp BuildPaddedDecode re-pads to the
  // captured batch size with -1), and the only other consumer slices [0, total)
  // (prepare_inputs.cpp). The tail-pad is therefore dead work — ~2×(max_num_
  // batched_tokens − total) int64 writes/step across the two KV groups — and is
  // dropped. The fill is bounded to [0, num_tokens); the tail keeps its prior
  // value and is never read.
}

void BlockTable::commit_block_table(int num_reqs) {
  const size_t count = static_cast<size_t>(num_reqs) * max_num_blocks_per_req;
  for (size_t i = 0; i < count; ++i) {
    block_table_device_[i] = block_table_cpu_[i];
  }
}

void BlockTable::clear() {
  std::fill(block_table_device_.begin(), block_table_device_.end(), 0);
  std::fill(block_table_cpu_.begin(), block_table_cpu_.end(), 0);
}

// ─── MultiGroupBlockTable ───────────────────────────────────────────────────

MultiGroupBlockTable::MultiGroupBlockTable(
    int max_num_reqs, int max_model_len, int max_num_batched_tokens,
    std::vector<int> block_sizes, std::vector<int> kernel_block_sizes,
    std::optional<std::vector<int>> max_num_blocks,
    int cp_kv_cache_interleave_size,
    std::optional<std::vector<SlotMappingMode>> slot_mapping_modes) {
  if (kernel_block_sizes.size() != block_sizes.size()) {
    throw std::invalid_argument(
        "kernel_block_sizes length must match block_sizes length");
  }
  // block_table.py:308-315.
  std::vector<SlotMappingMode> modes =
      slot_mapping_modes.has_value()
          ? std::move(*slot_mapping_modes)
          : std::vector<SlotMappingMode>(block_sizes.size(),
                                         SlotMappingMode::kTokenToKvSlot);
  if (modes.size() != block_sizes.size()) {
    throw std::invalid_argument(
        "slot_mapping_modes length must match block_sizes length");
  }

  std::vector<int> num_blocks;
  if (max_num_blocks.has_value()) {
    num_blocks = std::move(*max_num_blocks);
  } else {
    // total_cp_world_size == 1 at T0 (get_total_cp_world_size fallback).
    for (int bs : block_sizes) {
      num_blocks.push_back(cdiv(max_model_len, bs));
    }
  }
  if (num_blocks.size() != block_sizes.size()) {
    throw std::invalid_argument(
        "max_num_blocks length must match block_sizes length");
  }

  // block_table.py:322-331: a kTokenToKvSlot row is rounded to 128 tokens, a
  // kNone (Mamba/GDN) row keeps the width its spec asked for.
  for (size_t i = 0; i < num_blocks.size(); ++i) {
    num_blocks[i] =
        modes[i] == SlotMappingMode::kNone
            ? get_block_table_width(num_blocks[i], block_sizes[i], std::nullopt,
                                    /*token_alignment=*/std::nullopt)
            : get_block_table_width(num_blocks[i], block_sizes[i]);
  }

  block_tables.reserve(block_sizes.size());
  for (size_t i = 0; i < block_sizes.size(); ++i) {
    block_tables.emplace_back(block_sizes[i], max_num_reqs, num_blocks[i],
                              max_num_batched_tokens, kernel_block_sizes[i],
                              cp_kv_cache_interleave_size);
    block_tables.back().group_index = static_cast<int>(i);
  }
}

void MultiGroupBlockTable::append_row(
    const std::vector<std::vector<int>>& block_ids, int row_idx) {
  for (size_t i = 0; i < block_tables.size(); ++i) {
    block_tables[i].append_row(block_ids[i], row_idx);
  }
}

void MultiGroupBlockTable::add_row(
    const std::vector<std::vector<int>>& block_ids, int row_idx) {
  for (size_t i = 0; i < block_tables.size(); ++i) {
    block_tables[i].add_row(block_ids[i], row_idx);
  }
}

void MultiGroupBlockTable::clear_row(int row_idx) {
  for (auto& block_table : block_tables) {
    block_table.clear_row(row_idx);
  }
}

void MultiGroupBlockTable::move_row(int src, int tgt) {
  for (auto& block_table : block_tables) {
    block_table.move_row(src, tgt);
  }
}

void MultiGroupBlockTable::swap_row(int src, int tgt) {
  for (auto& block_table : block_tables) {
    block_table.swap_row(src, tgt);
  }
}

void MultiGroupBlockTable::compute_slot_mapping(
    int num_reqs, const std::vector<int32_t>& query_start_loc,
    const std::vector<int64_t>& positions) {
  for (auto& block_table : block_tables) {
    block_table.compute_slot_mapping(num_reqs, query_start_loc, positions);
  }
}

void MultiGroupBlockTable::commit_block_table(int num_reqs) {
  for (auto& block_table : block_tables) {
    block_table.commit_block_table(num_reqs);
  }
}

void MultiGroupBlockTable::clear() {
  for (auto& block_table : block_tables) {
    block_table.clear();
  }
}

}  // namespace vllm::v1
