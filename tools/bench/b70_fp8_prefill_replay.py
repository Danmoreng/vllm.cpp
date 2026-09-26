"""Replay one captured 4096-token FP8 attention operator in pinned Python."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dir', type=Path, required=True)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--samples', type=int, default=5)
    args = parser.parse_args()
    meta = json.loads((args.dir / 'metadata.json').read_text())

    def read(name, dtype):
        item = meta['files'][name]
        path = args.dir / item['name']
        data = path.read_bytes()
        if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
            raise RuntimeError(f'{name}: fixture hash/size mismatch')
        return np.frombuffer(data, dtype=dtype).reshape(item['shape']).copy()

    q_host = read('q', np.float16)
    k_host = read('k', np.uint8)
    v_host = read('v', np.uint8)
    expected = read('output', np.float16)
    tokens, q_heads, dim = q_host.shape
    pages, page, kv_heads, k_dim = k_host.shape
    page_ids = meta['selected_original_pages']
    physical_pages = meta['k_original_shape'][0]
    if (tokens != 4096 or k_dim != dim or v_host.shape != k_host.shape or
            len(page_ids) != pages or max(page_ids) >= physical_pages or
            min(page_ids) < 0 or meta['v_original_stride'] != meta['k_original_stride'] or
            meta['k_original_stride'] != [page * 2 * kv_heads * dim,
                                          2 * kv_heads * dim, 2 * dim, 1] or
            meta['table_original_stride'][1] != 1):
        raise RuntimeError('capture has an unsupported geometry/layout')

    physical = np.zeros((physical_pages, page, kv_heads, 2 * dim), dtype=np.uint8)
    physical[page_ids, :, :, :dim] = k_host
    physical[page_ids, :, :, dim:] = v_host
    q = torch.from_numpy(q_host).to('xpu')
    kv = torch.from_numpy(physical).to('xpu')
    k = kv[..., :dim].view(torch.float8_e4m3fn)
    v = kv[..., dim:].view(torch.float8_e4m3fn)
    out = torch.empty_like(q)
    table_width = meta['table_original_shape'][1]
    table_values = page_ids + [-1] * (table_width - pages)
    table = torch.tensor([table_values], dtype=torch.int32, device='xpu')
    cu = torch.tensor(meta['cu_seqlens_q'], dtype=torch.int32, device='xpu')
    used = torch.tensor([meta['seqused_k']], dtype=torch.int32, device='xpu')
    def descale(name):
        values = np.asarray(meta[name], dtype=np.float32)
        if not np.all(values == values.flat[0]):
            raise RuntimeError(f'{name}: pinned operator expects a broadcast scalar')
        return torch.tensor(float(values.flat[0]), dtype=torch.float32,
                            device='xpu').expand(*values.shape)

    k_scale = descale('k_descale')
    v_scale = descale('v_descale')

    def call():
        return flash_attn_varlen_func(
            q=q, k=k, v=v, out=out, cu_seqlens_q=cu,
            max_seqlen_q=meta['max_seqlen_q'], seqused_k=used,
            max_seqlen_k=meta['max_seqlen_k'],
            softmax_scale=meta['softmax_scale'], causal=meta['causal'],
            window_size=meta['window_size'], block_table=table,
            softcap=meta['softcap'], fa_version=meta['fa_version'],
            k_descale=k_scale, v_descale=v_scale)

    call()
    torch.xpu.synchronize()
    actual = out.cpu().numpy().astype(np.float32)
    reference = expected.astype(np.float32)
    delta = actual - reference
    rms = np.sqrt(np.sum(delta.astype(np.float64)**2) /
                  np.sum(reference.astype(np.float64)**2))
    worst = np.max(np.abs(delta))
    peak = np.max(np.abs(reference))
    print(json.dumps({'event': 'python_replay_quality', 'rms': float(rms),
                      'max_abs': float(worst), 'peak': float(peak),
                      'k_stride': list(k.stride()), 'v_stride': list(v.stride())}),
          flush=True)
    if rms > 0.001 or worst > 3e-6 + 0.002 * peak:
        raise RuntimeError('Python replay differs from captured output')

    for _ in range(args.warmups):
        call()
        torch.xpu.synchronize()
    measurements = []
    for _ in range(args.samples):
        start = time.perf_counter()
        call()
        torch.xpu.synchronize()
        measurements.append((time.perf_counter() - start) * 1000)
    print(json.dumps({'event': 'python_prefill_operator', 'page': page,
                      'tokens': tokens, 'samples_ms': measurements,
                      'median_ms': float(np.median(measurements))}), flush=True)


if __name__ == '__main__':
    main()
