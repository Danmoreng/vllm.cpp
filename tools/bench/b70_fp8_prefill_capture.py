"""Export one real 4096-token Xe2 FP8 attention call from pinned vLLM.

Run in the pinned Python image with the checkpoint and output directory mounted.
The exported bytes are intentionally kept outside the source tree.
"""

import argparse
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

import torch
from vllm.v1.attention.backends import flash_attn as fa

out_dir = Path(os.environ.get('VT_B70_FP8_PREFILL_CAPTURE_DIR', '/capture'))
target_tokens = 4096
original = fa.flash_attn_varlen_func
captured = False


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wrapped(*args, **kwargs):
    global captured
    q = kwargs.get('q', args[0] if args else None)
    result = original(*args, **kwargs)
    if captured or q is None or q.shape[0] != 4096:
        return result
    k, v = kwargs['k'], kwargs['v']
    table = kwargs['block_table'].detach().cpu().contiguous()
    length = int(kwargs['seqused_k'].detach().cpu().flatten()[0])
    page = k.shape[1]
    count = (length + page - 1) // page
    page_ids = table[0, :count].to(torch.long)
    if (page_ids < 0).any() or (page_ids >= k.shape[0]).any():
        raise RuntimeError('invalid captured page ids')
    take = page_ids.to(k.device)
    output = kwargs.get('out') if kwargs.get('out') is not None else result
    tensors = {
        'q': q.detach().cpu().contiguous(),
        'k': k.index_select(0, take).detach().cpu().contiguous().view(torch.uint8),
        'v': v.index_select(0, take).detach().cpu().contiguous().view(torch.uint8),
        'output': output.detach().cpu().contiguous(),
    }
    files = {}
    for name, tensor in tensors.items():
        path = out_dir / (name + '.bin')
        tensor.numpy().tofile(path)
        files[name] = {'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha(path),
                       'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}
    def scalar(name):
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        return value
    meta = {
        'files': files,
        'capture_script_sha256': sha(Path(__file__)),
        'model_revision': os.environ.get('VT_B70_MODEL_REVISION'),
        'requested_block_size': 64, 'actual_page_size': page,
        'q_original_shape': list(q.shape), 'q_original_stride': list(q.stride()),
        'output_original_shape': list(output.shape),
        'output_original_stride': list(output.stride()),
        'k_original_shape': list(k.shape), 'k_original_stride': list(k.stride()),
        'v_original_shape': list(v.shape), 'v_original_stride': list(v.stride()),
        'table_original_shape': list(kwargs['block_table'].shape),
        'table_original_stride': list(kwargs['block_table'].stride()),
        'selected_original_pages': page_ids.tolist(),
        'repacked_table': list(range(count)),
        'seqused_k': length, 'cu_seqlens_q': kwargs['cu_seqlens_q'].detach().cpu().tolist(),
        'max_seqlen_q': scalar('max_seqlen_q'), 'max_seqlen_k': scalar('max_seqlen_k'),
        'softmax_scale': scalar('softmax_scale'), 'k_descale': scalar('k_descale'),
        'v_descale': scalar('v_descale'), 'causal': scalar('causal'),
        'window_size': scalar('window_size'), 'softcap': scalar('softcap'),
        'fa_version': scalar('fa_version'),
    }
    binary = find_spec('vllm_xpu_kernels._vllm_fa2_C')
    if binary is not None and binary.origin is not None:
        meta['python_dso_sha256'] = sha(Path(binary.origin))
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    captured = True
    print('CAPTURED_ATTENTION ' + json.dumps({
        'q_shape': meta['q_original_shape'], 'k_shape': meta['k_original_shape'],
        'page': page, 'pages': count, 'length': length,
        'scale': meta['softmax_scale'], 'k_descale': meta['k_descale'],
        'v_descale': meta['v_descale'], 'causal': meta['causal']}), flush=True)
    return result


fa.flash_attn_varlen_func = wrapped

from vllm import LLM, SamplingParams, TokensPrompt

def main():
    global out_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='/models')
    parser.add_argument('--out', type=Path, default=out_dir)
    parser.add_argument('--revision', default='a47b0c6f0d756bc394c4cc629d5b0ded1acc7001')
    args = parser.parse_args()
    out_dir = args.out
    os.environ['VT_B70_FP8_PREFILL_CAPTURE_DIR'] = str(out_dir)
    os.environ['VT_B70_MODEL_REVISION'] = args.revision
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        model=args.model, tokenizer=args.model, revision=args.revision,
        quantization='gptq', dtype='float16', kv_cache_dtype='fp8',
        max_model_len=8192, max_num_seqs=1, max_num_batched_tokens=target_tokens,
        block_size=64, gpu_memory_utilization=0.85,
        enable_prefix_caching=False, enforce_eager=True, mamba_cache_mode='align',
    )
    print('CONFIG ' + json.dumps({'block_size': llm.llm_engine.vllm_config.cache_config.block_size,
                                  'kv_dtype': str(llm.llm_engine.vllm_config.cache_config.cache_dtype)}), flush=True)
    llm.generate([TokensPrompt(prompt_token_ids=[100 + i % 11 for i in range(target_tokens)])],
                 SamplingParams(temperature=0.0, max_tokens=1, min_tokens=1, ignore_eos=True))
    if not (out_dir / 'metadata.json').exists():
        raise RuntimeError('attention wrapper was not called for the 4096 prefill')


if __name__ == "__main__":
    main()
