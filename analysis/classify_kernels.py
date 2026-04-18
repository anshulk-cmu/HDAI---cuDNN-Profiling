"""Map a raw cuDNN/cuBLAS/CUDA kernel name to a coarse category.

Categories are chosen to make the final cross-model breakdown chart
(brief Phase 4) legible on a single bar: six buckets, no overlap.

Keyword order matters: more specific patterns must appear before
more general ones (e.g. 'winograd' before plain 'conv').
"""


def classify(name):
    n = name.lower()

    # --- convolution sub-families, tested most-specific first ---
    if 'winograd' in n:
        return 'conv_winograd'
    if 'depthwise' in n or 'conv_depthwise' in n:
        return 'conv_depthwise'
    if 'implicit_gemm' in n or 'implicit_precomp' in n:
        return 'conv_implicit_gemm'
    if 'implicit_convolve' in n:
        return 'conv_implicit_gemm'
    if 'fprop' in n and ('cutlass' in n or 'xmma' in n):
        return 'conv_implicit_gemm'
    if 'dgrad' in n or 'wgrad' in n:
        return 'conv_backward'

    # --- fused attention (FlashAttention / PyTorch Efficient Attention) ---
    # Must run before matmul_tensor_core: a future FMHA variant carrying a
    # 'tensorop'/'s1688' tile tag would otherwise land in matmul_tensor_core.
    if 'fmha' in n or 'attentionkernel' in n or 'flashattn' in n:
        return 'fused_attention'

    # --- matmul (cuBLAS / CUTLASS GEMM not inside a conv kernel name) ---
    if 'hmma' in n or 'tensorop_s16816gemm' in n or 'tensorop_s1688gemm' in n:
        return 'matmul_tensor_core'
    if 'gemm' in n:
        return 'matmul_fp32'

    # --- normalisation ---
    if 'bn_fw' in n or 'batch_norm' in n or 'batchnorm' in n or 'layer_norm' in n:
        return 'norm'

    # --- layout converters (NCHW <-> NHWC) ---
    if 'nchwtonhwc' in n or 'nhwctonchw' in n:
        return 'layout_convert'

    # --- pooling / rnn / softmax / elementwise / reduce / embedding ---
    if 'pool' in n:
        return 'pool'
    if 'rnn' in n or 'lstm' in n or 'gru' in n:
        return 'rnn'
    if 'softmax' in n:
        return 'softmax'
    if 'gather' in n:
        return 'embed_gather'
    if 'elementwise' in n or 'vectorized_elementwise' in n:
        return 'elementwise'
    if 'reduce' in n:
        return 'reduce'

    return 'other'


CATEGORY_ORDER = [
    'conv_winograd',
    'conv_implicit_gemm',
    'conv_depthwise',
    'conv_backward',
    'matmul_tensor_core',
    'matmul_fp32',
    'fused_attention',
    'norm',
    'layout_convert',
    'pool',
    'rnn',
    'softmax',
    'elementwise',
    'reduce',
    'embed_gather',
    'other',
]


def aggregate_by_category(per_name_totals):
    """per_name_totals: {name -> (total_us, n_calls)}.

    Returns {category -> (total_us, n_calls)} for categories present.
    """
    out = {}
    for name, (us, n) in per_name_totals.items():
        cat = classify(name)
        cur = out.setdefault(cat, [0.0, 0])
        cur[0] += us
        cur[1] += n
    return {c: (v[0], v[1]) for c, v in out.items()}
