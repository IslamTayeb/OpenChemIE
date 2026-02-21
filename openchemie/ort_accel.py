"""ONNX Runtime acceleration utilities for OpenChemIE models.

Handles CUDA library preloading, ONNX export, and ORT session management.

Note: Text model ORT (ChemNER, CRE product/role) was tested and reverted —
CPU<->GPU data transfer overhead made it slower than PyTorch FP16 for the
pipeline (Exp 21, 2026-02-20). The infrastructure here is kept for potential
future use with image encoder models (Pix2Seq, MolScribe Swin).
"""
import ctypes
import hashlib
import os
import sys

import torch
import numpy as np

_ORT_INITIALIZED = False
_ORT_AVAILABLE = False
ort = None  # lazy import


def _preload_cuda_libs():
    """Preload CUDA 11 .so files from nvidia pip packages for ORT CUDAExecutionProvider."""
    nvidia_dir = None
    for p in sys.path:
        candidate = os.path.join(p, 'nvidia')
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, 'cuda_runtime')):
            nvidia_dir = candidate
            break
    if nvidia_dir is None:
        return False

    for lib in [
        'cuda_runtime/lib/libcudart.so.11.0',
        'cublas/lib/libcublas.so.11',
        'cublas/lib/libcublasLt.so.11',
        'cudnn/lib/libcudnn.so.8',
        'cufft/lib/libcufft.so.10',
    ]:
        path = os.path.join(nvidia_dir, lib)
        if os.path.exists(path):
            ctypes.CDLL(path)
    return True


def init_ort():
    """Initialize ORT with CUDA support. Call once before creating any sessions.

    Set OPENCHEMIE_ORT=0 to disable ORT acceleration.
    """
    global _ORT_INITIALIZED, _ORT_AVAILABLE, ort
    if _ORT_INITIALIZED:
        return _ORT_AVAILABLE

    if os.environ.get('OPENCHEMIE_ORT', '1') == '0':
        _ORT_INITIALIZED = True
        _ORT_AVAILABLE = False
        return False

    _preload_cuda_libs()

    try:
        import onnxruntime as _ort
        ort = _ort
        providers = ort.get_available_providers()
        _ORT_AVAILABLE = 'CUDAExecutionProvider' in providers
    except ImportError:
        _ORT_AVAILABLE = False

    _ORT_INITIALIZED = True
    return _ORT_AVAILABLE


def get_ort_cache_dir():
    """Return cache directory for ONNX models."""
    cache_dir = os.path.expanduser('~/.cache/openchemie_ort')
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _model_hash(model):
    """Quick hash of model parameters to detect checkpoint changes."""
    h = hashlib.md5()
    for p in model.parameters():
        h.update(p.data.cpu().numpy().tobytes()[:64])
    return h.hexdigest()[:12]


def export_model(model_wrapper, dummy_inputs, onnx_path, input_names, output_names, dynamic_axes):
    """Export a PyTorch model to ONNX.

    Args:
        model_wrapper: nn.Module with clean forward() signature
        dummy_inputs: tuple of dummy input tensors
        onnx_path: output .onnx file path
        input_names: list of input names
        output_names: list of output names
        dynamic_axes: dict of dynamic axes
    """
    torch.onnx.export(
        model_wrapper, dummy_inputs, onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True,
    )


def create_session(onnx_path):
    """Create an ORT InferenceSession with CUDA acceleration."""
    if not init_ort():
        raise RuntimeError("ORT CUDA not available")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.log_severity_level = 3  # suppress warnings

    return ort.InferenceSession(
        onnx_path, sess_opts,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
