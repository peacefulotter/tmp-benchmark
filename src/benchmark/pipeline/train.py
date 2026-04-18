"""
ml_pipeline_benchmark.py  — production-grade hardware bottleneck profiler
Paste this AFTER the existing file (from the _infer_iter method onward).
Or use as a standalone drop-in replacement — the full file is self-contained.
"""

from __future__ import annotations

import json
import platform
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── optional NVML for live GPU metrics ────────────────────────────────────────
try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# ── optional rich for pretty output ───────────────────────────────────────────
try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH = True
    console = Console()
except ImportError:
    RICH = False

    class _FallbackConsole:
        def print(self, *a, **kw):
            print(*[str(x) for x in a])

        def rule(self, t=""):
            print("─" * 72 + (f" {t} " if t else ""))

    console = _FallbackConsole()

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "tiny": dict(layers=2, hidden=64, heads=2, name="Tiny   (~100K params)"),
    "small": dict(layers=4, hidden=256, heads=4, name="Small  (~2M params)"),
    "medium": dict(layers=8, hidden=512, heads=8, name="Medium (~25M params)"),
    "large": dict(layers=12, hidden=1024, heads=16, name="Large  (~120M params)"),
    "xl": dict(layers=24, hidden=2048, heads=16, name="XL     (~600M params)"),
}

PRECISION_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class BenchmarkConfig:
    device: str = "auto"
    model_size: str = "medium"
    batch_size: int = 64
    seq_len: int = 128
    dataset_size: int = 10_000
    num_workers: int = 4
    warmup_iters: int = 5
    bench_iters: int = 50
    precision: str = "fp32"
    use_amp: bool = False
    pin_memory: bool = True
    compile_model: bool = False
    export_path: Optional[str] = None
    quiet: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODELS
# ══════════════════════════════════════════════════════════════════════════════


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class BenchmarkModel(nn.Module):
    """Transformer-based model — stresses CPU, GPU compute, and memory bandwidth."""

    def __init__(self, cfg: dict, seq_len: int = 128, num_classes: int = 10):
        super().__init__()
        h = cfg["hidden"]
        self.embed = nn.Sequential(
            nn.Linear(seq_len, h),
            nn.LayerNorm(h),
        )
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, h) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(h, cfg["heads"]) for _ in range(cfg["layers"])]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) + self.pos_enc
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ══════════════════════════════════════════════════════════════════════════════
# 3. TIMER — CUDA-AWARE HIGH-PRECISION STOPWATCH
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class TimingResult:
    name: str
    hardware: str
    times_ms: List[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return float(np.mean(self.times_ms)) if self.times_ms else 0.0

    @property
    def std(self) -> float:
        return float(np.std(self.times_ms)) if self.times_ms else 0.0

    @property
    def median(self) -> float:
        return float(np.median(self.times_ms)) if self.times_ms else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times_ms, 95)) if self.times_ms else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self.times_ms, 99)) if self.times_ms else 0.0

    @property
    def cv(self) -> float:
        """Coefficient of variation — high CV signals jitter / instability."""
        return (self.std / self.mean * 100) if self.mean > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "hardware": self.hardware,
            "mean_ms": round(self.mean, 4),
            "std_ms": round(self.std, 4),
            "median_ms": round(self.median, 4),
            "p95_ms": round(self.p95, 4),
            "p99_ms": round(self.p99, 4),
            "cv_pct": round(self.cv, 2),
            "n_samples": len(self.times_ms),
        }


class CUDATimer:
    """Uses CUDA events for GPU timing; falls back to perf_counter on CPU."""

    def __init__(self, device: torch.device):
        self.device = device
        self.use_cuda = device.type == "cuda"
        self._elapsed_ms: float = 0.0

    @contextmanager
    def measure(self):
        if self.use_cuda:
            start_ev = torch.cuda.Event(enable_timing=True)
            end_ev = torch.cuda.Event(enable_timing=True)
            start_ev.record()
            yield
            end_ev.record()
            torch.cuda.synchronize()
            self._elapsed_ms = start_ev.elapsed_time(end_ev)
        else:
            t0 = time.perf_counter()
            yield
            self._elapsed_ms = (time.perf_counter() - t0) * 1000.0

    @property
    def elapsed_ms(self) -> float:
        return self._elapsed_ms


# ══════════════════════════════════════════════════════════════════════════════
# 4. HARDWARE MONITOR — background thread
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class HWSnapshot:
    ts: float
    cpu_pct: float
    ram_used_gb: float
    gpu_util_pct: float = 0.0
    gpu_mem_used_gb: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_power_w: float = 0.0


class HardwareMonitor:
    """Polls CPU / GPU metrics on a background thread at ~100 ms intervals."""

    def __init__(self, device_idx: int = 0):
        self._snapshots: List[HWSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._device_idx = device_idx
        self._nvml_handle = None
        if NVML_AVAILABLE:
            try:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            except Exception:
                pass

    def start(self):
        self._running = True
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> List[HWSnapshot]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self._snapshots

    def _poll(self):
        while self._running:
            snap = HWSnapshot(
                ts=time.perf_counter(),
                cpu_pct=psutil.cpu_percent(interval=None),
                ram_used_gb=psutil.virtual_memory().used / 1e9,
            )
            if self._nvml_handle:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    snap.gpu_util_pct = util.gpu
                    snap.gpu_mem_used_gb = mem.used / 1e9
                    snap.gpu_temp_c = pynvml.nvmlDeviceGetTemperature(
                        self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                    snap.gpu_power_w = (
                        pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                    )
                except Exception:
                    pass
            self._snapshots.append(snap)
            time.sleep(0.1)

    def summary(self) -> dict:
        if not self._snapshots:
            return {}
        cpu = [s.cpu_pct for s in self._snapshots]
        ram = [s.ram_used_gb for s in self._snapshots]
        gpu_u = [s.gpu_util_pct for s in self._snapshots]
        gpu_m = [s.gpu_mem_used_gb for s in self._snapshots]
        gpu_t = [s.gpu_temp_c for s in self._snapshots]
        gpu_p = [s.gpu_power_w for s in self._snapshots]
        return {
            "cpu_util_mean_pct": round(np.mean(cpu), 1),
            "cpu_util_peak_pct": round(np.max(cpu), 1),
            "ram_peak_gb": round(np.max(ram), 2),
            "gpu_util_mean_pct": round(np.mean(gpu_u), 1),
            "gpu_util_peak_pct": round(np.max(gpu_u), 1),
            "gpu_mem_peak_gb": round(np.max(gpu_m), 2),
            "gpu_temp_peak_c": round(np.max(gpu_t), 1),
            "gpu_power_mean_w": round(np.mean(gpu_p), 1),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════


def build_dataset(cfg: BenchmarkConfig) -> Tuple[TensorDataset, TensorDataset]:
    N_train = cfg.dataset_size
    N_val = max(512, N_train // 10)
    seq = cfg.seq_len
    X_train = torch.randn(N_train, seq, seq)
    y_train = torch.randint(0, 10, (N_train,))
    X_val = torch.randn(N_val, seq, seq)
    y_val = torch.randint(0, 10, (N_val,))
    return TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)


def build_loaders(
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    cfg: BenchmarkConfig,
) -> Tuple[DataLoader, DataLoader]:
    device = resolve_device(cfg.device)
    pin = cfg.pin_memory and device.type == "cuda"
    kw: dict[str, Any] = dict(
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False, drop_last=False, **kw
    )
    return train_loader, val_loader


def synthetic_augmentation(x: torch.Tensor) -> torch.Tensor:
    """CPU augmentation: Gaussian noise + random masking."""
    x = x + torch.randn_like(x) * 0.05
    return x * (torch.rand_like(x) > 0.1)


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEVICE UTILS
# ══════════════════════════════════════════════════════════════════════════════


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def get_gpu_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).total_memory
    return 0


def torch_mem_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {}
    alloc = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    total = get_gpu_memory_bytes(device)
    return {
        "allocated_gb": round(alloc / 1e9, 3),
        "reserved_gb": round(reserved / 1e9, 3),
        "total_gb": round(total / 1e9, 3),
        "util_pct": round(alloc / total * 100, 1) if total else 0,
    }


def hardware_info(device: torch.device) -> dict:
    info = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": str(device),
        "cpu_brand": platform.processor() or "unknown",
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 2),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "gpu_name": props.name,
                "gpu_vram_gb": round(props.total_memory / 1e9, 2),
                "gpu_sm_count": props.multi_processor_count,
                "gpu_compute_cap": f"{props.major}.{props.minor}",
                "cuda_version": torch.version.cuda,
                "cudnn_version": str(torch.backends.cudnn.version()),
            }
        )
    return info


# ══════════════════════════════════════════════════════════════════════════════
# 7. BENCHMARK ENGINE
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkReport:
    config: dict
    hardware: dict
    stages: Dict[str, TimingResult]
    hw_monitor: dict
    throughput_train: float  # samples/sec during training forward+backward
    throughput_inf: float  # samples/sec during inference
    total_pipeline_ms: float  # mean end-to-end per batch (train)
    bottlenecks: List[dict]  # sorted by pct of wall time descending
    gpu_memory: dict
    per_iter_throughput: List[float]  # drift / throttle detection
    throttle_detected: bool
    throttle_delta_pct: float

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "hardware": self.hardware,
            "stages": {k: v.to_dict() for k, v in self.stages.items()},
            "hw_monitor": self.hw_monitor,
            "throughput_train": round(self.throughput_train, 1),
            "throughput_inf": round(self.throughput_inf, 1),
            "total_pipeline_ms": round(self.total_pipeline_ms, 3),
            "bottlenecks": self.bottlenecks,
            "gpu_memory": self.gpu_memory,
            "per_iter_throughput": [round(v, 1) for v in self.per_iter_throughput],
            "throttle_detected": self.throttle_detected,
            "throttle_delta_pct": round(self.throttle_delta_pct, 1),
        }


class MLPipelineBenchmark:
    # Canonical stage order (determines table row ordering)
    STAGE_ORDER = [
        "data_load",
        "host_augment",
        "host_to_device",
        "forward_pass",
        "loss_compute",
        "backward_pass",
        "optimizer_step",
        "device_sync",
        "inference_forward",
        "inference_postprocess",
        "result_to_host",
    ]

    STAGE_META: Dict[str, Tuple[str, str]] = {
        "data_load": ("Data Load", "I/O / CPU"),
        "host_augment": ("Host Augmentation", "CPU"),
        "host_to_device": ("Host → Device", "PCIe"),
        "forward_pass": ("Forward Pass", "GPU/CPU"),
        "loss_compute": ("Loss Compute", "GPU/CPU"),
        "backward_pass": ("Backward Pass", "GPU/CPU"),
        "optimizer_step": ("Optimizer Step", "GPU/CPU"),
        "device_sync": ("Device Sync", "GPU/CPU"),
        "inference_forward": ("Inference Forward", "GPU/CPU"),
        "inference_postprocess": ("Inference Postproc", "GPU/CPU"),
        "result_to_host": ("Result → Host", "PCIe"),
    }

    # Bottleneck thresholds (pct of total wall time)
    _BOTTLENECK_CRITICAL = 30.0
    _BOTTLENECK_HIGH = 15.0
    _BOTTLENECK_ELEVATED = 8.0

    # Throttle detection: if last-quartile throughput drops >10% vs first quartile
    _THROTTLE_DROP_THRESHOLD = 10.0

    def __init__(self, cfg: BenchmarkConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.dtype = PRECISION_MAP.get(cfg.precision, torch.float32)
        self.timer = CUDATimer(self.device)
        self.stages: Dict[str, TimingResult] = {
            k: TimingResult(name=v[0], hardware=v[1])
            for k, v in self.STAGE_META.items()
        }

    # ── build model ───────────────────────────────────────────────────────────
    def _build_model(self) -> nn.Module:
        mcfg = MODEL_CONFIGS[self.cfg.model_size]
        model = BenchmarkModel(mcfg, seq_len=self.cfg.seq_len)
        model = model.to(device=self.device)
        if self.cfg.compile_model and hasattr(torch, "compile"):
            _p("torch.compile() enabled — first batch may be slow")
            model = torch.compile(model)
        return model  # type: ignore

    # ── warmup ────────────────────────────────────────────────────────────────
    def _warmup(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        scaler: "torch.cuda.amp.GradScaler",
    ):
        model.train()
        loader_iter = iter(loader)
        for _ in range(self.cfg.warmup_iters):
            try:
                x, y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                x, y = next(loader_iter)
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type, dtype=self.dtype, enabled=self.cfg.use_amp
            ):
                out = model(x)
                loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    # ── training iteration (fully instrumented) ───────────────────────────────
    def _train_iter(
        self,
        model: nn.Module,
        x_cpu: torch.Tensor,
        y_cpu: torch.Tensor,
        optimizer: optim.Optimizer,
        scaler: "torch.cuda.amp.GradScaler",
    ) -> float:
        t = self.timer

        # ① host augmentation (CPU)
        with t.measure():
            x_aug = synthetic_augmentation(x_cpu)
        self.stages["host_augment"].times_ms.append(t.elapsed_ms)

        # ② host → device transfer (PCIe / UMA)
        with t.measure():
            x = x_aug.to(self.device, non_blocking=True)
            y = y_cpu.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["host_to_device"].times_ms.append(t.elapsed_ms)

        # ③ forward pass
        optimizer.zero_grad(set_to_none=True)
        with t.measure():
            with torch.autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                out = model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["forward_pass"].times_ms.append(t.elapsed_ms)

        # ④ loss computation
        with t.measure():
            with torch.autocast(device_type=self.device.type, enabled=self.cfg.use_amp):
                loss = F.cross_entropy(out, y)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["loss_compute"].times_ms.append(t.elapsed_ms)

        # ⑤ backward pass
        with t.measure():
            scaler.scale(loss).backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["backward_pass"].times_ms.append(t.elapsed_ms)

        # ⑥ optimizer step (weight update + scaler update)
        with t.measure():
            scaler.step(optimizer)
            scaler.update()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["optimizer_step"].times_ms.append(t.elapsed_ms)

        # ⑦ explicit device sync (measures residual pipeline drain)
        with t.measure():
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["device_sync"].times_ms.append(t.elapsed_ms)

        return float(loss.item())

    # ── inference iteration (fully instrumented) ──────────────────────────────
    def _infer_iter(
        self,
        model: nn.Module,
        x_cpu: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t = self.timer

        # ① host → device
        with t.measure():
            x = x_cpu.to(self.device, non_blocking=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["host_to_device"].times_ms.append(t.elapsed_ms)

        # ② inference forward (no_grad + optional AMP)
        with t.measure():
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type, enabled=self.cfg.use_amp
                ):
                    logits = model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["inference_forward"].times_ms.append(t.elapsed_ms)

        # ③ postprocess on-device: softmax → top-k argmax
        with t.measure():
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
        self.stages["inference_postprocess"].times_ms.append(t.elapsed_ms)

        # ④ result → host (PCIe DMA back)
        with t.measure():
            preds_cpu = preds.cpu()
            probs_cpu = probs.cpu()
        self.stages["result_to_host"].times_ms.append(t.elapsed_ms)

        return preds_cpu, probs_cpu

    # ── data-load timing (wraps DataLoader iteration) ─────────────────────────
    def _timed_data_load(self, loader_iter) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Returns (x_cpu, y_cpu, elapsed_ms)."""
        t0 = time.perf_counter()
        try:
            x, y = next(loader_iter)
        except StopIteration:
            raise
        elapsed = (time.perf_counter() - t0) * 1000.0
        return x, y, elapsed

    # ══════════════════════════════════════════════════════════════════════════
    # 8. MAIN BENCHMARK LOOP
    # ══════════════════════════════════════════════════════════════════════════

    def run(self) -> BenchmarkReport:
        cfg = self.cfg

        # ── hardware fingerprint ──────────────────────────────────────────────
        hw = hardware_info(self.device)
        _section("Hardware")
        for k, v in hw.items():
            _kv(k, str(v))

        # ── dataset + loaders ─────────────────────────────────────────────────
        _section("Building Dataset")
        t_ds = time.perf_counter()
        train_ds, val_ds = build_dataset(cfg)
        _kv("dataset build time", f"{(time.perf_counter() - t_ds) * 1e3:.1f} ms")
        _kv("train samples", str(len(train_ds)))
        _kv("val samples", str(len(val_ds)))

        train_loader, val_loader = build_loaders(train_ds, val_ds, cfg)

        # ── model ─────────────────────────────────────────────────────────────
        _section("Model")
        model = self._build_model()
        n_params = sum(p.numel() for p in model.parameters())
        _kv("architecture", f"Transformer  [{MODEL_CONFIGS[cfg.model_size]['name']}]")
        _kv("parameters", f"{n_params:,}")
        _kv("precision", cfg.precision)
        _kv("device", str(self.device))
        _kv("AMP", str(cfg.use_amp))
        _kv("torch.compile", str(cfg.compile_model))

        # ── optimizer + scaler ────────────────────────────────────────────────
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler(
            enabled=cfg.use_amp and self.device.type == "cuda"
        )

        # ── warmup ────────────────────────────────────────────────────────────
        _section(f"Warmup  ({cfg.warmup_iters} iters)")
        self._warmup(model, train_loader, optimizer, scaler)
        _p("Warmup complete.")

        # ── start background hardware monitor ─────────────────────────────────
        device_idx = self.device.index or 0 if self.device.type == "cuda" else 0
        monitor = HardwareMonitor(device_idx=device_idx)
        monitor.start()

        # ══════════════════════════════════════════════════════════════════════
        # TRAINING BENCHMARK
        # ══════════════════════════════════════════════════════════════════════
        _section(f"Training Benchmark  ({cfg.bench_iters} iters)")
        model.train()
        train_iter = iter(train_loader)
        per_iter_thr: List[float] = []
        batch_wall_ms: List[float] = []

        _progress_start()
        for i in range(cfg.bench_iters):
            # ① data load (CPU-side timer — DataLoader runs in workers)
            try:
                x_cpu, y_cpu, load_ms = self._timed_data_load(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x_cpu, y_cpu, load_ms = self._timed_data_load(train_iter)
            self.stages["data_load"].times_ms.append(load_ms)

            # ② rest of the pipeline
            t_batch_start = time.perf_counter()
            self._train_iter(model, x_cpu, y_cpu, optimizer, scaler)
            wall_ms = (time.perf_counter() - t_batch_start) * 1000.0 + load_ms

            batch_wall_ms.append(wall_ms)
            thr = cfg.batch_size / (wall_ms / 1000.0)
            per_iter_thr.append(thr)

            _progress_tick(i, cfg.bench_iters, thr)

        _progress_end()

        # ══════════════════════════════════════════════════════════════════════
        # INFERENCE BENCHMARK
        # ══════════════════════════════════════════════════════════════════════
        _section(f"Inference Benchmark  ({cfg.bench_iters} iters)")
        model.eval()
        val_iter = iter(val_loader)
        inf_throughputs: List[float] = []

        for i in range(cfg.bench_iters):
            try:
                x_cpu, _, load_ms = self._timed_data_load(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                x_cpu, _, load_ms = self._timed_data_load(val_iter)

            # Inference batch size is batch_size*2 from val_loader
            t0 = time.perf_counter()
            self._infer_iter(model, x_cpu)
            inf_wall_ms = (time.perf_counter() - t0) * 1000.0 + load_ms
            inf_throughputs.append(x_cpu.shape[0] / (inf_wall_ms / 1000.0))

        monitor.stop()

        # ══════════════════════════════════════════════════════════════════════
        # ANALYSIS
        # ══════════════════════════════════════════════════════════════════════

        # ── gpu memory snapshot ───────────────────────────────────────────────
        gpu_mem = torch_mem_stats(self.device)

        # ── throughput summary ────────────────────────────────────────────────
        throughput_train = float(np.mean(per_iter_thr))
        throughput_inf = float(np.mean(inf_throughputs))

        # ── total pipeline (sum of measured training stages) ──────────────────
        train_stage_keys = [
            "data_load",
            "host_augment",
            "host_to_device",
            "forward_pass",
            "loss_compute",
            "backward_pass",
            "optimizer_step",
            "device_sync",
        ]
        total_ms = sum(
            self.stages[k].mean for k in train_stage_keys if self.stages[k].times_ms
        )

        # ── bottleneck analysis ───────────────────────────────────────────────
        bottlenecks: List[dict] = []
        for k in train_stage_keys:
            s = self.stages[k]
            if not s.times_ms:
                continue
            pct = (s.mean / total_ms * 100) if total_ms > 0 else 0.0
            if pct >= self._BOTTLENECK_CRITICAL:
                severity = "CRITICAL"
            elif pct >= self._BOTTLENECK_HIGH:
                severity = "HIGH"
            elif pct >= self._BOTTLENECK_ELEVATED:
                severity = "ELEVATED"
            else:
                severity = "OK"

            # Diagnose likely cause
            diagnosis = self._diagnose(k, s, pct, hw)

            bottlenecks.append(
                {
                    "stage": k,
                    "name": s.name,
                    "hardware": s.hardware,
                    "mean_ms": round(s.mean, 3),
                    "pct": round(pct, 2),
                    "severity": severity,
                    "cv_pct": round(s.cv, 2),
                    "diagnosis": diagnosis,
                }
            )

        bottlenecks.sort(key=lambda d: d["pct"], reverse=True)

        # ── thermal throttle detection ────────────────────────────────────────
        n = len(per_iter_thr)
        q1_mean = float(np.mean(per_iter_thr[: max(1, n // 4)]))
        q4_mean = float(np.mean(per_iter_thr[max(0, n * 3 // 4) :]))
        throttle_delta = (q1_mean - q4_mean) / q1_mean * 100 if q1_mean > 0 else 0.0
        throttle_detected = throttle_delta > self._THROTTLE_DROP_THRESHOLD

        report = BenchmarkReport(
            config=vars(cfg),
            hardware=hw,
            stages=self.stages,
            hw_monitor=monitor.summary(),
            throughput_train=throughput_train,
            throughput_inf=throughput_inf,
            total_pipeline_ms=total_ms,
            bottlenecks=bottlenecks,
            gpu_memory=gpu_mem,
            per_iter_throughput=per_iter_thr,
            throttle_detected=throttle_detected,
            throttle_delta_pct=throttle_delta,
        )

        self._print_report(report)

        if cfg.export_path:
            self._export(report, cfg.export_path)

        return report

    # ══════════════════════════════════════════════════════════════════════════
    # 9. DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════════

    def _diagnose(self, key: str, s: TimingResult, pct: float, hw: dict) -> str:
        """Return a human-readable diagnosis string for a bottleneck stage."""
        diag: List[str] = []

        if key == "data_load":
            if pct > self._BOTTLENECK_HIGH:
                diag.append(
                    "I/O-bound: consider increasing num_workers, using memory-mapped "
                    "datasets (e.g. HDF5/LMDB), or pre-loading into shared memory."
                )
            if s.cv > 30:
                diag.append(
                    "High jitter (CV={:.1f}%): likely disk seek variance — "
                    "SSD/NVMe strongly recommended.".format(s.cv)
                )

        elif key == "host_augment":
            diag.append(
                "CPU-bound augmentation: move transforms to GPU (torchvision.transforms.v2 "
                "with device=cuda) or use DALI/Albumentations GPU backend."
            )

        elif key == "host_to_device":
            if pct > self._BOTTLENECK_HIGH:
                diag.append(
                    "PCIe-bound: ensure pin_memory=True and non_blocking=True. "
                    "Check PCIe lane width (x16 vs x8) with 'nvidia-smi -q'. "
                    "Consider GPU-native data generation to skip host transfer."
                )
            if s.cv > 25:
                diag.append(
                    "High PCIe jitter: may indicate IOMMU interference or NUMA "
                    "memory-controller contention. Try numactl --cpunodebind=0."
                )

        elif key == "forward_pass":
            if pct > self._BOTTLENECK_CRITICAL:
                diag.append(
                    "Compute-bound forward pass: consider FP16/BF16 AMP, smaller model, "
                    "or torch.compile(). Flash-attention (xformers/F.scaled_dot_product_attention) "
                    "can 2–4× the attention throughput."
                )
            if s.cv > 20:
                diag.append(
                    "Forward pass jitter (CV={:.1f}%): possible GPU frequency scaling "
                    "or memory pressure from other processes.".format(s.cv)
                )

        elif key == "backward_pass":
            if pct > self._BOTTLENECK_CRITICAL:
                diag.append(
                    "Compute/memory-bound backward: gradient checkpointing "
                    "(torch.utils.checkpoint) trades VRAM for recompute; "
                    "consider gradient accumulation to amortise cost."
                )

        elif key == "optimizer_step":
            if pct > self._BOTTLENECK_HIGH:
                diag.append(
                    "Memory-bandwidth-bound optimizer step: try fused optimizers "
                    "(torch.optim.AdamW with fused=True on CUDA) or apex.optimizers.FusedAdam."
                )

        elif key == "device_sync":
            if s.mean > 2.0:
                diag.append(
                    "Non-trivial device sync overhead ({:.2f} ms): avoid explicit "
                    "synchronize() calls mid-pipeline; let CUDA streams do async work.".format(
                        s.mean
                    )
                )

        if not diag:
            diag.append("Within normal range — no action required.")

        return " | ".join(diag)

    # ══════════════════════════════════════════════════════════════════════════
    # 10. REPORTING
    # ══════════════════════════════════════════════════════════════════════════

    def _print_report(self, r: BenchmarkReport):
        _section("Results — Stage Timing")

        if RICH:
            self._rich_table(r)
        else:
            self._plain_table(r)

        _section("Hardware Utilisation (live-sampled)")
        for k, v in r.hw_monitor.items():
            _kv(k, str(v))

        if r.gpu_memory:
            _section("GPU Memory Snapshot (post-benchmark)")
            for k, v in r.gpu_memory.items():
                _kv(k, str(v))

        _section("Throughput Summary")
        _kv("Training throughput", f"{r.throughput_train:.1f} samples/sec")
        _kv("Inference throughput", f"{r.throughput_inf:.1f}  samples/sec")
        _kv("Total pipeline (train)", f"{r.total_pipeline_ms:.2f} ms/batch")

        _section("Bottleneck Analysis")
        for b in r.bottlenecks:
            sev_tag = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "ELEVATED": "🟡",
                "OK": "🟢",
            }.get(b["severity"], "")
            _p(f"  {sev_tag}  {b['name']:<26} {b['pct']:5.1f}%  [{b['hardware']}]")
            if b["severity"] != "OK":
                _p(f"       → {b['diagnosis']}")

        if r.throttle_detected:
            _p(
                f"\n  ⚠️  THERMAL THROTTLE DETECTED: throughput dropped "
                f"{r.throttle_delta_pct:.1f}% from first→last quartile "
                f"(threshold {self._THROTTLE_DROP_THRESHOLD}%). "
                "Check GPU temperature — consider improved cooling or power limits."
            )
        else:
            _p(
                f"\n  ✅  No thermal throttle detected "
                f"(drop={r.throttle_delta_pct:.1f}%, threshold={self._THROTTLE_DROP_THRESHOLD}%)."
            )

        console.rule()

    def _rich_table(self, r: BenchmarkReport):
        total = r.total_pipeline_ms
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_header=True,
            header_style="bold",
            title="ML Pipeline Stage Breakdown",
        )
        table.add_column("Stage", style="dim", no_wrap=True)
        table.add_column("Hardware", no_wrap=True)
        table.add_column("Mean (ms)", justify="right")
        table.add_column("±σ (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("p99 (ms)", justify="right")
        table.add_column("CV %", justify="right")
        table.add_column("Wall %", justify="right")
        table.add_column("Severity", justify="center")

        for b in r.bottlenecks:
            s = r.stages[b["stage"]]
            pct = b["pct"]
            sev = b["severity"]
            color_map = {
                "CRITICAL": "red",
                "HIGH": "yellow",
                "ELEVATED": "cyan",
                "OK": "green",
            }
            row_style = color_map.get(sev, "")
            table.add_row(
                s.name,
                s.hardware,
                f"{s.mean:.2f}",
                f"{s.std:.2f}",
                f"{s.p95:.2f}",
                f"{s.p99:.2f}",
                f"{s.cv:.1f}",
                f"{pct:.1f}%",
                sev,
                style=row_style,
            )

        # Inference-only stages
        for key in ("inference_forward", "inference_postprocess", "result_to_host"):
            s = r.stages[key]
            if not s.times_ms:
                continue
            pct = s.mean / total * 100 if total else 0
            table.add_row(
                s.name + " [inf]",
                s.hardware,
                f"{s.mean:.2f}",
                f"{s.std:.2f}",
                f"{s.p95:.2f}",
                f"{s.p99:.2f}",
                f"{s.cv:.1f}",
                f"{pct:.1f}%",
                "—",
                style="dim",
            )

        console.print(table)

    def _plain_table(self, r: BenchmarkReport):
        hdr = f"{'Stage':<26} {'HW':<14} {'Mean':>8} {'±σ':>7} {'p95':>7} {'p99':>7} {'CV%':>6} {'Wall%':>7} Sev"
        print(hdr)
        print("─" * len(hdr))
        total = r.total_pipeline_ms
        for b in r.bottlenecks:
            s = r.stages[b["stage"]]
            pct = b["pct"]
            print(
                f"{s.name:<26} {s.hardware:<14} {s.mean:>8.2f} {s.std:>7.2f} "
                f"{s.p95:>7.2f} {s.p99:>7.2f} {s.cv:>6.1f} {pct:>6.1f}% {b['severity']}"
            )

    # ══════════════════════════════════════════════════════════════════════════
    # 11. EXPORT
    # ══════════════════════════════════════════════════════════════════════════

    def _export(self, report: BenchmarkReport, path: str):
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = report.to_dict()
        with open(out, "w") as f:
            json.dump(data, f, indent=2, default=str)
        _p(f"\nReport exported → {out.resolve()}")


# ══════════════════════════════════════════════════════════════════════════════
# 12. PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _p(msg: str):
    if RICH:
        console.print(msg)
    else:
        print(msg)


def _kv(key: str, val: str):
    if RICH:
        console.print(f"  [dim]{key:<28}[/dim] {val}")
    else:
        print(f"  {key:<28} {val}")


def _section(title: str):
    if RICH:
        console.rule(f"[bold]{title}[/bold]")
    else:
        print(f"\n{'─' * 72}\n  {title}\n{'─' * 72}")


_PROGRESS = None


def _progress_start():
    global _PROGRESS
    if RICH:
        _PROGRESS = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=36),
            TextColumn("[cyan]{task.fields[thr]:>9.0f} samp/s"),
            TextColumn("[dim]{task.completed}/{task.total}"),
            transient=True,
        )
        _PROGRESS.__enter__()
        _PROGRESS._task = _PROGRESS.add_task("Training", total=None, thr=0.0)  # type: ignore


def _progress_tick(i: int, total: int, thr: float):
    if RICH and _PROGRESS:
        _PROGRESS.update(
            _PROGRESS._task,  # type: ignore
            advance=1,
            thr=thr,
            total=total,
            description=f"iter {i + 1:>4}/{total}",
        )
    elif i % max(1, total // 10) == 0:
        print(f"  iter {i + 1:>4}/{total}   {thr:>9.1f} samp/s")


def _progress_end():
    global _PROGRESS
    if RICH and _PROGRESS:
        _PROGRESS.__exit__(None, None, None)
        _PROGRESS = None


# ══════════════════════════════════════════════════════════════════════════════
# 13. CLI ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> BenchmarkConfig:
    import argparse

    ap = argparse.ArgumentParser(
        description="ML Pipeline Hardware Bottleneck Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--device", default="auto", help="cuda / cpu / auto")
    ap.add_argument("--model-size", default="medium", choices=list(MODEL_CONFIGS))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--dataset-size", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--warmup-iters", type=int, default=5)
    ap.add_argument("--bench-iters", type=int, default=50)
    ap.add_argument("--precision", default="fp32", choices=list(PRECISION_MAP))
    ap.add_argument("--amp", action="store_true", help="Enable torch AMP")
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile() the model")
    ap.add_argument(
        "--export", default=None, metavar="PATH", help="Export JSON report to this path"
    )
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    return BenchmarkConfig(
        device=a.device,
        model_size=a.model_size,
        batch_size=a.batch_size,
        seq_len=a.seq_len,
        dataset_size=a.dataset_size,
        num_workers=a.num_workers,
        warmup_iters=a.warmup_iters,
        bench_iters=a.bench_iters,
        precision=a.precision,
        use_amp=a.amp,
        pin_memory=not a.no_pin_memory,
        compile_model=a.compile,
        export_path=a.export,
        quiet=a.quiet,
    )


if __name__ == "__main__":
    cfg = parse_args()
    bench = MLPipelineBenchmark(cfg)
    report = bench.run()
