import json

import pytest

from llamalauncher.app.core import hardware
from llamalauncher.app.core.hardware import (
    GB, MB, Backend, GPUInfo, SystemInfo, Tier, parse_macos_displays, parse_nvidia_smi, parse_rocm_smi,
)


def make_system(total_ram_gb, gpus=(), perf=None, wired=None, physical=8):
    return SystemInfo(
        os_name="Test", os_version="1", arch="x86_64", cpu_name="CPU",
        physical_cores=physical, logical_cores=physical * 2, performance_cores=perf,
        total_ram=int(total_ram_gb * GB), available_ram=int(total_ram_gb * GB * 0.6),
        gpus=list(gpus), gpu_wired_limit=wired,
    )


def apple(total_gb, **kw):
    return make_system(total_gb, [GPUInfo("Apple M4", "apple", int(total_gb * GB), unified=True)], **kw)


def nvidia(vram_gb, ram_gb):
    return make_system(ram_gb, [GPUInfo("RTX", "nvidia", int(vram_gb * GB))])


M4_MAX_DISPLAYS = json.dumps({"SPDisplaysDataType": [{
    "_name": "Apple M4 Max", "spdisplays_vendor": "sppci_vendor_Apple",
    "sppci_cores": "40", "sppci_model": "Apple M4 Max",
}]})


def test_parse_macos_apple_silicon():
    gpus = parse_macos_displays(M4_MAX_DISPLAYS, arch="arm64", total_ram=48 * GB)
    assert len(gpus) == 1
    g = gpus[0]
    assert g.vendor == "apple" and g.unified and g.cores == 40
    assert g.backend == Backend.METAL


def test_parse_macos_intel_mac_with_amd():
    raw = json.dumps({"SPDisplaysDataType": [
        {"sppci_model": "Intel UHD Graphics 630", "spdisplays_vendor": "Intel", "spdisplays_vram": "1536 MB"},
        {"sppci_model": "AMD Radeon Pro 5500M", "spdisplays_vendor": "sppci_vendor_amd", "spdisplays_vram": "8 GB"},
    ]})
    gpus = parse_macos_displays(raw, arch="x86_64", total_ram=32 * GB)
    assert [g.vendor for g in gpus] == ["amd"]
    assert gpus[0].vram_bytes == 8 * GB


def test_parse_macos_bad_output_falls_back_on_arm():
    gpus = parse_macos_displays("not json", arch="arm64", total_ram=16 * GB)
    assert gpus[0].vendor == "apple"


def test_parse_nvidia_smi_multi_gpu():
    gpus = parse_nvidia_smi("NVIDIA GeForce RTX 4090, 24564\nNVIDIA GeForce RTX 3060, 12288\n")
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3060"]
    assert gpus[0].vram_bytes == 24564 * MB
    assert parse_nvidia_smi(None) == []
    assert parse_nvidia_smi("garbage") == []


def test_parse_rocm_smi():
    raw = json.dumps({"card0": {"Card series": "Radeon RX 7900 XTX", "VRAM Total Memory (B)": str(24 * GB)}})
    gpus = parse_rocm_smi(raw)
    assert gpus[0].vendor == "amd" and gpus[0].vram_bytes == 24 * GB


@pytest.mark.parametrize("name,vendor", [
    ("NVIDIA GeForce RTX 4070", "nvidia"), ("AMD Radeon RX 6800", "amd"),
    ("Intel(R) Arc(TM) A770", "intel"), ("Intel(R) UHD Graphics", "intel"),
    ("Microsoft Basic Display Adapter", "unknown"),
])
def test_vendor_from_name(name, vendor):
    assert hardware._vendor_from_name(name) == vendor


def test_drop_weak_integrated():
    gpus = [GPUInfo("Intel UHD", "intel", 128 * MB), GPUInfo("RTX 4070", "nvidia", 12 * GB)]
    assert [g.name for g in hardware._drop_weak_integrated(gpus)] == ["RTX 4070"]


def test_apple_gpu_budget_default_ratio():
    # 48GB > 36GB → 75% = 36GB
    assert apple(48).gpu_budget == int(48 * GB * 0.75)
    # 16GB → 67%
    assert apple(16).gpu_budget == int(16 * GB * 0.67)
    assert apple(48).cpu_budget == 0


def test_apple_gpu_budget_respects_wired_limit():
    assert apple(64, wired=56 * GB).gpu_budget == 56 * GB
    # 램보다 4GB 이상 여유는 남긴다
    assert apple(64, wired=63 * GB).gpu_budget == 60 * GB


def test_discrete_gpu_budget():
    s = nvidia(12, 32)
    assert s.gpu_budget == 12 * GB - GB // 2
    assert s.cpu_budget == 29 * GB
    assert s.backend == Backend.CUDA


def test_cpu_only():
    s = make_system(16)
    assert s.gpu_budget == 0 and s.backend == Backend.CPU
    assert s.fast_budget == 13 * GB


@pytest.mark.parametrize("system,tier", [
    (apple(8), Tier.ENTRY),
    (apple(16), Tier.STANDARD),
    (apple(36), Tier.HIGH),
    (apple(48), Tier.WORKSTATION),
    (nvidia(8, 32), Tier.ENTRY),
    (nvidia(24, 64), Tier.HIGH),
    (make_system(64), Tier.STANDARD),  # GPU 없으면 최대 '보통'
])
def test_tier(system, tier):
    assert system.tier == tier


def test_recommended_threads_prefers_perf_cores():
    assert apple(48, perf=12, physical=16).recommended_threads == 12
    assert nvidia(8, 16).recommended_threads == 8


def test_detect_macos_with_fake_runner(monkeypatch):
    outputs = {
        ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M4 Max\n",
        ("sysctl", "-n", "hw.perflevel0.physicalcpu"): "12\n",
        ("sysctl", "-n", "iogpu.wired_limit_mb"): "0\n",
        ("system_profiler", "SPDisplaysDataType", "-json"): M4_MAX_DISPLAYS,
    }
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "machine", lambda: "arm64")
    info = hardware.detect_system(runner=lambda cmd: outputs.get(tuple(cmd)))
    assert info.os_name == "macOS"
    assert info.cpu_name == "Apple M4 Max"
    assert info.performance_cores == 12
    assert info.gpu_wired_limit is None
    assert info.backend == Backend.METAL


def test_detect_windows_with_fake_runner(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware, "_windows_cpu_name", lambda: "AMD Ryzen 7 7800X3D")
    monkeypatch.setattr(hardware, "_windows_registry_gpus", lambda: [
        GPUInfo("NVIDIA GeForce RTX 4070", "nvidia", 12 * GB),
        GPUInfo("AMD Radeon(TM) Graphics", "amd", 512 * MB),
    ])
    smi = "NVIDIA GeForce RTX 4070, 12282\n"
    info = hardware.detect_system(runner=lambda cmd: smi if cmd[0] == "nvidia-smi" else None)
    assert info.os_name == "Windows"
    assert info.cpu_name == "AMD Ryzen 7 7800X3D"
    assert [g.name for g in info.gpus] == ["NVIDIA GeForce RTX 4070"]
    assert info.backend == Backend.CUDA


def test_detect_windows_amd_only_uses_vulkan(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hardware.sys, "platform", "win32")
    monkeypatch.setattr(hardware, "_windows_cpu_name", lambda: None)
    monkeypatch.setattr(hardware, "_windows_registry_gpus",
                        lambda: [GPUInfo("AMD Radeon RX 7800 XT", "amd", 16 * GB)])
    info = hardware.detect_system(runner=lambda cmd: None)
    assert info.backend == Backend.VULKAN


def test_detect_survives_missing_tools(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    info = hardware.detect_system(runner=lambda cmd: None)
    assert info.total_ram > 0 and info.backend == Backend.CPU
