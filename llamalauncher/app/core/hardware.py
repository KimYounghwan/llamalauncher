"""시스템 사양 자동 감지 (macOS / Windows / Linux).

모든 외부 명령은 `CommandRunner`를 통해 실행하므로 테스트에서 가짜 출력을 주입할 수 있다.
감지에 실패한 항목은 예외를 던지지 않고 기본값(None/0)으로 남긴다.
"""

from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import psutil

GB = 1024**3
MB = 1024**2

CommandRunner = Callable[[list[str]], Optional[str]]


def default_runner(cmd: list[str], timeout: float = 10.0) -> Optional[str]:
    """명령을 실행하고 stdout을 돌려준다. 실패하면 None."""
    if shutil.which(cmd[0]) is None:
        return None
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace", **kwargs,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


class Backend(str, Enum):
    METAL = "metal"
    CUDA = "cuda"
    ROCM = "rocm"
    VULKAN = "vulkan"
    CPU = "cpu"

    @property
    def label(self) -> str:
        return {
            "metal": "Metal (Apple GPU)",
            "cuda": "CUDA (NVIDIA)",
            "rocm": "ROCm (AMD)",
            "vulkan": "Vulkan",
            "cpu": "CPU 전용",
        }[self.value]


class Tier(str, Enum):
    ENTRY = "입문"
    STANDARD = "보통"
    HIGH = "고성능"
    WORKSTATION = "워크스테이션"


@dataclass
class GPUInfo:
    name: str
    vendor: str  # "apple" | "nvidia" | "amd" | "intel" | "unknown"
    vram_bytes: int = 0
    unified: bool = False  # True면 시스템 RAM을 공유 (Apple Silicon 등)
    cores: Optional[int] = None

    @property
    def backend(self) -> Backend:
        if self.vendor == "apple":
            return Backend.METAL
        if self.vendor == "nvidia":
            return Backend.CUDA
        if self.vendor == "amd":
            return Backend.ROCM if sys.platform.startswith("linux") else Backend.VULKAN
        if self.vendor == "intel":
            return Backend.VULKAN
        return Backend.CPU


@dataclass
class SystemInfo:
    os_name: str  # "macOS" | "Windows" | "Linux"
    os_version: str
    arch: str
    cpu_name: str
    physical_cores: int
    logical_cores: int
    performance_cores: Optional[int]  # Apple Silicon의 성능 코어 수 (없으면 None)
    total_ram: int
    available_ram: int
    gpus: list[GPUInfo] = field(default_factory=list)
    disk_path: str = ""
    disk_free: int = 0
    # Apple Silicon에서 GPU가 쓸 수 있는 메모리 한도 (iogpu.wired_limit_mb, 0이면 OS 기본값)
    gpu_wired_limit: Optional[int] = None

    # ---- 파생 값 -------------------------------------------------------
    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        usable = [g for g in self.gpus if g.backend != Backend.CPU]
        if not usable:
            return None
        return max(usable, key=lambda g: (g.unified, g.vram_bytes))

    @property
    def backend(self) -> Backend:
        gpu = self.primary_gpu
        return gpu.backend if gpu else Backend.CPU

    @property
    def is_unified_memory(self) -> bool:
        gpu = self.primary_gpu
        return bool(gpu and gpu.unified)

    @property
    def gpu_budget(self) -> int:
        """모델+캐시를 GPU에 올릴 수 있는 메모리 (바이트). GPU가 없으면 0."""
        gpu = self.primary_gpu
        if gpu is None:
            return 0
        if gpu.unified:
            if self.gpu_wired_limit:
                limit = self.gpu_wired_limit
            else:
                # macOS 기본값: 램 36GB 이하면 약 2/3, 그보다 크면 약 3/4
                ratio = 0.67 if self.total_ram <= 36 * GB else 0.75
                limit = int(self.total_ram * ratio)
            return max(0, min(limit, self.total_ram - 4 * GB))
        # 전용 GPU: 드라이버/화면 출력용 여유 0.5GB 제외
        return max(0, gpu.vram_bytes - GB // 2)

    @property
    def cpu_budget(self) -> int:
        """GPU에 못 올린 나머지를 시스템 RAM에 둘 수 있는 양 (전용 GPU / CPU 전용일 때)."""
        if self.is_unified_memory:
            return 0  # 통합 메모리는 gpu_budget과 같은 메모리를 쓰므로 따로 더하지 않음
        reserve = 3 * GB if self.total_ram >= 16 * GB else 2 * GB
        return max(0, self.total_ram - reserve)

    @property
    def total_budget(self) -> int:
        return self.gpu_budget + self.cpu_budget

    @property
    def fast_budget(self) -> int:
        """'원활하게' 돌릴 수 있는 메모리: GPU가 있으면 GPU 메모리, 없으면 RAM."""
        return self.gpu_budget if self.gpu_budget > 0 else self.cpu_budget

    @property
    def recommended_threads(self) -> int:
        if self.performance_cores:
            return self.performance_cores
        return max(1, self.physical_cores)

    @property
    def tier(self) -> Tier:
        fast = self.fast_budget / GB
        if fast < 8:
            t = Tier.ENTRY
        elif fast < 16:
            t = Tier.STANDARD
        elif fast < 32:
            t = Tier.HIGH
        else:
            t = Tier.WORKSTATION
        if self.backend == Backend.CPU and t in (Tier.HIGH, Tier.WORKSTATION):
            t = Tier.STANDARD  # GPU 없이는 큰 모델이 느리다
        return t


# ---------------------------------------------------------------------------
# 감지
# ---------------------------------------------------------------------------

def detect_system(disk_path: Optional[Path] = None, runner: CommandRunner = default_runner) -> SystemInfo:
    system = platform.system()
    if system == "Darwin":
        info = _detect_macos(runner)
    elif system == "Windows":
        info = _detect_windows(runner)
    else:
        info = _detect_linux(runner)

    path = Path(disk_path) if disk_path else Path.home()
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        info.disk_free = shutil.disk_usage(probe).free
        info.disk_path = str(path)
    except OSError:
        pass
    return info


def _base_info(os_name: str, os_version: str) -> SystemInfo:
    vm = psutil.virtual_memory()
    return SystemInfo(
        os_name=os_name,
        os_version=os_version,
        arch=platform.machine(),
        cpu_name=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or os.cpu_count() or 1,
        logical_cores=psutil.cpu_count(logical=True) or os.cpu_count() or 1,
        performance_cores=None,
        total_ram=vm.total,
        available_ram=vm.available,
    )


def _sysctl(runner: CommandRunner, key: str) -> Optional[str]:
    out = runner(["sysctl", "-n", key])
    return out.strip() if out and out.strip() else None


def _detect_macos(runner: CommandRunner) -> SystemInfo:
    info = _base_info("macOS", platform.mac_ver()[0])
    info.cpu_name = _sysctl(runner, "machdep.cpu.brand_string") or info.cpu_name
    perf = _sysctl(runner, "hw.perflevel0.physicalcpu")
    if perf and perf.isdigit():
        info.performance_cores = int(perf)
    wired = _sysctl(runner, "iogpu.wired_limit_mb")
    if wired and wired.isdigit() and int(wired) > 0:
        info.gpu_wired_limit = int(wired) * MB
    info.gpus = parse_macos_displays(runner(["system_profiler", "SPDisplaysDataType", "-json"]),
                                     arch=info.arch, total_ram=info.total_ram)
    return info


def parse_macos_displays(raw: Optional[str], arch: str, total_ram: int) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    entries = []
    if raw:
        try:
            entries = json.loads(raw).get("SPDisplaysDataType", [])
        except (json.JSONDecodeError, AttributeError):
            entries = []
    for e in entries:
        name = e.get("sppci_model") or e.get("_name") or "GPU"
        vendor_raw = (e.get("spdisplays_vendor") or "").lower() + " " + name.lower()
        cores = e.get("sppci_cores")
        cores = int(cores) if isinstance(cores, str) and cores.isdigit() else None
        if "apple" in vendor_raw:
            gpus.append(GPUInfo(name=name, vendor="apple", vram_bytes=total_ram, unified=True, cores=cores))
            continue
        vendor = _vendor_from_name(vendor_raw)
        vram = _parse_size(e.get("spdisplays_vram") or e.get("spdisplays_vram_shared") or "")
        gpus.append(GPUInfo(name=name, vendor=vendor, vram_bytes=vram, unified=False))
    if not gpus and arch == "arm64":
        gpus.append(GPUInfo(name="Apple GPU", vendor="apple", vram_bytes=total_ram, unified=True))
    # 인텔 맥의 내장 그래픽(Intel)은 Metal 가속 효과가 작으므로 제외
    return [g for g in gpus if not (g.vendor == "intel" and not g.unified)] or gpus


def _detect_windows(runner: CommandRunner) -> SystemInfo:
    info = _base_info("Windows", platform.version())
    name = _windows_cpu_name()
    if name:
        info.cpu_name = name
    gpus = parse_nvidia_smi(runner(["nvidia-smi", "--query-gpu=name,memory.total",
                                    "--format=csv,noheader,nounits"]))
    known = {g.name for g in gpus}
    for g in _windows_registry_gpus():
        if g.name not in known and g.vendor != "nvidia":
            gpus.append(g)
    info.gpus = _drop_weak_integrated(gpus)
    return info


def _windows_cpu_name() -> Optional[str]:
    try:
        import winreg  # type: ignore

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
    except Exception:
        return None


def _windows_registry_gpus() -> list[GPUInfo]:
    """레지스트리에서 GPU 이름과 정확한 VRAM(qwMemorySize)을 읽는다.

    WMI의 AdapterRAM은 4GB에서 잘리기 때문에 레지스트리를 사용한다.
    """
    gpus: list[GPUInfo] = []
    try:
        import winreg  # type: ignore

        base = r"SYSTEM\ControlSet001\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
            for i in range(winreg.QueryInfoKey(root)[0]):
                sub = winreg.EnumKey(root, i)
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as k:
                        name = str(winreg.QueryValueEx(k, "DriverDesc")[0])
                        try:
                            vram = int(winreg.QueryValueEx(k, "HardwareInformation.qwMemorySize")[0])
                        except OSError:
                            vram = 0
                except OSError:
                    continue
                gpus.append(GPUInfo(name=name, vendor=_vendor_from_name(name.lower()), vram_bytes=vram))
    except Exception:
        pass
    return gpus


def _detect_linux(runner: CommandRunner) -> SystemInfo:
    info = _base_info("Linux", platform.release())
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info.cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpus = parse_nvidia_smi(runner(["nvidia-smi", "--query-gpu=name,memory.total",
                                    "--format=csv,noheader,nounits"]))
    gpus += parse_rocm_smi(runner(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"]))
    info.gpus = gpus
    return info


def parse_nvidia_smi(raw: Optional[str]) -> list[GPUInfo]:
    gpus = []
    for line in (raw or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            mib = float(parts[1])
        except ValueError:
            continue
        gpus.append(GPUInfo(name=parts[0], vendor="nvidia", vram_bytes=int(mib * MB)))
    return gpus


def parse_rocm_smi(raw: Optional[str]) -> list[GPUInfo]:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    gpus = []
    for card in data.values():
        if not isinstance(card, dict):
            continue
        total = card.get("VRAM Total Memory (B)")
        name = card.get("Card series") or card.get("Card Series") or "AMD GPU"
        if total:
            gpus.append(GPUInfo(name=name, vendor="amd", vram_bytes=int(total)))
    return gpus


def _drop_weak_integrated(gpus: list[GPUInfo]) -> list[GPUInfo]:
    """전용 GPU가 있으면 VRAM이 작은 내장 그래픽은 목록에서 뺀다."""
    if any(g.vram_bytes >= 4 * GB for g in gpus):
        return [g for g in gpus if g.vram_bytes >= 2 * GB]
    return gpus


def _vendor_from_name(text: str) -> str:
    text = text.lower()
    if "nvidia" in text or "geforce" in text or "quadro" in text or "rtx" in text:
        return "nvidia"
    if re.search(r"\b(amd|radeon|ati)\b", text):
        return "amd"
    if re.search(r"\b(intel|arc)\b", text):
        return "intel"
    if "apple" in text:
        return "apple"
    return "unknown"


def _parse_size(text: str) -> int:
    m = re.search(r"([\d.]+)\s*(GB|MB)", text, re.I)
    if not m:
        return 0
    value = float(m.group(1))
    return int(value * (GB if m.group(2).upper() == "GB" else MB))


def format_bytes(n: int) -> str:
    if n >= GB:
        return f"{n / GB:.1f} GB"
    return f"{n / MB:.0f} MB"
