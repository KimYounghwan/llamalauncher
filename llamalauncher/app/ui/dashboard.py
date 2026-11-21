"""대시보드 탭: 시스템 사양 카드 + 모델 크기별 적합도 미리보기."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.hardware import SystemInfo, detect_system, format_bytes
from ..core.recommender import Fit, ModelSpec, Preset, recommend, summary_sentence

PREVIEW_MODELS = [(1, "Q8_0"), (4, "Q4_K_M"), (8, "Q4_K_M"), (14, "Q4_K_M"),
                  (32, "Q4_K_M"), (70, "Q4_K_M"), (120, "Q4_K_M")]


class _DetectWorker(QObject):
    done = Signal(object)

    def __init__(self, disk_path: Path):
        super().__init__()
        self.disk_path = disk_path

    def run(self):
        self.done.emit(detect_system(self.disk_path))


class InfoCard(QFrame):
    def __init__(self, icon: str, title: str):
        super().__init__()
        self.setObjectName("card")
        lay = QVBoxLayout(self)
        head = QLabel(f"{icon}  {title}")
        head.setObjectName("cardTitle")
        self.value = QLabel("확인 중…")
        self.value.setObjectName("cardValue")
        self.value.setWordWrap(True)
        self.detail = QLabel("")
        self.detail.setObjectName("cardDetail")
        self.detail.setWordWrap(True)
        lay.addWidget(head)
        lay.addWidget(self.value)
        lay.addWidget(self.detail)
        lay.addStretch()

    def set(self, value: str, detail: str = ""):
        self.value.setText(value)
        self.detail.setText(detail)


class DashboardTab(QWidget):
    system_detected = Signal(object)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir
        self.system: SystemInfo | None = None
        self._thread: QThread | None = None

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        root = QVBoxLayout(body)
        root.setSpacing(14)

        top = QHBoxLayout()
        title = QLabel("내 컴퓨터 사양")
        title.setObjectName("pageTitle")
        self.refresh_btn = QPushButton("🔄 다시 확인")
        self.refresh_btn.clicked.connect(self.refresh)
        top.addWidget(title)
        top.addStretch()
        top.addWidget(self.refresh_btn)
        root.addLayout(top)

        self.tier_label = QLabel("사양을 확인하고 있어요…")
        self.tier_label.setObjectName("tierBanner")
        self.tier_label.setWordWrap(True)
        root.addWidget(self.tier_label)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards = {
            "os": InfoCard("💻", "운영체제"),
            "cpu": InfoCard("⚙️", "CPU"),
            "ram": InfoCard("🧮", "메모리 (RAM)"),
            "gpu": InfoCard("🎮", "GPU"),
            "budget": InfoCard("📦", "모델에 쓸 수 있는 메모리"),
            "disk": InfoCard("💾", "저장 공간"),
        }
        for i, card in enumerate(self.cards.values()):
            grid.addWidget(card, i // 3, i % 3)
        root.addLayout(grid)

        sub = QLabel("모델 크기별로 이 컴퓨터에서 어떻게 동작할까요?")
        sub.setObjectName("sectionTitle")
        root.addWidget(sub)
        hint = QLabel("‘B’는 모델의 파라미터 수(십억 개)예요. 클수록 똑똑하지만 메모리를 많이 써요. "
                      "‘Q4_K_M’ 같은 양자화는 모델을 압축한 정도예요.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["모델 크기", "파일 크기(예상)", "적합도", "추천 문맥 길이", "GPU 레이어"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setMinimumHeight(300)
        root.addWidget(self.table)
        root.addStretch()

        self.refresh()

    def refresh(self):
        if self._thread and self._thread.isRunning():
            return
        self.refresh_btn.setEnabled(False)
        self.tier_label.setText("사양을 확인하고 있어요…")
        self._thread = QThread(self)
        self._worker = _DetectWorker(self.models_dir)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_detected)
        self._worker.done.connect(self._thread.quit)
        self._thread.start()

    def _on_detected(self, info: SystemInfo):
        self.system = info
        self.refresh_btn.setEnabled(True)
        self.tier_label.setText(f"이 컴퓨터 등급: <b>{info.tier.value}</b><br>{summary_sentence(info)}")

        c = self.cards
        c["os"].set(f"{info.os_name} {info.os_version}", f"아키텍처: {info.arch}")
        perf = f", 성능 코어 {info.performance_cores}개" if info.performance_cores else ""
        c["cpu"].set(info.cpu_name, f"물리 코어 {info.physical_cores}개{perf}, 논리 코어 {info.logical_cores}개")
        c["ram"].set(format_bytes(info.total_ram), f"현재 사용 가능: {format_bytes(info.available_ram)}")
        gpu = info.primary_gpu
        if gpu:
            mem = "시스템 메모리 공유 (통합 메모리)" if gpu.unified else f"전용 메모리 {format_bytes(gpu.vram_bytes)}"
            cores = f", GPU 코어 {gpu.cores}개" if gpu.cores else ""
            c["gpu"].set(gpu.name, f"{info.backend.label} · {mem}{cores}")
        else:
            c["gpu"].set("가속 가능한 GPU 없음", "CPU로 실행해요. 작은 모델을 권장해요.")
        if info.is_unified_memory:
            c["budget"].set(format_bytes(info.gpu_budget), "macOS가 GPU에 허용하는 메모리 한도 기준")
        elif info.gpu_budget:
            c["budget"].set(f"GPU {format_bytes(info.gpu_budget)} + RAM {format_bytes(info.cpu_budget)}",
                            "GPU를 넘치는 부분은 RAM에서 느리게 실행돼요")
        else:
            c["budget"].set(format_bytes(info.cpu_budget), "시스템 RAM 기준")
        c["disk"].set(f"남은 공간 {format_bytes(info.disk_free)}", f"모델 저장 위치: {info.disk_path}")

        self.table.setRowCount(len(PREVIEW_MODELS))
        for row, (params, quant) in enumerate(PREVIEW_MODELS):
            spec = ModelSpec.estimate(params, quant)
            rec = recommend(info, spec, Preset.BALANCED)
            ngl = rec.options.n_gpu_layers
            ok = rec.fit.fit != Fit.INSUFFICIENT
            values = [
                f"{params}B ({quant})",
                format_bytes(spec.file_size),
                rec.fit.fit.badge,
                f"{rec.options.ctx_size:,} 토큰" if ok else "—",
                ("전부" if ngl >= 999 else ("CPU 전용" if ngl == 0 else f"{ngl}/{spec.layers}층")) if ok else "—",
            ]
            for col, v in enumerate(values):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 2:
                    item.setToolTip(rec.fit.reason)
                self.table.setItem(row, col, item)
        self.system_detected.emit(info)
