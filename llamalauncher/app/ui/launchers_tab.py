"""런처 탭: 여러 실행 엔진의 설치 상태를 보여주고 설치/업데이트/제거한다."""

from __future__ import annotations

import html
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtWidgets import (
    QFileDialog, QFrame, QHBoxLayout, QLabel, QMenu, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from ..core.hardware import SystemInfo
from ..core.installer import Installer, InstallError
from ..core.launchers import FORMAT_LABELS, Launcher, LauncherStatus, all_launchers
from ..core.launchers.base import CommandStep, GithubReleaseStep, InstallPlan, OpenUrlStep, RemovePathStep, VenvPipStep
from ..core.settings import Settings
from .workers import run_in_background


def describe_plan(plan: InstallPlan) -> str:
    """설치 전에 무엇을 할지 사람이 읽을 수 있게."""
    lines = []
    for st in plan.steps:
        if isinstance(st, CommandStep):
            lines.append("• 명령 실행: " + " ".join(st.args))
        elif isinstance(st, GithubReleaseStep):
            lines.append(f"• GitHub {st.repo} 최신 배포에서 이 컴퓨터용 파일을 받아 {st.dest}에 설치")
        elif isinstance(st, VenvPipStep):
            lines.append(f"• 전용 파이썬 환경({st.venv})에 설치: pip install {' '.join(st.packages)}")
        elif isinstance(st, RemovePathStep):
            lines.append(f"• 삭제: {st.path}")
        elif isinstance(st, OpenUrlStep):
            lines.append(f"• 브라우저로 열기: {st.url}")
    if plan.note:
        lines.append("\n" + plan.note)
    return "\n".join(lines)


class _Bridge(QObject):
    log = Signal(str)
    finished = Signal(str, object)   # launcher id, 오류(None이면 성공)


class LauncherCard(QFrame):
    def __init__(self, tab: "LaunchersTab", launcher: Launcher):
        super().__init__()
        self.setObjectName("card")
        self.tab = tab
        self.launcher = launcher
        self.status: Optional[LauncherStatus] = None
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        kind = "서버형 — 라마런처에서 바로 실행" if launcher.runnable else "앱형 — 따로 열어서 사용"
        self.title = QLabel(f"<b style='font-size:15px'>{html.escape(launcher.name)}</b> "
                            f"<span style='color:gray'>· {kind}</span>")
        top.addWidget(self.title, 1)
        self.state = QLabel("확인 중…")
        top.addWidget(self.state)
        lay.addLayout(top)
        fmts = ", ".join(FORMAT_LABELS.get(f, f) for f in launcher.formats)
        desc = QLabel(f"{html.escape(launcher.description)}<br><span style='color:gray'>모델 형식: {fmts}</span>")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        self.detail = QLabel()
        self.detail.setObjectName("cardDetail")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.detail)

        row = QHBoxLayout()
        self.install_btn = QPushButton("⬇️ 설치")
        self.install_btn.setObjectName("primary")
        self.update_btn = QPushButton("🔄 업데이트")
        self.remove_btn = QPushButton("🗑 제거")
        self.folder_btn = QPushButton("📂 폴더 열기")
        self.path_btn = QPushButton("경로 지정…")
        self.open_btn = QPushButton("↗ 앱 열기")
        self.use_btn = QPushButton("▶ 이 엔진으로 실행")
        self.models_btn = QPushButton("🔍 맞는 모델 찾기")
        self.web_btn = QPushButton("🌐 홈페이지")
        for b in (self.install_btn, self.update_btn, self.remove_btn, self.use_btn, self.models_btn, self.open_btn,
                  self.folder_btn, self.path_btn, self.web_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.install_btn.clicked.connect(lambda: self._run_plans(self.launcher.install_plans(self.tab.system), "설치"))
        self.update_btn.clicked.connect(
            lambda: self._run_plans(self.launcher.update_plans(self.tab.system, self.status), "업데이트"))
        self.remove_btn.clicked.connect(lambda: self._run_plans(self.launcher.uninstall_plans(self.status), "제거"))
        self.folder_btn.clicked.connect(self._open_folder)
        self.path_btn.clicked.connect(self._pick_path)
        self.open_btn.clicked.connect(self._open_app)
        self.use_btn.clicked.connect(lambda: self.tab.use_engine.emit(self.launcher.id))
        self.models_btn.clicked.connect(lambda: self.tab.find_models.emit(self.launcher.formats[0]))
        self.web_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(self.launcher.homepage)))
        self.set_status(None)

    def set_status(self, st: Optional[LauncherStatus]) -> None:
        self.status = st
        ok, why = self.launcher.availability(self.tab.system)
        busy = self.tab.busy
        installed = bool(st and st.installed)
        if st is None:
            self.state.setText("확인 중…")
            self.detail.setText("")
        elif not ok and not installed:
            self.state.setText("🚫 사용 불가")
            self.detail.setText(html.escape(why))
        elif installed:
            run = f" · 🟢 실행 중 (포트 {self.launcher.default_port})" if st.running else ""
            self.state.setText("✅ 설치됨" + run)
            self.detail.setText(f"버전 {html.escape(st.version)} · {html.escape(str(st.path))}")
        else:
            self.state.setText("❌ 설치 안 됨" + (" · 🟢 실행 중" if st.running else ""))
            self.detail.setText(html.escape(st.message))
        plans = self.launcher.install_plans(self.tab.system) if ok else []
        self.install_btn.setVisible(not installed and bool(plans))
        self.install_btn.setEnabled(not busy)
        if plans and plans[0].manual:
            self.install_btn.setText("🌐 다운로드 페이지에서 설치")
        elif len(plans) > 1:
            self.install_btn.setText(f"⬇️ 설치 ({plans[0].label})")
        else:
            self.install_btn.setText("⬇️ 설치")
        self.update_btn.setVisible(installed and bool(st and self.launcher.update_plans(self.tab.system, st)))
        self.remove_btn.setVisible(installed and bool(st and self.launcher.uninstall_plans(st)))
        for b in (self.update_btn, self.remove_btn):
            b.setEnabled(not busy)
        self.folder_btn.setVisible(installed)
        self.open_btn.setVisible(installed and not self.launcher.runnable)
        self.use_btn.setVisible(installed and self.launcher.runnable)
        self.models_btn.setVisible(installed or ok)
        self.path_btn.setVisible(ok and self.launcher.kind == "server")

    def _run_plans(self, plans: list[InstallPlan], verb: str) -> None:
        if not plans:
            return
        if len(plans) == 1:
            self.tab.run_plan(self.launcher, plans[0], verb)
            return
        menu = QMenu(self)
        for p in plans:
            a = menu.addAction(p.label)
            a.triggered.connect(lambda _=False, p=p: self.tab.run_plan(self.launcher, p, verb))
        btn = self.sender() if isinstance(self.sender(), QPushButton) else self.install_btn
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _open_folder(self) -> None:
        if self.status and self.status.path:
            p = Path(self.status.path)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(p if p.is_dir() and p.suffix != ".app" else p.parent)))

    def _open_app(self) -> None:
        if not (self.status and self.status.path):
            return
        p = str(self.status.path)
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", p])
            else:
                QDesktopServices.openUrl(QUrl.fromLocalFile(p))
        except OSError as e:
            QMessageBox.warning(self, "열 수 없어요", str(e))

    def _pick_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, f"{self.launcher.name} 실행 파일 선택", "")
        if path:
            self.tab.settings.launcher_paths[self.launcher.id] = path
            self.tab.save_settings()
            self.tab.detect(self.launcher.id)


class LaunchersTab(QWidget):
    use_engine = Signal(str)          # 서버 탭에서 이 엔진 사용
    find_models = Signal(str)         # 모델 찾기 탭에서 이 형식으로 검색
    statuses_changed = Signal(object)  # {launcher id: LauncherStatus}

    def __init__(self, settings: Settings, save_settings: Callable[[], None]):
        super().__init__()
        self.settings = settings
        self.save_settings = save_settings
        self.system: Optional[SystemInfo] = None
        self.launchers = all_launchers()
        self.statuses: dict[str, LauncherStatus] = {}
        self._installer: Optional[Installer] = None
        self._bridge = _Bridge()
        self._bridge.log.connect(self._log)
        self._bridge.finished.connect(self._on_finished)

        root = QVBoxLayout(self)
        title = QLabel("🚀 런처 (실행 엔진)")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        hint = QLabel("AI 모델을 실제로 돌리는 프로그램들이에요. 처음이라면 <b>llama.cpp</b> 하나면 충분해요. "
                      "<b>서버형</b>은 라마런처의 [서버] 탭에서 바로 실행하고, <b>앱형</b>은 그 프로그램을 열어서 써요.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)
        bar = QHBoxLayout()
        bar.addStretch(1)
        self.refresh_btn = QPushButton("🔄 다시 확인")
        self.refresh_btn.clicked.connect(lambda: self.detect())
        bar.addWidget(self.refresh_btn)
        root.addLayout(bar)

        split = QSplitter(Qt.Vertical)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        il = QVBoxLayout(inner)
        self.cards: dict[str, LauncherCard] = {}
        for l in self.launchers:
            c = LauncherCard(self, l)
            self.cards[l.id] = c
            il.addWidget(c)
        il.addStretch(1)
        scroll.setWidget(inner)
        split.addWidget(scroll)

        logw = QWidget()
        ll = QVBoxLayout(logw)
        ll.setContentsMargins(0, 0, 0, 0)
        lrow = QHBoxLayout()
        self.log_title = QLabel("설치 기록")
        self.log_title.setObjectName("sectionTitle")
        lrow.addWidget(self.log_title, 1)
        self.cancel_btn = QPushButton("⏹ 취소")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        lrow.addWidget(self.cancel_btn)
        ll.addLayout(lrow)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Menlo" if sys.platform == "darwin" else "Consolas", 11))
        self.log_view.setPlaceholderText("설치·업데이트를 시작하면 진행 상황이 여기에 나와요.")
        ll.addWidget(self.log_view)
        split.addWidget(logw)
        split.setSizes([560, 200])
        root.addWidget(split, 1)
        self.detect()

    @property
    def busy(self) -> bool:
        return self._installer is not None

    def set_system(self, info: SystemInfo) -> None:
        self.system = info
        for l in self.launchers:
            self.cards[l.id].set_status(self.statuses.get(l.id))

    def detect(self, only: Optional[str] = None) -> None:
        for l in self.launchers:
            if only and l.id != only:
                continue
            custom = self.settings.launcher_paths.get(l.id)
            self.cards[l.id].set_status(None)
            run_in_background(lambda l=l, c=custom: (l.id, l.detect(c)), self._on_detected,
                              lambda e, l=l: self._on_detected((l.id, LauncherStatus(installed=False, message=str(e)))))

    def _on_detected(self, result) -> None:
        lid, st = result
        self.statuses[lid] = st
        self.cards[lid].set_status(st)
        self.statuses_changed.emit(dict(self.statuses))

    # ------------------------------------------------------------------ 설치 실행
    def run_plan(self, launcher: Launcher, plan: InstallPlan, verb: str) -> None:
        if self.busy:
            QMessageBox.information(self, "진행 중", "다른 설치가 진행 중이에요. 끝난 뒤 다시 시도해 주세요.")
            return
        if not plan.manual:
            warn = "\n\n⚠️ 지우면 되돌릴 수 없어요." if verb == "제거" else ""
            if QMessageBox.question(self, f"{launcher.name} {verb}",
                                    f"{launcher.name}을(를) {verb}할까요? 다음 작업을 해요:\n\n{describe_plan(plan)}{warn}"
                                    ) != QMessageBox.Yes:
                return
        self.log_view.clear()
        self.log_title.setText(f"설치 기록 — {launcher.name} {verb} 중…")
        self.cancel_btn.setEnabled(True)
        self._installer = Installer(log=self._bridge.log.emit,
                                    open_url=lambda u: self._bridge.log.emit("@open " + u))
        inst = self._installer

        def work():
            err = None
            try:
                inst.run(plan)
            except InstallError as e:
                err = str(e)
            except Exception as e:  # noqa: BLE001
                err = f"예상하지 못한 오류: {e}"
            self._bridge.finished.emit(launcher.id, err)

        threading.Thread(target=work, daemon=True, name=f"install-{launcher.id}").start()
        for c in self.cards.values():
            c.set_status(c.status)

    def cancel(self) -> None:
        if self._installer:
            self._installer.stop()

    def _log(self, line: str) -> None:
        if line.startswith("@open "):
            QDesktopServices.openUrl(QUrl(line[6:]))
            return
        self.log_view.appendPlainText(line)

    def _on_finished(self, lid: str, err) -> None:
        self._installer = None
        self.cancel_btn.setEnabled(False)
        name = self.cards[lid].launcher.name
        if err:
            self.log_view.appendPlainText(f"❌ {err}")
            self.log_title.setText(f"설치 기록 — {name}: 실패")
        else:
            self.log_title.setText(f"설치 기록 — {name}: 완료")
        self.detect(lid)

    def shutdown(self) -> None:
        self.cancel()
