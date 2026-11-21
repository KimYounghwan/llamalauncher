"""연동 탭: 실행 중인 모델을 Claude Code·OpenCode·Continue 등 개발 도구에 연결하는 설정을 만들고 적용한다."""

from __future__ import annotations

import html
import re
import shutil
import sys
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, QProcess, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QVBoxLayout, QWidget,
)

from ..core.integrations import CATEGORY_LABELS, Context, Integration, MergeError, StateStore, Tester, load_all
from ..core.settings import config_dir

MONO = QFont("Menlo" if sys.platform == "darwin" else "Consolas", 11)
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def copy_text(text: str) -> None:
    QGuiApplication.clipboard().setText(text)


class DiffDialog(QDialog):
    """적용 전에 바뀌는 내용을 보여주고 확인받는다."""

    def __init__(self, parent, title: str, text: str, warnings: list[str]):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 520)
        lay = QVBoxLayout(self)
        intro = QLabel("아래처럼 설정 파일이 바뀌어요. 기존 설정은 그대로 두고 필요한 부분만 추가·수정하며, "
                       "원본은 <code>.bak-날짜</code> 파일로 백업해요.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        for w in warnings:
            lb = QLabel("⚠️ " + html.escape(w))
            lb.setObjectName("errorBanner")
            lb.setWordWrap(True)
            lay.addWidget(lb)
        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setFont(MONO)
        lay.addWidget(view, 1)
        bb = QDialogButtonBox()
        ok = bb.addButton("적용하기", QDialogButtonBox.AcceptRole)
        ok.setObjectName("primary")
        bb.addButton("취소", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


class _Bridge(QObject):
    result = Signal(object)
    done = Signal()


class IntegrationsTab(QWidget):
    go_to_server = Signal()
    agent_preset_requested = Signal()
    find_models = Signal(str)   # 카테고리 id

    def __init__(self, store: Optional[StateStore] = None):
        super().__init__()
        self.store = store or StateStore(config_dir() / "integrations.json")
        self.items: list[Integration] = load_all()
        self.detected: dict[str, bool] = {}
        self.context_provider: Callable[[], Context] = Context
        self.ctx = Context()
        self.current: Optional[Integration] = None
        self._testing = False
        self._bridge = _Bridge()
        self._bridge.result.connect(self._on_test_result)
        self._bridge.done.connect(self._on_test_done)
        self.tunnel: Optional[QProcess] = None
        self.tunnel_url = ""

        root = QVBoxLayout(self)
        title = QLabel("🔌 개발 도구 연동")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        hint = QLabel("실행 중인 모델을 코딩 도구에서 바로 쓸 수 있게 설정을 만들어 줘요. 도구를 고르면 이 컴퓨터의 "
                      "서버 주소·모델 이름·키가 채워진 설정이 나오고, 설정 파일에 자동으로 적용할 수도 있어요.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        split = QSplitter(Qt.Horizontal)
        self.list = QListWidget()
        self.list.setMinimumWidth(240)
        self.list.currentRowChanged.connect(self._on_select)
        split.addWidget(self.list)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        self.body = QVBoxLayout(body)
        self._build_detail()
        scroll.setWidget(body)
        split.addWidget(scroll)
        split.setSizes([260, 900])
        root.addWidget(split, 1)

        self._fill_list()
        self.list.setCurrentRow(0)

    # ------------------------------------------------------------------ UI
    def _build_detail(self) -> None:
        b = self.body
        self.d_title = QLabel()
        self.d_title.setObjectName("sectionTitle")
        self.d_title.setTextFormat(Qt.RichText)
        self.d_title.setOpenExternalLinks(True)
        self.d_title.setWordWrap(True)
        b.addWidget(self.d_title)

        srv = QFrame()
        srv.setObjectName("card")
        sl = QHBoxLayout(srv)
        self.srv_lbl = QLabel()
        self.srv_lbl.setWordWrap(True)
        self.srv_lbl.setTextFormat(Qt.RichText)
        sl.addWidget(self.srv_lbl, 1)
        self.agent_btn = QPushButton("🪄 코딩 에이전트용으로 설정")
        self.agent_btn.setToolTip("서버 탭을 '코딩 에이전트용' 프리셋(64K 문맥, 도구 호출, API 키)으로 바꿔요.")
        self.agent_btn.clicked.connect(self.agent_preset_requested.emit)
        self.server_btn = QPushButton("▶️ 서버 탭")
        self.server_btn.clicked.connect(self.go_to_server.emit)
        sl.addWidget(self.agent_btn)
        sl.addWidget(self.server_btn)
        b.addWidget(srv)

        b.addWidget(self._section("호환성 점검"))
        self.checks_lbl = QLabel()
        self.checks_lbl.setWordWrap(True)
        self.checks_lbl.setTextFormat(Qt.RichText)
        b.addWidget(self.checks_lbl)
        self.models_btn = QPushButton("🔍 이 도구에 맞는 모델 찾기")
        self.models_btn.clicked.connect(lambda: self.find_models.emit(
            "coding" if self.current and self.current.category in ("agent", "edit") else "recommended"))
        row = QHBoxLayout()
        row.addWidget(self.models_btn)
        row.addStretch(1)
        b.addLayout(row)

        # 파일 설정
        self.file_box = QWidget()
        fl = QVBoxLayout(self.file_box)
        fl.setContentsMargins(0, 0, 0, 0)
        fl.addWidget(self._section("설정 내용"))
        self.files_lbl = QLabel()
        self.files_lbl.setObjectName("hint")
        self.files_lbl.setWordWrap(True)
        self.files_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl.addWidget(self.files_lbl)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setFont(MONO)
        self.preview.setMinimumHeight(220)
        fl.addWidget(self.preview)
        frow = QHBoxLayout()
        self.copy_btn = QPushButton("📋 복사")
        self.copy_btn.clicked.connect(lambda: (copy_text(self.preview.toPlainText()), self._flash("복사했어요.")))
        self.apply_btn = QPushButton("⚙️ 설정 파일에 자동 적용")
        self.apply_btn.setObjectName("primary")
        self.apply_btn.clicked.connect(self._apply)
        self.revert_btn = QPushButton("↩️ 원래대로 되돌리기")
        self.revert_btn.clicked.connect(self._revert)
        for w in (self.copy_btn, self.apply_btn, self.revert_btn):
            frow.addWidget(w)
        frow.addStretch(1)
        fl.addLayout(frow)
        b.addWidget(self.file_box)

        # GUI 안내
        self.gui_box = QWidget()
        gl = QVBoxLayout(self.gui_box)
        gl.setContentsMargins(0, 0, 0, 0)
        gl.addWidget(self._section("설정 방법 (도구 화면에서 직접 입력)"))
        self.steps_lbl = QLabel()
        self.steps_lbl.setWordWrap(True)
        self.steps_lbl.setTextFormat(Qt.RichText)
        gl.addWidget(self.steps_lbl)
        self.values_box = QVBoxLayout()
        gl.addLayout(self.values_box)
        b.addWidget(self.gui_box)

        # 터널 (Cursor)
        self.tunnel_box = QWidget()
        tl = QHBoxLayout(self.tunnel_box)
        tl.setContentsMargins(0, 0, 0, 0)
        self.tunnel_btn = QPushButton("🌐 터널 열기")
        self.tunnel_btn.clicked.connect(self._toggle_tunnel)
        self.tunnel_lbl = QLabel()
        self.tunnel_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.tunnel_lbl.setWordWrap(True)
        tl.addWidget(self.tunnel_btn)
        tl.addWidget(self.tunnel_lbl, 1)
        b.addWidget(self.tunnel_box)

        # 코드 예제
        self.snip_box = QWidget()
        nl = QVBoxLayout(self.snip_box)
        nl.setContentsMargins(0, 0, 0, 0)
        nl.addWidget(self._section("예제 코드"))
        nrow = QHBoxLayout()
        self.snip_combo = QComboBox()
        self.snip_combo.currentIndexChanged.connect(self._show_snippet)
        snip_copy = QPushButton("📋 복사")
        snip_copy.clicked.connect(lambda: (copy_text(self.snip_view.toPlainText()), self._flash("복사했어요.")))
        nrow.addWidget(self.snip_combo, 1)
        nrow.addWidget(snip_copy)
        nl.addLayout(nrow)
        self.snip_view = QPlainTextEdit()
        self.snip_view.setReadOnly(True)
        self.snip_view.setFont(MONO)
        self.snip_view.setMinimumHeight(200)
        nl.addWidget(self.snip_view)
        b.addWidget(self.snip_box)

        # 연결 테스트
        b.addWidget(self._section("연결 테스트"))
        trow = QHBoxLayout()
        self.test_btn = QPushButton("🧪 연결 테스트")
        self.test_btn.setToolTip("이 도구가 쓰는 API로 실제 요청을 보내 답변과 도구 호출을 확인해요.")
        self.test_btn.clicked.connect(self._test)
        trow.addWidget(self.test_btn)
        trow.addStretch(1)
        b.addLayout(trow)
        self.test_lbl = QLabel()
        self.test_lbl.setWordWrap(True)
        self.test_lbl.setTextFormat(Qt.RichText)
        self.test_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        b.addWidget(self.test_lbl)

        self.notes_lbl = QLabel()
        self.notes_lbl.setObjectName("hint")
        self.notes_lbl.setWordWrap(True)
        self.notes_lbl.setTextFormat(Qt.RichText)
        b.addWidget(self.notes_lbl)
        self.flash_lbl = QLabel()
        self.flash_lbl.setObjectName("hint")
        b.addWidget(self.flash_lbl)
        b.addStretch(1)

    @staticmethod
    def _section(text: str) -> QLabel:
        lb = QLabel(text)
        lb.setObjectName("sectionTitle")
        return lb

    def _flash(self, text: str) -> None:
        self.flash_lbl.setText(text)

    def _fill_list(self) -> None:
        cur = self.list.currentRow()
        self.list.blockSignals(True)
        self.list.clear()
        for it in self.items:
            mark = " ✅" if it.applied(self.store) else ""
            det = "" if not self.detected.get(it.id) else "  · 설치됨"
            li = QListWidgetItem(f"{it.icon}  {it.name}{mark}{det}")
            li.setToolTip(f"{CATEGORY_LABELS.get(it.category, '')} — {it.description}")
            self.list.addItem(li)
        self.list.blockSignals(False)
        if cur >= 0:
            self.list.setCurrentRow(cur)

    def detect_tools(self) -> None:
        from .workers import run_in_background
        items = list(self.items)
        run_in_background(lambda: {i.id: i.detect() for i in items}, self._on_detected)

    def _on_detected(self, res: dict) -> None:
        self.detected = res
        self._fill_list()

    # ------------------------------------------------------------------ 상태 갱신
    def refresh(self) -> None:
        """서버 상태가 바뀌거나 탭을 열 때 값을 다시 채운다."""
        try:
            self.ctx = self.context_provider()
        except Exception:  # noqa: BLE001
            self.ctx = Context()
        c = self.ctx
        state = "🟢 실행 중" if c.running else "⚪ 꺼짐 (서버 탭 설정 기준)"
        key = "🔑 API 키 있음" if c.api_key else "API 키 없음 (아무 값이나 넣어도 돼요)"
        tools = {True: "🛠 도구 호출 지원", False: "도구 호출 미지원", None: "도구 호출 확인 안 됨"}[c.supports_tools]
        self.srv_lbl.setText(
            f"<b>연결할 서버</b> {state} · {html.escape(c.engine_name)}<br>"
            f"주소 <code>{html.escape(c.api_base)}</code> · 모델 이름 <code>{html.escape(c.model)}</code> · "
            f"문맥 {c.context:,} · {key} · {tools}")
        self._render_current()

    def _on_select(self, row: int) -> None:
        self.current = self.items[row] if 0 <= row < len(self.items) else None
        self.test_lbl.clear()
        self.flash_lbl.clear()
        self.refresh()

    def _render_current(self) -> None:
        it = self.current
        if not it:
            return
        c = self.ctx
        docs = f" · <a href='{it.docs_url}'>공식 문서</a> (확인: {it.checked})" if it.docs_url else ""
        if it.verified:
            docs += f" · ✔️ {html.escape(it.verified)}"
        det = " · 설치됨" if self.detected.get(it.id) else ""
        self.d_title.setText(f"{it.icon} {html.escape(it.name)} <span style='color:gray; font-size:13px'>"
                             f"· {CATEGORY_LABELS.get(it.category, '')}{det}{docs}</span><br>"
                             f"<span style='font-size:13px; font-weight:400'>{html.escape(it.description)}</span>")
        checks = it.check(c)
        self.checks_lbl.setText("<br>".join(f"{ch.icon} {html.escape(ch.text)}" for ch in checks))
        self.agent_btn.setVisible(it.category == "agent" and (c.context < it.recommended_context or not c.api_key))

        self.file_box.setVisible(it.kind == "file")
        if it.kind == "file":
            self.files_lbl.setText("설정 파일: " + ", ".join(str(p) for p, _, _ in it.rendered_files(c)))
            self.preview.setPlainText(it.preview(c))
            applied = it.applied(self.store)
            self.revert_btn.setEnabled(applied)
            self.apply_btn.setText("⚙️ 다시 적용 (현재 값으로)" if applied else "⚙️ 설정 파일에 자동 적용")
        self.gui_box.setVisible(bool(it.steps or it.copy_values))
        if it.steps or it.copy_values:
            steps = it.rendered_steps(c)
            self.steps_lbl.setText("".join(f"<p>{i}. {self._md_bold(s)}</p>" for i, s in enumerate(steps, 1)))
            self._fill_values(it.gui_values(c))
        self.tunnel_box.setVisible(it.tunnel)
        if it.tunnel:
            self._update_tunnel_ui()
        self.snip_box.setVisible(bool(it.snippets))
        if it.snippets:
            self.snip_combo.blockSignals(True)
            idx = max(0, self.snip_combo.currentIndex())
            self.snip_combo.clear()
            for t, _ in it.rendered_snippets(c):
                self.snip_combo.addItem(t)
            self.snip_combo.setCurrentIndex(min(idx, self.snip_combo.count() - 1))
            self.snip_combo.blockSignals(False)
            self._show_snippet()
        self.test_btn.setEnabled(c.running and not self._testing)
        self.test_btn.setToolTip("" if c.running else "서버를 켜야 테스트할 수 있어요.")
        notes = list(it.notes)
        env = it.env_vars(c)
        if env and it.kind == "file":
            notes.append("API 키는 환경 변수로 전달해요. 터미널 설정 파일(~/.zshrc 등)에 위 'export' 줄을 추가하세요.")
        self.notes_lbl.setText("".join(f"<p>ℹ️ {html.escape(n)}</p>" for n in notes))

    @staticmethod
    def _md_bold(s: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(s))

    def _fill_values(self, values: list[tuple[str, str]]) -> None:
        while self.values_box.count():
            item = self.values_box.takeAt(0)
            w = item.widget() or (item.layout() and None)
            if item.layout():
                lay = item.layout()
                while lay.count():
                    ww = lay.takeAt(0).widget()
                    if ww:
                        ww.deleteLater()
            if w:
                w.deleteLater()
        for label, value in values:
            row = QHBoxLayout()
            lb = QLabel(label)
            lb.setMinimumWidth(120)
            ed = QLineEdit(str(value))
            ed.setReadOnly(True)
            ed.setFont(MONO)
            btn = QPushButton("📋")
            btn.setToolTip(f"{label} 복사")
            btn.clicked.connect(lambda _=False, v=str(value), l=label: (copy_text(v), self._flash(f"{l}을(를) 복사했어요.")))
            row.addWidget(lb)
            row.addWidget(ed, 1)
            row.addWidget(btn)
            self.values_box.addLayout(row)

    def _show_snippet(self) -> None:
        it = self.current
        if not it or not it.snippets:
            return
        snips = it.rendered_snippets(self.ctx)
        i = self.snip_combo.currentIndex()
        if 0 <= i < len(snips):
            self.snip_view.setPlainText(snips[i][1])

    # ------------------------------------------------------------------ 적용/되돌리기
    def _apply(self) -> None:
        it = self.current
        if not it:
            return
        errors = [ch for ch in it.check(self.ctx) if ch.level == "error"]
        if errors and QMessageBox.warning(
                self, "호환성 문제가 있어요", "\n".join("❌ " + e.text for e in errors) + "\n\n그래도 적용할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            plans = it.plans(self.ctx)
        except MergeError as e:
            QMessageBox.warning(self, "적용할 수 없어요", f"{e}\n\n위의 설정 내용을 복사해 직접 붙여 넣어 주세요.")
            return
        if not any(p.changed for p in plans):
            QMessageBox.information(self, "이미 적용됨", "설정 파일에 이미 같은 내용이 들어 있어요.")
            return
        diff = "\n\n".join(f"# {p.path}{' (새 파일)' if not p.existed else ''}\n{p.diff()}" for p in plans if p.changed)
        warnings = [w for p in plans for w in p.warnings]
        if DiffDialog(self, f"{it.name} 설정 적용", diff, warnings).exec() != QDialog.Accepted:
            return
        try:
            it.apply(plans, self.store)
        except OSError as e:
            QMessageBox.warning(self, "적용 실패", f"설정 파일을 쓰지 못했어요: {e}")
            return
        msg = f"{it.name} 설정을 적용했어요. 도구를 다시 시작하면 반영돼요."
        if it.env_vars(self.ctx):
            msg += "\n\nAPI 키 환경 변수도 설정해야 해요 (설정 내용 맨 아래 참고)."
        QMessageBox.information(self, "적용 완료", msg)
        self._fill_list()
        self._render_current()

    def _revert(self) -> None:
        it = self.current
        if not it or not it.applied(self.store):
            return
        if QMessageBox.question(self, "원래대로 되돌리기",
                                f"{it.name} 설정 파일을 라마런처가 처음 적용하기 전 상태로 되돌릴까요?") != QMessageBox.Yes:
            return
        try:
            msgs = it.revert(self.store)
        except (OSError, MergeError) as e:
            QMessageBox.warning(self, "되돌리기 실패", str(e))
            return
        QMessageBox.information(self, "되돌렸어요", "\n".join(msgs))
        self._fill_list()
        self._render_current()

    # ------------------------------------------------------------------ 연결 테스트
    def _test(self) -> None:
        it = self.current
        if not it or self._testing or not self.ctx.running:
            return
        self._testing = True
        self.test_btn.setEnabled(False)
        self.test_lbl.setText("테스트 중… (큰 모델은 몇십 초 걸릴 수 있어요)")
        self._results: list = []
        c = self.ctx
        base = (c.anthropic_base_url or c.base_url) if it.api == "messages" else c.base_url
        tester = Tester(base, c.model, c.api_key)
        api, needs = it.api, it.needs_tools or it.category == "agent"

        def work():
            tester.for_api(api, needs, self._bridge.result.emit)
            self._bridge.done.emit()

        threading.Thread(target=work, daemon=True, name="integration-test").start()

    def _on_test_result(self, r) -> None:
        self._results.append(r)
        self.test_lbl.setText("<br>".join(f"{x.icon} <b>{html.escape(x.name)}</b> — {html.escape(x.detail)} "
                                          f"<span style='color:gray'>({x.seconds:.1f}초)</span>" for x in self._results))

    def _on_test_done(self) -> None:
        self._testing = False
        ok = all(r.ok for r in self._results)
        self.test_lbl.setText(self.test_lbl.text() + ("<br><b>🎉 이 도구에서 쓸 준비가 됐어요.</b>" if ok else
                                                      "<br><b>일부 항목이 실패했어요. 위 안내를 확인해 주세요.</b>"))
        self.test_btn.setEnabled(self.ctx.running)

    # ------------------------------------------------------------------ 터널 (Cursor)
    def _update_tunnel_ui(self) -> None:
        exe = shutil.which("cloudflared")
        running = self.tunnel is not None and self.tunnel.state() != QProcess.NotRunning
        self.tunnel_btn.setEnabled(bool(exe) and (running or self.ctx.running))
        self.tunnel_btn.setText("⏹ 터널 닫기" if running else "🌐 터널 열기")
        if not exe:
            how = "brew install cloudflared" if sys.platform == "darwin" else (
                "winget install Cloudflare.cloudflared" if sys.platform == "win32" else "패키지 관리자로 cloudflared 설치")
            self.tunnel_lbl.setText(f"cloudflared가 필요해요: {how}")
        elif running:
            self.tunnel_lbl.setText(f"공개 주소: {self.tunnel_url}/v1" if self.tunnel_url else "터널을 여는 중…")
        else:
            self.tunnel_lbl.setText("터널을 열면 Cursor에 넣을 인터넷 주소가 여기에 나와요.")

    def _toggle_tunnel(self) -> None:
        if self.tunnel is not None and self.tunnel.state() != QProcess.NotRunning:
            self.tunnel.kill()
            self.tunnel.waitForFinished(3000)
            self.tunnel, self.tunnel_url = None, ""
            self._update_tunnel_ui()
            return
        if not self.ctx.api_key and QMessageBox.warning(
                self, "API 키가 없어요", "API 키 없이 터널을 열면 인터넷의 누구나 이 컴퓨터의 모델을 쓸 수 있어요.\n"
                "서버 탭에서 API 키를 설정한 뒤 다시 시작하는 것을 강력히 권장해요.\n\n그래도 열까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.tunnel = QProcess(self)
        self.tunnel.setProcessChannelMode(QProcess.MergedChannels)
        self.tunnel.readyReadStandardOutput.connect(self._on_tunnel_output)
        self.tunnel.finished.connect(lambda *_: self._update_tunnel_ui())
        self.tunnel.start(shutil.which("cloudflared"), ["tunnel", "--url", self.ctx.base_url])
        self._update_tunnel_ui()

    def _on_tunnel_output(self) -> None:
        text = bytes(self.tunnel.readAllStandardOutput()).decode("utf-8", errors="replace")
        m = TUNNEL_RE.search(text)
        if m and not self.tunnel_url:
            self.tunnel_url = m.group(0)
            self._update_tunnel_ui()

    def shutdown(self) -> None:
        if self.tunnel is not None and self.tunnel.state() != QProcess.NotRunning:
            self.tunnel.kill()
            self.tunnel.waitForFinished(2000)

    def showEvent(self, e):
        super().showEvent(e)
        self.refresh()
