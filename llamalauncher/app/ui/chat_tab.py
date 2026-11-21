"""채팅 테스트 탭: 실행 중인 서버에 질문하고 답변을 실시간(스트리밍)으로 보여준다."""

from __future__ import annotations

import html
import re
import threading
from dataclasses import dataclass
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QTextBrowser,
    QVBoxLayout, QWidget,
)

from ..core.chat_client import ChatEvent, ChatRequest, stream_chat
from ..core.server import ServerState
from ..core.settings import Settings

STARTERS = [
    ("👋 자기소개", "안녕! 너는 누구고 무엇을 잘하는지 세 줄로 소개해 줘."),
    ("🧮 간단한 계산", "사과가 3개 있었는데 2개를 더 사고 1개를 먹었어. 몇 개 남았는지 풀이와 함께 알려줘."),
    ("💻 코드 작성", "파이썬으로 피보나치 수열의 n번째 값을 구하는 함수를 짧게 작성해 줘."),
    ("🌏 번역", "다음 문장을 영어로 번역해 줘: 오늘은 날씨가 좋아서 산책하기 딱 좋아요."),
]


@dataclass
class Message:
    role: str            # user | assistant | error
    content: str = ""
    reasoning: str = ""
    stats: str = ""


# ---------------------------------------------------------------- 표시용 변환
def split_think(text: str) -> tuple[str, str]:
    """서버가 생각 과정을 분리해 주지 않을 때 <think>...</think>를 직접 나눈다. (생각, 답변)"""
    m = re.match(r"\s*<think>(.*?)(</think>|$)(.*)", text, re.S)
    if not m:
        return "", text
    return m.group(1).strip(), m.group(3).lstrip()


_CODE_RE = re.compile(r"```[^\n`]*\n?(.*?)(```|$)", re.S)


def render_markdown_lite(text: str, code_bg: str) -> str:
    """코드 블록·인라인 코드·굵은 글씨·줄바꿈 정도만 처리하는 가벼운 변환."""
    out, pos = [], 0
    for m in _CODE_RE.finditer(text):
        before = text[pos:m.start()]
        out.append(_inline(before[:-1] if before.endswith("\n") else before))  # <pre> 앞뒤 빈 줄 방지
        code = html.escape(m.group(1).rstrip("\n"))
        out.append(f"<pre style='background:{code_bg}; font-family:Menlo,Consolas,monospace'>{code}</pre>")
        pos = m.end() + (1 if text[m.end():m.end() + 1] == "\n" else 0)
    out.append(_inline(text[pos:]))
    return "".join(out)


def _inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"`([^`\n]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*\n]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?m)^#{1,6}\s+(.+)$", r"<b>\1</b>", t)
    return t.replace("\n", "<br>")


def stats_text(stats: dict) -> str:
    parts = []
    if stats.get("tps"):
        parts.append(f"{stats['tps']:.1f} 토큰/초")
    if stats.get("first_token") is not None:
        parts.append(f"첫 글자까지 {stats['first_token']:.1f}초")
    usage = stats.get("usage") or {}
    if usage.get("completion_tokens"):
        parts.append(f"답변 {usage['completion_tokens']:,} 토큰")
    if usage.get("prompt_tokens"):
        parts.append(f"대화 누적 {usage['prompt_tokens'] + usage.get('completion_tokens', 0):,} 토큰")
    if stats.get("stopped"):
        parts.append("⏹ 중간에 멈춤")
    return " · ".join(parts)


class _Bridge(QObject):
    event = Signal(object)


class ChatTab(QWidget):
    go_to_server = Signal()

    def __init__(self, settings: Settings, save_settings: Callable[[], None]):
        super().__init__()
        self.settings = settings
        self.save_settings = save_settings
        self.messages: list[Message] = []
        self.base_url = ""
        self.server_info: dict = {}
        self.state = ServerState.STOPPED
        self._stop: Optional[threading.Event] = None
        self._thread: Optional[threading.Thread] = None
        self._bridge = _Bridge()
        self._bridge.event.connect(self._on_event)
        self._render_timer = QTimer(self, singleShot=True, interval=60)
        self._render_timer.timeout.connect(self._render)

        root = QVBoxLayout(self)
        title = QLabel("💬 채팅 테스트")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        status = QHBoxLayout()
        self.light = QLabel()
        self.light.setFixedSize(14, 14)
        status.addWidget(self.light)
        self.status_lbl = QLabel()
        self.status_lbl.setTextFormat(Qt.RichText)
        status.addWidget(self.status_lbl, 1)
        self.server_btn = QPushButton("▶️ 서버 탭으로 가기")
        self.server_btn.clicked.connect(self.go_to_server.emit)
        status.addWidget(self.server_btn)
        root.addLayout(status)

        sys_box = QGroupBox("시스템 프롬프트 (AI에게 주는 기본 지시)")
        sys_box.setToolTip("모든 대화 앞에 붙는 지시문이에요. 말투, 역할, 답변 언어 등을 정할 수 있어요.")
        sl = QVBoxLayout(sys_box)
        self.system_edit = QPlainTextEdit(settings.chat_system_prompt)
        self.system_edit.setFixedHeight(54)
        self.system_edit.textChanged.connect(self._on_system_changed)
        sl.addWidget(self.system_edit)
        root.addWidget(sys_box)

        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        root.addWidget(self.view, 1)

        starters = QHBoxLayout()
        starters.addWidget(QLabel("예시 질문:"))
        self.starter_btns = []
        for label, text in STARTERS:
            b = QPushButton(label)
            b.setToolTip(text)
            b.clicked.connect(lambda _=False, t=text: self.send(t))
            starters.addWidget(b)
            self.starter_btns.append(b)
        starters.addStretch(1)
        self.reasoning_chk = QCheckBox("생각 과정 보기")
        self.reasoning_chk.setToolTip("추론(Thinking) 모델이 답하기 전에 생각한 내용을 회색으로 보여줘요.")
        self.reasoning_chk.setChecked(settings.chat_show_reasoning)
        self.reasoning_chk.toggled.connect(self._on_reasoning_toggled)
        starters.addWidget(self.reasoning_chk)
        root.addLayout(starters)

        inp = QHBoxLayout()
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("메시지를 입력하세요. Enter로 보내고, Shift+Enter로 줄을 바꿔요.")
        self.input.setFixedHeight(72)
        self.input.installEventFilter(self)
        inp.addWidget(self.input, 1)
        btns = QVBoxLayout()
        self.send_btn = QPushButton("보내기")
        self.send_btn.setObjectName("primary")
        self.send_btn.clicked.connect(lambda: self.send())
        self.stop_btn = QPushButton("⏹ 멈추기")
        self.stop_btn.clicked.connect(self.stop)
        self.clear_btn = QPushButton("🧹 대화 초기화")
        self.clear_btn.clicked.connect(self.clear)
        for b in (self.send_btn, self.stop_btn, self.clear_btn):
            btns.addWidget(b)
        inp.addLayout(btns)
        root.addLayout(inp)

        self.stats_lbl = QLabel()
        self.stats_lbl.setObjectName("hint")
        root.addWidget(self.stats_lbl)

        self.set_server(ServerState.STOPPED, "", {})

    # ------------------------------------------------------------------ 서버 상태
    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_server(self, state: ServerState, base_url: str, info: dict) -> None:
        self.state = state
        self.base_url = base_url
        self.server_info = info or {}
        running = state == ServerState.RUNNING
        color = {ServerState.RUNNING: "#2e7d32", ServerState.STARTING: "#f9a825",
                 ServerState.ERROR: "#c62828"}.get(state, "#9e9e9e")
        self.light.setStyleSheet(f"background:{color}; border-radius:7px;")
        if running:
            name = html.escape(self.server_info.get("model_name") or "모델")
            self.status_lbl.setText(f"<b>{name}</b>에 연결됐어요 · <code>{html.escape(base_url)}</code>")
        elif state == ServerState.STARTING:
            self.status_lbl.setText("서버가 모델을 불러오는 중이에요. 준비되면 바로 질문할 수 있어요…")
        else:
            self.status_lbl.setText("서버가 꺼져 있어요. [서버] 탭에서 모델을 고르고 ▶ 서버 시작을 눌러 주세요.")
        self.server_btn.setVisible(not running)
        if not running and self.busy:
            self.stop()
        self._update_buttons()
        if not self.messages:
            self._render()

    def _update_buttons(self) -> None:
        ok = self.state == ServerState.RUNNING and not self.busy
        self.send_btn.setEnabled(ok)
        for b in self.starter_btns:
            b.setEnabled(ok)
        self.stop_btn.setEnabled(self.busy)
        self.clear_btn.setEnabled(bool(self.messages) and not self.busy)

    # ------------------------------------------------------------------ 입력
    def eventFilter(self, obj, ev):
        if obj is self.input and ev.type() == QEvent.KeyPress and ev.key() in (Qt.Key_Return, Qt.Key_Enter):
            if not (ev.modifiers() & Qt.ShiftModifier) and not self._ime_composing():
                self.send()
                return True
        return super().eventFilter(obj, ev)

    def _ime_composing(self) -> bool:
        """한글 입력 중(조합 중)에 Enter를 누르면 글자 확정만 하고 보내지 않는다."""
        layout = self.input.textCursor().block().layout()
        return bool(layout and layout.preeditAreaText())

    def _on_system_changed(self) -> None:
        self.settings.chat_system_prompt = self.system_edit.toPlainText()
        self.save_settings()

    def _on_reasoning_toggled(self, on: bool) -> None:
        self.settings.chat_show_reasoning = on
        self.save_settings()
        self._render()

    def history(self) -> list[dict]:
        msgs: list[dict] = []
        sp = self.system_edit.toPlainText().strip()
        if sp:
            msgs.append({"role": "system", "content": sp})
        for m in self.messages:
            if m.role in ("user", "assistant") and (m.content or m.role == "user"):
                msgs.append({"role": m.role, "content": m.content})
        return msgs

    def send(self, text: Optional[str] = None) -> None:
        if self.busy or self.state != ServerState.RUNNING:
            return
        text = (text if text is not None else self.input.toPlainText()).strip()
        if not text:
            return
        if text and self.input.toPlainText().strip() == text:
            self.input.clear()
        self.messages.append(Message("user", text))
        req = ChatRequest(base_url=self.base_url, messages=self.history(),
                          model=self.server_info.get("alias") or "local-model",
                          api_key=self.server_info.get("api_key") or "")
        self.messages.append(Message("assistant"))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=stream_chat, args=(req, self._bridge.event.emit, self._stop),
                                        daemon=True, name="chat-stream")
        self._thread.start()
        self.stats_lbl.setText("답변을 기다리는 중…")
        self._update_buttons()
        self._render()

    def stop(self) -> None:
        if self._stop:
            self._stop.set()

    def clear(self) -> None:
        if self.busy:
            return
        self.messages.clear()
        self.stats_lbl.clear()
        self._update_buttons()
        self._render()

    def shutdown(self) -> None:
        self.stop()

    # ------------------------------------------------------------------ 응답 처리
    def _on_event(self, ev: ChatEvent) -> None:
        if not self.messages or self.messages[-1].role != "assistant":
            return
        cur = self.messages[-1]
        if ev.kind == "content":
            cur.content += ev.text
        elif ev.kind == "reasoning":
            cur.reasoning += ev.text
        elif ev.kind == "done":
            if not cur.reasoning:
                think, answer = split_think(cur.content)
                if think:
                    cur.reasoning, cur.content = think, answer
            cur.stats = stats_text(ev.stats)
            if not cur.content and not cur.reasoning:
                cur.content = "(빈 답변이에요. 다시 보내 보세요.)"
            self.stats_lbl.setText(cur.stats)
            QTimer.singleShot(0, self._update_buttons)  # 스레드가 완전히 끝난 뒤
        elif ev.kind == "error":
            # 실패한 질문은 대화에서 빼고 입력칸에 되돌려 다시 보낼 수 있게 한다
            self.messages.pop()
            user = self.messages.pop() if self.messages and self.messages[-1].role == "user" else None
            self.messages.append(Message("error", ev.text))
            if user and not self.input.toPlainText().strip():
                self.input.setPlainText(user.content)
            self.stats_lbl.setText("")
            QTimer.singleShot(0, self._update_buttons)
        if not self._render_timer.isActive():
            self._render_timer.start()

    # ------------------------------------------------------------------ 그리기
    def _colors(self) -> dict:
        dark = QApplication.palette().color(QPalette.Window).lightness() < 128
        if dark:
            return {"user": "#1f3a5f", "bot": "#2b2b2b", "err": "#5c2323", "think": "#9e9e9e", "code": "#1e1e1e"}
        return {"user": "#e3eefc", "bot": "#f4f4f4", "err": "#fde7e7", "think": "#757575", "code": "#eaeaea"}

    def _render(self) -> None:
        sb = self.view.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 30
        c = self._colors()
        if not self.messages:
            if self.state == ServerState.RUNNING:
                body = ("<p style='color:gray'>아래에 질문을 입력하거나 예시 질문 버튼을 눌러 보세요. "
                        "답변이 글자 단위로 나오면 서버가 잘 동작하는 거예요. 🎉</p>")
            else:
                body = "<p style='color:gray'>서버를 켜면 여기서 AI와 대화해 볼 수 있어요.</p>"
            self.view.setHtml(body)
            return
        parts = []
        show_think = self.reasoning_chk.isChecked()
        for i, m in enumerate(self.messages):
            streaming = self.busy and i == len(self.messages) - 1
            if m.role == "user":
                parts.append(self._bubble("🙋 나", html.escape(m.content).replace("\n", "<br>"), c["user"]))
            elif m.role == "error":
                parts.append(self._bubble("⚠️ 오류", html.escape(m.content), c["err"]))
            else:
                reasoning, content = m.reasoning, m.content
                if not reasoning and content.lstrip().startswith("<think>"):
                    reasoning, content = split_think(content)
                inner = ""
                if reasoning:
                    if show_think:
                        inner += (f"<p style='color:{c['think']}'><i>💭 생각 과정</i><br>"
                                  f"{html.escape(reasoning).replace(chr(10), '<br>')}</p>")
                    elif streaming and not content:
                        inner += f"<p style='color:{c['think']}'><i>💭 생각하는 중… ({len(reasoning):,}자)</i></p>"
                    else:
                        inner += f"<p style='color:{c['think']}'><i>💭 생각 과정 {len(reasoning):,}자 (숨김)</i></p>"
                inner += render_markdown_lite(content, c["code"])
                if streaming:
                    inner += " ▍"
                elif m.stats:
                    inner += f"<p style='color:{c['think']}; font-size:11px'>{html.escape(m.stats)}</p>"
                parts.append(self._bubble("🤖 AI", inner or "…", c["bot"]))
        self.view.setHtml("".join(parts))
        if at_bottom:
            sb.setValue(sb.maximum())

    @staticmethod
    def _bubble(who: str, inner: str, bg: str) -> str:
        return (f"<table width='100%' cellpadding='8' cellspacing='0' style='margin-bottom:6px'>"
                f"<tr><td bgcolor='{bg}'><b>{who}</b><br>{inner}</td></tr></table>")
