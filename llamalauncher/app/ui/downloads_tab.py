"""다운로드 탭: 진행률, 일시정지/이어받기/취소, 완료된 모델을 바로 서버에서 실행."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..core.downloader import DownloadError, DownloadItem, DownloadManager, FileSpec, Status, human_eta, repo_dir
from ..core.hardware import format_bytes
from ..core.hf_client import HFModel, QuantFile
from ..core.secrets_store import get_hf_token, set_hf_token
from ..core.settings import Settings, config_dir

COLS = ["모델", "상태", "진행률", "속도", "남은 시간", ""]


class _Bridge(QObject):
    """작업 스레드의 콜백을 UI 스레드 시그널로 바꾼다."""
    updated = Signal(object)


class DownloadsTab(QWidget):
    run_model = Signal(str)            # 완료된 모델 경로 → 서버 탭
    active_count_changed = Signal(int)
    token_changed = Signal(object)     # 새 토큰 (없으면 None)

    def __init__(self, settings: Settings, save_settings):
        super().__init__()
        self.settings = settings
        self.save_settings = save_settings
        self._bridge = _Bridge()
        self._bridge.updated.connect(self._on_item_updated, Qt.QueuedConnection)
        self.manager = DownloadManager(
            config_dir() / "downloads.json", token_getter=get_hf_token,
            max_concurrent=settings.max_downloads, on_update=self._bridge.updated.emit,
        )
        self._rows: dict[str, int] = {}
        self._dirty: set[str] = set()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._flush)
        self._refresh_timer.start(250)  # 진행률 갱신을 묶어서 처리 (화면 깜빡임 방지)

        root = QVBoxLayout(self)
        top = QHBoxLayout()
        title = QLabel("다운로드")
        title.setObjectName("pageTitle")
        top.addWidget(title)
        top.addStretch()
        top.addWidget(QLabel("동시 다운로드"))
        self.conc = QSpinBox()
        self.conc.setRange(1, 4)
        self.conc.setValue(settings.max_downloads)
        self.conc.valueChanged.connect(self._on_concurrency)
        top.addWidget(self.conc)
        self.token_btn = QPushButton()
        self.token_btn.clicked.connect(self._edit_token)
        top.addWidget(self.token_btn)
        root.addLayout(top)

        dir_row = QHBoxLayout()
        dir_row.addWidget(QLabel("모델 저장 위치:"))
        self.dir_edit = QLineEdit(settings.models_dir)
        self.dir_edit.setReadOnly(True)
        dir_row.addWidget(self.dir_edit, 1)
        change = QPushButton("변경…")
        change.clicked.connect(self._change_dir)
        open_dir = QPushButton("폴더 열기")
        open_dir.clicked.connect(lambda: self._open_folder(Path(self.settings.models_dir)))
        dir_row.addWidget(change)
        dir_row.addWidget(open_dir)
        root.addLayout(dir_row)

        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        for i in (1, 3, 4):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        root.addWidget(self.table, 1)

        self.empty_lbl = QLabel("아직 받은 모델이 없어요. [🔍 모델 찾기] 탭에서 모델을 골라 보세요.")
        self.empty_lbl.setObjectName("hint")
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        root.addWidget(self.empty_lbl)
        bottom = QHBoxLayout()
        bottom.addStretch()
        clear = QPushButton("완료된 항목 목록에서 지우기")
        clear.clicked.connect(self._clear_finished)
        bottom.addWidget(clear)
        root.addLayout(bottom)

        self._update_token_btn()
        for item in self.manager.items.values():
            self._ensure_row(item)
        self._flush_all()

    # ------------------------------------------------------------------ 외부에서 호출
    def enqueue(self, model: HFModel, quant: QuantFile, mmproj: Optional[QuantFile]) -> Optional[DownloadItem]:
        keep = quant.kind != "gguf"
        files = [FileSpec(p, s, h, keep_path=keep)
                 for p, s, h in zip(quant.paths, quant.sizes or [quant.size], quant.sha256 or [None] * len(quant.paths))]
        mm_name = None
        if mmproj:
            files.append(FileSpec(mmproj.paths[0], mmproj.size, (mmproj.sha256 or [None])[0]))
            mm_name = Path(mmproj.paths[0]).name
        dest = repo_dir(Path(self.settings.models_dir), model.id)
        if keep:  # 폴더형 모델은 형식별 하위 폴더에 (같은 저장소의 GGUF와 섞이지 않게)
            dest = Path(self.settings.models_dir) / quant.kind / model.id.replace("/", "__")
        try:
            item = self.manager.add(model.id, quant.quant, files, dest, mmproj=mm_name, kind=quant.kind)
        except DownloadError as e:
            QMessageBox.warning(self, "다운로드할 수 없어요", str(e))
            return None
        self._ensure_row(item)
        self._dirty.add(item.id)
        return item

    def shutdown(self) -> None:
        self.manager.shutdown()

    def has_active(self) -> bool:
        return self.manager.active_count() > 0

    # ------------------------------------------------------------------ 표
    def _ensure_row(self, item: DownloadItem) -> None:
        if item.id in self._rows:
            return
        row = self.table.rowCount()
        self.table.insertRow(row)
        self._rows[item.id] = row
        name = QTableWidgetItem(f"{item.display_name}\n{item.repo}")
        name.setToolTip(f"{item.repo}\n저장 위치: {item.dest_dir}\n파일: " + ", ".join(f.filename for f in item.files))
        self.table.setItem(row, 0, name)
        for col in (1, 3, 4):
            it = QTableWidgetItem("")
            it.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, col, it)
        bar = QProgressBar()
        bar.setRange(0, 1000)
        bar.setTextVisible(True)
        self.table.setCellWidget(row, 2, bar)

        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(4, 0, 4, 0)
        toggle = QPushButton()
        toggle.setObjectName("toggle")
        toggle.clicked.connect(lambda _=False, i=item.id: self._toggle(i))
        cancel = QPushButton("취소")
        cancel.setObjectName("cancel")
        cancel.clicked.connect(lambda _=False, i=item.id: self._cancel(i))
        folder = QPushButton("폴더")
        folder.clicked.connect(lambda _=False, d=item.dest_dir: self._open_folder(Path(d)))
        run = QPushButton("▶ 실행")
        run.setObjectName("run")
        run.setToolTip("서버 탭에서 이 모델로 실행 준비")
        run.clicked.connect(lambda _=False, i=item.id: self._run(i))
        for b in (toggle, cancel, folder, run):
            bl.addWidget(b)
        self.table.setCellWidget(row, 5, btns)
        self.table.setRowHeight(row, 48)
        self.empty_lbl.hide()

    def _on_item_updated(self, item: DownloadItem) -> None:
        self._dirty.add(item.id)
        if item.status in (Status.DONE, Status.ERROR, Status.CANCELED, Status.PAUSED):
            self._flush()  # 상태 변화는 바로 반영

    def _flush_all(self) -> None:
        self._dirty.update(self.manager.items.keys())
        self._flush()

    def _flush(self) -> None:
        if not self._dirty:
            return
        dirty, self._dirty = self._dirty, set()
        for item_id in dirty:
            item = self.manager.items.get(item_id)
            if item is None:
                self._remove_row(item_id)
                continue
            self._ensure_row(item)
            self._render(item)
        self.active_count_changed.emit(self.manager.active_count() +
                                       sum(1 for i in self.manager.items.values() if i.status == Status.QUEUED))

    def _render(self, item: DownloadItem) -> None:
        row = self._rows[item.id]
        status_txt = item.status.label
        if item.status == Status.ERROR and item.error:
            status_txt = "⚠️ 오류"
        st = self.table.item(row, 1)
        st.setText(status_txt)
        st.setToolTip(item.error)
        bar: QProgressBar = self.table.cellWidget(row, 2)
        frac = item.downloaded / item.total if item.total else 0
        bar.setValue(int(frac * 1000))
        bar.setFormat(f"{frac * 100:.1f}%  ({format_bytes(item.downloaded)} / {format_bytes(item.total)})")
        downloading = item.status == Status.DOWNLOADING
        self.table.item(row, 3).setText(f"{format_bytes(int(item.speed))}/s" if downloading and item.speed else "—")
        self.table.item(row, 4).setText(human_eta(item.eta_seconds) if downloading else "—")
        if item.status == Status.ERROR and item.error:
            self.table.item(row, 0).setToolTip(item.error)
            self.table.item(row, 0).setText(f"{item.display_name}\n⚠️ {item.error}")
        elif item.status == Status.DONE:
            self.table.item(row, 0).setText(f"{item.display_name}\n✅ {item.main_path.name}")

        w = self.table.cellWidget(row, 5)
        toggle = w.findChild(QPushButton, "toggle")
        cancel = w.findChild(QPushButton, "cancel")
        run = w.findChild(QPushButton, "run")
        done = item.status == Status.DONE
        toggle.setVisible(not done)
        cancel.setVisible(not done)
        run.setVisible(done)
        if item.status.active or item.status == Status.QUEUED:
            toggle.setText("일시정지")
        else:
            toggle.setText("이어받기" if item.downloaded else "시작")

    def _remove_row(self, item_id: str) -> None:
        row = self._rows.pop(item_id, None)
        if row is None:
            return
        self.table.removeRow(row)
        for k, r in self._rows.items():
            if r > row:
                self._rows[k] = r - 1
        self.empty_lbl.setVisible(not self._rows)

    # ------------------------------------------------------------------ 버튼 동작
    def _toggle(self, item_id: str) -> None:
        item = self.manager.items.get(item_id)
        if not item:
            return
        if item.status.active or item.status == Status.QUEUED:
            self.manager.pause(item_id)
        else:
            self.manager.resume(item_id)
        self._dirty.add(item_id)

    def _cancel(self, item_id: str) -> None:
        item = self.manager.items.get(item_id)
        if not item:
            return
        if QMessageBox.question(self, "다운로드 취소",
                                f"{item.display_name} 다운로드를 취소하고 받던 파일을 지울까요?") == QMessageBox.Yes:
            self.manager.cancel(item_id)
            self._dirty.add(item_id)

    def _run(self, item_id: str) -> None:
        item = self.manager.items.get(item_id)
        if item and item.main_path.exists():
            self.run_model.emit(str(item.main_path))
        elif item:
            QMessageBox.warning(self, "파일 없음", f"모델 파일을 찾을 수 없어요:\n{item.main_path}")

    def _clear_finished(self) -> None:
        for item in list(self.manager.items.values()):
            if item.status == Status.DONE:
                self.manager.remove(item.id)
                self._remove_row(item.id)

    def _on_concurrency(self, n: int) -> None:
        self.settings.max_downloads = n
        self.manager.max_concurrent = n
        self.save_settings()
        self.manager._schedule()

    def _change_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "모델 저장 위치 선택", self.settings.models_dir)
        if d:
            self.settings.models_dir = d
            self.dir_edit.setText(d)
            self.save_settings()

    @staticmethod
    def _open_folder(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _update_token_btn(self) -> None:
        self.token_btn.setText("🔑 허깅페이스 토큰: " + ("설정됨" if get_hf_token() else "없음"))
        self.token_btn.setToolTip("로그인이 필요한 모델(🔒)을 받거나 검색 한도를 늘릴 때 필요해요.\n"
                                  "huggingface.co → Settings → Access Tokens 에서 'Read' 토큰을 만드세요.")

    def _edit_token(self) -> None:
        tok, ok = QInputDialog.getText(
            self, "허깅페이스 토큰",
            "huggingface.co → Settings → Access Tokens 에서 만든 토큰(hf_로 시작)을 붙여 넣으세요.\n"
            "비워 두고 확인을 누르면 저장된 토큰을 지워요. 토큰은 OS 키체인에 안전하게 저장돼요.",
            QLineEdit.Password)
        if not ok:
            return
        if not set_hf_token(tok.strip() or None):
            QMessageBox.warning(self, "저장 실패", "이 컴퓨터의 키체인에 토큰을 저장하지 못했어요. "
                                "환경 변수 HF_TOKEN 으로 설정할 수도 있어요.")
        self._update_token_btn()
        self.token_changed.emit(get_hf_token())
