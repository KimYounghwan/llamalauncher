"""내 모델 탭: 받은 모델·직접 추가한 파일·LM Studio/Ollama 모델을 한곳에서 보고 실행/삭제한다."""

from __future__ import annotations

import html
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMenu, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QToolButton, QVBoxLayout,
    QWidget,
)

from ..core import model_library as lib
from ..core.hardware import SystemInfo, format_bytes
from ..core.recommender import Fit, ModelSpec, assess_fit
from ..core.settings import Settings
from .workers import run_in_background

FIT_CTX = 8192  # 모델 찾기 탭과 같은 기준
COLS = ["이름", "양자화", "크기", "출처", "날짜", "이 컴퓨터에서"]


class _SortItem(QTableWidgetItem):
    """표시 글자와 별개로 정렬 기준값(UserRole)을 쓰는 칸."""

    def __lt__(self, other):
        a, b = self.data(Qt.UserRole), other.data(Qt.UserRole)
        if a is not None and b is not None:
            try:
                return a < b
            except TypeError:
                pass
        return super().__lt__(other)


def reveal_in_file_manager(path: Path) -> None:
    """파일이면 그 파일이 선택된 상태로 폴더를 연다 (가능한 OS에서)."""
    import subprocess
    path = Path(path)
    try:
        if sys.platform == "darwin" and path.is_file():
            subprocess.Popen(["open", "-R", str(path)])
            return
        if sys.platform == "win32" and path.is_file():
            subprocess.Popen(["explorer", "/select,", str(path)])
            return
    except OSError:
        pass
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))


class ModelsTab(QWidget):
    run_model = Signal(str, object)   # 모델 경로, 비전 파일 경로(없으면 None)
    models_changed = Signal()

    def __init__(self, settings: Settings, save_settings: Callable[[], None]):
        super().__init__()
        self.settings = settings
        self.save_settings = save_settings
        self.system: Optional[SystemInfo] = None
        self.models: list[lib.LocalModel] = []
        self.in_use: Callable[[str], bool] = lambda path: False
        self._seq = 0
        self._loading = False
        self._refresh_timer = QTimer(self, singleShot=True, interval=800)
        self._refresh_timer.timeout.connect(self.refresh)

        root = QVBoxLayout(self)
        title = QLabel("📁 내 모델")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        hint = QLabel("이 컴퓨터에 있는 모델이에요. 라마런처로 받은 모델, 직접 추가한 파일, "
                      "LM Studio·Ollama에서 받은 모델까지 한곳에서 보고 바로 실행할 수 있어요.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        root.addWidget(hint)

        bar = QHBoxLayout()
        self.add_btn = QPushButton("📄 파일 직접 선택…")
        self.add_btn.setToolTip("컴퓨터에 있는 .gguf 모델 파일을 목록에 추가해요. 파일은 옮기지 않아요.")
        self.add_btn.clicked.connect(self._pick_files)
        self.folder_btn = QPushButton("📁 폴더 가져오기…")
        self.folder_btn.setToolTip("고른 폴더(하위 폴더 포함)에 있는 .gguf 모델을 모두 목록에 보여줘요.")
        self.folder_btn.clicked.connect(self._pick_folder)
        self.folders_menu_btn = QToolButton()
        self.folders_menu_btn.setText("가져온 폴더 ▾")
        self.folders_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self.folders_menu = QMenu(self)
        self.folders_menu_btn.setMenu(self.folders_menu)
        self.external_chk = QCheckBox("LM Studio·Ollama 모델도 보기")
        self.external_chk.setChecked(settings.show_external_models)
        self.external_chk.toggled.connect(self._toggle_external)
        self.refresh_btn = QPushButton("🔄 새로고침")
        self.refresh_btn.clicked.connect(self.refresh)
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("🔎 목록에서 찾기")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        for w in (self.add_btn, self.folder_btn, self.folders_menu_btn, self.external_chk):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.filter_edit)
        bar.addWidget(self.refresh_btn)
        root.addLayout(bar)

        self.sources_lbl = QLabel()
        self.sources_lbl.setObjectName("hint")
        self.sources_lbl.setWordWrap(True)
        root.addWidget(self.sources_lbl)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.table = QTableWidget(0, len(COLS))
        self.table.setHorizontalHeaderLabels(COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(True)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, len(COLS)):
            h.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.doubleClicked.connect(lambda *_: self._run())
        lv.addWidget(self.table)
        for key in (QKeySequence.Delete, QKeySequence(Qt.Key_Backspace)):
            sc = QShortcut(key, self.table, context=Qt.WidgetShortcut)
            sc.activated.connect(self._delete)
        self.usage_lbl = QLabel()
        self.usage_lbl.setObjectName("hint")
        lv.addWidget(self.usage_lbl)
        self.empty_lbl = QLabel("아직 모델이 없어요.\n[🔍 모델 찾기] 탭에서 받거나 [📄 파일 직접 선택…]으로 추가해 보세요.")
        self.empty_lbl.setAlignment(Qt.AlignCenter)
        self.empty_lbl.setObjectName("hint")
        lv.addWidget(self.empty_lbl)
        split.addWidget(left)

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(8, 0, 0, 0)
        self.detail = QLabel("왼쪽 목록에서 모델을 고르면 자세한 정보가 나와요.")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        rv.addWidget(self.detail, 1)
        self.run_btn = QPushButton("▶ 서버에서 실행")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self._run)
        self.reveal_btn = QPushButton("📂 폴더 열기")
        self.reveal_btn.clicked.connect(self._reveal)
        self.rename_btn = QPushButton("✏️ 이름 바꾸기")
        self.rename_btn.setToolTip("목록과 서버 탭에 보이는 이름만 바꿔요. 파일 이름은 그대로예요.")
        self.rename_btn.clicked.connect(self._rename)
        self.forget_btn = QPushButton("➖ 목록에서 빼기")
        self.forget_btn.setToolTip("파일은 지우지 않고 목록에서만 빼요.")
        self.forget_btn.clicked.connect(self._forget)
        self.delete_btn = QPushButton("🗑 파일 삭제")
        self.delete_btn.clicked.connect(self._delete)
        for b in (self.run_btn, self.reveal_btn, self.rename_btn, self.forget_btn, self.delete_btn):
            rv.addWidget(b)
        split.addWidget(right)
        split.setSizes([760, 380])
        root.addWidget(split, 1)

        self._update_buttons(None)
        self._rebuild_folders_menu()
        self.refresh()

    # ------------------------------------------------------------------ 데이터
    def set_system(self, info: SystemInfo) -> None:
        self.system = info
        self._fill_table()

    def display_name_for(self, path: str) -> str:
        custom = self.settings.model_names.get(str(path))
        if custom:
            return custom
        m = self._find(path)
        return m.display_name if m else Path(path).name

    def mmproj_for(self, path: str) -> Optional[str]:
        m = self._find(path)
        return str(m.mmproj) if m and m.mmproj else None

    def _find(self, path: str) -> Optional[lib.LocalModel]:
        return next((m for m in self.models if str(m.path) == str(path)), None)

    def refresh_later(self) -> None:
        self._refresh_timer.start()

    def refresh(self) -> None:
        self._seq += 1
        seq = self._seq
        self._loading = True
        self.refresh_btn.setEnabled(False)
        self.sources_lbl.setText("모델을 찾는 중이에요…")
        src = lib.LibrarySources(models_dir=Path(self.settings.models_dir), added=list(self.settings.added_models),
                                 folders=list(self.settings.model_folders),
                                 include_external=self.settings.show_external_models)

        def work():
            models = lib.collect(src)
            ext = (lib.lmstudio_dirs(), lib.ollama_dirs()) if src.include_external else ([], [])
            return models, ext

        def done(result):
            if seq != self._seq:
                return
            self._loading = False
            self.refresh_btn.setEnabled(True)
            self.models, (lm, ol) = result
            self._update_sources_label(lm, ol)
            self._fill_table()
            self.models_changed.emit()

        def failed(e):
            if seq != self._seq:
                return
            self._loading = False
            self.refresh_btn.setEnabled(True)
            self.sources_lbl.setText(f"모델 목록을 읽지 못했어요: {e}")

        run_in_background(work, done, failed)

    def _update_sources_label(self, lm: list[Path], ol: list[Path]) -> None:
        parts = [f"라마런처 모델 폴더: <code>{html.escape(self.settings.models_dir)}</code>"]
        if self.settings.show_external_models:
            parts.append("LM Studio: " + (", ".join(f"<code>{html.escape(str(p))}</code>" for p in lm) or "못 찾음"))
            parts.append("Ollama: " + (", ".join(f"<code>{html.escape(str(p))}</code>" for p in ol) or "못 찾음"))
        self.sources_lbl.setText(" · ".join(parts))

    def _fit_of(self, m: lib.LocalModel):
        if not self.system:
            return None
        spec = ModelSpec.from_gguf(m.info) if m.info else ModelSpec(file_size=m.size)
        return assess_fit(self.system, spec, FIT_CTX)

    def _fill_table(self) -> None:
        selected = [m.key for m in self.selected_models()]
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for m in self.models:
            r = self.table.rowCount()
            self.table.insertRow(r)
            name = self.display_name_for(str(m.path))
            badges = ""
            if m.error:
                badges += " ⚠️"
            if m.is_embedding:
                badges += " 📚"
            if m.info and m.info.supports_tools:
                badges += " 🛠"
            if m.mmproj:
                badges += " 👁"
            name_item = _SortItem(name + badges)
            name_item.setData(Qt.UserRole, name.lower())
            name_item.setData(Qt.UserRole + 1, m.key)
            name_item.setToolTip(str(m.path))
            size_item = _SortItem(format_bytes(m.size) if m.size else "—")
            size_item.setData(Qt.UserRole, m.size)
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            date_item = _SortItem(m.date.strftime("%Y-%m-%d") if m.date else "—")
            date_item.setData(Qt.UserRole, m.date.timestamp() if m.date else 0)
            fit = None if m.error else self._fit_of(m)
            if m.error:
                fit_text, fit_key = "⚠️ 읽을 수 없음", 9
            elif m.is_embedding:
                fit_text, fit_key = "📚 임베딩 전용", 5
            elif fit:
                fit_text, fit_key = fit.fit.badge, list(Fit).index(fit.fit)
            else:
                fit_text, fit_key = "…", 8
            fit_item = _SortItem(fit_text)
            fit_item.setData(Qt.UserRole, fit_key)
            if fit:
                fit_item.setToolTip(fit.reason)
            src_item = _SortItem(lib.Source.LABELS.get(m.source, m.source))
            src_item.setData(Qt.UserRole, m.source)
            qtext = (m.quant or "—") + {"mlx": " · MLX", "safetensors": " · safetensors"}.get(m.fmt, "")
            for c, it in enumerate([name_item, QTableWidgetItem(qtext), size_item, src_item,
                                    date_item, fit_item]):
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self.empty_lbl.setVisible(not self.models and not self._loading)
        self.table.setVisible(bool(self.models) or self._loading)
        self._apply_filter()
        self._select_keys(selected)
        self._update_usage()
        self._on_select()

    def _apply_filter(self) -> None:
        q = self.filter_edit.text().strip().lower()
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            hay = f"{it.text()} {it.toolTip()} {self.table.item(r, 1).text()}".lower()
            self.table.setRowHidden(r, bool(q) and q not in hay)

    def _selected_key(self) -> Optional[str]:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.table.item(rows[0].row(), 0).data(Qt.UserRole + 1)

    def _select_keys(self, keys: list[str]) -> None:
        if not keys:
            return
        from PySide6.QtCore import QItemSelection, QItemSelectionModel
        sel = QItemSelection()
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0).data(Qt.UserRole + 1) in keys:
                sel.select(self.table.model().index(r, 0), self.table.model().index(r, self.table.columnCount() - 1))
        self.table.blockSignals(True)
        self.table.selectionModel().select(sel, QItemSelectionModel.ClearAndSelect)
        self.table.blockSignals(False)

    def current(self) -> Optional[lib.LocalModel]:
        key = self._selected_key()
        return self._find(key) if key else None

    def selected_models(self) -> list[lib.LocalModel]:
        out = []
        for idx in self.table.selectionModel().selectedRows():
            m = self._find(self.table.item(idx.row(), 0).data(Qt.UserRole + 1))
            if m:
                out.append(m)
        return out

    def _update_usage(self) -> None:
        if not self.models:
            self.usage_lbl.setText("")
            return
        mine = [m for m in self.models if m.source in (lib.Source.DOWNLOAD,)]
        total = sum(m.size for m in self.models)
        text = f"모델 {len(self.models)}개 · 모두 {format_bytes(total)}"
        if mine:
            text += f" (라마런처로 받은 모델 {format_bytes(sum(m.size for m in mine))})"
        try:
            free = shutil.disk_usage(self.settings.models_dir).free
            text += f" · 모델 폴더 디스크 남은 공간 {format_bytes(free)}"
        except OSError:
            pass
        self.usage_lbl.setText(text + "  —  여러 개를 골라(⌘/Ctrl·Shift+클릭) 한 번에 지울 수 있어요.")

    # ------------------------------------------------------------------ 상세 정보
    def _on_select(self) -> None:
        sel = self.selected_models()
        if len(sel) > 1:
            self._update_buttons(None)
            n_del = sum(1 for m in sel if m.deletable)
            self.delete_btn.setEnabled(n_del > 0)
            self.delete_btn.setText(f"🗑 {n_del}개 파일 삭제")
            self.detail.setText(f"<b>{len(sel)}개 모델을 골랐어요</b><br>모두 {format_bytes(sum(m.size for m in sel))}<br><br>"
                                "[🗑 파일 삭제]를 누르면 한 번에 지워요.")
            return
        self.delete_btn.setText("🗑 파일 삭제")
        m = self.current()
        self._update_buttons(m)
        if not m:
            self.detail.setText("왼쪽 목록에서 모델을 고르면 자세한 정보가 나와요.")
            return
        e = html.escape
        rows = [f"<h3 style='margin:0'>{e(self.display_name_for(str(m.path)))}</h3>"]
        info = m.info
        if m.error:
            rows.append(f"<p style='color:#c62828'>⚠️ {e(m.error)}</p>")
        kv: list[tuple[str, str]] = []
        if m.repo:
            kv.append(("허깅페이스", f"<a href='https://huggingface.co/{e(m.repo)}'>{e(m.repo)}</a>"))
        if info:
            if info.name:
                kv.append(("모델 이름", e(info.name)))
            if info.architecture:
                kv.append(("구조(아키텍처)", e(info.architecture)))
            params = info.size_label or (f"{info.n_params / 1e9:.1f}B" if info.n_params else "")
            if params:
                kv.append(("크기(파라미터)", e(params)))
        if m.is_dir:
            kv.append(("형식", "MLX (Apple Silicon 전용, MLX-LM으로 실행)" if m.fmt == "mlx"
                       else "safetensors (vLLM으로 실행)"))
            if m.dir_meta.get("model_type"):
                kv.append(("구조(아키텍처)", e(str(m.dir_meta["model_type"]))))
            if m.dir_meta.get("ctx"):
                kv.append(("최대 문맥 길이", f"{int(m.dir_meta['ctx']):,} 토큰"))
            if m.dir_meta.get("layers"):
                kv.append(("레이어 수", str(m.dir_meta["layers"])))
        kv.append(("양자화", e(m.quant or "알 수 없음")))
        kv.append(("파일 크기", format_bytes(m.size) if m.size else "—"))
        if info:
            if info.n_ctx_train:
                kv.append(("최대 문맥 길이", f"{info.n_ctx_train:,} 토큰"))
            if info.n_layers:
                kv.append(("레이어 수", str(info.n_layers)))
            kv.append(("채팅 템플릿", "있음 ✅" if info.chat_template else "없음 (대화 형식을 직접 맞춰야 할 수 있어요)"))
            kv.append(("도구 호출", "지원 🛠 (코딩 에이전트에 연결 가능)" if info.supports_tools else "지원 안 함 / 확인 안 됨"))
            if info.missing_parts:
                kv.append(("⚠️ 빠진 조각", e(", ".join(p.name for p in info.missing_parts))))
        kv.append(("이미지 이해", f"가능 👁 ({e(m.mmproj.name)})" if m.mmproj else "—"))
        kv.append(("출처", lib.Source.LABELS.get(m.source, m.source)))
        if m.date:
            kv.append(("날짜", m.date.strftime("%Y-%m-%d %H:%M")))
        rows.append("<table cellspacing=4>" + "".join(
            f"<tr><td style='color:gray'>{k}</td><td>{v}</td></tr>" for k, v in kv) + "</table>")
        fit = None if m.error else self._fit_of(m)
        if m.is_embedding:
            rows.append("<p>📚 문서 검색(RAG)용 <b>임베딩 모델</b>이에요. 대화에는 쓸 수 없어요.</p>")
        elif fit:
            rows.append(f"<p><b>{fit.fit.badge}</b> — {e(fit.reason)}<br>"
                        f"<span style='color:gray'>문맥 {FIT_CTX:,} 기준 예상 메모리 {format_bytes(fit.needed)}</span></p>")
        if m.source == lib.Source.OLLAMA:
            rows.append("<p style='color:gray'>🦙 Ollama가 관리하는 파일이라 이름이 암호처럼 보여요. "
                        "llama.cpp로 바로 실행할 수 있어요. 일부 최신 모델은 Ollama 전용 형식이라 실행이 안 될 수 있어요.</p>")
        rows.append(f"<p style='color:gray; font-size:11px'>{e(str(m.path))}</p>")
        self.detail.setText("".join(rows))
        self.detail.setOpenExternalLinks(True)

    def _update_buttons(self, m: Optional[lib.LocalModel]) -> None:
        has = m is not None
        self.run_btn.setEnabled(has and not m.error)
        self.reveal_btn.setEnabled(has)
        self.rename_btn.setEnabled(has)
        self.forget_btn.setVisible(has and m.source in (lib.Source.ADDED, lib.Source.FOLDER))
        self.delete_btn.setEnabled(has and m.deletable)
        self.delete_btn.setToolTip("모델 파일을 디스크에서 완전히 지워요 (휴지통에 남지 않아요). 단축키: Delete")

    # ------------------------------------------------------------------ 동작
    def add_file(self, path: str, quiet: bool = True) -> None:
        """다른 탭에서 직접 고른 파일도 목록에 남긴다."""
        p = str(Path(path))
        if any(str(m.path) == p for m in self.models) or p in self.settings.added_models:
            return
        self.settings.added_models.append(p)
        self.save_settings()
        self.refresh_later()

    def _pick_files(self) -> None:
        start = Path(self.settings.added_models[-1]).parent if self.settings.added_models else Path.home()
        paths, _ = QFileDialog.getOpenFileNames(self, "모델 파일 선택", str(start), "GGUF 모델 (*.gguf)")
        added = 0
        for p in paths:
            if "mmproj" in Path(p).name.lower():
                QMessageBox.information(self, "비전 파일", f"{Path(p).name}은(는) 모델이 아니라 이미지 이해용 보조 파일이에요.\n"
                                        "모델 파일과 같은 폴더에 두면 자동으로 함께 쓰여요.")
                continue
            if p not in self.settings.added_models:
                self.settings.added_models.append(p)
                added += 1
        if added:
            self.save_settings()
            self.refresh()

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "모델이 들어 있는 폴더 선택", str(Path.home()))
        if not d:
            return
        n = len(lib.scan_folder(Path(d)))
        if n == 0:
            QMessageBox.information(self, "모델 없음", "이 폴더(하위 폴더 포함)에서 .gguf 모델을 찾지 못했어요.")
            return
        if d not in self.settings.model_folders:
            self.settings.model_folders.append(d)
            self.save_settings()
            self._rebuild_folders_menu()
        self.refresh()

    def _rebuild_folders_menu(self) -> None:
        self.folders_menu.clear()
        if not self.settings.model_folders:
            a = self.folders_menu.addAction("가져온 폴더가 없어요")
            a.setEnabled(False)
        for f in self.settings.model_folders:
            a = self.folders_menu.addAction(f"✕  {f}")
            a.setToolTip("목록에서 이 폴더를 빼요 (파일은 그대로)")
            a.triggered.connect(lambda _=False, f=f: self._remove_folder(f))
        self.folders_menu_btn.setVisible(bool(self.settings.model_folders))

    def _remove_folder(self, f: str) -> None:
        if f in self.settings.model_folders:
            self.settings.model_folders.remove(f)
            self.save_settings()
            self._rebuild_folders_menu()
            self.refresh()

    def _toggle_external(self, on: bool) -> None:
        self.settings.show_external_models = on
        self.save_settings()
        self.refresh()

    def _run(self) -> None:
        m = self.current()
        if not m or m.error:
            return
        if m.is_embedding:
            QMessageBox.information(self, "임베딩 모델", "이 모델은 문서 검색(RAG)용 임베딩 모델이라 대화는 할 수 없어요.\n"
                                    "서버는 실행되지만 채팅 테스트에서는 답을 받을 수 없어요.")
        fit = self._fit_of(m)
        if fit and fit.fit == Fit.INSUFFICIENT and QMessageBox.warning(
                self, "메모리가 부족할 수 있어요", f"{fit.reason}\n\n그래도 실행할까요?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self.run_model.emit(str(m.path), str(m.mmproj) if m.mmproj else None)

    def _reveal(self) -> None:
        m = self.current()
        if m:
            reveal_in_file_manager(m.path)

    def _rename(self) -> None:
        m = self.current()
        if not m:
            return
        cur = self.display_name_for(str(m.path))
        name, ok = QInputDialog.getText(self, "이름 바꾸기", "목록에 보일 이름 (비우면 원래 이름으로):", text=cur)
        if not ok:
            return
        name = name.strip()
        if name and name != m.display_name:
            self.settings.model_names[str(m.path)] = name
        else:
            self.settings.model_names.pop(str(m.path), None)
        self.save_settings()
        self._fill_table()
        self.models_changed.emit()

    def _forget(self) -> None:
        m = self.current()
        if not m:
            return
        if m.source == lib.Source.ADDED and str(m.path) in self.settings.added_models:
            self.settings.added_models.remove(str(m.path))
        elif m.source == lib.Source.FOLDER:
            QMessageBox.information(self, "폴더에서 가져온 모델",
                                    "이 모델은 가져온 폴더에 들어 있어요. [가져온 폴더 ▾]에서 폴더째 뺄 수 있어요.")
            return
        self.save_settings()
        self.refresh()

    def _delete(self) -> None:
        targets = [m for m in self.selected_models() if m.deletable]
        if not targets:
            return
        busy = [m for m in targets if self.in_use(str(m.path))]
        if busy:
            QMessageBox.warning(self, "사용 중인 모델",
                                f"'{self.display_name_for(str(busy[0].path))}'은(는) 지금 서버에서 실행 중이에요. "
                                "서버를 먼저 중지해 주세요.")
            return
        plan = [(m, lib.files_of(m, [o for o in self.models if o not in targets])) for m in targets]
        total = lib.size_on_disk([f for _, fs in plan for f in fs])
        lines, warns = [], set()
        for m, fs in plan:
            name = self.display_name_for(str(m.path))
            lines.append(f"• {name}" + (f"  ({len(fs)}개 파일)" if len(fs) > 1 else ""))
            if m.source == lib.Source.LMSTUDIO:
                warns.add("LM Studio에서 받은 모델은 LM Studio에서도 사라져요.")
            elif m.source in (lib.Source.ADDED, lib.Source.FOLDER):
                warns.add("직접 추가한 파일은 원본 파일이 지워져요.")
            elif m.source == lib.Source.OLLAMA:
                warns.add("Ollama 모델은 Ollama에서도 사라져요 (다른 모델과 같이 쓰는 파일은 남겨요).")
            elif m.source == lib.Source.HFCACHE:
                warns.add("허깅페이스 캐시의 이 저장소 폴더가 통째로 지워져요.")
        if len(lines) > 12:
            lines = lines[:12] + [f"… 외 {len(lines) - 12}개"]
        msg = (f"{len(targets)}개 모델을 디스크에서 완전히 지울까요? ({format_bytes(total)} 확보)\n"
               "휴지통에 남지 않아 되돌릴 수 없어요.\n\n" + "\n".join(lines))
        if warns:
            msg += "\n\n" + "\n".join(f"⚠️ {w}" for w in sorted(warns))
        if QMessageBox.warning(self, "모델 파일 삭제", msg, QMessageBox.Yes | QMessageBox.No,
                               QMessageBox.No) != QMessageBox.Yes:
            return
        failed = []
        remaining = list(self.models)
        for m, _ in plan:
            try:
                lib.delete_model(m, Path(self.settings.models_dir), [o for o in remaining if o is not m])
                remaining.remove(m)
            except OSError as e:
                failed.append(f"{self.display_name_for(str(m.path))}: {e}")
                continue
            self._forget_paths(str(m.path))
        self.save_settings()
        self.models = remaining
        self._fill_table()
        self.models_changed.emit()
        if failed:
            QMessageBox.warning(self, "일부 삭제 실패", "다음 모델을 지우지 못했어요:\n\n" + "\n".join(failed))
        self.refresh()

    def _forget_paths(self, p: str) -> None:
        """지운 모델을 설정(직접 추가 목록, 이름, 최근 사용)에서도 뺀다."""
        if p in self.settings.added_models:
            self.settings.added_models.remove(p)
        self.settings.model_names.pop(p, None)
        self.settings.recent_models = [x for x in self.settings.recent_models if x != p]
        if self.settings.last_model == p:
            self.settings.last_model = ""

    def _context_menu(self, pos) -> None:
        if not self.selected_models():
            return
        menu = QMenu(self)
        for btn in (self.run_btn, self.reveal_btn, self.rename_btn, self.forget_btn, self.delete_btn):
            if not btn.isHidden():
                a = menu.addAction(btn.text())
                a.setEnabled(btn.isEnabled())
                a.triggered.connect(btn.click)
        menu.exec(self.table.viewport().mapToGlobal(pos))
