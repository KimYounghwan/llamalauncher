"""모델 찾기 탭: 허깅페이스에서 카테고리별로 모델을 찾고, 이 컴퓨터에 맞는 파일을 골라 다운로드한다."""

from __future__ import annotations

import html
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFrame, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)

from ..core.hardware import SystemInfo, format_bytes
from ..core.hf_client import (
    Catalog, HFClient, HFError, HFModel, QuantFile, RepoFiles, model_fit, recommend_quant,
)
from ..core.recommender import Fit, ModelSpec, assess_fit
from ..core.settings import Settings
from .workers import run_in_background

SORTS = [("내 컴퓨터에 추천순", "recommended"), ("다운로드 많은 순", "downloads"),
         ("좋아요 많은 순", "likes"), ("최근 업데이트 순", "updated")]

QUANT_HELP = ("💡 같은 모델을 압축 정도(양자화)만 다르게 만든 파일들이에요. 숫자가 클수록(Q8 > Q6 > Q4 > Q2) "
              "품질이 좋지만 파일이 크고 느려요. 잘 모르면 ⭐ 추천 파일을 받으세요.")


def _num(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class _SortItem(QTableWidgetItem):
    """표시 텍스트와 별개로 숫자 기준 정렬."""

    def __init__(self, text: str, key):
        super().__init__(text)
        self.key = key

    def __lt__(self, other):
        if isinstance(other, _SortItem):
            return (self.key is None, self.key) < (other.key is None, other.key)
        return super().__lt__(other)


class BrowseTab(QWidget):
    download_requested = Signal(object, object, object)   # (HFModel, QuantFile, Optional[QuantFile] mmproj)
    ollama_pull_requested = Signal(str, str)              # (저장소, 양자화) → ollama pull hf.co/저장소:양자화

    def __init__(self, settings: Settings, client: HFClient, save_settings):
        super().__init__()
        self.settings = settings
        self.client = client
        self.save_settings = save_settings
        self.catalog: Catalog = client.catalog
        self.system: Optional[SystemInfo] = None
        self.models: list[HFModel] = []
        self.current: Optional[HFModel] = None
        self.files: Optional[RepoFiles] = None
        self._search_seq = 0
        self._detail_seq = 0
        self.fmt = settings.hf_format or "gguf"
        self.ollama_available = False

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._build_sidebar())
        split.addWidget(self._build_results())
        split.addWidget(self._build_detail())
        split.setSizes([210, 560, 460])
        split.setStretchFactor(1, 1)

        self.cat_list.setCurrentRow(0)

    # ------------------------------------------------------------------ UI
    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("카테고리")
        title.setObjectName("sectionTitle")
        lay.addWidget(title)
        self.cat_list = QListWidget()
        for c in self.catalog.categories:
            it = QListWidgetItem(c.label)
            it.setData(Qt.UserRole, c.id)
            it.setToolTip(c.description)
            self.cat_list.addItem(it)
        self.cat_list.currentRowChanged.connect(lambda _: self.search())
        lay.addWidget(self.cat_list, 1)

        self.trusted_cb = QCheckBox("검증된 제작자만")
        self.trusted_cb.setToolTip("공식 개발사와 널리 쓰이는 변환 제작자(unsloth, bartowski 등)의 모델만 보여줘요.")
        self.trusted_cb.setChecked(self.settings.hf_trusted_only)
        self.risky_cb = QCheckBox("탈옥·성인용 변형 숨기기")
        self.risky_cb.setToolTip("uncensored, abliterated 같은 안전장치를 없앤 변형 모델을 숨겨요.")
        self.risky_cb.setChecked(self.settings.hf_hide_risky)
        for cb in (self.trusted_cb, self.risky_cb):
            cb.toggled.connect(self._on_filter_changed)
            lay.addWidget(cb)
        lay.addWidget(QLabel("<b>모델 형식</b>"))
        self.fmt_combo = QComboBox()
        self.fmt_combo.setToolTip("쓰려는 실행 엔진에 맞는 형식을 고르세요. 잘 모르면 GGUF예요.")
        self.fmt_combo.activated.connect(lambda i: self.set_format(self.fmt_combo.itemData(i)))
        lay.addWidget(self.fmt_combo)
        self.set_format_options([("gguf", "GGUF — llama.cpp·Ollama·LM Studio 등")])
        return w

    def set_format_options(self, options: list[tuple[str, str]]) -> None:
        """런처 설치 상태에 따라 고를 수 있는 형식 목록을 바꾼다."""
        self.fmt_combo.blockSignals(True)
        self.fmt_combo.clear()
        for fmt, label in options:
            self.fmt_combo.addItem(label, fmt)
        idx = self.fmt_combo.findData(self.fmt)
        if idx < 0:
            self.fmt = "gguf"
            idx = 0
        self.fmt_combo.setCurrentIndex(idx)
        self.fmt_combo.blockSignals(False)

    def set_format(self, fmt: str) -> None:
        idx = self.fmt_combo.findData(fmt)
        if idx < 0:
            return
        self.fmt_combo.setCurrentIndex(idx)
        if fmt == self.fmt:
            return
        self.fmt = fmt
        self.settings.hf_format = fmt
        self.save_settings()
        self.search()

    def _build_results(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("모델 이름으로 검색 (예: qwen, gemma, exaone)")
        self.search_edit.returnPressed.connect(self.search)
        self.search_edit.setClearButtonEnabled(True)
        go = QPushButton("검색")
        go.clicked.connect(self.search)
        self.sort_combo = QComboBox()
        for label, v in SORTS:
            self.sort_combo.addItem(label, v)
        self.sort_combo.currentIndexChanged.connect(lambda _: self.search())
        refresh = QPushButton("↻")
        refresh.setToolTip("허깅페이스에서 새로 불러오기 (캐시 무시)")
        refresh.clicked.connect(lambda: self.search(use_cache=False))
        bar.addWidget(self.search_edit, 1)
        bar.addWidget(go)
        bar.addWidget(self.sort_combo)
        bar.addWidget(refresh)
        lay.addLayout(bar)

        self.cat_desc = QLabel("")
        self.cat_desc.setObjectName("hint")
        self.cat_desc.setWordWrap(True)
        lay.addWidget(self.cat_desc)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["모델", "크기", "이 컴퓨터", "다운로드", "좋아요", "업데이트"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setSortingEnabled(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 6):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self.table.itemSelectionChanged.connect(self._on_select)
        lay.addWidget(self.table, 1)

        self.status_lbl = QLabel("")
        self.status_lbl.setObjectName("hint")
        self.status_lbl.setWordWrap(True)
        lay.addWidget(self.status_lbl)
        return w

    def _build_detail(self) -> QWidget:
        w = QFrame()
        w.setObjectName("card")
        lay = QVBoxLayout(w)
        self.d_title = QLabel("모델을 선택하면 자세한 정보가 나와요.")
        self.d_title.setObjectName("sectionTitle")
        self.d_title.setWordWrap(True)
        self.d_title.setTextFormat(Qt.RichText)
        self.d_title.setOpenExternalLinks(True)
        lay.addWidget(self.d_title)
        self.d_meta = QLabel("")
        self.d_meta.setWordWrap(True)
        self.d_meta.setTextFormat(Qt.RichText)
        lay.addWidget(self.d_meta)
        self.d_summary = QTextBrowser()
        self.d_summary.setOpenExternalLinks(True)
        self.d_summary.setMaximumHeight(150)
        lay.addWidget(self.d_summary)

        files_title = QLabel("파일 고르기")
        files_title.setObjectName("sectionTitle")
        lay.addWidget(files_title)
        help_lbl = QLabel(QUANT_HELP)
        help_lbl.setObjectName("hint")
        help_lbl.setWordWrap(True)
        lay.addWidget(help_lbl)
        self.f_table = QTableWidget(0, 3)
        self.f_table.setHorizontalHeaderLabels(["양자화", "크기", "이 컴퓨터"])
        self.f_table.verticalHeader().setVisible(False)
        self.f_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.f_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.f_table.setSelectionMode(QAbstractItemView.SingleSelection)
        fh = self.f_table.horizontalHeader()
        fh.setSectionResizeMode(0, QHeaderView.Stretch)
        fh.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        fh.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.f_table.itemSelectionChanged.connect(self._update_download_btn)
        lay.addWidget(self.f_table, 1)

        self.mmproj_cb = QCheckBox("비전 파일(mmproj)도 함께 받기 — 이미지 입력에 필요")
        self.mmproj_cb.setChecked(True)
        self.mmproj_cb.hide()
        lay.addWidget(self.mmproj_cb)
        btns = QHBoxLayout()
        self.rec_btn = QPushButton("⭐ 추천 파일 받기")
        self.rec_btn.setObjectName("primary")
        self.rec_btn.clicked.connect(self._download_recommended)
        self.dl_btn = QPushButton("⬇ 선택한 파일 받기")
        self.dl_btn.clicked.connect(self._download_selected)
        self.ollama_btn = QPushButton("🦙 Ollama로 받기")
        self.ollama_btn.setToolTip("선택한 파일을 Ollama에 바로 받아요 (ollama pull hf.co/…). Ollama 앱·연동 도구에서 바로 쓸 수 있어요.")
        self.ollama_btn.clicked.connect(self._ollama_pull)
        self.ollama_btn.hide()
        btns.addWidget(self.ollama_btn)
        self.page_btn = QPushButton("허깅페이스에서 보기")
        self.page_btn.clicked.connect(lambda: self.current and QDesktopServices.openUrl(QUrl(self.current.url)))
        for b in (self.rec_btn, self.dl_btn, self.page_btn):
            b.setEnabled(False)
            btns.addWidget(b)
        lay.addLayout(btns)
        return w

    # ------------------------------------------------------------------ 검색
    def set_system(self, info: SystemInfo) -> None:
        first = self.system is None
        self.system = info
        if first:
            self.search()

    def _on_filter_changed(self) -> None:
        self.settings.hf_trusted_only = self.trusted_cb.isChecked()
        self.settings.hf_hide_risky = self.risky_cb.isChecked()
        self.save_settings()
        self.search()

    def _category_id(self) -> Optional[str]:
        it = self.cat_list.currentItem()
        return it.data(Qt.UserRole) if it else None

    def search(self, use_cache: bool = True) -> None:
        cat_id = self._category_id()
        if cat_id:
            cat = self.catalog.get(cat_id)
            self.cat_desc.setText(f"{cat.icon} {cat.description}")
        text = self.search_edit.text()
        self._search_seq += 1
        seq = self._search_seq
        self.status_lbl.setText("허깅페이스에서 찾는 중…")
        system = self.system
        kwargs = dict(category=cat_id, text=text, fmt=self.fmt, system=system, hide_risky=self.risky_cb.isChecked(),
                      trusted_only=self.trusted_cb.isChecked(), sort=self.sort_combo.currentData(),
                      use_cache=use_cache)
        run_in_background(lambda: self.client.search(**kwargs),
                          lambda res: seq == self._search_seq and self._show_results(res),
                          lambda e: seq == self._search_seq and self._show_error(e))

    def _show_error(self, e: Exception) -> None:
        msg = str(e) if isinstance(e, HFError) else f"모델 목록을 가져오지 못했어요: {e}"
        self.status_lbl.setText("⚠️ " + msg)

    def _show_results(self, models: list[HFModel]) -> None:
        # 선택을 먼저 지워야 새 목록의 첫 줄을 선택할 때 상세 정보가 갱신된다
        # (같은 줄 번호가 이미 선택돼 있으면 선택 변경 신호가 오지 않음)
        self.table.blockSignals(True)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self._clear_detail()
        self.models = models
        self.table.setRowCount(len(models))
        for row, m in enumerate(models):
            fit = model_fit(m, self.system, self.fmt)
            trusted = self.catalog.is_trusted(m.author)
            icons = ("🛠" if m.supports_tools else "") + ("👁" if m.is_vision else "") + ("🔒" if m.gated else "")
            name = QTableWidgetItem(f"{m.name}  {icons}".rstrip())
            name.setToolTip(f"{m.id}\n제작자: {m.author}{' ✔ 검증됨' if trusted else ''}\n"
                            "🛠 도구 호출 지원  👁 이미지 이해  🔒 로그인 필요")
            name.setData(Qt.UserRole, row)
            sub = f"{'✔ ' if trusted else ''}{m.author}"
            name.setText(f"{m.name}  {icons}\n{sub}".strip())
            p = m.params_b
            size = _SortItem(f"{p:.1f}B" if p and p < 10 else (f"{p:.0f}B" if p else "?"), p)
            fit_item = _SortItem(fit.badge if fit else "?", {Fit.COMFORTABLE: 0, Fit.SLOW: 1,
                                                              Fit.INSUFFICIENT: 2}.get(fit, 3))
            if fit and m.estimated_spec():
                fit_item.setToolTip(f"Q4_K_M 기준 예상 파일 크기 {format_bytes(m.estimated_spec().file_size)}")
            days = m.days_since_update()
            upd = _SortItem("—" if days is None else (f"{days}일 전" if days < 60 else f"{days // 30}개월 전"),
                            days)
            cells = [name, size, fit_item, _SortItem(_num(m.downloads), -m.downloads),
                     _SortItem(_num(m.likes), -m.likes), upd]
            for col, it in enumerate(cells):
                if col:
                    it.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, col, it)
        self.table.resizeRowsToContents()
        if models:
            self.status_lbl.setText(f"{len(models)}개 모델 · 🛠 도구 호출  👁 이미지 이해  🔒 로그인 필요 · ✔ 검증된 제작자")
            self.table.selectRow(0)
        else:
            self.status_lbl.setText("조건에 맞는 모델이 없어요. 검색어나 필터를 바꿔 보세요.")
            self._clear_detail()

    # ------------------------------------------------------------------ 상세
    def _clear_detail(self) -> None:
        self._detail_seq += 1  # 이전 모델의 파일 목록이 늦게 도착해도 무시
        self.current = None
        self.files = None
        self.d_title.setText("모델을 선택하면 자세한 정보가 나와요.")
        self.d_meta.setText("")
        self.d_summary.clear()
        self.f_table.setRowCount(0)
        self.mmproj_cb.hide()
        for b in (self.rec_btn, self.dl_btn, self.page_btn):
            b.setEnabled(False)

    def _on_select(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        m = self.models[rows[0].row()]
        if self.current and self.current.id == m.id:
            return
        self.current = m
        self.files = None
        self.d_title.setText(f"<a href='{m.url}'>{html.escape(m.id)}</a>")
        meta = []
        if m.params_b:
            meta.append(f"파라미터 {m.params_b:.1f}B")
        if m.context_length:
            meta.append(f"최대 문맥 {m.context_length:,}")
        if m.architecture:
            meta.append(html.escape(m.architecture))
        if m.license:
            meta.append(f"라이선스 {html.escape(m.license)}")
        if m.supports_tools:
            meta.append("🛠 도구 호출(코딩 에이전트) 지원")
        if m.is_vision:
            meta.append("👁 이미지 이해")
        if m.gated:
            meta.append("🔒 허깅페이스 로그인·동의 필요")
        if m.base_model:
            meta.append(f"원본: {html.escape(m.base_model)}")
        if m.is_embedding:
            meta.append("📚 임베딩 전용 (대화 불가)")
        self.d_meta.setText(" · ".join(meta))
        self.d_summary.setPlainText("설명을 불러오는 중…")
        self.f_table.setRowCount(0)
        self.page_btn.setEnabled(True)
        self.rec_btn.setEnabled(False)
        self.dl_btn.setEnabled(False)
        self._detail_seq += 1
        seq = self._detail_seq
        repo = m.id
        fmt = self.fmt
        run_in_background(lambda: self.client.list_files(repo, fmt=fmt),
                          lambda f: seq == self._detail_seq and self._show_files(f),
                          lambda e: seq == self._detail_seq and self._files_error(e))
        run_in_background(lambda: self.client.readme_summary(repo),
                          lambda s: seq == self._detail_seq and self.d_summary.setPlainText(
                              s or "모델 카드에 설명이 없어요. [허깅페이스에서 보기]로 확인해 주세요."),
                          lambda e: None)

    def _files_error(self, e: Exception) -> None:
        self.d_summary.append("\n⚠️ " + (str(e) if isinstance(e, HFError) else f"파일 목록 오류: {e}"))

    def _show_files(self, files: RepoFiles) -> None:
        self.files = files
        m = self.current
        params = m.params_b * 1e9 if m and m.params_b else None
        rec = recommend_quant(files, self.system, params, n_ctx_train=m.context_length) if self.system else None
        self.f_table.setRowCount(len(files.quants))
        rec_row = None
        for row, q in enumerate(files.quants):
            label = q.quant + (f"  (분할 {len(q.paths)}개)" if q.is_split else "")
            if rec is q:
                label = "⭐ " + label + "  추천"
                rec_row = row
            it = QTableWidgetItem(label)
            it.setToolTip("\n".join(q.paths))
            size = QTableWidgetItem(format_bytes(q.size))
            size.setTextAlignment(Qt.AlignCenter)
            fit_txt = "?"
            if self.system:
                fr = assess_fit(self.system, ModelSpec(file_size=q.size, n_ctx_train=m.context_length), 8192)
                fit_txt = fr.fit.badge
                size.setToolTip(fr.reason)
            fit_item = QTableWidgetItem(fit_txt)
            fit_item.setTextAlignment(Qt.AlignCenter)
            for col, x in enumerate((it, size, fit_item)):
                self.f_table.setItem(row, col, x)
        self.mmproj_cb.setVisible(bool(files.mmproj))
        if files.mmproj:
            self.mmproj_cb.setText(f"비전 파일(mmproj, {format_bytes(files.mmproj[0].size)})도 함께 받기 — 이미지 입력에 필요")
        self.rec_btn.setEnabled(rec is not None)
        self.rec_btn.setToolTip("" if rec else "이 컴퓨터에서 쾌적하게 돌아가는 파일이 없어요. 더 작은 모델을 찾아 보세요.")
        if rec_row is not None:
            self.f_table.selectRow(rec_row)
        if not files.quants:
            what = {"gguf": "GGUF 모델 파일", "mlx": "MLX 모델", "safetensors": "safetensors 모델"}.get(self.fmt, "모델")
            self.d_summary.append(f"\n⚠️ 이 저장소에는 받을 수 있는 {what}이 없어요.")
        elif self.fmt != "gguf":
            self.d_summary.append("\nℹ️ 이 형식은 저장소의 모델 폴더 전체(설정·토크나이저 포함)를 받아요.")
        self._update_download_btn()

    def _selected_quant(self) -> Optional[QuantFile]:
        rows = self.f_table.selectionModel().selectedRows() if self.files else []
        return self.files.quants[rows[0].row()] if rows else None

    def _update_download_btn(self) -> None:
        q = self._selected_quant()
        self.dl_btn.setEnabled(q is not None)
        self.ollama_btn.setVisible(self.ollama_available and self.fmt == "gguf")
        self.ollama_btn.setEnabled(q is not None and not q.is_split)
        if q is not None and q.is_split:
            self.ollama_btn.setToolTip("분할된 파일은 Ollama로 바로 받을 수 없어요.")

    def _ollama_pull(self) -> None:
        q = self._selected_quant()
        if q and self.current:
            self.ollama_pull_requested.emit(self.current.id, q.quant)

    def _mmproj_choice(self) -> Optional[QuantFile]:
        if self.files and self.files.mmproj and self.mmproj_cb.isChecked():
            for pref in ("F16", "BF16"):
                for mm in self.files.mmproj:
                    if mm.quant == pref:
                        return mm
            return self.files.mmproj[0]
        return None

    def _download_recommended(self) -> None:
        if not (self.files and self.current and self.system):
            return
        m = self.current
        rec = recommend_quant(self.files, self.system, m.params_b * 1e9 if m.params_b else None,
                              n_ctx_train=m.context_length)
        if rec:
            self.download_requested.emit(m, rec, self._mmproj_choice())

    def _download_selected(self) -> None:
        q = self._selected_quant()
        if q and self.current:
            self.download_requested.emit(self.current, q, self._mmproj_choice())
