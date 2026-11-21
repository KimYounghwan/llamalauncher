"""모델 관리: 내 모델 탭에서 파일 삭제(여러 개, HF 캐시), 서버 탭 모델 목록 자동 연동."""

import json
import time
from pathlib import Path

from PySide6.QtWidgets import QMessageBox

from llamalauncher.app.core import model_library as lib
from llamalauncher.app.core.downloader import manifest_path
from tests.test_stage4 import make_model
from tests.test_stage5 import _model_dir


def _window(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui.main_window import MainWindow
    monkeypatch.setattr(Settings, "save", lambda self, path=None: None)
    w = MainWindow(Settings(models_dir=str(tmp_path / "models"), show_external_models=False))
    return w


def _wait(qapp, cond, timeout=15):
    end = time.time() + timeout
    while time.time() < end and not cond():
        qapp.processEvents()
        time.sleep(0.02)
    return cond()


def test_hf_cache_delete_removes_repo(tmp_path):
    hub = tmp_path / "hub"
    snap = _model_dir(hub / "models--a--b" / "snapshots" / "rev1")
    (hub / "models--a--b" / "blobs").mkdir()
    other = _model_dir(hub / "models--c--d" / "snapshots" / "rev1")
    m = lib.LocalModel(path=snap, source=lib.Source.HFCACHE, fmt="mlx")
    assert lib.files_of(m) == [hub / "models--a--b"]
    lib.delete_model(m, tmp_path / "models")
    assert not (hub / "models--a--b").exists() and other.exists()
    # 캐시 구조가 아닌 곳이면 지우지 않는다
    assert not lib.LocalModel(path=other.parent, source=lib.Source.HFCACHE).deletable


def test_engine_accepts_by_format():
    g = lib.LocalModel(path=Path("/x/a.gguf"), source=lib.Source.DOWNLOAD)
    mlx = lib.LocalModel(path=Path("/x/m"), source=lib.Source.DOWNLOAD, fmt="mlx")
    bad = lib.LocalModel(path=Path("/x/b.gguf"), source=lib.Source.DOWNLOAD, error="broken")
    assert lib.engine_accepts(("gguf",), g) and not lib.engine_accepts(("gguf",), mlx)
    assert lib.engine_accepts(("mlx",), mlx) and not lib.engine_accepts(("gguf",), bad)


def test_server_list_follows_library_and_delete(qapp, tmp_path, monkeypatch):
    mdir = tmp_path / "models"
    a = make_model(mdir / "org" / "A-GGUF" / "A-Q4_K_M.gguf")
    b = make_model(mdir / "org" / "B-GGUF" / "B-Q8_0.gguf")
    _model_dir(mdir / "mlx" / "mlx-community__C-4bit")
    manifest_path(mdir).write_text(json.dumps({"models": [
        {"path": str(a), "repo": "org/A-GGUF", "quant": "Q4_K_M"}, {"path": str(b), "repo": "org/B-GGUF"}]}))
    w = _window(qapp, tmp_path, monkeypatch)
    try:
        assert _wait(qapp, lambda: len(w.models.models) == 3)
        # 받은 모델이 서버 탭 목록에 자동으로 보인다 (llama.cpp에는 GGUF만)
        assert _wait(qapp, lambda: w.server.model_combo.count() == 2)
        paths = [w.server.model_combo.itemData(i) for i in range(w.server.model_combo.count())]
        assert set(paths) == {str(a), str(b)}
        assert "Q4_K_M" in w.server.model_combo.itemText(paths.index(str(a)))
        w.server.set_engine("mlx")
        assert [p for _, p, _ in w.server.combo_entries()] == [str(mdir / "mlx" / "mlx-community__C-4bit")]
        w.server.set_engine("llamacpp")

        # 하나 고른 뒤 두 개를 한꺼번에 삭제
        w.server.select_model_path(str(a))
        assert w.server.model_path == str(a)
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: QMessageBox.Yes))
        w.models.table.selectAll()
        w.models._fill_table()  # 백그라운드 새로고침이 와도 여러 개 선택이 유지된다
        sel = w.models.selected_models()
        assert len(sel) == 3 and "3개" in w.models.delete_btn.text()
        w.models._delete()
        assert not a.exists() and not b.exists() and not (mdir / "mlx" / "mlx-community__C-4bit").exists()
        assert _wait(qapp, lambda: w.server.model_combo.count() == 0)
        assert w.server.model_path is None and str(a) not in w.settings.recent_models

        # 새로 받은 모델도 바로 목록에 뜬다
        c = make_model(mdir / "org" / "D-GGUF" / "D-Q4_K_M.gguf")
        w.models.refresh()
        assert _wait(qapp, lambda: w.server.model_combo.count() == 1)
        assert w.server.model_combo.itemData(0) == str(c)
    finally:
        w.downloads.shutdown()


def test_delete_refuses_running_model(qapp, tmp_path, monkeypatch):
    mdir = tmp_path / "models"
    a = make_model(mdir / "org" / "A-GGUF" / "A-Q4_K_M.gguf")
    w = _window(qapp, tmp_path, monkeypatch)
    try:
        assert _wait(qapp, lambda: len(w.models.models) == 1)
        w.models.in_use = lambda p: p == str(a)
        shown = []
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *x, **k: shown.append(x[2]) or QMessageBox.Yes))
        w.models.table.selectAll()
        w.models._delete()
        assert a.exists() and "실행 중" in shown[0]
    finally:
        w.downloads.shutdown()
