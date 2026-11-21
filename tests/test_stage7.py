"""7단계: 첫 실행 마법사, 테마, 앱 시작(스모크 테스트), 빌드 설정."""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from llamalauncher.app.core.hf_client import HFModel, QuantFile, RepoFiles
from tests.test_stage5 import system

ROOT = Path(__file__).resolve().parent.parent


def _model(mid, params="8B", gated=False, pipeline="text-generation"):
    m = HFModel.from_api({"id": mid, "downloads": 1000, "pipeline_tag": pipeline, "gated": gated})
    return m


def test_pick_starter_skips_gated_embedding_and_too_big():
    from llamalauncher.app.ui.wizard import pick_starter
    s = system()
    s.total_ram = 16 << 30
    s.gpus[0].vram_bytes = 16 << 30
    files = {
        "a/Gated-8B-GGUF": RepoFiles("a", [QuantFile("Q4_K_M", ["x.gguf"], 5 << 30)], [], []),
        "a/Embed-GGUF": RepoFiles("a", [QuantFile("F16", ["e.gguf"], 1 << 28)], [], []),
        "a/Huge-70B-GGUF": RepoFiles("a", [QuantFile("Q4_K_M", ["h.gguf"], 40 << 30)], [], []),
        "a/Good-4B-GGUF": RepoFiles("a", [QuantFile("Q4_K_M", ["g4.gguf"], 2500 << 20),
                                          QuantFile("Q8_0", ["g8.gguf"], 4300 << 20)], [], []),
    }
    models = [_model("a/Gated-8B-GGUF", gated=True), _model("a/Embed-GGUF", pipeline="feature-extraction"),
              _model("a/Huge-70B-GGUF"), _model("a/Good-4B-GGUF")]
    m, q = pick_starter(models, lambda rid: files[rid], s)
    assert m.id == "a/Good-4B-GGUF" and q.quant in ("Q4_K_M", "Q8_0")
    assert pick_starter(models[:3], lambda rid: files[rid], s) is None
    # 큰 컴퓨터라도 첫 모델은 10GB 이하
    big = system()
    big.total_ram = big.gpus[0].vram_bytes = 128 << 30
    files["a/Huge-70B-GGUF"].quants.append(QuantFile("Q2_K", ["h2.gguf"], 26 << 30))
    m, q = pick_starter([_model("a/Huge-70B-GGUF"), _model("a/Good-4B-GGUF")], lambda rid: files[rid], big)
    assert m.id == "a/Good-4B-GGUF" and q.size <= 10 << 30
    # 1~2비트 양자화는 첫 모델로 고르지 않는다
    files["a/Huge-70B-GGUF"].quants.append(QuantFile("UD-IQ1_M", ["h1.gguf"], 9 << 30))
    m, q = pick_starter([_model("a/Huge-70B-GGUF"), _model("a/Good-4B-GGUF")], lambda rid: files[rid], big)
    assert m.id == "a/Good-4B-GGUF"


def test_theme_apply(qapp):
    from llamalauncher.app.ui.theme import apply_theme, is_dark
    apply_theme(qapp, "dark")
    assert is_dark(qapp)
    apply_theme(qapp, "light")
    assert not is_dark(qapp)
    apply_theme(qapp, "system")


def test_wizard_first_steps(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    from llamalauncher.app.core.settings import Settings
    from llamalauncher.app.ui.main_window import MainWindow
    from llamalauncher.app.ui.wizard import SetupWizard
    monkeypatch.setattr(Settings, "save", lambda self, path=None: None)
    w = MainWindow(Settings(models_dir=str(tmp_path / "models")))
    end = time.time() + 20
    while time.time() < end and (w.dashboard.system is None or "llamacpp" not in w.launchers.statuses):
        qapp.processEvents()
        time.sleep(0.02)
    wiz = SetupWizard(w)
    wiz.show()
    qapp.processEvents()
    assert wiz.step == 0 and "메모리" in wiz.body.text() and wiz.next_btn.isEnabled()
    wiz._next()
    st = w.launchers.statuses["llamacpp"]
    if st.installed:
        assert "준비되어 있어요" in wiz.body.text() and wiz.next_btn.isEnabled()
    else:
        assert wiz.action_btn.text().startswith("⬇️") and not wiz.next_btn.isEnabled()
    wiz.reject()
    assert w.settings.first_run_done  # 닫으면 다시 묻지 않는다 (메뉴에서 다시 열 수 있음)
    w.downloads.shutdown()


def test_app_smoke_test(tmp_path):
    """실제로 앱을 실행해 모든 탭이 뜨는지 확인 (빌드 결과 확인에도 같은 옵션 사용)."""
    env = {**os.environ, "HOME": str(tmp_path), "QT_QPA_PLATFORM": "offscreen",
           "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring"}
    p = subprocess.run([sys.executable, "-m", "llamalauncher.main", "--smoke-test"], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=90)
    assert "SMOKE-TEST OK" in p.stdout, p.stdout + p.stderr
    assert "tabs=8" in p.stdout and "integrations=12" in p.stdout


def test_pyinstaller_spec_includes_data():
    spec = (ROOT / "llamalauncher.spec").read_text(encoding="utf-8")
    assert "llamalauncher/app/data" in spec and "catalog.json" not in spec.split("datas")[0]
    for f in ("build_mac.sh", "build_windows.bat"):
        assert (ROOT / f).exists()


def test_draft_models_not_offered_as_quants():
    """ggml-org 저장소의 dflash-/mtp- 파일은 보조(초안) 모델이라 양자화 목록에서 빠져야 한다 (실제 저장소 구조)."""
    from llamalauncher.app.core.hf_client import parse_tree
    names = {"Qwen3.8-27B-Q4_K_M.gguf": 18504, "Qwen3.8-27B-Q8_0.gguf": 27271, "dflash-Qwen3.8-27B-BF16.gguf": 3681,
             "mtp-Qwen3.8-27B-Q4_0.gguf": 1602, "mmproj-Qwen3.8-27B-Q8_0.gguf": 600, "draft/x-Q4_0.gguf": 100,
             "Qwen3-0.6B-draft-Q8_0.gguf": 600}
    f = parse_tree("ggml-org/Qwen3.8-27B-GGUF", [{"type": "file", "path": n, "size": s << 20} for n, s in names.items()])
    assert [q.quant for q in f.quants] == ["Q4_K_M", "Q8_0"] and len(f.other_gguf) == 4 and len(f.mmproj) == 1
