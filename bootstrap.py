"""仮想環境と依存パッケージを自動セットアップする（初回実行時のみ時間がかかる）"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import REQUIREMENTS, VENV_DIR

_COMPATIBLE_PYTHONS = (
    ("python3.12", ()),
    ("python3.11", ()),
    ("python3.10", ()),
    ("python3", ()),
)
_WINDOWS_COMPATIBLE_PYTHONS = (
    ("py", ("-3.12",)),
    ("py", ("-3.11",)),
    ("py", ("-3.10",)),
    ("py", ("-3",)),
)
_MAX_MINOR_VERSION = 12


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _python_for_venv() -> str:
    """TensorFlow 向けに 3.10〜3.12 の Python を優先して選ぶ"""
    candidates = _COMPATIBLE_PYTHONS
    if sys.platform == "win32":
        candidates = _WINDOWS_COMPATIBLE_PYTHONS + candidates

    for cmd, extra_args in candidates:
        exe = shutil.which(cmd)
        if not exe:
            continue
        minor = int(
            subprocess.check_output(
                [exe, *extra_args, "-c", "import sys; print(sys.version_info.minor)"],
                text=True,
            )
        )
        if minor <= _MAX_MINOR_VERSION:
            return exe
    return sys.executable


def _in_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == venv_python().resolve()
    except FileNotFoundError:
        return False


def _deps_installed() -> bool:
    try:
        import tensorflow  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def _pip_install(python: Path | str) -> None:
    subprocess.check_call(
        [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)]
    )


def ensure_environment() -> None:
    """venv と依存関係を整え、必要なら venv の Python でスクリプトを再起動する"""
    py = venv_python()

    if _in_venv():
        if _deps_installed():
            return
        print("依存パッケージをインストールしています...")
        _pip_install(sys.executable)
        return

    if not VENV_DIR.exists():
        print("仮想環境を作成しています（初回のみ）...")
        subprocess.check_call([_python_for_venv(), "-m", "venv", str(VENV_DIR)])

    print("依存パッケージをインストールしています（初回のみ）...")
    _pip_install(py)

    os.execv(str(py), [str(py), *sys.argv])
