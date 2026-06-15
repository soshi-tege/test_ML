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
_MIN_MINOR_VERSION = 10
_MAX_MINOR_VERSION = 12
_PREFERRED_PYTHON = "3.12"


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _python_minor_version_from_cmd(command: list[str]) -> int | None:
    try:
        return int(
            subprocess.check_output(
                [*command, "-c", "import sys; print(sys.version_info.minor)"],
                text=True,
            )
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _find_compatible_python_command() -> list[str] | None:
    candidates = _COMPATIBLE_PYTHONS
    if sys.platform == "win32":
        candidates = _WINDOWS_COMPATIBLE_PYTHONS + candidates

    for cmd, extra_args in candidates:
        exe = shutil.which(cmd)
        if not exe:
            continue
        command = [exe, *extra_args]
        minor = _python_minor_version_from_cmd(command)
        if minor is None:
            continue
        if _MIN_MINOR_VERSION <= minor <= _MAX_MINOR_VERSION:
            return command
    return None


def _try_install_compatible_python() -> bool:
    if sys.platform == "win32":
        winget = shutil.which("winget")
        if winget:
            try:
                subprocess.check_call(
                    [
                        winget,
                        "install",
                        "-e",
                        "--id",
                        "Python.Python.3.12",
                        "--accept-package-agreements",
                        "--accept-source-agreements",
                    ]
                )
                return True
            except subprocess.CalledProcessError:
                return False
        return False

    if sys.platform == "darwin":
        brew = shutil.which("brew")
        if brew:
            try:
                subprocess.check_call([brew, "install", "python@3.12"])
                return True
            except subprocess.CalledProcessError:
                return False
        return False

    apt = shutil.which("apt-get")
    if apt:
        try:
            subprocess.check_call([apt, "update"])
            subprocess.check_call([apt, "install", "-y", "python3.12", "python3.12-venv"])
            return True
        except subprocess.CalledProcessError:
            return False
    return False


def _python_for_venv() -> list[str]:
    """TensorFlow 向けに 3.10〜3.12 の Python を優先して選ぶ"""
    command = _find_compatible_python_command()
    if command:
        return command

    print(
        "TensorFlow 対応の Python 3.10〜3.12 が見つからないため、"
        "自動インストールを試みます..."
    )
    if _try_install_compatible_python():
        command = _find_compatible_python_command()
        if command:
            return command

    raise RuntimeError(
        "TensorFlow に対応した Python 3.10〜3.12 が見つかりません。"
        f" Python {_PREFERRED_PYTHON} をインストールして PATH に追加してから、"
        "もう一度このスクリプトを実行してください。"
    )


def _python_minor_version(python: Path | str) -> int | None:
    try:
        return int(
            subprocess.check_output(
                [str(python), "-c", "import sys; print(sys.version_info.minor)"],
                text=True,
            )
        )
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def _is_compatible_python(python: Path | str) -> bool:
    minor = _python_minor_version(python)
    return minor is not None and _MIN_MINOR_VERSION <= minor <= _MAX_MINOR_VERSION


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

    if VENV_DIR.exists() and not _is_compatible_python(py):
        raise RuntimeError(
            "既存の .venv が TensorFlow 非対応の Python で作られています。"
            " .venv を削除してから、Python 3.10〜3.12 が使える状態で"
            " もう一度実行してください。"
        )

    if _in_venv():
        if _deps_installed():
            return
        print("依存パッケージをインストールしています...")
        _pip_install(sys.executable)
        return

    if not VENV_DIR.exists():
        print("仮想環境を作成しています（初回のみ）...")
        venv_cmd = _python_for_venv()
        subprocess.check_call([*venv_cmd, "-m", "venv", str(VENV_DIR)])

    print("依存パッケージをインストールしています（初回のみ）...")
    _pip_install(py)

    os.execv(str(py), [str(py), *sys.argv])
