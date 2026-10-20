"""インストール済みメタデータ(git タグ由来)から版数を得る。未インストール時は不明版。"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kernelsmith")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
