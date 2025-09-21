import tomli as tomllib
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Mapping, Union

BASE_DIR = Path(__file__).resolve().parent

# path to your toml file
CONFIG_FILE = Path(__file__).with_name("rag2_config.toml")

def _to_ns(d: Mapping[str, Any]) -> Any:
    return SimpleNamespace(
        **{k: _to_ns(v) if isinstance(v, dict) else v for k, v in d.items()}
    )

with open(CONFIG_FILE, "rb") as f:
    raw: dict[str, Any] = tomllib.load(f)

cfg = _to_ns(raw)

# ---------- auto-create paths on import ----------
def _as_path(val: Any) -> Path | None:
    if isinstance(val, Path):
        return (BASE_DIR / val).expanduser().resolve()
    if isinstance(val, str):
        return (BASE_DIR / Path(val)).expanduser().resolve()
    return None

def _ensure_dirs() -> None:
    """
    Create folders declared under cfg.paths.
    - If a value has no suffix (e.g., 'data/processed'), create that dir.
    - If it has a suffix (e.g., 'data/processed/parsed.json'), create its parent.
    Resolves all paths relative to this script’s location.
    """
    paths = getattr(cfg, "paths", None)
    if not isinstance(paths, SimpleNamespace):
        return

    for name, val in vars(paths).items():
        p = _as_path(val)
        if p is None:
            continue
        target = p if p.suffix == "" else p.parent
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[paths] warn: could not create '{target}' for '{name}': {e}")
            
            
def get_rag2_config_path(name_or_path: Union[str, Path]) -> Path:
    """
    Resolve a path relative to this config's BASE_DIR.

    - If `name_or_path` is a key under cfg.paths (e.g., "raw", "proc"),
      use that value from the TOML.
    - Otherwise, treat it as a literal path.

    Returns an absolute Path.
    """
    # try as a key under [paths]
    paths = getattr(cfg, "paths", None)
    if isinstance(name_or_path, str) and isinstance(paths, SimpleNamespace) and hasattr(paths, name_or_path):
        val = getattr(paths, name_or_path)
        p = _as_path(val)
        if p is not None:
            return p

    raise ValueError(f"Cannot resolve path from: {name_or_path!r}")


# auto-run on import
_ensure_dirs()