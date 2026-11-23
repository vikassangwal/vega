"""
JARVIS OMEGA - File Tools
Built-in tools for file system operations.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from tools.tool_registry import ToolRegistry, ToolMetadata, ToolParameter
from tools.builtin.file_tools import _workspace

logger = logging.getLogger("vega.tools.file")


# =============================================================================
# WORKSPACE MANAGEMENT
# =============================================================================

class WorkspaceManager:
    """Manages safe file access within workspace boundaries."""

    def __init__(self, workspace_root: Path | None = None):
        self.workspace_root = workspace_root or Path.home() / ".vega" / "workspace"
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str) -> Path:
        """Resolve path relative to workspace, preventing traversal."""
        target = Path(path)

        if target.is_absolute():
            try:
                target.relative_to(self.workspace_root)
            except ValueError:
                raise PermissionError(f"Path outside workspace: {path}")
            return target

        full_path = (self.workspace_root / path).resolve()
        try:
            full_path.relative_to(self.workspace_root)
        except ValueError:
            raise PermissionError(f"Path traversal attempt: {path}")

        return full_path

    def ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)


_workspace = WorkspaceManager()


def get_workspace() -> WorkspaceManager:
    return _workspace


def set_workspace_root(root: Path) -> None:
    global _workspace
    _workspace = WorkspaceManager(root)


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

async def file_read(params: dict) -> dict:
    path = _workspace.resolve_path(params["path"])
    encoding = params.get("encoding", "utf-8")
    max_size = params.get("max_size", 10 * 1024 * 1024)

    if not path.exists():
        return {"success": False, "error": "File not found"}

    if path.stat().st_size > max_size:
        return {"success": False, "error": f"File too large (max {max_size} bytes)"}

    try:
        if encoding == "binary":
            content = path.read_bytes()
        else:
            content = path.read_text(encoding=encoding)

        return {
            "success": True,
            "content": content,
            "size": path.stat().st_size,
            "encoding": encoding,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_write(params: dict) -> dict:
    path = _workspace.resolve_path(params["path"])
    content = params["content"]
    encoding = params.get("encoding", "utf-8")
    mode = params.get("mode", "w")

    _workspace.ensure_parent(path)

    try:
        if isinstance(content, str):
            if encoding == "binary":
                import base64
                data = base64.b64decode(content)
                path.write_bytes(data)
            else:
                path.write_text(content, encoding=encoding)
        else:
            return {"success": False, "error": "Content must be string"}

        return {"success": True, "path": str(path), "size": path.stat().st_size}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_append(params: dict) -> dict:
    params["mode"] = "a"
    return await file_write(params)


async def file_delete(params: dict) -> dict:
    path = _workspace.resolve_path(params["path"])
    recursive = params.get("recursive", False)

    if not path.exists():
        return {"success": False, "error": "Path not found"}

    try:
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                path.rmdir()
        else:
            path.unlink()

        return {"success": True, "path": str(path)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_list(params: dict) -> dict:
    path = _workspace.resolve_path(params.get("path", "."))
    pattern = params.get("pattern", "*")
    recursive = params.get("recursive", False)
    include_hidden = params.get("include_hidden", False)

    if not path.exists() or not path.is_dir():
        return {"success": False, "error": "Directory not found"}

    try:
        if recursive:
            files = path.rglob(pattern)
        else:
            files = path.glob(pattern)

        result = []
        for f in files:
            if not include_hidden and f.name.startswith("."):
                continue

            stat = f.stat()
            result.append({
                "name": f.name,
                "path": str(f.relative_to(_workspace.workspace_root)),
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_dir": f.is_dir(),
                "extension": f.suffix,
            })

        return {"success": True, "files": result, "count": len(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_search(params: dict) -> dict:
    path = _workspace.resolve_path(params.get("path", "."))
    pattern = params.get("pattern", "*")
    content = params.get("content")
    file_types = params.get("file_types")
    max_results = params.get("max_results", 50)

    if not path.exists():
        return {"success": False, "error": "Path not found"}

    try:
        matches = []

        for f in path.rglob(pattern):
            if not f.is_file():
                continue

            if file_types and f.suffix not in file_types:
                continue

            if content:
                try:
                    if content not in f.read_text(encoding="utf-8", errors="ignore"):
                        continue
                except Exception:
                    continue

            matches.append({
                "path": str(f.relative_to(_workspace.workspace_root)),
            })

            if len(matches) >= max_results:
                break

        return {"success": True, "matches": matches, "count": len(matches)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_copy(params: dict) -> dict:
    src = _workspace.resolve_path(params["src"])
    dst = _workspace.resolve_path(params["dst"])
    overwrite = params.get("overwrite", False)

    if not src.exists():
        return {"success": False, "error": "Source not found"}

    if dst.exists() and not overwrite:
        return {"success": False, "error": "Destination exists (use overwrite=true)"}

    _workspace.ensure_parent(dst)

    try:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(src, dst)

        return {"success": True, "src": str(src), "dst": str(dst)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_move(params: dict) -> dict:
    src = _workspace.resolve_path(params["src"])
    dst = _workspace.resolve_path(params["dst"])
    overwrite = params.get("overwrite", False)

    if not src.exists():
        return {"success": False, "error": "Source not found"}

    if dst.exists() and not overwrite:
        return {"success": False, "error": "Destination exists (use overwrite=true)"}

    _workspace.ensure_parent(dst)

    try:
        shutil.move(str(src), str(dst))
        return {"success": True, "src": str(src), "dst": str(dst)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_info(params: dict) -> dict:
    path = _workspace.resolve_path(params["path"])

    if not path.exists():
        return {"success": False, "error": "Path not found"}

    try:
        stat = path.stat()
        return {
            "success": True,
            "path": str(path.relative_to(_workspace.workspace_root)),
            "name": path.name,
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
            "extension": path.suffix,
            "permissions": oct(stat.st_mode)[-3:],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_archive(params: dict) -> dict:
    paths = [_workspace.resolve_path(p) for p in params["paths"]]
    output = _workspace.resolve_path(params["output"])
    format = params.get("format", "zip")

    _workspace.ensure_parent(output)

    try:
        if format == "zip":
            import zipfile
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
                for p in paths:
                    if p.is_dir():
                        for f in p.rglob("*"):
                            if f.is_file():
                                zf.write(f, f.relative_to(_workspace.workspace_root))
                    else:
                        zf.write(p, p.relative_to(_workspace.workspace_root))
        elif format in ["tar.gz", "tgz"]:
            import tarfile
            with tarfile.open(output, "w:gz") as tf:
                for p in paths:
                    tf.add(p, arcname=str(p.relative_to(_workspace.workspace_root)))
        else:
            return {"success": False, "error": f"Unsupported format: {format}"}

        return {"success": True, "output": str(output), "size": output.stat().st_size}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_extract(params: dict) -> dict:
    archive = _workspace.resolve_path(params["archive"])
    output = _workspace.resolve_path(params["output"])
    format = params.get("format")

    if not archive.exists():
        return {"success": False, "error": "Archive not found"}

    _workspace.ensure_parent(output)

    try:
        if format == "zip" or (not format and archive.suffix == ".zip"):
            import zipfile
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(output)
        elif format in ["tar.gz", "tgz"] or (not format and archive.suffix in [".tar.gz", ".tgz"]):
            import tarfile
            with tarfile.open(archive, "r:gz") as tf:
                tf.extractall(output)
        else:
            return {"success": False, "error": f"Unsupported format: {format}"}

        return {"success": True, "output": str(output)}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def file_temp(params: dict) -> dict:
    prefix = params.get("prefix", "vega_")
    suffix = params.get("suffix", "")
    directory = params.get("directory", False)

    try:
        if directory:
            path = Path(tempfile.mkdtemp(prefix=prefix, suffix=suffix))
        else:
            fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            os.close(fd)
            path = Path(tmp_path)

        return {"success": True, "path": str(path)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# REGISTRATION
# =============================================================================

def register_file_tools(registry: ToolRegistry) -> None:
    tools = [
        ("file_read", file_read, "Read file contents", [
            ToolParameter("path", "string", "File path relative to workspace", required=True),
            ToolParameter("encoding", "string", "Text encoding", default="utf-8"),
            ToolParameter("max_size", "integer", "Max bytes to read", default=10485760),
        ]),
        ("file_write", file_write, "Write content to file", [
            ToolParameter("path", "string", "File path", required=True),
            ToolParameter("content", "string", "Content to write", required=True),
            ToolParameter("encoding", "string", "Text encoding", default="utf-8"),
            ToolParameter("mode", "string", "Write mode: w=overwrite, a=append", default="w"),
        ]),
        ("file_append", file_append, "Append content to file", [
            ToolParameter("path", "string", "File path", required=True),
            ToolParameter("content", "string", "Content to append", required=True),
            ToolParameter("encoding", "string", "Text encoding", default="utf-8"),
        ]),
        ("file_delete", file_delete, "Delete file or directory", [
            ToolParameter("path", "string", "Path to delete", required=True),
            ToolParameter("recursive", "boolean", "Delete directories recursively", default=False),
        ]),
        ("file_list", file_list, "List files in directory", [
            ToolParameter("path", "string", "Directory path", default="."),
            ToolParameter("pattern", "string", "Glob pattern", default="*"),
            ToolParameter("recursive", "boolean", "Search recursively", default=False),
            ToolParameter("include_hidden", "boolean", "Include hidden files", default=False),
        ]),
        ("file_search", file_search, "Search files by name or content", [
            ToolParameter("pattern", "string", "Filename pattern", default="*"),
            ToolParameter("content", "string", "Text to search in contents"),
            ToolParameter("path", "string", "Search root", default="."),
            ToolParameter("file_types", "array", "File extensions to include"),
            ToolParameter("max_results", "integer", "Maximum results", default=50),
        ]),
        ("file_copy", file_copy, "Copy file or directory", [
            ToolParameter("src", "string", "Source path", required=True),
            ToolParameter("dst", "string", "Destination path", required=True),
            ToolParameter("overwrite", "boolean", "Overwrite if exists", default=False),
        ]),
        ("file_move", file_move, "Move/rename file or directory", [
            ToolParameter("src", "string", "Source path", required=True),
            ToolParameter("dst", "string", "Destination path", required=True),
            ToolParameter("overwrite", "boolean", "Overwrite if exists", default=False),
        ]),
        ("file_info", file_info, "Get file metadata", [
            ToolParameter("path", "string", "File path", required=True),
        ]),
        ("file_archive", file_archive, "Create archive (zip/tar.gz)", [
            ToolParameter("paths", "array", "Paths to archive", required=True),
            ToolParameter("output", "string", "Output archive path", required=True),
            ToolParameter("format", "string", "Archive format: zip or tar.gz", default="zip"),
        ]),
        ("file_extract", file_extract, "Extract archive", [
            ToolParameter("archive", "string", "Archive path", required=True),
            ToolParameter("output", "string", "Output directory", required=True),
            ToolParameter("format", "string", "Archive format: zip or tar.gz"),
        ]),
        ("file_temp", file_temp, "Create temporary file/directory", [
            ToolParameter("prefix", "string", "Temp name prefix", default="vega_"),
            ToolParameter("suffix", "string", "Temp name suffix", default=""),
            ToolParameter("directory", "boolean", "Create temp directory", default=False),
        ]),
    ]

    for name, func, desc, params in tools:
        metadata = ToolMetadata(
            name=name,
            description=desc,
            category="file",
            parameters=params,
            returns={"type": "object"},
            sandboxed=True,
            tags=["file", "filesystem"],
        )
        registry.register(name, func, metadata)

    logger.info("Registered file tools")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "register_file_tools",
    "WorkspaceManager",
    "get_workspace",
    "set_workspace_root",
]