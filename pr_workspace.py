"""
pr_workspace.py - Isolated PR Workspace Creation and Secure Archive Extraction.

Milestone 4 — Stage 3:
  1. Commit-bound workspace: Prepares local workspace strictly bound to the exact PR head SHA.
  2. Untrusted archive defense: Pre-extraction validation blocking directory traversal,
     Windows drive specifiers, absolute paths, symlinks, hardlinks, and special device files.
  3. Collision & normalization defense: Detects duplicate paths, Windows case-folding collisions
     (e.g. foo.py vs FOO.py), ambiguous trailing dot/space components, invalid characters,
     and file/directory hierarchy conflicts before writing anything to disk.
  4. Decompression bomb limits: Enforces entry count, archive size, expanded size, and path length.
  5. Automatic cleanup: Guarantees partial-extraction cleanup on error and disposable workspace lifecycle.
  6. Sandbox boundary: Prepares filesystem layout only; never executes arbitrary code on the host.
"""

from __future__ import annotations

import datetime
import logging
import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from github_client import GitHubClient

logger = logging.getLogger(__name__)

# Default security limits for untrusted archive extraction
DEFAULT_MAX_ARCHIVE_BYTES = 50 * 1024 * 1024  # 50 MB compressed
DEFAULT_MAX_EXPANDED_BYTES = 250 * 1024 * 1024  # 250 MB decompressed
DEFAULT_MAX_ENTRIES = 10_000  # Max files/directories in archive
DEFAULT_MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024  # 50 MB max per single file
DEFAULT_MAX_PATH_LENGTH = 260  # Safe path length on Windows

WINDOWS_RESERVED_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CLOCK$",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)

_DRIVE_PATTERN = re.compile(r"^[a-zA-Z]:")
_INVALID_PATH_CHARS = frozenset('<>:"|?*')


# ============================================================================
# Exceptions
# ============================================================================


class PRWorkspaceError(Exception):
    """Base exception for all PR workspace errors."""


class CommitBindingError(PRWorkspaceError):
    """Raised when expected commit SHA cannot be established or verified."""


class ArchiveSecurityError(PRWorkspaceError):
    """Raised when an untrusted archive contains malicious entries or exceeds limits."""


class ArchiveExtractionError(PRWorkspaceError):
    """Raised when an archive is corrupted or extraction fails."""


# ============================================================================
# Data Models
# ============================================================================


@dataclass(frozen=True)
class WorkspaceLimits:
    """Configurable security boundaries for untrusted archive extraction."""

    max_archive_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES
    max_expanded_bytes: int = DEFAULT_MAX_EXPANDED_BYTES
    max_entries: int = DEFAULT_MAX_ENTRIES
    max_single_file_bytes: int = DEFAULT_MAX_SINGLE_FILE_BYTES
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH


def _safe_delete_workspace(path: Path | str | None) -> None:
    """
    Safely delete an isolated workspace directory, ensuring it never deletes
    the root filesystem, current working directory, or system temp root.
    """
    if not path:
        return
    try:
        p = Path(path).resolve()
        cwd = Path.cwd().resolve()
        temp_root = Path(tempfile.gettempdir()).resolve()
        # Must not be cwd, temp root itself, filesystem root, or empty path
        if p == cwd or p == temp_root or p == Path("/").resolve() or p.parent == p:
            logger.warning("Refusing to delete unsafe root path: %s", p)
            return
        if str(p) in ("/", "\\", ""):
            return
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    except (OSError, RuntimeError) as e:
        logger.warning("Error deleting workspace directory %s: %s", path, e)


@dataclass
class PRWorkspace:
    """
    Represents an isolated, commit-bound local workspace for a PR.

    Can be used as a context manager for automatic disposable cleanup:
        with prepare_pr_workspace(owner, repo, pr_number) as ws:
            # Inspect workspace files at ws.workspace_path
            ...
    """

    owner: str
    repo: str
    pr_number: int
    head_sha: str
    workspace_path: Path
    created_at: str
    _cleanup_on_exit: bool = True
    _cleaned: bool = False

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self._cleanup_on_exit:
            self.cleanup()

    def cleanup(self) -> None:
        """Safely delete the isolated workspace directory."""
        if not self._cleaned and self.workspace_path.exists():
            _safe_delete_workspace(self.workspace_path)
            self._cleaned = True


# ============================================================================
# Path Conflict & Collision Tracker
# ============================================================================


class PathConflictTracker:
    """
    Pre-extraction tracker that detects and rejects:
      1. Duplicate paths (e.g. 'foo.py' and 'foo.py').
      2. Windows case-folding collisions (e.g. 'foo.py' and 'FOO.py').
      3. Ambiguous alternate representations.
      4. File/directory conflicts (e.g. 'foo' as file and 'foo/' as directory,
         or 'foo' as file and 'foo/bar.txt' as file).
    """

    def __init__(self) -> None:
        self.seen_files: dict[str, str] = {}  # casefold -> original_path
        self.seen_dirs: dict[str, str] = {}  # casefold -> original_path

    def register_entry(self, clean_rel: str, is_dir: bool, original_name: str) -> None:
        casefold = clean_rel.casefold()
        parts = [p for p in casefold.split("/") if p]
        if not parts:
            return

        # Check all ancestor components to ensure no ancestor is an existing file
        orig_parts = clean_rel.split("/")
        for i in range(1, len(parts)):
            parent = "/".join(parts[:i])
            if parent in self.seen_files:
                raise ArchiveSecurityError(
                    f"File/directory conflict: path '{clean_rel}' is inside file '{self.seen_files[parent]}'."
                )
            # Record implicit parent directory in seen_dirs
            if parent not in self.seen_dirs:
                orig_parent = "/".join(orig_parts[:i])
                self.seen_dirs[parent] = orig_parent

        if is_dir:
            if casefold in self.seen_files:
                raise ArchiveSecurityError(
                    f"File/directory conflict: directory '{clean_rel}' conflicts with existing file '{self.seen_files[casefold]}'."
                )
            if casefold in self.seen_dirs:
                if self.seen_dirs[casefold] != clean_rel:
                    raise ArchiveSecurityError(
                        f"Case-folding collision for directory: '{clean_rel}' collides with '{self.seen_dirs[casefold]}'."
                    )
            else:
                self.seen_dirs[casefold] = clean_rel
        else:
            if casefold in self.seen_files:
                raise ArchiveSecurityError(
                    f"Duplicate or colliding path detected: '{clean_rel}' collides with '{self.seen_files[casefold]}'."
                )
            if casefold in self.seen_dirs:
                raise ArchiveSecurityError(
                    f"File/directory conflict: file '{clean_rel}' conflicts with existing directory '{self.seen_dirs[casefold]}'."
                )
            # Check if this file name is an ancestor of any existing file or directory
            for existing_dir in self.seen_dirs:
                if existing_dir == casefold or existing_dir.startswith(casefold + "/"):
                    raise ArchiveSecurityError(
                        f"File/directory conflict: file '{clean_rel}' conflicts with directory path '{self.seen_dirs[existing_dir]}'."
                    )
            self.seen_files[casefold] = clean_rel


# ============================================================================
# Path Sanitization & Validation
# ============================================================================


def _normalize_archive_path(
    raw_name: str,
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
) -> tuple[str, bool]:
    """
    Validate and normalize a raw archive member path string.

    Returns:
        tuple[str, bool]: (cleaned_relative_posix_path, is_directory)

    Raises:
        ArchiveSecurityError: On any invalid, ambiguous, or traversal component.
    """
    if "\x00" in raw_name:
        raise ArchiveSecurityError(f"Null byte detected in archive path: {raw_name!r}")

    # Normalize backslashes to forward slashes
    normalized = raw_name.replace("\\", "/")
    if normalized.startswith(" ") or normalized.endswith(" "):
        raise ArchiveSecurityError(
            f"Ambiguous Windows path component with leading/trailing space or dot: {raw_name!r}"
        )
    is_dir = normalized.endswith("/")

    # Reject drive letters (e.g., C:/foo or C:foo)
    if _DRIVE_PATTERN.match(normalized):
        raise ArchiveSecurityError(f"Windows drive specifier detected: {raw_name!r}")

    # Reject absolute paths
    if normalized.startswith("/"):
        raise ArchiveSecurityError(f"Absolute path detected: {raw_name!r}")

    # Check for invalid characters on Windows: <>:"|?* and control characters
    for ch in normalized:
        if ch in _INVALID_PATH_CHARS or ord(ch) < 32:
            raise ArchiveSecurityError(
                f"Invalid character {ch!r} detected in archive path: {raw_name!r}"
            )

    raw_parts = normalized.split("/")
    clean_parts: list[str] = []

    for part in raw_parts:
        if not part or part == ".":
            continue
        if part == "..":
            raise ArchiveSecurityError(
                f"Directory traversal ('..') detected: {raw_name!r}"
            )

        # Check for leading/trailing space or trailing dot (NTFS normalization exploit)
        if part.startswith(" ") or part.endswith((" ", ".")):
            raise ArchiveSecurityError(
                f"Ambiguous Windows path component with leading/trailing space or dot: {part!r}"
            )

        # Check Windows reserved device names
        stem = part.split(".")[0].upper()
        if stem in WINDOWS_RESERVED_NAMES:
            raise ArchiveSecurityError(
                f"Windows reserved device name detected: {part!r}"
            )

        clean_parts.append(part)

    if not clean_parts:
        return "", True

    rel_str = "/".join(clean_parts)
    if len(rel_str) > max_path_length:
        raise ArchiveSecurityError(
            f"Archive member path exceeds maximum length of {max_path_length}: {rel_str!r}"
        )

    return rel_str, is_dir


def sanitize_archive_member_path(
    raw_name: str,
    dest_root: Path,
    max_path_length: int = DEFAULT_MAX_PATH_LENGTH,
) -> tuple[Path, bool]:
    """
    Validate and resolve an archive member path strictly inside dest_root.

    Rejects:
      - Null bytes in filename.
      - Windows drive paths (e.g. C:file.txt).
      - Absolute paths (e.g. /etc/passwd or \\windows\\win.ini).
      - Path traversal ('..') segments anywhere in the path.
      - Ambiguous components (trailing dots or spaces).
      - Windows reserved device names (CON, PRN, NUL, AUX, COM1-9, LPT1-9, CLOCK$).
      - Invalid Windows characters (< > : " | ? * or ASCII < 32).
      - Path lengths exceeding max_path_length.
      - Paths that resolve outside dest_root.

    Returns:
        tuple[Path, bool]: (resolved_destination_path, is_directory)

    Raises:
        ArchiveSecurityError: On any security boundary violation.
    """
    rel_str, is_dir = _normalize_archive_path(raw_name, max_path_length=max_path_length)
    if not rel_str:
        return dest_root, True

    # Resolve target path and verify workspace boundary
    dest_root_resolved = dest_root.resolve()
    target_resolved = (dest_root_resolved / rel_str).resolve()

    try:
        if not target_resolved.is_relative_to(dest_root_resolved):
            raise ArchiveSecurityError(
                f"Path traversal escape detected: '{raw_name}' resolves outside workspace."
            )
    except (ValueError, RuntimeError) as e:
        raise ArchiveSecurityError(
            f"Path traversal escape detected: '{raw_name}' escapes destination root."
        ) from e

    return target_resolved, is_dir


def _detect_common_prefix(entry_names: list[str]) -> str:
    """
    Detect if all archive entries share a single common top-level directory.

    GitHub zipball and tarball archives package repositories under a single top-level
    directory like `{owner}-{repo}-{short_sha}/`.

    Security constraints:
      - The candidate common prefix must be a clean, valid directory name (no '..', '.',
        drive letters, reserved device names, trailing dots/spaces, or control characters).
      - Every non-empty entry must either match the prefix exactly (the directory entry)
        or reside strictly within the prefix (start with prefix + '/').
      - Cannot conceal directory traversal or alternate path escapes.
    """
    non_empty = [e.replace("\\", "/").strip() for e in entry_names if e.strip()]
    if not non_empty:
        return ""

    first_components: set[str] = set()
    for name in non_empty:
        # If any entry contains traversal or forbidden characters, do not strip prefix
        if "\x00" in name or ".." in name or ":" in name or name.startswith("/"):
            return ""
        parts = [p for p in name.split("/") if p]
        if parts:
            first_components.add(parts[0])

    if len(first_components) != 1:
        return ""

    candidate = next(iter(first_components))

    # Validate that candidate is a safe single directory name
    if candidate in (".", "..") or not candidate:
        return ""
    if candidate.startswith(" ") or candidate.endswith((" ", ".")):
        return ""
    for ch in candidate:
        if ch in _INVALID_PATH_CHARS or ord(ch) < 32:
            return ""
    stem = candidate.split(".")[0].upper()
    if stem in WINDOWS_RESERVED_NAMES:
        return ""

    prefix = candidate + "/"

    # Verify that every non-empty entry either is the directory itself or is inside prefix
    has_child = False
    for name in non_empty:
        if name == candidate:
            # File entry with same name as candidate: not a directory!
            return ""
        if name == prefix:
            # Directory entry for candidate/
            continue
        if name.startswith(prefix):
            has_child = True
        else:
            return ""

    if not has_child:
        return ""

    return prefix


# ============================================================================
# Safe Extraction Helpers
# ============================================================================


def _verify_safe_extraction_target(target_p: Path, dest_root: Path) -> None:
    """Ensure target path and its ancestor directories do not traverse through symlinks."""
    dest_root_resolved = dest_root.resolve()
    target_resolved = target_p.resolve()

    if not target_resolved.is_relative_to(dest_root_resolved):
        raise ArchiveSecurityError(
            f"Extraction target '{target_p}' escapes destination root '{dest_root}'."
        )

    if os.path.islink(target_p) or target_p.is_symlink():
        raise ArchiveSecurityError(
            f"Target path cannot be a symbolic link: '{target_p}'."
        )

    curr = target_p.parent
    while curr != dest_root and curr != curr.parent:
        if os.path.islink(curr) or curr.is_symlink():
            raise ArchiveSecurityError(
                f"Pre-existing symlink encountered in extraction path: '{curr}'."
            )
        curr = curr.parent


def _ensure_safe_directory(target_p: Path, dest_root: Path) -> None:
    """Create directory target_p safely, ensuring no ancestor is a symlink."""
    dest_root_resolved = dest_root.resolve()
    if not target_p.resolve().is_relative_to(dest_root_resolved):
        raise ArchiveSecurityError(f"Directory '{target_p}' escapes destination root.")

    parts_to_create: list[Path] = []
    curr = target_p
    while curr != dest_root and curr != curr.parent:
        if os.path.islink(curr) or curr.is_symlink():
            raise ArchiveSecurityError(f"Symlink encountered in directory: '{curr}'.")
        parts_to_create.append(curr)
        curr = curr.parent

    for d in reversed(parts_to_create):
        if os.path.islink(d) or d.is_symlink():
            raise ArchiveSecurityError(f"Symlink encountered in directory: '{d}'.")
        d.mkdir(exist_ok=True)


def _ensure_safe_parent_directory(target_p: Path, dest_root: Path) -> None:
    """Ensure parent directory of target_p exists safely without traversing symlinks."""
    _ensure_safe_directory(target_p.parent, dest_root)


# ============================================================================
# Secure Archive Extraction
# ============================================================================


def extract_archive_securely(
    archive_path: Path | str,
    destination_dir: Path | str,
    limits: WorkspaceLimits | None = None,
) -> list[Path]:
    """
    Securely extract an untrusted ZIP or TAR archive into destination_dir.

    Pre-validates the entire archive before extracting:
      1. Inspects all members for traversal, absolute paths, drive letters, reserved names.
      2. Rejects duplicate paths, Windows case collisions, and file/directory conflicts.
      3. Rejects symbolic links, hard links, and special device files.
      4. Validates entry count, archive size, expanded size, and single file size.
      5. Strips single common top-level directory prefix safely.
      6. Verifies destination is a fresh, empty directory and not a symlink.
      7. Automatically cleans up partial extractions on any validation or write error.

    Returns:
        list[Path]: List of all extracted file paths.

    Raises:
        ArchiveSecurityError: If any security limit or path constraint is violated.
        ArchiveExtractionError: If archive is corrupted or extraction fails.
    """
    limits = limits or WorkspaceLimits()
    archive_p = Path(archive_path).resolve()
    dest_p = Path(destination_dir).resolve()

    if not archive_p.is_file():
        raise ArchiveExtractionError(f"Archive file not found: '{archive_p}'.")

    # Destination directory must not be a symbolic link
    if os.path.islink(destination_dir) or dest_p.is_symlink():
        raise ArchiveSecurityError(
            f"Destination directory cannot be a symbolic link: '{destination_dir}'."
        )

    # Destination directory must be empty if it exists
    if dest_p.exists():
        if not dest_p.is_dir():
            raise ArchiveSecurityError(
                f"Destination path is not a directory: '{dest_p}'."
            )
        if any(dest_p.iterdir()):
            raise ArchiveSecurityError(
                f"Destination directory must be empty: '{dest_p}'."
            )
    else:
        dest_p.mkdir(parents=True, exist_ok=True)

    # Check compressed archive size
    archive_size = archive_p.stat().st_size
    if archive_size > limits.max_archive_bytes:
        raise ArchiveSecurityError(
            f"Archive size ({archive_size} bytes) exceeds limit ({limits.max_archive_bytes} bytes)."
        )

    try:
        if zipfile.is_zipfile(archive_p):
            extracted_files = _extract_zip_securely(archive_p, dest_p, limits)
        elif tarfile.is_tarfile(archive_p):
            extracted_files = _extract_tar_securely(archive_p, dest_p, limits)
        else:
            raise ArchiveExtractionError(
                f"Unsupported or corrupted archive format: '{archive_p.name}'."
            )
    except Exception:
        # Atomic cleanup on error: remove partial files from destination safely
        _safe_delete_workspace(dest_p)
        raise

    return extracted_files


def _extract_zip_securely(
    archive_path: Path,
    dest_dir: Path,
    limits: WorkspaceLimits,
) -> list[Path]:
    """Pre-validate and safely extract a ZIP archive."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            infolist = zf.infolist()
            total_entries = len(infolist)
            if total_entries > limits.max_entries:
                raise ArchiveSecurityError(
                    f"Archive entry count ({total_entries}) exceeds limit ({limits.max_entries})."
                )

            total_expanded = 0
            names = [info.filename for info in infolist]
            prefix = _detect_common_prefix(names)

            # Pre-validation pass across all members
            validated_members: list[tuple[zipfile.ZipInfo, Path, bool]] = []
            conflict_tracker = PathConflictTracker()

            for info in infolist:
                # Check for symlink in external attributes (POSIX mode)
                mode = (info.external_attr >> 16) & 0o177777
                if (mode & 0o170000) == 0o120000:
                    raise ArchiveSecurityError(
                        f"Symbolic link forbidden in untrusted archive: {info.filename!r}"
                    )
                if (mode & 0o170000) in (0o020000, 0o060000, 0o010000):
                    raise ArchiveSecurityError(
                        f"Special device file forbidden in untrusted archive: {info.filename!r}"
                    )

                if info.file_size > limits.max_single_file_bytes:
                    raise ArchiveSecurityError(
                        f"File '{info.filename}' size ({info.file_size}) exceeds limit ({limits.max_single_file_bytes})."
                    )

                total_expanded += info.file_size
                if total_expanded > limits.max_expanded_bytes:
                    raise ArchiveSecurityError(
                        f"Total expanded size ({total_expanded}) exceeds limit ({limits.max_expanded_bytes})."
                    )

                # Strip common top-level directory prefix if applicable
                rel_name = info.filename.replace("\\", "/").strip()
                if prefix:
                    if rel_name.rstrip("/") == prefix.rstrip("/"):
                        # Skip root directory entry itself
                        continue
                    if rel_name.startswith(prefix):
                        rel_name = rel_name[len(prefix) :]
                    else:
                        raise ArchiveSecurityError(
                            f"Archive member '{info.filename}' does not start with expected common prefix '{prefix}'."
                        )
                if not rel_name or rel_name == ".":
                    continue

                target_p, is_dir = sanitize_archive_member_path(
                    rel_name, dest_dir, max_path_length=limits.max_path_length
                )
                clean_rel, clean_is_dir = _normalize_archive_path(
                    rel_name, max_path_length=limits.max_path_length
                )
                effective_is_dir = clean_is_dir or info.is_dir()
                conflict_tracker.register_entry(
                    clean_rel, effective_is_dir, info.filename
                )
                validated_members.append((info, target_p, effective_is_dir))

            # Extraction pass
            extracted: list[Path] = []
            bytes_written = 0
            for info, target_p, is_dir in validated_members:
                _verify_safe_extraction_target(target_p, dest_dir)
                if is_dir:
                    _ensure_safe_directory(target_p, dest_dir)
                else:
                    _ensure_safe_parent_directory(target_p, dest_dir)
                    if os.path.islink(target_p) or target_p.is_symlink():
                        raise ArchiveSecurityError(
                            f"Target path cannot be a symbolic link: '{target_p}'."
                        )
                    file_bytes_written = 0
                    with zf.open(info) as src, open(target_p, "wb") as dst:
                        while chunk := src.read(64 * 1024):
                            file_bytes_written += len(chunk)
                            if file_bytes_written > limits.max_single_file_bytes:
                                raise ArchiveSecurityError(
                                    f"File '{info.filename}' decompressed stream exceeded single-file limit ({limits.max_single_file_bytes} bytes)."
                                )
                            bytes_written += len(chunk)
                            if bytes_written > limits.max_expanded_bytes:
                                raise ArchiveSecurityError(
                                    f"Total decompressed stream exceeded limit ({limits.max_expanded_bytes} bytes)."
                                )
                            dst.write(chunk)
                    extracted.append(target_p)

            return extracted
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
        raise ArchiveExtractionError(f"Corrupted or invalid zip archive: {e}") from e


def _extract_tar_securely(
    archive_path: Path,
    dest_dir: Path,
    limits: WorkspaceLimits,
) -> list[Path]:
    """Pre-validate and safely extract a TAR / TAR.GZ archive."""
    try:
        with tarfile.open(archive_path, "r:*") as tf:
            members = tf.getmembers()
            total_entries = len(members)
            if total_entries > limits.max_entries:
                raise ArchiveSecurityError(
                    f"Archive entry count ({total_entries}) exceeds limit ({limits.max_entries})."
                )

            total_expanded = 0
            names = [m.name for m in members]
            prefix = _detect_common_prefix(names)

            validated_members: list[tuple[tarfile.TarInfo, Path, bool]] = []
            conflict_tracker = PathConflictTracker()

            for member in members:
                # Reject symlinks, hardlinks, devices, fifos
                if member.issym() or member.islnk():
                    raise ArchiveSecurityError(
                        f"Links forbidden in untrusted archive: {member.name!r}"
                    )
                if member.ischr() or member.isblk() or member.isfifo():
                    raise ArchiveSecurityError(
                        f"Special device forbidden in untrusted archive: {member.name!r}"
                    )

                if member.size > limits.max_single_file_bytes:
                    raise ArchiveSecurityError(
                        f"File '{member.name}' size ({member.size}) exceeds limit ({limits.max_single_file_bytes})."
                    )

                total_expanded += member.size
                if total_expanded > limits.max_expanded_bytes:
                    raise ArchiveSecurityError(
                        f"Total expanded size ({total_expanded}) exceeds limit ({limits.max_expanded_bytes})."
                    )

                rel_name = member.name.replace("\\", "/").strip()
                if prefix:
                    if rel_name.rstrip("/") == prefix.rstrip("/"):
                        continue
                    if rel_name.startswith(prefix):
                        rel_name = rel_name[len(prefix) :]
                    else:
                        raise ArchiveSecurityError(
                            f"Archive member '{member.name}' does not start with expected common prefix '{prefix}'."
                        )
                if not rel_name or rel_name == ".":
                    continue

                target_p, is_dir = sanitize_archive_member_path(
                    rel_name, dest_dir, max_path_length=limits.max_path_length
                )
                clean_rel, clean_is_dir = _normalize_archive_path(
                    rel_name, max_path_length=limits.max_path_length
                )
                effective_is_dir = clean_is_dir or member.isdir()
                conflict_tracker.register_entry(
                    clean_rel, effective_is_dir, member.name
                )
                validated_members.append((member, target_p, effective_is_dir))

            extracted: list[Path] = []
            bytes_written = 0
            for member, target_p, is_dir in validated_members:
                _verify_safe_extraction_target(target_p, dest_dir)
                if is_dir:
                    _ensure_safe_directory(target_p, dest_dir)
                else:
                    _ensure_safe_parent_directory(target_p, dest_dir)
                    if os.path.islink(target_p) or target_p.is_symlink():
                        raise ArchiveSecurityError(
                            f"Target path cannot be a symbolic link: '{target_p}'."
                        )
                    f_obj = tf.extractfile(member)
                    if f_obj is None:
                        continue
                    file_bytes_written = 0
                    with f_obj as src, open(target_p, "wb") as dst:
                        while chunk := src.read(64 * 1024):
                            file_bytes_written += len(chunk)
                            if file_bytes_written > limits.max_single_file_bytes:
                                raise ArchiveSecurityError(
                                    f"File '{member.name}' decompressed stream exceeded single-file limit ({limits.max_single_file_bytes} bytes)."
                                )
                            bytes_written += len(chunk)
                            if bytes_written > limits.max_expanded_bytes:
                                raise ArchiveSecurityError(
                                    f"Total decompressed stream exceeded limit ({limits.max_expanded_bytes} bytes)."
                                )
                            dst.write(chunk)
                    extracted.append(target_p)

            return extracted
    except (tarfile.TarError, EOFError) as e:
        raise ArchiveExtractionError(f"Corrupted or invalid tar archive: {e}") from e


# ============================================================================
# PR Workspace Preparation
# ============================================================================


def prepare_pr_workspace(
    owner: str,
    repo: str,
    pr_number: int,
    expected_head_sha: str | None = None,
    client: GitHubClient | None = None,
    base_dir: str | Path | None = None,
    limits: WorkspaceLimits | None = None,
    cleanup_on_exit: bool = True,
) -> PRWorkspace:
    """
    Prepare an isolated local workspace containing code for a specific PR commit.

    Workflow:
      1. Fetch PR metadata to identify live head SHA.
      2. If expected_head_sha is specified, strictly verifies that live head SHA matches.
      3. Downloads source archive targeting the exact commit SHA (never a moving branch).
      4. Pre-validates and extracts the archive into an isolated temporary directory.
      5. Cleans up archive file and returns a disposable PRWorkspace.

    Raises:
        CommitBindingError: If head SHA cannot be established or mismatches expected_head_sha.
        ArchiveSecurityError: If archive contains unsafe paths, links, or exceeds limits.
        PRWorkspaceError: On general network or workspace failures.
    """
    limits = limits or WorkspaceLimits()
    gh_client = client or GitHubClient()

    # 1. Fetch metadata
    meta = gh_client.get_pr_metadata(owner, repo, pr_number)
    live_sha = (meta.head_sha or "").strip()

    if not live_sha or len(live_sha) < 7:
        raise CommitBindingError(
            f"Could not establish valid head SHA from PR metadata for {owner}/{repo} #{pr_number}."
        )

    # 2. Bind exact commit
    if expected_head_sha:
        clean_expected = expected_head_sha.strip().lower()
        if clean_expected != live_sha.lower():
            raise CommitBindingError(
                f"PR head SHA mismatch for {owner}/{repo} #{pr_number}: expected commit "
                f"'{expected_head_sha}', but PR metadata reports '{live_sha}'. Aborting."
            )

    # 3. Create isolated temp directory
    target_dir = Path(base_dir).resolve() if base_dir else Path(tempfile.gettempdir())
    ws_dir = Path(
        tempfile.mkdtemp(prefix=f"pr_ws_{owner}_{repo}_{pr_number}_", dir=target_dir)
    )

    archive_temp_path: Path | None = None
    try:
        # 4. Download source archive for the EXACT commit SHA
        archive_temp_path = gh_client.download_commit_archive(
            owner=owner,
            repo=repo,
            commit_sha=live_sha,
            max_bytes=limits.max_archive_bytes,
        )

        # 5. Secure extraction
        extract_archive_securely(archive_temp_path, ws_dir, limits=limits)

        # 6. Delete downloaded archive
        if archive_temp_path and archive_temp_path.exists():
            archive_temp_path.unlink(missing_ok=True)
            archive_temp_path = None

        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        return PRWorkspace(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            head_sha=live_sha,
            workspace_path=ws_dir,
            created_at=now_str,
            _cleanup_on_exit=cleanup_on_exit,
        )
    except Exception:
        # Atomic cleanup on failure
        if archive_temp_path and archive_temp_path.exists():
            archive_temp_path.unlink(missing_ok=True)
        _safe_delete_workspace(ws_dir)
        raise


def create_pr_workspace_from_archive(
    owner: str,
    repo: str,
    pr_number: int,
    head_sha: str,
    archive_path: Path | str,
    base_dir: str | Path | None = None,
    limits: WorkspaceLimits | None = None,
    cleanup_on_exit: bool = True,
) -> PRWorkspace:
    """
    Offline helper to extract an existing archive directly into an isolated PR workspace.
    Useful for offline testing, hermetic environments, and local analysis.
    """
    limits = limits or WorkspaceLimits()
    if not head_sha or len(head_sha.strip()) < 7:
        raise CommitBindingError(f"Invalid head SHA provided: {head_sha!r}")

    target_dir = Path(base_dir).resolve() if base_dir else Path(tempfile.gettempdir())
    ws_dir = Path(
        tempfile.mkdtemp(prefix=f"pr_ws_{owner}_{repo}_{pr_number}_", dir=target_dir)
    )

    try:
        extract_archive_securely(archive_path, ws_dir, limits=limits)
        now_str = datetime.datetime.now(datetime.UTC).isoformat()
        return PRWorkspace(
            owner=owner,
            repo=repo,
            pr_number=pr_number,
            head_sha=head_sha.strip(),
            workspace_path=ws_dir,
            created_at=now_str,
            _cleanup_on_exit=cleanup_on_exit,
        )
    except Exception:
        _safe_delete_workspace(ws_dir)
        raise
