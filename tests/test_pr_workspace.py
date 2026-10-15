"""
tests/test_pr_workspace.py - Unit tests for Isolated PR Workspace Creation and Archive Security.

Tests:
  - Successful archive preparation bound to exact commit SHA.
  - Commit binding enforcement, detecting mismatches or missing SHA metadata.
  - Archive download failure and timeout handling with atomic cleanup.
  - Directory traversal attacks (unix ../ and windows ..\\).
  - Absolute path and Windows drive specifier attacks.
  - Windows reserved device name rejections (CON, NUL, AUX, COM1-9, LPT1-9, CLOCK$).
  - Symlink, hardlink, and device file escape attempts in zip and tar archives.
  - Windows path collision detection: foo.py vs FOO.py, duplicate normalized paths.
  - File/directory hierarchy conflicts before disk writes.
  - Ambiguous Windows path components (trailing dots/spaces, colons, invalid characters).
  - Destination safety: rejecting symlinks, non-empty directories, and traversing symlinks.
  - Common root prefix stripping safety: ensuring traversal cannot be concealed.
  - Archive bomb defenses (entry count, expanded size, single-file size, archive size).
  - Malformed and unsupported archive formats with partial extraction cleanup.
  - Context manager and disposable lifecycle cleanup on success and failure.
  - Boundary-checked cleanup refusing to delete unsafe system directories.
"""

from __future__ import annotations

import io
import tarfile
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from github_client import GitHubClient, GitHubNetworkError, PRMetadata
from pr_workspace import (
    ArchiveExtractionError,
    ArchiveSecurityError,
    CommitBindingError,
    PRWorkspace,
    WorkspaceLimits,
    _detect_common_prefix,
    _safe_delete_workspace,
    create_pr_workspace_from_archive,
    extract_archive_securely,
    prepare_pr_workspace,
    sanitize_archive_member_path,
)

# ============================================================================
# Archive Construction Helpers
# ============================================================================


def create_zip_bytes(entries: dict[str, str | bytes]) -> bytes:
    """Create in-memory zip archive bytes with arbitrary filenames."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(name, data)
    return buf.getvalue()


def create_tar_bytes(entries: dict[str, str | bytes]) -> bytes:
    """Create in-memory tar.gz archive bytes."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, content in entries.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            ti = tarfile.TarInfo(name=name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


# ============================================================================
# Path Sanitization Tests
# ============================================================================


class TestSanitizeArchiveMemberPath:
    """Test path sanitization against directory traversal and platform exploits."""

    def test_valid_relative_paths(self, tmp_path):
        target, is_dir = sanitize_archive_member_path("src/main.py", tmp_path)
        assert is_dir is False
        assert target == (tmp_path / "src" / "main.py").resolve()

        target_dir, is_dir = sanitize_archive_member_path("src/subdir/", tmp_path)
        assert is_dir is True
        assert target_dir == (tmp_path / "src" / "subdir").resolve()

    def test_reject_null_bytes(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="Null byte"):
            sanitize_archive_member_path("foo.txt\x00bar.py", tmp_path)

    def test_reject_directory_traversal(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            sanitize_archive_member_path("../escape.py", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            sanitize_archive_member_path("sub/../../escape.py", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            sanitize_archive_member_path("..\\escape.py", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            sanitize_archive_member_path("sub\\..\\..\\escape.py", tmp_path)

    def test_reject_absolute_paths(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="Absolute path"):
            sanitize_archive_member_path("/etc/passwd", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Absolute path"):
            sanitize_archive_member_path("/var/log/syslog", tmp_path)

    def test_reject_windows_drive_specifiers(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="Windows drive specifier"):
            sanitize_archive_member_path("C:/Windows/win.ini", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Windows drive specifier"):
            sanitize_archive_member_path("D:script.py", tmp_path)

    def test_reject_windows_reserved_device_names(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="reserved device name"):
            sanitize_archive_member_path("CON", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="reserved device name"):
            sanitize_archive_member_path("src/NUL.txt", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="reserved device name"):
            sanitize_archive_member_path("com1.py", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="reserved device name"):
            sanitize_archive_member_path("aux/data", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="reserved device name"):
            sanitize_archive_member_path("clock$", tmp_path)

    def test_reject_excessive_path_length(self, tmp_path):
        long_name = "a" * 261
        with pytest.raises(ArchiveSecurityError, match="exceeds maximum length"):
            sanitize_archive_member_path(long_name, tmp_path, max_path_length=260)


# ============================================================================
# Windows Path Ambiguities & Invalid Characters
# ============================================================================


class TestAmbiguousWindowsPathsAndCharacters:
    """Test rejection of alternate, ambiguous Windows path forms and characters."""

    def test_reject_trailing_dots_and_spaces(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="leading/trailing space or dot"):
            sanitize_archive_member_path("foo.py.", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="leading/trailing space"):
            sanitize_archive_member_path("foo.py ", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="leading/trailing space"):
            sanitize_archive_member_path(" foo.py", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="leading/trailing space or dot"):
            sanitize_archive_member_path("dir./file.txt", tmp_path)

    def test_reject_invalid_windows_characters(self, tmp_path):
        for bad_char in '<>:"|?*':
            with pytest.raises(ArchiveSecurityError, match="Invalid character"):
                sanitize_archive_member_path(f"foo{bad_char}bar.py", tmp_path)

    def test_reject_ntfs_alternate_data_streams(self, tmp_path):
        with pytest.raises(ArchiveSecurityError, match="Invalid character"):
            sanitize_archive_member_path("secret.py:hidden_stream", tmp_path)
        with pytest.raises(ArchiveSecurityError, match="Invalid character"):
            sanitize_archive_member_path("payload.py::$DATA", tmp_path)


# ============================================================================
# Windows Path Collisions & File/Directory Conflicts
# ============================================================================


class TestWindowsPathCollisionsAndConflicts:
    """Test detection and rejection of case-folding collisions and file/dir conflicts."""

    def test_reject_case_folding_collision_in_zip(self, tmp_path):
        # Case collision: 'foo.py' and 'FOO.py'
        zip_bytes = create_zip_bytes(
            {
                "foo.py": "content1",
                "FOO.py": "content2",
            }
        )
        archive = tmp_path / "case_collision.zip"
        archive.write_bytes(zip_bytes)
        dest_dir = tmp_path / "dest_case"

        with pytest.raises(ArchiveSecurityError, match="Duplicate or colliding path"):
            extract_archive_securely(archive, dest_dir)
        assert not (dest_dir / "foo.py").exists()

    def test_reject_case_folding_collision_in_tar(self, tmp_path):
        tar_bytes = create_tar_bytes(
            {
                "utils/helper.py": "helper1",
                "UTILS/helper.py": "helper2",
            }
        )
        archive = tmp_path / "case_collision.tar.gz"
        archive.write_bytes(tar_bytes)
        dest_dir = tmp_path / "dest_tar_case"

        with pytest.raises(ArchiveSecurityError, match="Duplicate or colliding path"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_duplicate_normalized_paths(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("script.py", "version 1")
            zf.writestr("script.py", "version 2")

        archive = tmp_path / "duplicate.zip"
        archive.write_bytes(buf.getvalue())
        dest_dir = tmp_path / "dest_dup"

        with pytest.raises(ArchiveSecurityError, match="Duplicate or colliding path"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_file_dir_conflict_file_then_dir(self, tmp_path):
        # File 'pkg' then directory 'pkg/'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("pkg", "I am a file")
            zinfo = zipfile.ZipInfo("pkg/")
            zf.writestr(zinfo, "")

        archive = tmp_path / "file_dir_conflict1.zip"
        archive.write_bytes(buf.getvalue())
        dest_dir = tmp_path / "dest_conflict1"

        with pytest.raises(ArchiveSecurityError, match="File/directory conflict"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_file_dir_conflict_dir_then_file(self, tmp_path):
        # Directory 'pkg/' then file 'pkg'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zinfo = zipfile.ZipInfo("pkg/")
            zf.writestr(zinfo, "")
            zf.writestr("pkg", "I am a file")

        archive = tmp_path / "file_dir_conflict2.zip"
        archive.write_bytes(buf.getvalue())
        dest_dir = tmp_path / "dest_conflict2"

        with pytest.raises(ArchiveSecurityError, match="File/directory conflict"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_file_dir_conflict_file_then_child_file(self, tmp_path):
        # File 'pkg' then file 'pkg/init.py'
        zip_bytes = create_zip_bytes(
            {
                "pkg": "file content",
                "pkg/init.py": "child file",
            }
        )
        archive = tmp_path / "file_child_conflict.zip"
        archive.write_bytes(zip_bytes)
        dest_dir = tmp_path / "dest_conflict3"

        with pytest.raises(ArchiveSecurityError, match="File/directory conflict"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_file_dir_conflict_child_file_then_file(self, tmp_path):
        # File 'pkg/init.py' then file 'pkg'
        zip_bytes = create_zip_bytes(
            {
                "pkg/init.py": "child file",
                "pkg": "file content",
            }
        )
        archive = tmp_path / "child_file_conflict.zip"
        archive.write_bytes(zip_bytes)
        dest_dir = tmp_path / "dest_conflict4"

        with pytest.raises(ArchiveSecurityError, match="File/directory conflict"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_directory_case_collision(self, tmp_path):
        # Directory 'src/' and 'SRC/'
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(zipfile.ZipInfo("src/"), "")
            zf.writestr(zipfile.ZipInfo("SRC/"), "")

        archive = tmp_path / "dir_case_collision.zip"
        archive.write_bytes(buf.getvalue())
        dest_dir = tmp_path / "dest_dir_case"

        with pytest.raises(ArchiveSecurityError, match="Case-folding collision"):
            extract_archive_securely(archive, dest_dir)


# ============================================================================
# Destination Safety & Cleanup
# ============================================================================


class TestDestinationSafetyAndCleanup:
    """Test destination directory safety, empty checks, and boundary protection."""

    def test_reject_non_empty_destination(self, tmp_path):
        dest_dir = tmp_path / "non_empty"
        dest_dir.mkdir()
        (dest_dir / "pre_existing.txt").write_text("existing", encoding="utf-8")

        archive = tmp_path / "simple.zip"
        archive.write_bytes(create_zip_bytes({"new.txt": "new"}))

        with pytest.raises(ArchiveSecurityError, match="must be empty"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_destination_is_symlink(self, tmp_path, monkeypatch):
        dest_dir = tmp_path / "symlink_dest"
        dest_dir.mkdir()
        monkeypatch.setattr("pr_workspace.os.path.islink", lambda p: True)

        archive = tmp_path / "simple.zip"
        archive.write_bytes(create_zip_bytes({"file.txt": "data"}))

        with pytest.raises(ArchiveSecurityError, match="symbolic link"):
            extract_archive_securely(archive, dest_dir)

    def test_reject_preexisting_symlink_inside_destination(self, tmp_path, monkeypatch):
        dest_dir = tmp_path / "dest_with_symlink"
        dest_dir.mkdir()

        # Mock islink so check for sneaky_symlink reports a symlink
        def mock_islink(path):
            return "sneaky_symlink" in str(path)

        monkeypatch.setattr("pr_workspace.os.path.islink", mock_islink)

        archive = tmp_path / "nested.zip"
        archive.write_bytes(
            create_zip_bytes(
                {
                    "root/README.md": "readme",
                    "root/sneaky_symlink/file.txt": "data",
                }
            )
        )

        with pytest.raises(ArchiveSecurityError, match=r"(?i)symbolic link|symlink"):
            extract_archive_securely(archive, dest_dir)

    def test_detect_common_prefix_does_not_conceal_traversal(self):
        # Traversal entry must not result in prefix stripping that conceals traversal
        assert _detect_common_prefix(["../evil.txt"]) == ""
        assert _detect_common_prefix(["C:/evil.txt"]) == ""
        assert _detect_common_prefix(["CON/evil.txt"]) == ""
        assert _detect_common_prefix(["..\\evil.txt"]) == ""
        assert _detect_common_prefix(["safe-repo-123/main.py"]) == "safe-repo-123/"

    def test_safe_delete_workspace_guards(self, tmp_path):
        # Ensure _safe_delete_workspace refuses to delete temp root or filesystem root
        temp_root = Path(tempfile.gettempdir())
        _safe_delete_workspace(temp_root)
        assert temp_root.exists()

        _safe_delete_workspace(Path("/"))
        _safe_delete_workspace("")
        _safe_delete_workspace(Path.cwd())


# ============================================================================
# Secure Extraction Tests (Zip & Tar)
# ============================================================================


class TestSecureArchiveExtraction:
    """Test extraction with security validations across zip and tar archives."""

    def test_extract_valid_zip_with_github_prefix(self, tmp_path):
        prefix = "octocat-hello-world-0123456/"
        zip_bytes = create_zip_bytes(
            {
                f"{prefix}README.md": "# Hello World",
                f"{prefix}src/main.py": "print('hello')",
            }
        )
        archive_file = tmp_path / "repo.zip"
        archive_file.write_bytes(zip_bytes)

        dest_dir = tmp_path / "extracted"
        extracted = extract_archive_securely(archive_file, dest_dir)

        # Prefix must be stripped so files are at root of dest_dir
        assert (dest_dir / "README.md").is_file()
        assert (dest_dir / "src" / "main.py").is_file()
        assert (dest_dir / "README.md").read_text(encoding="utf-8") == "# Hello World"
        assert len(extracted) == 2

    def test_extract_valid_tar_with_github_prefix(self, tmp_path):
        prefix = "octocat-hello-world-0123456/"
        tar_bytes = create_tar_bytes(
            {
                f"{prefix}app.py": "x = 1",
            }
        )
        archive_file = tmp_path / "repo.tar.gz"
        archive_file.write_bytes(tar_bytes)

        dest_dir = tmp_path / "extracted_tar"
        extracted = extract_archive_securely(archive_file, dest_dir)

        assert (dest_dir / "app.py").is_file()
        assert len(extracted) == 1

    def test_reject_zip_with_traversal_and_clean_up(self, tmp_path):
        bad_zip = create_zip_bytes(
            {
                "good.txt": "legit",
                "../evil.txt": "malicious",
            }
        )
        archive_file = tmp_path / "bad.zip"
        archive_file.write_bytes(bad_zip)

        dest_dir = tmp_path / "dest"
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            extract_archive_securely(archive_file, dest_dir)

        # Pre-validation must prevent any files from remaining extracted
        assert not (dest_dir / "good.txt").exists()
        assert not (tmp_path / "evil.txt").exists()

    def test_reject_tar_with_traversal_and_clean_up(self, tmp_path):
        bad_tar = create_tar_bytes(
            {
                "legit.py": "pass",
                "../../bad.py": "bad",
            }
        )
        archive_file = tmp_path / "bad.tar.gz"
        archive_file.write_bytes(bad_tar)

        dest_dir = tmp_path / "dest_tar"
        with pytest.raises(ArchiveSecurityError, match="Directory traversal"):
            extract_archive_securely(archive_file, dest_dir)

        assert not (dest_dir / "legit.py").exists()

    def test_reject_symlink_in_zip(self, tmp_path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zinfo = zipfile.ZipInfo("symlink_entry")
            zinfo.create_system = 3  # Unix
            zinfo.external_attr = 0o120777 << 16
            zf.writestr(zinfo, "/etc/passwd")

        archive_file = tmp_path / "symlink.zip"
        archive_file.write_bytes(buf.getvalue())

        dest_dir = tmp_path / "symlink_dest"
        with pytest.raises(ArchiveSecurityError, match="Symbolic link forbidden"):
            extract_archive_securely(archive_file, dest_dir)

    def test_reject_symlink_in_tar(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            ti = tarfile.TarInfo(name="symlink_tar")
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/shadow"
            tf.addfile(ti)

        archive_file = tmp_path / "symlink.tar.gz"
        archive_file.write_bytes(buf.getvalue())

        dest_dir = tmp_path / "symlink_tar_dest"
        with pytest.raises(ArchiveSecurityError, match="Links forbidden"):
            extract_archive_securely(archive_file, dest_dir)

    def test_reject_hardlink_in_tar(self, tmp_path):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            ti = tarfile.TarInfo(name="hardlink_tar")
            ti.type = tarfile.LNKTYPE
            ti.linkname = "some_target"
            tf.addfile(ti)

        archive_file = tmp_path / "hardlink.tar.gz"
        archive_file.write_bytes(buf.getvalue())

        dest_dir = tmp_path / "hardlink_dest"
        with pytest.raises(ArchiveSecurityError, match="Links forbidden"):
            extract_archive_securely(archive_file, dest_dir)

    def test_reject_entry_count_limit_exceeded(self, tmp_path):
        entries = {f"file_{i}.txt": f"content {i}" for i in range(15)}
        zip_bytes = create_zip_bytes(entries)
        archive_file = tmp_path / "too_many.zip"
        archive_file.write_bytes(zip_bytes)

        limits = WorkspaceLimits(max_entries=10)
        dest_dir = tmp_path / "dest_count"
        with pytest.raises(ArchiveSecurityError, match="entry count .* exceeds limit"):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

    def test_reject_expanded_size_limit_exceeded(self, tmp_path):
        # 3 files of 1000 bytes each = 3000 bytes
        entries = {f"file_{i}.txt": "a" * 1000 for i in range(3)}
        zip_bytes = create_zip_bytes(entries)
        archive_file = tmp_path / "too_large.zip"
        archive_file.write_bytes(zip_bytes)

        limits = WorkspaceLimits(max_expanded_bytes=2000)
        dest_dir = tmp_path / "dest_large"
        with pytest.raises(
            ArchiveSecurityError, match="expanded size .* exceeds limit"
        ):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

    def test_reject_single_file_size_limit_exceeded(self, tmp_path):
        entries = {"large.bin": "b" * 5000}
        zip_bytes = create_zip_bytes(entries)
        archive_file = tmp_path / "single_large.zip"
        archive_file.write_bytes(zip_bytes)

        limits = WorkspaceLimits(max_single_file_bytes=2000)
        dest_dir = tmp_path / "dest_single"
        with pytest.raises(ArchiveSecurityError, match="exceeds limit"):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

    def test_zip_streaming_per_file_limit_enforced_and_atomic_cleanup(self, tmp_path):
        """Streaming per-file limit enforced on ZIP when metadata reports small size but stream exceeds limit."""
        # Member metadata has size 100 bytes (passes Phase 1 validation)
        zip_bytes = create_zip_bytes({"sneaky.bin": b"A" * 100})
        archive_file = tmp_path / "sneaky_stream.zip"
        archive_file.write_bytes(zip_bytes)

        limits = WorkspaceLimits(max_single_file_bytes=2000, max_expanded_bytes=10000)
        dest_dir = tmp_path / "dest_zip_stream"

        # Stream decompression produces 5000 bytes (exceeding 2000 byte single-file limit)
        def mock_open(self, name, *args, **kwargs):
            return io.BytesIO(b"A" * 5000)

        with (
            patch.object(zipfile.ZipFile, "open", mock_open),
            pytest.raises(
                ArchiveSecurityError,
                match="decompressed stream exceeded single-file limit",
            ),
        ):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

        # Atomic cleanup verification: destination workspace must not retain partially written files
        assert not dest_dir.exists()

    def test_tar_streaming_per_file_limit_enforced_and_atomic_cleanup(self, tmp_path):
        """Streaming per-file limit enforced on TAR when metadata reports small size but stream exceeds limit."""
        # Member metadata has size 100 bytes (passes Phase 1 validation)
        tar_bytes = create_tar_bytes({"sneaky.bin": b"A" * 100})
        archive_file = tmp_path / "sneaky_stream.tar.gz"
        archive_file.write_bytes(tar_bytes)

        limits = WorkspaceLimits(max_single_file_bytes=2000, max_expanded_bytes=10000)
        dest_dir = tmp_path / "dest_tar_stream"

        def mock_extractfile(self, member):
            return io.BytesIO(b"A" * 5000)

        with (
            patch.object(tarfile.TarFile, "extractfile", mock_extractfile),
            pytest.raises(
                ArchiveSecurityError,
                match="decompressed stream exceeded single-file limit",
            ),
        ):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

        # Atomic cleanup verification: destination workspace must not retain partially written files
        assert not dest_dir.exists()

    def test_streaming_total_expanded_bytes_limit_enforced_independently(
        self, tmp_path
    ):
        """Total expanded size is enforced across multiple files that individually pass single-file limit."""
        # Two files of 1200 bytes each. Each is < 2000 single file limit, but sum (2400) > 2000 total limit.
        entries = {"part1.txt": "a" * 1200, "part2.txt": "b" * 1200}
        zip_bytes = create_zip_bytes(entries)
        archive_file = tmp_path / "multi_part.zip"
        archive_file.write_bytes(zip_bytes)

        limits = WorkspaceLimits(max_single_file_bytes=2000, max_expanded_bytes=2000)
        dest_dir = tmp_path / "dest_total_independent"
        with pytest.raises(
            ArchiveSecurityError,
            match="exceeds limit",
        ):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

        assert not dest_dir.exists()

    def test_reject_archive_file_size_exceeded(self, tmp_path):
        archive_file = tmp_path / "archive_limit.zip"
        archive_file.write_bytes(b"PK\x03\x04" + b"0" * 1000)

        limits = WorkspaceLimits(max_archive_bytes=500)
        dest_dir = tmp_path / "dest_arch"
        with pytest.raises(ArchiveSecurityError, match="Archive size .* exceeds limit"):
            extract_archive_securely(archive_file, dest_dir, limits=limits)

    def test_corrupted_archive_format_raises_extraction_error(self, tmp_path):
        bad_file = tmp_path / "corrupted.zip"
        bad_file.write_bytes(b"not a valid zip or tar archive content at all")

        dest_dir = tmp_path / "dest_corrupted"
        with pytest.raises(ArchiveExtractionError, match="Unsupported or corrupted"):
            extract_archive_securely(bad_file, dest_dir)


# ============================================================================
# Commit Binding & PR Workspace Tests
# ============================================================================


class TestPreparePRWorkspace:
    """Test full prepare_pr_workspace lifecycle and commit binding."""

    def test_successful_workspace_preparation_with_exact_sha(self, tmp_path):
        commit_sha = "0123456789abcdef0123456789abcdef01234567"
        zip_bytes = create_zip_bytes(
            {
                f"owner-repo-{commit_sha[:7]}/setup.py": "print('setup')",
                f"owner-repo-{commit_sha[:7]}/src/app.py": "def run(): pass",
            }
        )

        mock_client = MagicMock(spec=GitHubClient)
        mock_meta = MagicMock(spec=PRMetadata)
        mock_meta.head_sha = commit_sha
        mock_client.get_pr_metadata.return_value = mock_meta

        archive_path = tmp_path / "downloaded.zip"
        archive_path.write_bytes(zip_bytes)
        mock_client.download_commit_archive.return_value = archive_path

        ws = prepare_pr_workspace(
            owner="owner",
            repo="repo",
            pr_number=12,
            expected_head_sha=commit_sha,
            client=mock_client,
            base_dir=tmp_path,
        )

        assert isinstance(ws, PRWorkspace)
        assert ws.owner == "owner"
        assert ws.repo == "repo"
        assert ws.pr_number == 12
        assert ws.head_sha == commit_sha
        assert ws.workspace_path.is_dir()
        assert (ws.workspace_path / "setup.py").is_file()
        assert (ws.workspace_path / "src" / "app.py").is_file()

        mock_client.get_pr_metadata.assert_called_once_with("owner", "repo", 12)
        mock_client.download_commit_archive.assert_called_once_with(
            owner="owner",
            repo="repo",
            commit_sha=commit_sha,
            max_bytes=50 * 1024 * 1024,
        )

        ws.cleanup()
        assert not ws.workspace_path.exists()

    def test_head_sha_mismatch_fails_closed(self, tmp_path):
        expected_sha = "1111111111111111111111111111111111111111"
        newer_sha = "2222222222222222222222222222222222222222"

        mock_client = MagicMock(spec=GitHubClient)
        mock_meta = MagicMock(spec=PRMetadata)
        mock_meta.head_sha = newer_sha
        mock_client.get_pr_metadata.return_value = mock_meta

        with pytest.raises(CommitBindingError, match="PR head SHA mismatch"):
            prepare_pr_workspace(
                owner="owner",
                repo="repo",
                pr_number=5,
                expected_head_sha=expected_sha,
                client=mock_client,
                base_dir=tmp_path,
            )

        mock_client.download_commit_archive.assert_not_called()

    def test_missing_or_short_head_sha_fails_closed(self, tmp_path):
        mock_client = MagicMock(spec=GitHubClient)
        mock_meta = MagicMock(spec=PRMetadata)
        mock_meta.head_sha = ""
        mock_client.get_pr_metadata.return_value = mock_meta

        with pytest.raises(
            CommitBindingError, match="Could not establish valid head SHA"
        ):
            prepare_pr_workspace(
                owner="owner",
                repo="repo",
                pr_number=5,
                client=mock_client,
                base_dir=tmp_path,
            )

    def test_archive_download_failure_cleans_up_workspace(self, tmp_path):
        commit_sha = "abcdef1234567890abcdef1234567890abcdef12"
        mock_client = MagicMock(spec=GitHubClient)
        mock_meta = MagicMock(spec=PRMetadata)
        mock_meta.head_sha = commit_sha
        mock_client.get_pr_metadata.return_value = mock_meta
        mock_client.download_commit_archive.side_effect = GitHubNetworkError(
            "Download failed"
        )

        with pytest.raises(GitHubNetworkError):
            prepare_pr_workspace(
                owner="owner",
                repo="repo",
                pr_number=1,
                client=mock_client,
                base_dir=tmp_path,
            )

        subdirs = [p for p in tmp_path.iterdir() if p.is_dir()]
        assert len(subdirs) == 0

    def test_workspace_context_manager_cleans_up(self, tmp_path):
        commit_sha = "1234567890abcdef"
        zip_bytes = create_zip_bytes({"test.py": "x = 1"})
        archive = tmp_path / "test.zip"
        archive.write_bytes(zip_bytes)

        ws_path = None
        with create_pr_workspace_from_archive(
            owner="org",
            repo="repo",
            pr_number=1,
            head_sha=commit_sha,
            archive_path=archive,
            base_dir=tmp_path,
        ) as ws:
            ws_path = ws.workspace_path
            assert ws_path.is_dir()
            assert (ws_path / "test.py").is_file()

        assert ws_path is not None
        assert not ws_path.exists()

    def test_workspace_context_manager_cleans_up_on_exception(self, tmp_path):
        commit_sha = "1234567890abcdef"
        zip_bytes = create_zip_bytes({"test.py": "x = 1"})
        archive = tmp_path / "test.zip"
        archive.write_bytes(zip_bytes)

        ws_path = None
        try:
            with create_pr_workspace_from_archive(
                owner="org",
                repo="repo",
                pr_number=1,
                head_sha=commit_sha,
                archive_path=archive,
                base_dir=tmp_path,
            ) as ws:
                ws_path = ws.workspace_path
                assert ws_path.is_dir()
                raise RuntimeError("Simulated failure during analysis")
        except RuntimeError:
            pass

        assert ws_path is not None
        assert not ws_path.exists()
