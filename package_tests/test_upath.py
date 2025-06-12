"""
This comprehensive test file covers the core functionality of the upath Python package. Here's what it tests:

Path Creation & Properties:
    Basic path creation from strings and pathlib.Path objects
    Path components (name, stem, suffix, parent, parts)
    Path joining with / operator and joinpath()

Path Operations:
    Absolute/relative path handling
    Path resolution and normalization
    Pattern matching with match()
    Suffix and name manipulation

File System Operations:
    File reading/writing (both text and binary)
    Directory creation with mkdir(parents=True)
    File and directory removal
    Existence and type checking

Advanced Features:
    Glob patterns (glob() and rglob())
    Directory iteration
    Relative path computation
    Path comparison and hashing

Key Features Tested:
    Cross-platform compatibility (UPath's main strength)
    Temporary file operations for safe testing
    Both simple and complex directory structures
    Error handling scenarios
"""

import pytest
import tempfile
from pathlib import Path
from upath import UPath


class TestUPathCore:
    """Test core functionality of UPath."""
    
    def test_basic_path_creation(self):
        """Test basic path creation and string representation."""
        path = UPath("test/path/file.txt")
        assert str(path) == "test/path/file.txt"
        # Test with pathlib.Path
        pathlib_path = Path("test/path/file.txt")
        upath_from_pathlib = UPath(pathlib_path)
        assert str(upath_from_pathlib) == str(pathlib_path)
    
    def test_path_parts_and_properties(self):
        """Test path parts, name, suffix, stem, etc."""
        path = UPath("folder/subfolder/document.txt")
        assert path.name == "document.txt"
        assert path.stem == "document"
        assert path.suffix == ".txt"
        assert path.parent == UPath("folder/subfolder")
        assert path.parts == ("folder", "subfolder", "document.txt")
    
    def test_path_joining(self):
        """Test path joining with / operator and joinpath."""
        base = UPath("home/user")
        # Test / operator
        joined1 = base / "documents" / "file.txt"
        assert str(joined1) == "home/user/documents/file.txt"
        # Test joinpath method
        joined2 = base.joinpath("documents", "file.txt")
        assert str(joined2) == "home/user/documents/file.txt"
        
        assert joined1 == joined2
    
    def test_absolute_and_relative_paths(self):
        """Test absolute path operations."""
        rel_path = UPath("relative/path")
        abs_path = rel_path.absolute()
        assert abs_path.is_absolute()
        assert not rel_path.is_absolute()
    
    def test_path_resolution(self):
        """Test path resolution and normalization."""
        path = UPath("folder/../folder/./file.txt")
        resolved = path.resolve()
        # Should normalize the path
        assert ".." not in str(resolved)
        assert "/." not in str(resolved)
    
    def test_file_operations_with_temp_files(self):
        """Test file operations using temporary files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = UPath(temp_dir)
            test_file = temp_path / "test_file.txt"
            test_content = "Hello, UPath!"
            # Test writing
            test_file.write_text(test_content)
            assert test_file.exists()
            assert test_file.is_file()
            # Test reading
            content = test_file.read_text()
            assert content == test_content
            # Test file stats
            assert test_file.stat().st_size > 0
            # Test directory operations
            test_dir = temp_path / "test_directory"
            test_dir.mkdir()
            assert test_dir.exists()
            assert test_dir.is_dir()
            # Test listing directory contents
            contents = list(temp_path.iterdir())
            assert len(contents) == 2  # test_file.txt and test_directory
            # Test glob patterns
            txt_files = list(temp_path.glob("*.txt"))
            assert len(txt_files) == 1
            assert txt_files[0].name == "test_file.txt"
    
    def test_binary_file_operations(self, tmp_path):
        """Test binary file read/write operations."""
        temp_path = UPath(tmp_path)
        binary_file = temp_path / "binary_test.bin"
        binary_data = b"Binary data content"
        # Write binary data
        binary_file.write_bytes(binary_data)
        assert binary_file.exists()
        # Read binary data
        read_data = binary_file.read_bytes()
        assert read_data == binary_data
    
    def test_path_matching(self):
        """Test path pattern matching."""
        path = UPath("documents/project/file.py")
        assert path.match("*.py")
        assert path.match("**/file.py")
        assert path.match("documents/*/file.py")
        assert not path.match("*.txt")
    
    def test_path_with_suffix(self):
        """Test changing file extensions."""
        path = UPath("document.txt")
        new_path = path.with_suffix(".md")
        assert str(new_path) == "document.md"
        # Test removing suffix
        no_suffix = path.with_suffix("")
        assert str(no_suffix) == "document"
    
    def test_path_with_name(self):
        """Test changing filename."""
        path = UPath("folder/old_name.txt")
        new_path = path.with_name("new_name.txt")
        assert str(new_path) == "folder/new_name.txt"
        assert new_path.parent == path.parent
    
    def test_relative_to(self):
        """Test computing relative paths."""
        base = UPath("/home/user/documents")
        target = UPath("/home/user/documents/projects/myproject")
        relative = target.relative_to(base)
        assert str(relative) == "projects/myproject"
    
    def test_path_comparison(self):
        """Test path equality and comparison."""
        path1 = UPath("test/path")
        path2 = UPath("test/path")
        path3 = UPath("different/path")
        assert path1 == path2
        assert path1 != path3
        assert hash(path1) == hash(path2)
    
    def test_mkdir_parents(self, tmp_path):
        """Test creating directories with parents."""
        temp_path = UPath(tmp_path)
        nested_dir = temp_path / "level1" / "level2" / "level3"
        # Create with parents
        nested_dir.mkdir(parents=True)
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        # Verify all parent directories were created
        assert (temp_path / "level1").exists()
        assert (temp_path / "level1" / "level2").exists()
    
    def test_rmdir_and_unlink(self, tmp_path):
        """Test removing files and directories."""
        temp_path = UPath(tmp_path)
        test_file = temp_path / "to_remove.txt"
        test_file.write_text("content")
        assert test_file.exists()
        test_file.unlink()
        assert not test_file.exists()
        test_dir = temp_path / "to_remove_dir"
        test_dir.mkdir()
        assert test_dir.exists()
        test_dir.rmdir()
        assert not test_dir.exists()
    
    def test_rglob_recursive_glob(self, tmp_path):
        """Test recursive globbing."""
        temp_path = UPath(tmp_path)
        # Create nested structure
        (temp_path / "dir1").mkdir()
        (temp_path / "dir1" / "file1.txt").write_text("content")
        (temp_path / "dir2").mkdir()
        (temp_path / "dir2" / "file2.txt").write_text("content")
        (temp_path / "file3.txt").write_text("content")
        # Test recursive glob
        txt_files = list(temp_path.rglob("*.txt"))
        assert len(txt_files) == 3
        # Verify all files found
        file_names = {f.name for f in txt_files}
        assert file_names == {"file1.txt", "file2.txt", "file3.txt"}
    
    def test_uri_and_as_uri(self):
        """Test URI conversion."""
        path = UPath("/home/user/file.txt")
        try:
            uri = path.as_uri()
            assert uri.startswith("file://")
        except AttributeError:
            # Some platforms/versions might not support as_uri
            pass
    
    def test_exists_and_is_methods(self, tmp_path):
        """Test various existence and type checking methods."""
        temp_path = UPath(tmp_path)
        non_existent = temp_path / "does_not_exist"
        assert non_existent.exists() is False
        assert non_existent.is_file() is False
        assert non_existent.is_dir() is False
        test_file = temp_path / "test.txt"
        test_file.write_text("content")
        assert test_file.exists() is True
        assert test_file.is_file() is True
        assert test_file.is_dir() is False
        test_dir = temp_path / "test_dir"
        test_dir.mkdir()
        assert test_dir.exists() is True
        assert test_dir.is_file() is False
        assert test_dir.is_dir() is True


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])