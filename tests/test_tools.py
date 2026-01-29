import os
import pytest
from pathlib import Path
from gpal.server import list_directory, read_file, search_project

def test_list_directory(tmp_path):
    # Create a dummy structure
    (tmp_path / "subdir").mkdir()
    (tmp_path / "file1.txt").write_text("hello")
    (tmp_path / "file2.py").write_text("print('world')")
    
    # Test listing
    results = list_directory(str(tmp_path))
    assert "subdir" in results
    assert "file1.txt" in results
    assert "file2.py" in results
    assert len(results) == 3

def test_list_directory_nonexistent():
    results = list_directory("/non/existent/path/at/all")
    assert len(results) == 1
    assert "Error" in results[0]

def test_read_file(tmp_path):
    test_file = tmp_path / "test.txt"
    content = "Sample content for testing."
    test_file.write_text(content)
    
    result = read_file(str(test_file))
    assert result == content

def test_read_file_error():
    result = read_file("this_file_does_not_exist_at_all.txt")
    assert "Error reading file" in result

def test_search_project(tmp_path, monkeypatch):
    # Setup dummy files
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("def my_function(): pass")
    (tmp_path / "app" / "utils.py").write_text("def other_function(): pass")
    (tmp_path / "README.md").write_text("This is the main project readme.")
    
    # Use monkeypatch to change CWD to tmp_path for the search
    monkeypatch.chdir(tmp_path)
    
    # Test finding a term
    result = search_project("my_function")
    assert "app/main.py" in result
    assert "app/utils.py" not in result
    
    # Test globbing
    result = search_project("def", glob_pattern="app/*.py")
    assert "app/main.py" in result
    assert "app/utils.py" in result
    assert "README.md" not in result

def test_search_project_no_matches(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = search_project("non_existent_term_xyz_123")
    assert result == "No matches found."
