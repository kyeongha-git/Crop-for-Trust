#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive unit tests for:
    • `utils.load_config`
    • `utils.logging`
    • `utils.model_hub`
"""

import logging
import pytest
import yaml
import hashlib
from unittest.mock import MagicMock, patch, call

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging
from utils import model_hub

# Tests for load_yaml_config
def test_load_valid_yaml(tmp_path):
    """
    Verify that a valid YAML file is successfully loaded.

    Ensures:
        - Keys and nested structures are parsed correctly.
        - Values match expected content.
    """
    config_content = """
    data_augmentor:
      data:
        input_dir: "data/original"
        output_dir: "data/output"
      split:
        train_ratio: 0.8
        valid_ratio: 0.1
        test_ratio: 0.1
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(config_content, encoding="utf-8")

    cfg = load_yaml_config(yaml_path)
    assert "data_augmentor" in cfg
    assert cfg["data_augmentor"]["split"]["train_ratio"] == 0.8


def test_file_not_found(tmp_path):
    """
    Verify that FileNotFoundError is raised when the YAML file does not exist.
    """
    nonexistent_path = tmp_path / "no_such_file.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml_config(nonexistent_path)


def test_invalid_yaml_syntax(tmp_path):
    """
    Verify that YAML syntax errors raise `yaml.YAMLError`.

    This ensures that malformed YAML content is properly detected.
    """
    invalid_yaml_content = """
    data_augmentor:
      data:
        input_dir: "data/original"
        output_dir: "data/output"
      split:
        train_ratio: 0.8
        valid_ratio: 0.1
        test_ratio: 0.1
        test_ratio: 0.1   # duplicated key
      invalid_block: [ unclosed_bracket
    """
    yaml_path = tmp_path / "invalid.yaml"
    yaml_path.write_text(invalid_yaml_content, encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        load_yaml_config(yaml_path)


def test_invalid_yaml_structure(tmp_path):
    """
    Verify that ValueError is raised when the top-level YAML structure is not a dictionary.

    Example:
        A YAML file starting with a list (e.g. `- item1`) should raise an error.
    """
    yaml_path = tmp_path / "invalid_type.yaml"
    yaml_path.write_text(
        "- item1\n- item2", encoding="utf-8"
    )  # List type instead of dict

    with pytest.raises(ValueError):
        load_yaml_config(yaml_path)


def test_path_is_resolved(tmp_path):
    """
    Ensure the path is resolved to an absolute path and the file exists.

    Confirms:
        - YAML is parsed correctly.
        - The file path resolves properly via Path.resolve().
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("root: test", encoding="utf-8")

    cfg = load_yaml_config(yaml_path)

    assert cfg["root"] == "test"
    assert yaml_path.resolve().exists()


# Tests for utils.logging
def test_setup_logging_creates_log_dir_and_file(tmp_path):
    """
    Verify that setup_logging() creates a log directory and a log file.

    Checks:
        - Directory existence
        - One `.log` file is generated
    """
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)

    assert log_dir.exists(), "Log directory was not created"

    log_files = list(log_dir.glob("run_*.log"))
    assert len(log_files) == 1, "Log file was not created"
    assert log_files[0].suffix == ".log"


def test_setup_logging_registers_handlers(tmp_path):
    """
    Verify that both StreamHandler and FileHandler are registered to the root logger.
    """
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)

    root_logger = logging.getLogger()
    handler_types = [type(h).__name__ for h in root_logger.handlers]

    assert "StreamHandler" in handler_types, "StreamHandler not found"
    assert "FileHandler" in handler_types, "FileHandler not found"


def test_logging_writes_to_file(tmp_path):
    """
    Verify that log messages are written to the log file.

    Checks:
        - Log message presence
        - INFO level inclusion
    """
    log_dir = tmp_path / "logs"
    setup_logging(log_dir)
    logger = get_logger("test_logger")

    logger.info("Hello, logging test!")

    log_file = next(log_dir.glob("run_*.log"))
    content = log_file.read_text(encoding="utf-8")

    assert "Hello, logging test!" in content, "Message not found in log file"
    assert "INFO" in content, "INFO level missing from log file"


def test_get_logger_returns_same_instance():
    """
    Verify that calling get_logger() with the same name returns the same Logger instance.
    """
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_a")

    assert logger_a is logger_b, "get_logger did not return the same instance"


def test_get_logger_returns_different_instances_for_different_names():
    """
    Verify that get_logger() returns different Logger instances for different names.
    """
    logger_a = get_logger("module_a")
    logger_b = get_logger("module_b")

    assert logger_a is not logger_b, "Different names returned the same logger"
    assert isinstance(logger_a, logging.Logger)
    assert isinstance(logger_b, logging.Logger)


# Tests for utils.model_hub
@pytest.fixture
def mock_logger():
    return MagicMock()

def test_sha256_of(tmp_path):
    """Verify SHA256 calculation correctness."""
    p = tmp_path / "test_file.txt"
    content = b"hello world"
    p.write_bytes(content)
    
    expected_hash = hashlib.sha256(content).hexdigest()
    assert model_hub.sha256_of(p) == expected_hash

def test_verify_sha256(tmp_path):
    """Verify hash comparison logic."""
    p = tmp_path / "test.bin"
    p.write_bytes(b"data")
    correct_hash = hashlib.sha256(b"data").hexdigest()
    
    # Case 1: Correct hash
    assert model_hub.verify_sha256(p, correct_hash) is True
    # Case 2: Incorrect hash
    assert model_hub.verify_sha256(p, "wronghash") is False
    # Case 3: Prefix handling
    assert model_hub.verify_sha256(p, f"sha256:{correct_hash}") is True

@patch("subprocess.run")
def test_download_file_curl_call(mock_run, tmp_path):
    """Verify curl command construction."""
    save_path = tmp_path / "model.pt"
    url = "http://example.com/file"
    
    model_hub.download_file(url, save_path)
    
    # Check if subprocess.run was called correctly
    expected_cmd = ["curl", "-L", "-f", "-o", str(save_path), url]
    mock_run.assert_called_once_with(expected_cmd, check=True)

@patch("subprocess.run")
def test_download_file_dropbox_fix(mock_run, tmp_path):
    """Verify Dropbox URL query parameter fix."""
    save_path = tmp_path / "db.pt"
    
    # Case 1: dl=0 -> dl=1
    model_hub.download_file("http://dropbox.com/s/xyz?dl=0", save_path)
    args, _ = mock_run.call_args
    assert "http://dropbox.com/s/xyz?dl=1" in args[0]
    
    # Case 2: No param -> append ?dl=1
    model_hub.download_file("http://dropbox.com/s/xyz", save_path)
    args, _ = mock_run.call_args
    assert "http://dropbox.com/s/xyz?dl=1" in args[0]

def test_download_weights_skips_existing(mock_logger, tmp_path):
    """Should skip download if file exists."""
    p = tmp_path / "exist.pt"
    p.touch()
    
    cfg = {"weights": {"yolo": "url"}}
    
    with patch("utils.model_hub.download_file") as mock_dl:
        model_hub.download_fine_tuned_weights(
            cfg=cfg, model_name="yolo", saved_model_path=p, logger=mock_logger
        )
        mock_dl.assert_not_called()
        mock_logger.info.assert_any_call(f"[WEIGHT] Using existing weight: {p}")

def test_download_weights_sha_mismatch_cleanup(mock_logger, tmp_path):
    """
    Should raise RuntimeError and delete file if SHA mismatch occurs.
    """
    save_path = tmp_path / "corrupt.pt"
    
    # Config with a hash that won't match our dummy file
    cfg = {
        "weights": {"yolo": "http://url"},
        "sha256": {"yolo": "deadbeef"}
    }
    
    # Mock download_file to 'create' a file (simulate download)
    def side_effect_dl(url, path):
        path.write_bytes(b"corrupted_content")
        
    with patch("utils.model_hub.download_file", side_effect=side_effect_dl):
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            model_hub.download_fine_tuned_weights(
                cfg=cfg, model_name="yolo", saved_model_path=save_path, logger=mock_logger
            )
            
    # Verify file was deleted (unlink called)
    assert not save_path.exists(), "Corrupted file should be deleted after mismatch"