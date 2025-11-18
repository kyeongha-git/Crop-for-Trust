#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_load_config.py & test_logging.py
-------------------------------------
Comprehensive unit tests for:
    • `utils.load_config.load_yaml_config`
    • `utils.logging` (setup_logging, get_logger)

Goals:
    - Validate YAML loading and error handling
    - Ensure logging setup and file generation work correctly
    - Verify logger instances and handler behavior
"""

import logging

import pytest
import yaml

from utils.load_config import load_yaml_config
from utils.logging import get_logger, setup_logging

# ==============================================================
# Tests for load_yaml_config
# ==============================================================
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


def test_stdout_message_contains_loaded_path(tmp_path, capsys):
    """
    Verify that console output includes the 'Loaded configuration' message and resolved path.
    """
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("root: test", encoding="utf-8")

    _ = load_yaml_config(yaml_path)
    out, _ = capsys.readouterr()

    assert "Loaded configuration from:" in out
    assert str(yaml_path.resolve()) in out


# ==============================================================
# Tests for utils.logging
# ==============================================================
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
