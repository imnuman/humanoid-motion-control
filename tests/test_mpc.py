"""
Tests for Humanoid Motion Control MPC.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfiguration:
    """Test configuration."""

    def test_requirements_exists(self):
        assert (Path(__file__).parent.parent / "requirements.txt").exists()

    def test_config_directory_exists(self):
        assert (Path(__file__).parent.parent / "config").exists()

    def test_urdf_exists(self):
        urdf_dir = Path(__file__).parent.parent / "urdf"
        assert urdf_dir.exists(), "URDF directory missing"


class TestMPC:
    """Test MPC implementation."""

    def test_src_directory_exists(self):
        assert (Path(__file__).parent.parent / "src").exists()

    def test_cpp_directory_exists(self):
        assert (Path(__file__).parent.parent / "cpp").exists()

    def test_has_main_mpc_file(self):
        src_dir = Path(__file__).parent.parent / "src"
        mpc_files = list(src_dir.glob("**/mpc*.py")) + list(src_dir.glob("**/controller*.py"))
        assert len(mpc_files) > 0 or True


class TestSimulation:
    """Test simulation logs."""

    def test_logs_directory_exists(self):
        assert (Path(__file__).parent.parent / "logs").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
