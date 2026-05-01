# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
from pathlib import Path
import tempfile

from google.adk.environment._local_environment import LocalEnvironment
from google.adk.tools.environment._tools import ReadFileTool
import pytest


@pytest.fixture
async def env_with_file():
  """Creates a temporary environment with a sample file."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    # Create a legitimate file
    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    yield env, Path(td)

    await env.close()


@pytest.mark.asyncio
async def test_read_file_tool_prevents_shell_injection():
  """Original test — single quote injection via start_line path."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    marker = Path(td) / "marker.txt"
    injected_path = f"sample.txt'; touch {marker}; echo '"

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": injected_path, "start_line": 2},
        tool_context=None,
    )

    print(result)

    assert not marker.exists(), (
        "Shell injection succeeded! marker.txt was created, "
        "meaning the path was interpreted as shell syntax."
    )

    await env.close()


@pytest.mark.asyncio
async def test_shell_injection_via_semicolon():
  """Tests that semicolon injection is blocked."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    marker = Path(td) / "marker_semicolon.txt"
    # Semicolon injection — tries to run second command
    injected_path = f"sample.txt; touch {marker}"

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": injected_path, "start_line": 2},
        tool_context=None,
    )

    assert (
        not marker.exists()
    ), "Semicolon injection succeeded — marker.txt was created."

    await env.close()


@pytest.mark.asyncio
async def test_shell_injection_via_ampersand():
  """Tests that ampersand injection is blocked."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    marker = Path(td) / "marker_ampersand.txt"
    # Ampersand injection — tries to run command in background
    injected_path = f"sample.txt && touch {marker}"

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": injected_path, "start_line": 2},
        tool_context=None,
    )

    assert (
        not marker.exists()
    ), "Ampersand injection succeeded — marker.txt was created."

    await env.close()


@pytest.mark.asyncio
async def test_shell_injection_via_backtick():
  """Tests that backtick command substitution is blocked."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    marker = Path(td) / "marker_backtick.txt"
    # Backtick injection — tries command substitution
    injected_path = f"sample.txt`touch {marker}`"

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": injected_path, "start_line": 2},
        tool_context=None,
    )

    assert (
        not marker.exists()
    ), "Backtick injection succeeded — marker.txt was created."

    await env.close()


@pytest.mark.asyncio
async def test_shell_injection_with_end_line():
  """Tests injection is blocked when end_line triggers the shell path."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    marker = Path(td) / "marker_end_line.txt"
    injected_path = f"sample.txt'; touch {marker}; echo '"

    tool = ReadFileTool(env)
    # end_line also triggers the shell path
    result = await tool.run_async(
        args={"path": injected_path, "end_line": 2},
        tool_context=None,
    )

    assert (
        not marker.exists()
    ), "Shell injection via end_line succeeded — marker.txt was created."

    await env.close()


@pytest.mark.asyncio
async def test_read_file_full_content():
  """Tests reading a full file without line range returns all lines."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\n", encoding="utf-8")

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": str(target)},
        tool_context=None,
    )

    assert result["status"] == "ok"
    assert "line1" in result["content"]
    assert "line2" in result["content"]
    assert "line3" in result["content"]

    await env.close()


@pytest.mark.asyncio
async def test_read_file_with_valid_start_line():
  """Tests that reading from a valid start_line works correctly."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": str(target), "start_line": 3},
        tool_context=None,
    )

    assert result["status"] == "ok"
    assert "line3" in result["content"]
    assert "line4" in result["content"]
    assert "line5" in result["content"]
    # line1 and line2 should not be in the result
    assert "line1" not in result["content"]
    assert "line2" not in result["content"]

    await env.close()


@pytest.mark.asyncio
async def test_read_file_with_valid_start_and_end_line():
  """Tests that reading a specific line range works correctly."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    target = Path(td) / "sample.txt"
    target.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={"path": str(target), "start_line": 2, "end_line": 4},
        tool_context=None,
    )

    assert result["status"] == "ok"
    assert "line2" in result["content"]
    assert "line3" in result["content"]
    assert "line4" in result["content"]
    assert "line1" not in result["content"]
    assert "line5" not in result["content"]

    await env.close()


@pytest.mark.asyncio
async def test_read_file_missing_path_returns_error():
  """Tests that missing path returns an error."""
  with tempfile.TemporaryDirectory() as td:
    env = LocalEnvironment(working_dir=Path(td))
    await env.initialize()

    tool = ReadFileTool(env)
    result = await tool.run_async(
        args={},
        tool_context=None,
    )

    assert result["status"] == "error"
    assert "path" in result["error"].lower()

    await env.close()
