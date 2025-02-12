import pytest
from pathlib import Path
import json
from typing import Dict, Any
from dataclasses import dataclass

from tiny_eval.task.base import Task, BaseTaskConfig, TaskResult

# Test fixtures
@dataclass
class MockConfig(BaseTaskConfig):
    """Simple config for testing"""
    name: str
    value: int
    
    def get_id(self) -> str:
        return f"mock_{self.name}_{self.value}"

class MockTask(Task[MockConfig, Dict[str, Any]]):
    """Simple task that returns config values and tracks runs"""
    
    def __init__(self, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        self.run_count = 0  # Track number of actual runs
    
    async def run_single(self, config: MockConfig) -> TaskResult[Dict[str, Any]]:
        self.run_count += 1
        return TaskResult(
            status="success",
            data={
                "name": config.name,
                "value": config.value,
                "processed": True
            }
        )

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide temporary cache directory"""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir

@pytest.fixture
def mock_configs():
    """Sample configs for testing"""
    return [
        MockConfig(name="test1", value=1),
        MockConfig(name="test2", value=2),
    ]

@pytest.mark.asyncio
async def test_task_without_cache():
    """Test task execution without caching"""
    task = MockTask(cache_dir=None)
    configs = [MockConfig(name="test", value=1)]
    
    results = await task.run(configs)
    
    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].data["name"] == "test"
    assert results[0].data["value"] == 1
    assert task.run_count == 1  # Verify task was actually run

@pytest.mark.asyncio
async def test_task_with_cache(temp_cache_dir, mock_configs):
    """Test task execution with caching"""
    task = MockTask(cache_dir=temp_cache_dir)
    
    # First run - should execute tasks and cache results
    results1 = await task.run(mock_configs)
    assert task.run_count == 2  # Both tasks should have run
    
    # Verify cache files were created
    for config, result in zip(mock_configs, results1):
        cache_path = temp_cache_dir / f"{config.get_id()}.json"
        assert cache_path.exists()
        with open(cache_path) as f:
            cached_data = json.load(f)
            assert cached_data["status"] == "success"
            assert cached_data["data"]["name"] == config.name
            assert cached_data["data"]["value"] == config.value
    
    # Second run - should use cached results
    task.run_count = 0  # Reset counter
    results2 = await task.run(mock_configs)
    assert task.run_count == 0  # No tasks should have run
    
    # Compare results
    for r1, r2 in zip(results1, results2):
        assert r1.status == r2.status
        assert r1.data == r2.data

@pytest.mark.asyncio
async def test_partial_cache(temp_cache_dir):
    """Test behavior with mix of cached and uncached configs"""
    task = MockTask(cache_dir=temp_cache_dir)
    
    # Run first config
    config1 = MockConfig(name="test1", value=1)
    await task.run([config1])
    assert task.run_count == 1
    
    # Run both configs
    task.run_count = 0  # Reset counter
    config2 = MockConfig(name="test2", value=2)
    results = await task.run([config1, config2])
    
    assert task.run_count == 1  # Only config2 should have run
    assert len(results) == 2
    assert results[0].data["name"] == "test1"
    assert results[1].data["name"] == "test2"

@pytest.mark.asyncio
async def test_cache_directory_creation():
    """Test that cache directory is created if it doesn't exist"""
    cache_dir = Path("temp_test_cache")
    try:
        task = MockTask(cache_dir=cache_dir)
        assert cache_dir.exists()
        assert cache_dir.is_dir()
    finally:
        # Cleanup
        if cache_dir.exists():
            cache_dir.rmdir()

@pytest.mark.asyncio
async def test_invalid_cache_handling(temp_cache_dir):
    """Test handling of corrupted cache files"""
    task = MockTask(cache_dir=temp_cache_dir)
    config = MockConfig(name="test", value=1)
    
    # Create invalid cache file
    cache_path = temp_cache_dir / f"{config.get_id()}.json"
    cache_path.write_text("invalid json")
    
    # Should run task despite cache file existing
    results = await task.run([config])
    assert task.run_count == 1
    assert results[0].status == "success"
    assert results[0].data["name"] == "test"
    assert results[0].data["value"] == 1

@pytest.mark.asyncio
async def test_error_result():
    """Test handling of error results"""
    class ErrorTask(Task[MockConfig, Dict[str, Any]]):
        async def run_single(self, config: MockConfig) -> TaskResult[Dict[str, Any]]:
            return TaskResult(
                status="error",
                error="Test error message"
            )
    
    task = ErrorTask()
    config = MockConfig(name="test", value=1)
    results = await task.run([config])
    
    assert len(results) == 1
    assert results[0].status == "error"
    assert results[0].error == "Test error message"
    assert results[0].data is None 