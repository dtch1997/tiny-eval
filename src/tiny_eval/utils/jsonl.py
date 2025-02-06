import json
from pathlib import Path
from typing import Any, Iterator

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of dictionaries.
    
    Args:
        path: Path to the JSONL file
        
    Returns:
        List of dictionaries parsed from the JSONL file
    """
    path = Path(path)
    data = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data.append(json.loads(line))
    return data

def read_jsonl_iter(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read a JSONL file and yield dictionaries one at a time.
    
    Args:
        path: Path to the JSONL file
        
    Yields:
        Dictionaries parsed from the JSONL file
    """
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                yield json.loads(line)

def write_jsonl(data: list[dict[str, Any]], path: str | Path) -> None:
    """Write a list of dictionaries to a JSONL file.
    
    Args:
        data: List of dictionaries to write
        path: Path to write the JSONL file to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def append_jsonl(item: dict[str, Any], path: str | Path) -> None:
    """Append a single dictionary to a JSONL file.
    
    Args:
        item: Dictionary to append
        path: Path to the JSONL file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n') 