"""Data loaders for various input formats."""
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from .dataset_schema import Example, Message


class BaseLoader:
    """Base class for data loaders."""
    
    def load(self, path: str) -> Iterator[Example]:
        """Load examples from file.
        
        Args:
            path: Path to input file
        
        Yields:
            Example objects
        """
        raise NotImplementedError
    
    def detect(self, path: str) -> bool:
        """Check if this loader can handle the given file.
        
        Args:
            path: Path to input file
        
        Returns:
            True if this loader can handle the file
        """
        raise NotImplementedError


class JSONLLoader(BaseLoader):
    """Loader for JSONL files with messages format."""
    
    def load(self, path: str) -> Iterator[Example]:
        """Load examples from JSONL."""
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "messages" in data:
                        # Already in target format
                        example = Example.from_dict(data)
                        yield example
                    else:
                        # Try to convert
                        yield self._convert(data, line_num)
                except Exception as e:
                    print(f"Warning: Skipping line {line_num} in {path}: {e}")
    
    def _convert(self, data: Dict[str, Any], line_num: int) -> Example:
        """Convert single data dict to Example."""
        # Try common field names
        if "conversation" in data:
            messages = []
            for turn in data["conversation"]:
                role = turn.get("from", turn.get("role", "user"))
                if role == "human":
                    role = "user"
                elif role == "gpt" or role == "assistant":
                    role = "assistant"
                messages.append(Message(role=role, content=turn.get("value", "")))
            return Example(
                id=data.get("id", f"line_{line_num}"),
                messages=messages,
                meta={"source": "jsonl", "original": data}
            )
        else:
            raise ValueError(f"Cannot convert line {line_num}: missing 'messages' or 'conversation'")
    
    def detect(self, path: str) -> bool:
        """Check if file is JSONL."""
        return path.endswith(".jsonl") or path.endswith(".json")


class AlpacaLoader(BaseLoader):
    """Loader for Alpaca-format JSON files."""
    
    def load(self, path: str) -> Iterator[Example]:
        """Load Alpaca format examples."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        if isinstance(data, list):
            items = data
        else:
            items = [data]
        
        for idx, item in enumerate(items):
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            # Combine instruction and input
            if input_text:
                user_content = f"{instruction}\n\n{input_text}"
            else:
                user_content = instruction
            
            messages = [
                Message(role="user", content=user_content),
                Message(role="assistant", content=output)
            ]
            
            # Add system message if present
            if "system" in item and item["system"]:
                messages.insert(0, Message(role="system", content=item["system"]))
            
            yield Example(
                id=item.get("id", f"alpaca_{idx}"),
                messages=messages,
                meta={"source": "alpaca", "original": item}
            )
    
    def detect(self, path: str) -> bool:
        """Check if file looks like Alpaca format."""
        if not (path.endswith(".json") or path.endswith(".jsonl")):
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    first = data[0]
                    return "instruction" in first and "output" in first
                elif isinstance(data, dict):
                    return "instruction" in data and "output" in data
        except Exception:
            pass
        
        return False


class ShareGPTLoader(BaseLoader):
    """Loader for ShareGPT format."""
    
    def load(self, path: str) -> Iterator[Example]:
        """Load ShareGPT format examples."""
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    conversations = data.get("conversations", [])
                    
                    messages = []
                    for turn in conversations:
                        role = turn.get("from", "").lower()
                        if role == "human" or role == "user":
                            role = "user"
                        elif role == "gpt" or role == "assistant" or role == "chatgpt":
                            role = "assistant"
                        else:
                            continue  # Skip unknown roles
                        
                        content = turn.get("value", "")
                        if content:
                            messages.append(Message(role=role, content=content))
                    
                    if messages and messages[-1].role == "assistant":
                        yield Example(
                            id=data.get("id", f"sharegpt_{line_num}"),
                            messages=messages,
                            meta={"source": "sharegpt", "original": data}
                        )
                except Exception as e:
                    print(f"Warning: Skipping line {line_num} in {path}: {e}")
    
    def detect(self, path: str) -> bool:
        """Check if file looks like ShareGPT format."""
        if not path.endswith(".jsonl"):
            return False
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line:
                    data = json.loads(first_line.strip())
                    return "conversations" in data
        except Exception:
            pass
        
        return False


class CSVLoader(BaseLoader):
    """Loader for CSV files with instruction/response columns."""
    
    def __init__(self, instruction_col: str = "instruction", response_col: str = "response"):
        """Initialize CSV loader.
        
        Args:
            instruction_col: Name of instruction column
            response_col: Name of response column
        """
        self.instruction_col = instruction_col
        self.response_col = response_col
    
    def load(self, path: str) -> Iterator[Example]:
        """Load CSV format examples."""
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                instruction = row.get(self.instruction_col, "")
                response = row.get(self.response_col, "")
                
                if not instruction or not response:
                    continue
                
                messages = [
                    Message(role="user", content=instruction),
                    Message(role="assistant", content=response)
                ]
                
                # Add system message if column exists
                if "system" in row and row["system"]:
                    messages.insert(0, Message(role="system", content=row["system"]))
                
                yield Example(
                    id=row.get("id", f"csv_{idx}"),
                    messages=messages,
                    meta={"source": "csv", "row": row}
                )
    
    def detect(self, path: str) -> bool:
        """Check if file is CSV."""
        return path.endswith(".csv")


def detect_loader(path: str) -> Optional[BaseLoader]:
    """Auto-detect appropriate loader for file.
    
    Args:
        path: Path to input file
    
    Returns:
        Loader instance or None if no match
    """
    loaders = [
        ShareGPTLoader(),
        AlpacaLoader(),
        CSVLoader(),
        JSONLLoader(),  # Last as it's most permissive
    ]
    
    for loader in loaders:
        if loader.detect(path):
            return loader
    
    return None


def load_examples(path: str, loader: Optional[BaseLoader] = None) -> List[Example]:
    """Load examples from file using auto-detection or specified loader.
    
    Args:
        path: Path to input file or directory
        loader: Optional specific loader to use
    
    Returns:
        List of examples
    """
    path_obj = Path(path)
    
    if path_obj.is_file():
        files = [path]
    elif path_obj.is_dir():
        files = list(path_obj.glob("*.json")) + list(path_obj.glob("*.jsonl")) + list(path_obj.glob("*.csv"))
    else:
        raise ValueError(f"Path does not exist: {path}")
    
    all_examples = []
    for file_path in files:
        if loader is None:
            detected_loader = detect_loader(str(file_path))
            if detected_loader is None:
                print(f"Warning: Could not detect format for {file_path}, skipping")
                continue
        else:
            detected_loader = loader
        
        examples = list(detected_loader.load(str(file_path)))
        all_examples.extend(examples)
        print(f"Loaded {len(examples)} examples from {file_path}")
    
    return all_examples

