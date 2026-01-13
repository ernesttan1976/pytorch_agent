"""Dataset schema definitions and validators."""
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class Message:
    """Single message in a conversation."""
    role: str
    content: str
    
    def __post_init__(self):
        """Validate message after initialization."""
        if self.role not in ["system", "user", "assistant"]:
            raise ValueError(f"Invalid role: {self.role}. Must be 'system', 'user', or 'assistant'")
        if not self.content or not self.content.strip():
            raise ValueError("Message content cannot be empty")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}
    
    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "Message":
        """Create from dictionary."""
        return cls(role=data["role"], content=data["content"])


@dataclass
class Example:
    """Single training example."""
    id: str
    messages: List[Message]
    meta: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate example after initialization."""
        if not self.messages:
            raise ValueError("Example must have at least one message")
        
        # Last message must be from assistant
        if self.messages[-1].role != "assistant":
            raise ValueError("Last message must be from assistant")
        
        # Validate message sequence
        for i, msg in enumerate(self.messages):
            if i == 0 and msg.role == "assistant":
                raise ValueError("First message cannot be from assistant")
            if i > 0 and msg.role == "system":
                raise ValueError("System message can only be first message")
        
        if self.meta is None:
            self.meta = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSONL serialization."""
        result = {
            "id": self.id,
            "messages": [msg.to_dict() for msg in self.messages],
            "meta": self.meta or {}
        }
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        """Create from dictionary."""
        messages = [Message.from_dict(msg) for msg in data["messages"]]
        return cls(
            id=data["id"],
            messages=messages,
            meta=data.get("meta")
        )
    
    def to_jsonl(self) -> str:
        """Serialize to JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)
    
    @classmethod
    def from_jsonl(cls, line: str) -> "Example":
        """Parse from JSONL line."""
        data = json.loads(line)
        return cls.from_dict(data)


def validate_example(example: Example) -> List[str]:
    """Validate an example and return list of errors (empty if valid).
    
    Args:
        example: Example to validate
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    try:
        example.__post_init__()
    except ValueError as e:
        errors.append(str(e))
    
    return errors


def load_jsonl(path: str) -> List[Example]:
    """Load examples from JSONL file.
    
    Args:
        path: Path to JSONL file
    
    Returns:
        List of examples
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                example = Example.from_jsonl(line)
                examples.append(example)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {path}: {e}")
    
    return examples


def save_jsonl(examples: List[Example], path: str) -> None:
    """Save examples to JSONL file.
    
    Args:
        examples: List of examples to save
        path: Output file path
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(example.to_jsonl() + "\n")

