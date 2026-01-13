"""Message formatting for training with chat templates."""
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer


def format_messages(
    messages: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = False
) -> str:
    """Format messages using model's chat template.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: Tokenizer instance (must have apply_chat_template)
        add_generation_prompt: Whether to add generation prompt at end
    
    Returns:
        Formatted string ready for tokenization
    """
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
            return formatted
        except Exception as e:
            # Fallback to manual formatting if template fails
            print(f"Warning: Could not apply chat template: {e}. Using fallback.")
            return _format_messages_fallback(messages)
    else:
        return _format_messages_fallback(messages)


def _format_messages_fallback(messages: List[Dict[str, str]]) -> str:
    """Fallback formatting when chat template is not available.
    
    Uses simple format:
    <|system|>...<|user|>...<|assistant|>...
    """
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        
        if role == "system":
            parts.append(f"<|system|>\n{content}<|end|>\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}<|end|>\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}<|end|>\n")
    
    return "".join(parts)


def prepare_training_example(
    example: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    packing: bool = True
) -> Dict[str, Any]:
    """Prepare a single example for training.
    
    Args:
        example: Example dict with 'messages' field
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        packing: Whether to pack multiple examples (not implemented yet)
    
    Returns:
        Dictionary with 'input_ids' and 'labels'
    """
    messages = example["messages"]
    
    # Format messages using chat template
    text = format_messages(messages, tokenizer, add_generation_prompt=False)
    
    # Tokenize
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None
    )
    
    input_ids = tokenized["input_ids"]
    
    # For causal LM, labels are the same as input_ids
    # We'll mask out non-assistant tokens during training
    labels = input_ids.copy()
    
    return {
        "input_ids": input_ids,
        "labels": labels
    }


def get_model_max_length(tokenizer: AutoTokenizer) -> int:
    """Get model's maximum sequence length.
    
    Args:
        tokenizer: Tokenizer instance
    
    Returns:
        Maximum sequence length
    """
    if hasattr(tokenizer, "model_max_length"):
        return tokenizer.model_max_length
    elif hasattr(tokenizer, "max_len"):
        return tokenizer.max_len
    else:
        return 2048  # Safe default

