import time
try:
    import tiktoken
    _tiktoken_present = True
except ImportError:
    tiktoken = None
    _tiktoken_present = False

class Message:
    """Standard message format for all LLM clients"""

    def __init__(self, role: str, content: str, timestamp: float | None = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp if timestamp else time.time()

    @classmethod
    def from_dict(cls, data: dict) -> 'Message':
        """Create a Message from a dictionary"""
        role = data.get("role", "user")
        content = cls._extract_content(data)
        timestamp = data.get("timestamp", time.time())
        return cls(role, content, timestamp)

    @staticmethod
    def _extract_content(data: dict) -> str:
        """Extract content from the data dictionary"""
        content = data.get("content", "")
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
        return str(content) # Ensure content is always a string

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    def to_api_format(self) -> dict:
        """Convert to API-compatible format (without timestamp)"""
        return {"role": self.role, "content": self.content}

    def get_token_count(self, encoding, base_overhead: int) -> int:
        """
        Calculates the token count for this message using a given tiktoken encoding.
        Assumes tiktoken is present and encoding is a valid tiktoken encoding object.
        """
        if not _tiktoken_present or not encoding:
            # Fallback or raise error if tiktoken/encoding is not available
            # For now, let's assume a simple character count as a rough estimate
            # or indicate that token counting is not possible.
            # This behavior might need to be more sophisticated depending on requirements.
            # print("Warning: tiktoken not available for accurate token counting.")
            return len(self.content) // 4 # Very rough estimate

        num_tokens = base_overhead
        if self.content: # Ensure content is not None
            num_tokens += len(encoding.encode(self.content))
        return num_tokens