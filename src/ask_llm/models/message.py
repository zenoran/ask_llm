import time

class Message:
    """Standard message format for all LLM clients"""

    def __init__(self, role, content, timestamp=None):
        self.role = role
        self.content = content
        self.timestamp = timestamp if timestamp else time.time()

    @classmethod
    def from_dict(cls, data):
        """Create a Message from a dictionary"""
        role = data.get("role", "user")
        content = cls._extract_content(data)
        timestamp = data.get("timestamp", time.time())
        return cls(role, content, timestamp)

    @staticmethod
    def _extract_content(data):
        """Extract content from the data dictionary"""
        content = data.get("content", "")
        if isinstance(content, list):
            return " ".join(item.get("text", "") for item in content if item.get("type") == "text")
        return content

    def to_dict(self):
        """Convert to dictionary format"""
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    def to_api_format(self):
        """Convert to API-compatible format (without timestamp)"""
        return {"role": self.role, "content": self.content}