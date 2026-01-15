#!/usr/bin/env python3
"""Test profile extraction from facts."""

import logging
logging.basicConfig(level=logging.DEBUG)

from ask_llm.utils.config import Config
from ask_llm.memory_server.extraction import extract_profile_attributes_from_fact, _extract_attribute_key

config = Config()

# Test key extraction first
test_facts = [
    "User has 2 dogs named Nora and Cabbie",
    "User is 45 years old",
    "User works as a software engineer",
    "User likes Python programming",
    "User's favorite color is blue",
    "User lives in Seattle",
]

print("=== Testing key extraction ===")
for content in test_facts:
    key = _extract_attribute_key(content)
    print(f"  '{content[:50]}...' -> key={key}")

print("\n=== Testing full extraction ===")
# Test with a mock fact
test_fact = {
    "content": "User has 2 dogs named Nora and Cabbie",
    "tags": ["fact", "relationship"],
    "importance": 0.8,
}

result = extract_profile_attributes_from_fact(test_fact, user_id="default", config=config)
print(f"Result: {result}")

# Check what's in the database now
print("\n=== Database contents ===")
from ask_llm.profiles import ProfileManager, EntityType
manager = ProfileManager(config)
attrs = manager.get_all_attributes(EntityType.USER, "default")
for a in attrs:
    print(f"  {a.category}.{a.key} = {a.value[:60] if len(str(a.value)) > 60 else a.value}")
