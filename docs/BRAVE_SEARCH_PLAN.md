# Brave Search Integration - Implementation Plan

## Overview

Add Brave Search as a search provider option. Brave Search offers a privacy-focused search API with web, news, and video search capabilities, plus AI-generated summaries similar to Tavily's answer feature.

**API Documentation**: https://api.search.brave.com/app/documentation

## Why Brave Search?

- **Privacy-focused**: No user tracking, independent index
- **AI Summaries**: Optional summarized answers (like Tavily)
- **Rich Results**: Supports web, news, videos, images
- **Free Tier**: 2,000 queries/month free (good for development)
- **Relevance Scores**: Results include ranking scores

---

## Task 1: Add SearchProvider Enum Value

**File**: `src/ask_llm/search/base.py`

Add Brave to the `SearchProvider` enum:

```python
class SearchProvider(str, Enum):
    """Available search providers."""
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"
    BRAVE = "brave"  # ADD THIS
```

---

## Task 2: Add Configuration Settings

**File**: `src/ask_llm/utils/config.py`

Add Brave API key field to the `Config` class (near other API keys):

```python
# Brave Search settings
BRAVE_API_KEY: str = Field(default="", description="Brave Search API key")
```

The existing `SEARCH_PROVIDER`, `SEARCH_MAX_RESULTS`, and `SEARCH_TIMEOUT` settings will be reused.

---

## Task 3: Create Brave Search Client

**File**: `src/ask_llm/search/brave_client.py` (NEW FILE)

```python
"""Brave Search client implementation."""

import logging
from typing import TYPE_CHECKING

import httpx

from .base import SearchClient, SearchProvider, SearchResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Brave Search API base URL
BRAVE_API_BASE = "https://api.search.brave.com/res/v1"


class BraveSearchClient(SearchClient):
    """
    Brave Search API client.

    Brave Search provides:
    - Web search with relevance scores
    - News search with time filtering
    - Optional AI-generated summaries
    - Privacy-focused, independent index

    API docs: https://api.search.brave.com/app/documentation
    """

    PROVIDER = SearchProvider.BRAVE
    REQUIRES_API_KEY = True

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
        timeout: int = 10,
        include_summary: bool = False,
        safesearch: str = "moderate",
    ):
        """
        Initialize Brave Search client.

        Args:
            api_key: Brave Search API key
            max_results: Default number of results to return
            timeout: Request timeout in seconds
            include_summary: Include AI-generated summary (requires Pro plan)
            safesearch: Filter level - "off", "moderate", or "strict"
        """
        super().__init__(max_results=max_results)
        self._api_key = api_key
        self._timeout = timeout
        self._include_summary = include_summary
        self._safesearch = safesearch
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=BRAVE_API_BASE,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": self._api_key,
                },
                timeout=self._timeout,
            )
        return self._client

    def search(
        self,
        query: str,
        max_results: int | None = None,
        region: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform a Brave web search.

        Args:
            query: Search query
            max_results: Override default max results
            region: Country code (e.g., "US", "GB", "DE")

        Returns:
            List of SearchResult objects
        """
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
            }

            if region:
                params["country"] = region

            # Request summary if enabled (Pro plan feature)
            if self._include_summary:
                params["summary"] = "1"

            response = client.get("/web/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            # Parse web results
            web_results = data.get("web", {}).get("results", [])
            for raw in web_results:
                # Brave provides age and extra_snippets for context
                snippet = raw.get("description", "")

                # Add extra snippets if available for richer context
                extra = raw.get("extra_snippets", [])
                if extra:
                    snippet = f"{snippet}\n{' '.join(extra[:2])}"

                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=snippet,
                    score=self._normalize_score(raw),
                    source=self.PROVIDER,
                    raw=raw,
                ))

            return results[:max_results]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                logger.error("Brave API key is invalid")
            elif e.response.status_code == 429:
                logger.warning("Brave rate limit exceeded")
            else:
                logger.error(f"Brave search HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Brave search failed: {e}")
            return []

    def search_news(
        self,
        query: str,
        max_results: int | None = None,
        time_range: str | None = None,
    ) -> list[SearchResult]:
        """
        Search Brave news.

        Args:
            query: Search query
            max_results: Override default max results
            time_range: Time filter - "d" (day), "w" (week), "m" (month), "y" (year)

        Returns:
            List of SearchResult objects
        """
        if not self._api_key:
            logger.error("Brave API key not configured")
            return []

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
            }

            # Map time_range to Brave's freshness parameter
            if time_range:
                freshness_map = {
                    "d": "pd",   # Past day
                    "w": "pw",   # Past week
                    "m": "pm",   # Past month
                    "y": "py",   # Past year
                }
                if time_range in freshness_map:
                    params["freshness"] = freshness_map[time_range]

            response = client.get("/news/search", params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            news_results = data.get("results", [])
            for raw in news_results:
                # News results have age field for recency
                snippet = raw.get("description", "")
                age = raw.get("age", "")
                if age:
                    snippet = f"[{age}] {snippet}"

                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=snippet,
                    score=None,  # News results don't have scores
                    source=self.PROVIDER,
                    raw=raw,
                ))

            return results[:max_results]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Brave rate limit exceeded")
            else:
                logger.error(f"Brave news search HTTP error: {e}")
            return []
        except Exception as e:
            logger.error(f"Brave news search failed: {e}")
            return []

    def search_with_summary(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], str | None]:
        """
        Search with AI-generated summary (requires Pro plan).

        Similar to Tavily's search_with_answer feature.

        Args:
            query: Search query
            max_results: Override default max results

        Returns:
            Tuple of (results, summary_text or None)
        """
        if not self._api_key:
            return [], None

        max_results = max_results or self.max_results

        try:
            client = self._get_client()

            params = {
                "q": query,
                "count": max_results,
                "safesearch": self._safesearch,
                "summary": "1",  # Request summary
            }

            response = client.get("/web/search", params=params)
            response.raise_for_status()
            data = response.json()

            # Parse results
            results = []
            web_results = data.get("web", {}).get("results", [])
            for raw in web_results:
                results.append(SearchResult(
                    title=raw.get("title", ""),
                    url=raw.get("url", ""),
                    snippet=raw.get("description", ""),
                    score=self._normalize_score(raw),
                    source=self.PROVIDER,
                    raw=raw,
                ))

            # Extract summary if present
            summary = None
            summarizer = data.get("summarizer", {})
            if summarizer:
                summary = summarizer.get("summary", None)

            return results[:max_results], summary

        except Exception as e:
            logger.error(f"Brave search with summary failed: {e}")
            return [], None

    def _normalize_score(self, raw: dict) -> float | None:
        """
        Normalize Brave's ranking to 0.0-1.0 score.

        Brave doesn't provide explicit relevance scores, but we can
        derive one from the result position or other signals.
        """
        # Brave's API doesn't include explicit scores
        # Could potentially use page_age, language confidence, etc.
        # For now, return None to indicate no score available
        return None

    def is_available(self) -> bool:
        """Check if Brave Search is properly configured."""
        return bool(self._api_key)

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if self._client:
            self._client.close()


def is_brave_available() -> bool:
    """Check if httpx is installed (always True since it's a core dependency)."""
    try:
        import httpx  # noqa: F401
        return True
    except ImportError:
        return False
```

---

## Task 4: Update Factory

**File**: `src/ask_llm/search/factory.py`

### 4.1 Update imports and availability check

```python
# Add to imports section
from .brave_client import BraveSearchClient, is_brave_available
```

### 4.2 Update `get_search_client` function

Add Brave to the auto-selection logic (after Tavily, before DuckDuckGo):

```python
def get_search_client(
    config: "Config",
    provider: SearchProvider | str | None = None,
    max_results: int | None = None,
) -> SearchClient | None:
    """Create a search client based on configuration.

    Provider selection priority:
    1. Explicit provider argument
    2. Config SEARCH_PROVIDER setting
    3. Tavily (if API key configured)
    4. Brave (if API key configured)  # ADD THIS
    5. DuckDuckGo (free fallback)
    """
    # ... existing normalization code ...

    # Auto-select based on availability
    if provider is None:
        tavily_key = getattr(config, "TAVILY_API_KEY", None)
        brave_key = getattr(config, "BRAVE_API_KEY", None)  # ADD

        if tavily_key and is_search_available(SearchProvider.TAVILY):
            provider = SearchProvider.TAVILY
            logger.debug("Auto-selected Tavily (API key configured)")
        elif brave_key and is_search_available(SearchProvider.BRAVE):  # ADD
            provider = SearchProvider.BRAVE
            logger.debug("Auto-selected Brave (API key configured)")
        elif is_search_available(SearchProvider.DUCKDUCKGO):
            provider = SearchProvider.DUCKDUCKGO
            logger.debug("Auto-selected DuckDuckGo (free fallback)")
        else:
            logger.warning("No search provider available")
            return None

    # ... existing max_results code ...

    # ADD: Brave client instantiation
    elif provider == SearchProvider.BRAVE:
        if not is_brave_available():
            logger.warning("httpx not installed for Brave, falling back")
            # Fall back to DuckDuckGo...

        api_key = getattr(config, "BRAVE_API_KEY", None)
        if not api_key:
            logger.warning("Brave API key not configured, falling back to DuckDuckGo")
            # Fall back to DuckDuckGo...

        return BraveSearchClient(
            api_key=api_key,
            max_results=max_results,
            timeout=getattr(config, "SEARCH_TIMEOUT", 10),
            include_summary=getattr(config, "SEARCH_INCLUDE_ANSWER", False),
            safesearch=getattr(config, "BRAVE_SAFESEARCH", "moderate"),
        )
```

### 4.3 Update `is_search_available` function

```python
def is_search_available(provider: SearchProvider | str | None = None) -> bool:
    """Check if search is available for given provider or any provider."""
    if provider is None:
        # Check any provider
        return (
            is_ddgs_available()
            or is_tavily_available()
            or is_brave_available()  # ADD
        )

    if isinstance(provider, str):
        try:
            provider = SearchProvider(provider.lower())
        except ValueError:
            return False

    if provider == SearchProvider.DUCKDUCKGO:
        return is_ddgs_available()
    elif provider == SearchProvider.TAVILY:
        return is_tavily_available()
    elif provider == SearchProvider.BRAVE:  # ADD
        return is_brave_available()

    return False
```

### 4.4 Update `get_search_unavailable_reason` function

```python
def get_search_unavailable_reason(
    config: "Config",
    provider: SearchProvider | str | None = None,
) -> str:
    """Return human-readable reason why search isn't available."""
    # ... existing code ...

    # ADD: Brave case
    if provider == SearchProvider.BRAVE:
        api_key = getattr(config, "BRAVE_API_KEY", None)
        if not api_key:
            return (
                "Brave Search requires BRAVE_API_KEY to be set in your config. "
                "Get a free API key at: https://api.search.brave.com/"
            )
        if not is_brave_available():
            return (
                "Brave Search requires httpx. This should be installed by default. "
                "Try: pip install httpx"
            )
        return "Brave Search is available but encountered an unknown issue."
```

---

## Task 5: Update Package Exports (Optional)

**File**: `src/ask_llm/search/__init__.py`

The client classes aren't currently exported directly (factory handles instantiation), but for completeness:

```python
# Optional: export BraveSearchClient if users want direct access
from .brave_client import BraveSearchClient, is_brave_available

__all__ = [
    # ... existing exports ...
    "BraveSearchClient",  # Optional
]
```

---

## Task 6: Add Optional Configuration Fields

**File**: `src/ask_llm/utils/config.py`

Additional Brave-specific settings (optional):

```python
# Brave Search additional settings
BRAVE_SAFESEARCH: str = Field(
    default="moderate",
    description="Brave safesearch level: off, moderate, strict"
)
```

Note: `SEARCH_INCLUDE_ANSWER` is already used by Tavily and can be reused for Brave summaries.

---

## Dependencies

**No new dependencies required.** Brave Search uses `httpx` which is already a dependency (used by the service client). The implementation uses the REST API directly rather than an SDK.

If we wanted to add an optional Brave SDK in the future:

```toml
# pyproject.toml - under [project.optional-dependencies]
search = [
    "ddgs>=6.0.0",
    "tavily-python>=0.3.0",
    # "brave-search>=0.1.0",  # If SDK exists in future
]
```

---

## Testing

### Manual Testing

```bash
# 1. Set API key
export ASK_LLM_BRAVE_API_KEY="your-api-key-here"

# 2. Test with explicit provider
llm --search-provider brave "current weather in Seattle"

# 3. Test news search
llm --search-provider brave "latest AI news"

# 4. Test auto-selection (remove Tavily key to test fallback)
unset ASK_LLM_TAVILY_API_KEY
llm "search for python tutorials"

# 5. Check status
llm --status  # Should show Brave as available
```

### Unit Tests

**File**: `tests/test_brave_search.py` (NEW FILE)

```python
"""Tests for Brave Search client."""

import pytest
from unittest.mock import Mock, patch

from ask_llm.search.brave_client import BraveSearchClient, is_brave_available
from ask_llm.search.base import SearchProvider


class TestBraveSearchClient:
    """Tests for BraveSearchClient."""

    def test_provider_constant(self):
        """Verify provider is set correctly."""
        assert BraveSearchClient.PROVIDER == SearchProvider.BRAVE
        assert BraveSearchClient.REQUIRES_API_KEY is True

    def test_init_with_api_key(self):
        """Test client initialization."""
        client = BraveSearchClient(api_key="test-key", max_results=10)
        assert client._api_key == "test-key"
        assert client.max_results == 10
        assert client.is_available() is True

    def test_init_without_api_key(self):
        """Test client without API key."""
        client = BraveSearchClient(api_key="", max_results=5)
        assert client.is_available() is False

    def test_search_no_api_key(self):
        """Search should return empty list without API key."""
        client = BraveSearchClient(api_key="")
        results = client.search("test query")
        assert results == []

    @patch("ask_llm.search.brave_client.httpx.Client")
    def test_search_success(self, mock_client_class):
        """Test successful search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "Test description",
                    }
                ]
            }
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search("test query")

        assert len(results) == 1
        assert results[0].title == "Test Result"
        assert results[0].url == "https://example.com"
        assert results[0].source == SearchProvider.BRAVE

    @patch("ask_llm.search.brave_client.httpx.Client")
    def test_search_news(self, mock_client_class):
        """Test news search."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "News Article",
                    "url": "https://news.example.com",
                    "description": "Breaking news",
                    "age": "2 hours ago",
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        mock_client = Mock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        client = BraveSearchClient(api_key="test-key")
        results = client.search_news("test query", time_range="d")

        assert len(results) == 1
        assert "2 hours ago" in results[0].snippet


class TestBraveAvailability:
    """Tests for availability checks."""

    def test_is_brave_available(self):
        """httpx should always be available."""
        assert is_brave_available() is True
```

### Integration Test

```python
# tests/integration/test_brave_integration.py

import os
import pytest

from ask_llm.search import get_search_client, SearchProvider
from ask_llm.utils.config import Config


@pytest.mark.skipif(
    not os.getenv("ASK_LLM_BRAVE_API_KEY"),
    reason="BRAVE_API_KEY not set"
)
class TestBraveIntegration:
    """Integration tests requiring real API key."""

    def test_real_search(self):
        """Test real search with Brave API."""
        config = Config()
        client = get_search_client(config, provider=SearchProvider.BRAVE)

        assert client is not None
        results = client.search("python programming", max_results=3)

        assert len(results) > 0
        assert all(r.url.startswith("http") for r in results)
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/ask_llm/search/base.py` | MODIFY | Add `BRAVE` to `SearchProvider` enum |
| `src/ask_llm/utils/config.py` | MODIFY | Add `BRAVE_API_KEY` and optional settings |
| `src/ask_llm/search/brave_client.py` | CREATE | `BraveSearchClient` implementation |
| `src/ask_llm/search/factory.py` | MODIFY | Add Brave to factory and availability checks |
| `src/ask_llm/search/__init__.py` | MODIFY | Optional: export `BraveSearchClient` |
| `tests/test_brave_search.py` | CREATE | Unit tests |

---

## Implementation Order

1. **Task 1** - Add enum value (no dependencies)
2. **Task 2** - Add config settings (no dependencies)
3. **Task 3** - Create Brave client (depends on Task 1)
4. **Task 4** - Update factory (depends on Tasks 1, 2, 3)
5. **Task 5** - Update exports (depends on Task 3)
6. **Task 6** - Optional config fields (no dependencies)
7. **Testing** - After all tasks complete

---

## API Key Setup

1. Go to https://api.search.brave.com/
2. Create account and get free API key (2,000 queries/month)
3. Add to config:
   ```bash
   # ~/.config/ask-llm/.env
   ASK_LLM_BRAVE_API_KEY=your-key-here
   ```

---

## Notes for Implementer

- **No SDK needed**: Uses REST API directly with `httpx` (already a dependency)
- **Fallback behavior**: Follow existing pattern - fall back to DuckDuckGo if Brave unavailable
- **Summary feature**: Requires Brave Pro plan; gracefully handle if not available
- **Rate limits**: Free tier has 2,000 queries/month; log warnings on 429 errors
- **Error handling**: Follow patterns in `tavily_client.py` for consistency
- **Safesearch**: Default to "moderate" to match typical user expectations
