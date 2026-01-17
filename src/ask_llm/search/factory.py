"""Factory for creating search clients.

Handles provider selection, configuration, and fallback logic.
"""

import logging
from typing import TYPE_CHECKING

from .base import SearchClient, SearchProvider

if TYPE_CHECKING:
    from ..utils.config import Config

logger = logging.getLogger(__name__)


def is_search_available(provider: SearchProvider | str | None = None) -> bool:
    """Check if search is available for the given provider.
    
    Args:
        provider: Specific provider to check, or None to check any
        
    Returns:
        True if search is available
    """
    if provider is None:
        # Check if any provider is available
        from .ddgs_client import is_ddgs_available
        from .tavily_client import is_tavily_available
        return is_ddgs_available() or is_tavily_available()
    
    if isinstance(provider, str):
        provider = SearchProvider(provider.lower())
    
    if provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import is_ddgs_available
        return is_ddgs_available()
    elif provider == SearchProvider.TAVILY:
        from .tavily_client import is_tavily_available
        return is_tavily_available()
    
    return False


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
    4. DuckDuckGo (free fallback)
    
    Args:
        config: Application config
        provider: Override provider selection
        max_results: Override default max results
        
    Returns:
        Configured SearchClient or None if search unavailable
    """
    # Determine provider
    if provider is not None:
        if isinstance(provider, str):
            try:
                provider = SearchProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown search provider: {provider}")
                provider = None
    
    if provider is None:
        # Check config for preferred provider
        config_provider = getattr(config, "SEARCH_PROVIDER", None)
        if config_provider:
            try:
                provider = SearchProvider(config_provider.lower())
            except ValueError:
                logger.warning(f"Unknown search provider in config: {config_provider}")
    
    if provider is None:
        # Auto-select based on availability
        tavily_key = getattr(config, "TAVILY_API_KEY", None)
        if tavily_key and is_search_available(SearchProvider.TAVILY):
            provider = SearchProvider.TAVILY
            logger.debug("Auto-selected Tavily (API key configured)")
        elif is_search_available(SearchProvider.DUCKDUCKGO):
            provider = SearchProvider.DUCKDUCKGO
            logger.debug("Auto-selected DuckDuckGo (free fallback)")
        else:
            logger.warning("No search provider available")
            return None
    
    # Get max results from config if not specified
    if max_results is None:
        max_results = getattr(config, "SEARCH_MAX_RESULTS", 5)
    
    # Ensure max_results is an int at this point
    max_results = int(max_results)
    
    # Create client
    if provider == SearchProvider.TAVILY:
        from .tavily_client import TavilyClient, is_tavily_available
        
        if not is_tavily_available():
            logger.error("Tavily requested but tavily-python not installed")
            return None
        
        api_key = getattr(config, "TAVILY_API_KEY", None)
        if not api_key:
            logger.error("Tavily requested but TAVILY_API_KEY not configured")
            return None
        
        include_answer = getattr(config, "SEARCH_INCLUDE_ANSWER", False)
        search_depth = getattr(config, "SEARCH_DEPTH", "basic")
        
        return TavilyClient(
            api_key=api_key,
            max_results=max_results,
            include_answer=include_answer,
            search_depth=search_depth,
        )
    
    elif provider == SearchProvider.DUCKDUCKGO:
        from .ddgs_client import DuckDuckGoClient, is_ddgs_available
        
        if not is_ddgs_available():
            logger.error("DuckDuckGo requested but ddgs not installed")
            return None
        
        # Get timeout from config (default 10 seconds)
        timeout = getattr(config, "SEARCH_TIMEOUT", 10)
        
        return DuckDuckGoClient(max_results=max_results, timeout=timeout)
    
    logger.error(f"Unknown search provider: {provider}")
    return None
