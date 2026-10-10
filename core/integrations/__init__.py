"""Optional external services, all env-gated, all with honest offline fallbacks."""

from .config import ALL, NotConfigured, get, statuses, summary
from .firecrawl import FirecrawlClient, MockFirecrawl
from .firecrawl import capabilities as firecrawl_capabilities
from .firecrawl import get_client as get_firecrawl_client
from .ratelimit import get_limiter, reset_limiter
from .ratelimit import status as ratelimit_status
from .supabase import EmptySource as SupabaseEmpty
from .supabase import SupabaseSource
from .supabase import get_source as get_supabase_source
from .supabase import reset_source as reset_supabase_source
from .tavily import MockTavily, TavilyClient
from .tavily import capabilities as tavily_capabilities
from .tavily import get_client as get_tavily_client

__all__ = [
    "ALL", "NotConfigured", "get", "statuses", "summary",
    "MockTavily", "TavilyClient", "tavily_capabilities", "get_tavily_client",
    "MockFirecrawl", "FirecrawlClient", "firecrawl_capabilities", "get_firecrawl_client",
    "get_limiter", "reset_limiter", "ratelimit_status",
    "SupabaseSource", "SupabaseEmpty", "get_supabase_source", "reset_supabase_source",
]
