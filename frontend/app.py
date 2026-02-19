"""Growth Lab Deep Search — Streamlit frontend."""

from __future__ import annotations

import os

import streamlit as st

from frontend.api_client import (
    AgentSearchResponse,
    APIError,
    ChunkSearchResponse,
    SearchClient,
)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

EXAMPLE_QUERIES = [
    "How does economic complexity predict growth?",
    "What are the main binding constraints in growth diagnostics?",
    "Product space methodology and data sources",
    "Role of industrial policy in economic diversification",
    "Atlas of Economic Complexity methodology",
]


@st.cache_resource
def get_client() -> SearchClient:
    return SearchClient(base_url=BACKEND_URL)


@st.cache_data(ttl=60)
def cached_health(_client_id: int) -> object:
    """Fetch health status, cached for 60s. _client_id is used for cache key."""
    return get_client().health()


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "last_query": "",
        "last_results": None,
        "last_error": None,
        "trigger_search": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Growth Lab Deep Search")

search_mode = st.sidebar.radio(
    "Search mode",
    options=["Agent Search", "Raw Chunks"],
    captions=[
        "AI-synthesized answer with citations",
        "Direct vector similarity results",
    ],
)

st.sidebar.markdown("### Filters")
year = st.sidebar.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=None,
    step=1,
    placeholder="All years",
)
top_k = st.sidebar.slider("Number of results", min_value=1, max_value=30, value=10)

# Health indicator
st.sidebar.divider()
client = get_client()
health = cached_health(id(client))
if isinstance(health, APIError):
    st.sidebar.error(f"Backend: {health.message}")
else:
    if health.healthy:
        st.sidebar.success(
            f"Backend connected ({health.points_count:,} chunks indexed)"
        )
    else:
        st.sidebar.warning("Backend unhealthy — Qdrant may be disconnected")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("Growth Lab Deep Search")
query = st.text_input(
    "Search Growth Lab research documents",
    placeholder="Enter your research question...",
    label_visibility="collapsed",
)

# Search button row
search_clicked = st.button(
    "Search", type="primary", disabled=not query and not st.session_state.trigger_search
)

# Example queries (hidden once results exist)
if st.session_state.last_results is None and st.session_state.last_error is None:
    st.markdown("**Try searching for:**")
    cols = st.columns(2)
    for i, eq in enumerate(EXAMPLE_QUERIES):
        col = cols[i % 2]
        if col.button(eq, key=f"example_{i}", use_container_width=True):
            st.session_state.last_query = eq
            st.session_state.trigger_search = True
            st.rerun()


# ---------------------------------------------------------------------------
# Execute search
# ---------------------------------------------------------------------------
def _run_search(q: str) -> None:
    """Execute search and store results in session state."""
    st.session_state.last_query = q
    st.session_state.last_error = None
    st.session_state.last_results = None

    if search_mode == "Agent Search":
        with st.spinner("Searching and synthesizing answer..."):
            result = client.agent_search(q, top_k=top_k, year=year)
    else:
        with st.spinner("Searching..."):
            result = client.chunk_search(q, top_k=top_k, year=year)

    if isinstance(result, APIError):
        st.session_state.last_error = result
    else:
        st.session_state.last_results = result


# Determine if we should search
if st.session_state.trigger_search and st.session_state.last_query:
    st.session_state.trigger_search = False
    _run_search(st.session_state.last_query)
elif search_clicked and query:
    _run_search(query)


# ---------------------------------------------------------------------------
# Render errors
# ---------------------------------------------------------------------------
if st.session_state.last_error is not None:
    err = st.session_state.last_error
    if "connect" in err.message.lower():
        st.error(
            f"{err.message}\n\nStart the backend with: "
            "`uv run uvicorn backend.service.main:app`"
        )
    elif "timed out" in err.message.lower():
        st.warning(err.message)
    else:
        st.error(err.message)


# ---------------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------------
def _render_agent_results(result: AgentSearchResponse) -> None:
    """Render agent search results with answer and citations."""
    st.markdown("## Answer")
    st.markdown(result.answer)

    if result.citations:
        st.markdown("## Sources")
        for cite in result.citations:
            title = cite.document_title or "Untitled"
            year_str = f" ({cite.document_year})" if cite.document_year else ""
            authors = ", ".join(cite.document_authors) if cite.document_authors else ""
            author_str = f" — {authors}" if authors else ""

            with st.expander(f"[{cite.source_number}] {title}{year_str}{author_str}"):
                if cite.relevant_quote:
                    st.markdown(f"> {cite.relevant_quote}")
                if cite.document_url:
                    st.markdown(f"[View original document]({cite.document_url})")

    # Search metadata
    with st.expander("Search details"):
        if result.search_queries_used:
            st.markdown("**Queries used:** " + ", ".join(result.search_queries_used))
        st.markdown(f"**Chunks retrieved:** {result.chunks_retrieved}")


def _render_chunk_results(result: ChunkSearchResponse) -> None:
    """Render raw chunk search results."""
    if result.total_results == 0:
        st.info(
            "No results found. Try broadening your search terms or removing filters."
        )
        return

    st.markdown(f"## Results ({result.total_results} chunks)")
    for i, chunk in enumerate(result.results):
        title = chunk.document_title or "Untitled"
        year_str = f" ({chunk.document_year})" if chunk.document_year else ""
        score_str = f"Score: {chunk.score:.3f}"

        with st.expander(
            f"{title}{year_str} | {score_str}",
            expanded=i < 3,
        ):
            # Metadata line
            meta_parts: list[str] = []
            if chunk.document_authors:
                meta_parts.append("Authors: " + ", ".join(chunk.document_authors))
            if chunk.section_title:
                meta_parts.append(f"Section: {chunk.section_title}")
            if chunk.page_numbers:
                pages = ", ".join(str(p) for p in chunk.page_numbers)
                meta_parts.append(f"Pages: {pages}")
            if meta_parts:
                st.markdown(" | ".join(meta_parts))

            st.markdown(chunk.text_content)

            if chunk.document_url:
                st.markdown(f"[View original document]({chunk.document_url})")


results = st.session_state.last_results
if results is not None:
    if isinstance(results, AgentSearchResponse):
        _render_agent_results(results)
    elif isinstance(results, ChunkSearchResponse):
        _render_chunk_results(results)
