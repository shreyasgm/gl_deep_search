# Circular Import Analysis: publication_tracker.py ↔ growthlab.py

## Current Workflow (Pseudocode)

### publication_tracker.py
```python
# Business logic layer for managing publication lifecycle

class PublicationTracker:
    def __init__(ensure_db: bool):
        if ensure_db:
            initialize_database()

    async def discover_publications(
        growthlab_scraper: GrowthLabScraper | None,
        openalex_client: OpenAlexClient | None
    ) -> list[tuple[Publication, str]]:
        """Orchestrates discovery from multiple sources"""

        # Create scrapers if not provided
        gl_scraper = growthlab_scraper or GrowthLabScraper()
        oa_client = openalex_client or OpenAlexClient()

        # Discover from each source
        growthlab_pubs = await gl_scraper.extract_and_enrich_publications()
        openalex_pubs = await oa_client.fetch_publications()

        return combined_publications_with_source_tags

    def add_publication(publication: Publication) -> TrackingRecord:
        """Add or update publication in manifest"""
        existing = db.query(publication.paper_id)

        if existing:
            plan = generate_processing_plan(publication, existing)
            if plan.needs_updates:
                update_existing_record_and_reset_stages()
        else:
            create_new_tracking_record()

        return tracking_record
```

### growthlab.py
```python
# Scraper for Growth Lab website

class GrowthLabScraper:
    def __init__(
        config_path: Path,
        concurrency_limit: int,
        tracker: PublicationTracker | None  # OPTIONAL tracker injection
    ):
        load_config()
        self.tracker = tracker  # May be None

    async def extract_and_enrich_publications(limit: int) -> list[Publication]:
        """Scrapes and enriches publications from website"""

        # Scrape publication listings
        publications = []
        for page in pages:
            page_html = fetch_page(page_num)
            page_pubs = parse_publications(page_html)
            publications.extend(page_pubs)

        # Enrich with metadata
        for pub in publications:
            endnote_url = get_endnote_file_url(pub.url)
            if endnote_url:
                enrich_from_endnote(pub, endnote_url)
            else:
                enrich_from_publication_page(pub)

        return publications

    async def update_publications(
        existing_path: Path,
        output_path: Path,
        limit: int
    ) -> list[Publication]:
        """Full workflow: scrape, enrich, compare, register"""

        # Load existing publications
        existing_pubs = load_from_csv(existing_path)

        # Scrape and enrich new publications
        new_pubs = await extract_and_enrich_publications(limit)

        # Merge and detect changes
        updated_pubs = merge_publications(existing_pubs, new_pubs)

        # Save to CSV
        save_to_csv(updated_pubs, output_path)

        # OPTIONAL: Register with tracker if provided
        if self.tracker:
            for pub in updated_pubs:
                self.tracker.add_publication(pub)

        return updated_pubs
```

## Current Architecture Issues

### 1. **Bidirectional Dependency (The Core Problem)**
```
publication_tracker.py → growthlab.py (needs to instantiate/call scrapers)
       ↑                        ↓
       └────────────────────────┘ (scraper optionally registers with tracker)
```

This creates a circular dependency where:
- Tracker needs to import and instantiate scrapers
- Scrapers optionally need to import and use tracker

### 2. **Confused Responsibilities**

**PublicationTracker** is doing two things:
1. **Orchestration**: Coordinating multiple scrapers (`discover_publications`)
2. **State Management**: Managing publication tracking database

**GrowthLabScraper** is also doing two things:
1. **Data Extraction**: Scraping and parsing publications
2. **Workflow Orchestration**: Managing full update workflow (`update_publications`)

Both components are trying to "orchestrate" the workflow, leading to unclear boundaries.

### 3. **Tight Coupling**

`PublicationTracker.discover_publications()` directly instantiates concrete scraper classes:
```python
gl_scraper = growthlab_scraper or GrowthLabScraper()
oa_client = openalex_client or OpenAlexClient()
```

This violates the dependency inversion principle - the high-level tracker module depends on low-level scraper implementations.

### 4. **Dual Entry Points**

There are two ways to run the workflow:

**Option A: Through Tracker**
```python
tracker = PublicationTracker(ensure_db=True)
publications = await tracker.discover_publications()
for pub, source in publications:
    tracker.add_publication(pub)
```

**Option B: Through Scraper**
```python
tracker = PublicationTracker(ensure_db=True)
scraper = GrowthLabScraper(tracker=tracker)
await scraper.update_publications()  # Internally registers with tracker
```

This ambiguity makes the API confusing and error-prone.

## How It Fits Into the Larger Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    ETL PIPELINE STAGES                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. DISCOVERY (Current Issue Here)                          │
│     ├─ GrowthLabScraper.extract_and_enrich()               │
│     ├─ OpenAlexClient.fetch_publications()                 │
│     └─ PublicationTracker.discover_publications()          │
│                                                              │
│  2. REGISTRATION/MANIFEST                                    │
│     └─ PublicationTracker.add_publication()                │
│        (decides what processing is needed)                  │
│                                                              │
│  3. DOWNLOAD                                                 │
│     └─ Downloads files based on tracking status            │
│                                                              │
│  4. PROCESSING                                               │
│     └─ OCR, chunking, etc.                                  │
│                                                              │
│  5. EMBEDDING                                                │
│     └─ Generate vector embeddings                           │
│                                                              │
│  6. INGESTION                                                │
│     └─ Ingest into search index (Qdrant)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

The circular import occurs at **Stage 1 (Discovery)** because:
- The tracker wants to orchestrate multiple discovery sources
- The scrapers want to automatically register their discoveries with the tracker

## Proposed Architecture Solutions

### Solution 1: **Dependency Inversion (Recommended)**

Create an abstract interface that both depend on, eliminating direct dependencies.

```python
# backend/etl/interfaces/scraper.py (NEW)
from typing import Protocol, runtime_checkable

@runtime_checkable
class PublicationScraper(Protocol):
    """Interface for publication scrapers"""

    async def extract_and_enrich_publications(
        self, limit: int | None = None
    ) -> list[Publication]:
        """Extract and enrich publications from source"""
        ...

    @property
    def source_name(self) -> str:
        """Return source identifier (e.g., 'growthlab', 'openalex')"""
        ...
```

```python
# backend/etl/utils/publication_tracker.py
from backend.etl.interfaces.scraper import PublicationScraper

class PublicationTracker:
    async def discover_publications(
        self,
        scrapers: list[PublicationScraper]  # Generic interface, not concrete classes
    ) -> list[tuple[Publication, str]]:
        """Discover from any scrapers implementing the interface"""

        publications = []
        for scraper in scrapers:
            pubs = await scraper.extract_and_enrich_publications()
            publications.extend([(pub, scraper.source_name) for pub in pubs])

        return publications
```

```python
# backend/etl/scrapers/growthlab.py
from backend.etl.models.publications import GrowthLabPublication
# NO IMPORT of PublicationTracker

class GrowthLabScraper:
    # Remove tracker parameter entirely
    def __init__(self, config_path: Path | None = None, ...):
        load_config()
        # No tracker reference

    async def extract_and_enrich_publications(...) -> list[GrowthLabPublication]:
        # Just scrape and return - don't register anywhere
        ...

    @property
    def source_name(self) -> str:
        return "growthlab"
```

**Benefits:**
- No circular imports - tracker depends on interface, scrapers implement interface
- Scrapers are purely focused on data extraction
- Tracker is purely focused on orchestration and state management
- Easy to add new scrapers without modifying tracker

### Solution 2: **Separate Orchestration Layer**

Create a third component that coordinates both scrapers and tracker.

```python
# backend/etl/orchestration/pipeline.py (NEW)
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.scrapers.openalex import OpenAlexClient
from backend.etl.utils.publication_tracker import PublicationTracker

class ETLOrchestrator:
    """Coordinates the entire ETL pipeline"""

    def __init__(self):
        self.tracker = PublicationTracker(ensure_db=True)
        self.scrapers = {
            "growthlab": GrowthLabScraper(),
            "openalex": OpenAlexClient(),
        }

    async def discover_and_register(self, limit: int | None = None):
        """Run discovery and registration"""

        # Discover from all sources
        all_publications = []
        for name, scraper in self.scrapers.items():
            pubs = await scraper.extract_and_enrich_publications(limit)
            all_publications.extend([(pub, name) for pub in pubs])

        # Register with tracker
        for pub, source in all_publications:
            self.tracker.add_publication(pub)

        return all_publications
```

```python
# Remove discover_publications from PublicationTracker entirely
# Remove tracker parameter from GrowthLabScraper entirely
# Remove update_publications from GrowthLabScraper (move to orchestrator)
```

**Benefits:**
- Clear separation of concerns
- No circular dependencies
- Single entry point for the workflow
- Easy to test each component in isolation

**Drawbacks:**
- Adds another layer of abstraction
- May be overkill for simple workflows

### Solution 3: **Remove Bidirectional Dependency**

Keep tracker's `discover_publications()` but remove tracker injection from scrapers.

```python
# backend/etl/utils/publication_tracker.py
class PublicationTracker:
    # Keep discover_publications (imports scrapers)
    async def discover_publications(
        self,
        growthlab_scraper: GrowthLabScraper | None = None,
        openalex_client: OpenAlexClient | None = None,
    ) -> list[tuple[Publication, str]]:
        gl_scraper = growthlab_scraper or GrowthLabScraper()
        # ... rest of method
```

```python
# backend/etl/scrapers/growthlab.py
class GrowthLabScraper:
    # REMOVE tracker parameter completely
    def __init__(self, config_path: Path | None = None, ...):
        # No tracker reference

    # REMOVE update_publications method
    # OR move it to a separate utility/orchestration module
```

**Benefits:**
- Minimal changes
- Clear unidirectional dependency: tracker → scrapers
- Scrapers remain focused on extraction

**Drawbacks:**
- Loses some convenience of `scraper.update_publications()`
- Need to move that functionality elsewhere

### Solution 4: **TYPE_CHECKING Guard (Quick Fix, Not Recommended)**

Use runtime vs. type-checking imports to break the cycle.

```python
# backend/etl/scrapers/growthlab.py
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.etl.utils.publication_tracker import PublicationTracker

class GrowthLabScraper:
    def __init__(self, tracker: PublicationTracker | None = None):
        self.tracker = tracker
```

**Benefits:**
- Quick fix with minimal code changes
- Type hints still work

**Drawbacks:**
- Doesn't address the underlying design issue
- Still have bidirectional runtime dependency (tracker instantiates scrapers)
- Confusing for developers (why is import conditional?)
- Masks architectural problems

## Recommendation

**Use Solution 1 (Dependency Inversion) with elements of Solution 2.**

### Proposed Structure:

```
backend/etl/
├── interfaces/
│   └── scraper.py               # Protocol definition for scrapers
├── scrapers/
│   ├── growthlab.py            # Implements PublicationScraper protocol
│   └── openalex.py             # Implements PublicationScraper protocol
├── utils/
│   └── publication_tracker.py  # Depends on PublicationScraper protocol
└── orchestration/
    └── discovery.py            # Optional: high-level workflow coordination
```

### Key Changes:

1. **Create `interfaces/scraper.py`**: Define `PublicationScraper` protocol
2. **Modify `publication_tracker.py`**:
   - Change `discover_publications` to accept `list[PublicationScraper]`
   - Remove direct imports of concrete scraper classes
3. **Modify `growthlab.py`** (and `openalex.py`):
   - Remove `tracker` parameter from `__init__`
   - Remove or move `update_publications` to orchestration layer
   - Implement `PublicationScraper` protocol
4. **Create `orchestration/discovery.py`** (optional):
   - High-level workflow for discovery + registration
   - Instantiates scrapers and tracker
   - Coordinates the full workflow

### Example Usage After Refactoring:

```python
# Clean, unidirectional dependencies
from backend.etl.scrapers.growthlab import GrowthLabScraper
from backend.etl.scrapers.openalex import OpenAlexClient
from backend.etl.utils.publication_tracker import PublicationTracker

# Create components
tracker = PublicationTracker(ensure_db=True)
scrapers = [GrowthLabScraper(), OpenAlexClient()]

# Discover and register
publications = await tracker.discover_publications(scrapers)
for pub, source in publications:
    tracker.add_publication(pub)
```

Or with orchestration layer:

```python
from backend.etl.orchestration.discovery import DiscoveryOrchestrator

orchestrator = DiscoveryOrchestrator()
await orchestrator.run(limit=100)
```

## Implementation Priority

1. **Immediate fix**: Use `TYPE_CHECKING` guard to unblock pytest (5 minutes)
2. **Short-term**: Remove `tracker` parameter from scrapers (30 minutes)
3. **Medium-term**: Implement protocol-based dependency inversion (2-3 hours)
4. **Long-term**: Add orchestration layer if needed as pipeline grows (4-6 hours)

## Testing Implications

After refactoring:
- Each scraper can be tested in complete isolation (no tracker needed)
- Tracker can be tested with mock scrapers implementing the protocol
- Integration tests become clearer about what they're testing
- No more circular import issues in test files
