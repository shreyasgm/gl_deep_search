# Supabase Publication Manifest Implementation Plan

**Project**: Growth Lab Deep Search - Collaborative Publication Management System
**Date**: November 16, 2025
**Status**: In Progress
**Related PRs**: #37 (Manifest System), #38 (FastAPI Status Endpoint)

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Requirements Analysis](#requirements-analysis)
3. [Architecture Decision](#architecture-decision)
4. [Implementation Plan](#implementation-plan)
5. [Cost Analysis](#cost-analysis)
6. [Timeline & Phases](#timeline--phases)
7. [Files to Create/Modify](#files-to-createmodify)
8. [Testing Strategy](#testing-strategy)
9. [Deployment Guide](#deployment-guide)
10. [Success Criteria](#success-criteria)

---

## Problem Statement

### Original Understanding (Incorrect)
Initially, PR #37 was understood to be a simple ETL status tracking system - tracking publications as they move through the pipeline stages (download, processing, embedding, ingestion) for observability purposes only.

### Actual Requirements (Complete Picture)
The system is a **collaborative publication management platform** that serves two distinct user groups:

1. **Automated ETL Pipeline**:
   - Discovers publications via scrapers (Growth Lab website, OpenAlex API)
   - Downloads PDFs
   - Processes documents (OCR, text extraction, chunking)
   - Generates embeddings
   - Ingests into vector database
   - Tracks status at each stage

2. **Growth Lab Communications Team (5-10 users)**:
   - Manual publication entry (publications not found by scrapers)
   - Metadata correction (fix errors in title, authors, year, etc.)
   - PDF uploads (add missing PDFs)
   - Status monitoring (view processing progress)
   - Error investigation (check why publications failed)
   - **Concurrent 24/7 access** (not just during ETL runs)

### Key Insight: This Changes Everything
The requirement for a 24/7 web interface with multi-user concurrent access, manual uploads, and metadata correction means:

- ❌ SQLite in GCS bucket won't work (file locking issues with concurrent users)
- ❌ Simple status tracking is insufficient (need full CRUD + file management)
- ✅ Need always-on database with built-in admin UI
- ✅ Need authentication and user management
- ✅ Need webhook triggers for manual edits → ETL reprocessing

---

## Requirements Analysis

### Functional Requirements

#### For ETL Pipeline
1. **Publication Discovery**: Track newly discovered publications from scrapers
2. **Status Tracking**: Record progress through all pipeline stages
3. **Error Handling**: Log errors and retry attempts
4. **Change Detection**: Detect content changes to avoid reprocessing
5. **Selective Processing**: Only process stages that need updating
6. **Concurrent Access**: Multiple ETL containers accessing database simultaneously

#### For Communications Team
1. **Admin Interface**: Web UI to view/edit publications (no custom dev work preferred)
2. **User Authentication**: Email/password login for 5-10 team members
3. **Manual Entry**: Add publications not found by scrapers
4. **Metadata Editing**: Correct errors in publication metadata
5. **PDF Upload**: Upload missing PDFs to GCS
6. **Status Monitoring**: View real-time processing status
7. **Error Investigation**: View error messages and retry counts
8. **Search & Filter**: Find publications by title, author, year, status
9. **Webhook Triggers**: Metadata edits trigger selective ETL reprocessing

### Non-Functional Requirements

1. **Availability**: 24/7 uptime for comms team access
2. **Concurrency**: 1-5 concurrent users (low)
3. **Performance**: Sub-second query response times
4. **Cost**: Minimize monthly costs (prefer free tier if possible)
5. **Maintainability**: Low operational overhead, managed service preferred
6. **Scalability**: Handle 1,400 current publications, grow to 10,000+
7. **Integration**: Work with existing GCP infrastructure (GCS, Cloud Run)
8. **Security**: Row-level security, audit logging

---

## Architecture Decision

### Options Considered

| Solution | Admin UI | Auth | Cost/Month | Dev Effort | Concurrent Access | GCP Native |
|----------|----------|------|------------|------------|-------------------|------------|
| **Supabase** | ✅ Built-in Studio | ✅ Built-in | $0-25 | Low (12 days) | ✅ Native | ⚠️ External |
| **Firestore** | ❌ Need custom | ⚠️ Manual | $2 | High (20 days) | ✅ Native | ✅ Yes |
| **Cloud SQL** | ❌ Need custom | ⚠️ Manual | $7 | Medium (18 days) | ✅ Native | ✅ Yes |
| **SQLite+GCS** | ❌ Need custom | ❌ None | $0.03 | Medium (15 days) | ❌ File locks | ✅ Yes |

### Decision: Supabase with Hybrid Storage

**Winner**: **Supabase** - Only option with zero-config admin UI + built-in auth

**Rationale**:

1. **Zero Admin UI Development**: Supabase Studio provides ready-to-use admin interface
   - Table editor for publication CRUD operations
   - User management built-in
   - Query builder for advanced searches
   - No React/Vue.js development needed

2. **Built-in Authentication**: Email/password auth out-of-box
   - Invite team members via email
   - Row Level Security (RLS) for fine-grained permissions
   - No need to build auth flows

3. **Cost-Effective for Our Scale**:
   - FREE tier: 500 MB database (we need ~14 MB), 5 GB egress/month
   - With low concurrency (1-5 users), likely stay within free tier
   - Can upgrade to Pro ($25/month) if needed

4. **Webhook Support**: Native triggers for manual edits → ETL pipeline

5. **Hybrid Storage Strategy**: Best of both worlds
   - **Supabase**: Publication metadata (~14 MB)
   - **GCS**: PDF files (~2.8 GB, already integrated)
   - Database stores GCS URLs, not actual files

6. **PostgreSQL**: Full SQL capabilities, familiar for team

7. **Real-time Updates**: Team sees ETL progress updates live

8. **Low Maintenance**: Fully managed service, no ops overhead

### Why NOT Other Options

**Firestore**:
- ❌ No built-in admin UI (need custom React app)
- ❌ NoSQL query limitations
- ⚠️ Requires manual auth setup
- ✅ Cheaper ($2/month) but 20 days dev time vs 12 days

**Cloud SQL**:
- ❌ No admin UI (need custom app)
- ❌ More expensive ($7/month minimum)
- ⚠️ Requires manual auth
- ✅ Native GCP but higher overhead

**SQLite in GCS**:
- ❌ No concurrent access (file locking)
- ❌ No admin UI
- ❌ No auth
- ❌ Not suitable for multi-user web interface
- ✅ Cheapest but fundamentally incompatible with requirements

---

## Implementation Plan

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Growth Lab Team                           │
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Comms Team      │────────>│  Supabase Studio │          │
│  │  (5-10 users)    │         │   (Admin UI)     │          │
│  │  - Add pubs      │         │   - Built-in     │          │
│  │  - Edit metadata │         │   - Zero config  │          │
│  │  - Upload PDFs   │         └──────────────────┘          │
│  └──────────────────┘                  │                     │
│                                        ↓                      │
│                              ┌──────────────────┐            │
│                              │   Supabase DB    │            │
│                              │  (PostgreSQL)    │            │
│                              │  - Metadata      │            │
│                              │  - Status        │            │
│                              └──────────────────┘            │
│                                   ↑     ↓                     │
│                    ┌──────────────┴─────┴──────────┐         │
│                    │                                │         │
│         ┌──────────▼───────┐          ┌───────────▼──────┐  │
│         │  FastAPI Service │          │   ETL Pipeline   │  │
│         │  (PR#38)         │          │   (Cloud Run)    │  │
│         │  - Status API    │          │   - Scraping     │  │
│         │  - Queries       │          │   - Processing   │  │
│         └──────────────────┘          │   - Embeddings   │  │
│                                        └──────────────────┘  │
│                                                 ↓             │
│                                        ┌──────────────────┐  │
│                                        │   GCS Bucket     │  │
│                                        │   (PDF Storage)  │  │
│                                        └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: Manual Upload Workflow

```
1. Comms team logs into Supabase Studio (https://app.supabase.com)
   ↓
2. Team member adds publication or uploads PDF to GCS
   ↓
3. Team member updates file_urls in Supabase with GCS path
   ↓
4. Supabase webhook triggers (via pg_notify)
   ↓
5. FastAPI webhook endpoint receives notification
   ↓
6. Webhook handler triggers Cloud Run Job for that publication
   ↓
7. ETL Pipeline:
   - Reads publication from Supabase
   - Downloads PDF from GCS
   - Runs needed stages (OCR → chunking → embeddings → ingestion)
   - Updates status back to Supabase after each stage
   ↓
8. Comms team sees real-time status updates in Supabase Studio
```

### Component Architecture

#### Database Adapter Pattern

To decouple ETL logic from database implementation:

```python
# Abstract interface
class DatabaseAdapter(ABC):
    @abstractmethod
    async def add_publication(self, pub: PublicationTracking) -> PublicationTracking

    @abstractmethod
    async def get_publication(self, pub_id: str) -> PublicationTracking | None

    @abstractmethod
    async def get_publications_by_status(...) -> list[PublicationTracking]

    # ... other methods

# Concrete implementation
class SupabaseAdapter(DatabaseAdapter):
    """Supabase-specific implementation using supabase-py client"""

# Factory pattern
def get_database_adapter() -> DatabaseAdapter:
    db_type = os.environ.get("DATABASE_TYPE", "supabase")
    if db_type == "supabase":
        return SupabaseAdapter()
    else:
        raise ValueError(f"Unknown DATABASE_TYPE: {db_type}")
```

**Benefits**:
- ETL code doesn't depend on specific database
- Easy to swap implementations for testing
- Consistent interface across codebase
- Enables local dev with different backend if needed

---

## Cost Analysis

### Monthly Cost Breakdown

#### Supabase FREE Tier Limits
- **Database Storage**: 500 MB (we need ~14 MB for 1,400 publications) ✅
- **Egress**: 5 GB/month
- **File Storage**: 1 GB (not using Supabase Storage, using GCS instead) ✅
- **Projects**: 2 active projects ✅
- **Pausing**: Projects pause after 7 days inactivity (won't happen with 24/7 team access) ✅

#### Estimated Usage
- **Database Size**: ~14 MB (1,400 publications × ~10 KB each)
- **Egress**: ~1-2 GB/month (low concurrent users, minimal queries)
- **Growth**: ~10 MB/year as publications accumulate

#### Hybrid Storage Strategy
```
Supabase (FREE tier):
  Publication metadata: ~14 MB
  Cost: $0/month ✅

GCS (existing):
  PDF files: ~2.8 GB
  Storage cost: $0.026/GB × 2.8 GB = $0.07/month
  Egress cost: Negligible (internal GCP traffic)

Total Cost: $0.07/month (essentially FREE)
```

#### When to Upgrade to Pro ($25/month)
- Hit 5 GB egress limit (unlikely with low concurrency)
- Need production SLA guarantees
- Need technical support
- Need advanced features (edge functions, realtime at scale)

**Recommendation**: Start with FREE tier, monitor usage, upgrade only if needed (likely 6+ months).

---

## Timeline & Phases

### Total Duration: 12 Days (2.5 weeks)

#### Phase 1: Supabase Setup & Schema (Day 1)
**Tasks**:
- [x] Create Supabase project at supabase.com
- [x] Run migration SQL (`backend/storage/supabase_migrations/001_initial_schema.sql`)
- [ ] Configure Row Level Security (RLS) policies
- [ ] Create team member accounts
- [ ] Test table editor in Supabase Studio

**Deliverables**:
- Supabase project configured
- Database schema created
- Team members invited
- Credentials stored in GCP Secret Manager

---

#### Phase 2: Database Adapter Implementation (Days 2-3)
**Tasks**:
- [ ] Create `backend/storage/adapters/base.py` - Abstract interface
- [ ] Create `backend/storage/adapters/supabase_adapter.py` - Implementation (~400 lines)
- [ ] Create `backend/storage/adapters/factory.py` - Factory pattern
- [ ] Add helper methods for common operations

**Key Files**:
```
backend/storage/adapters/
├── __init__.py
├── base.py              # Abstract DatabaseAdapter class
├── supabase_adapter.py  # SupabaseAdapter implementation
└── factory.py           # get_database_adapter() factory
```

**Deliverables**:
- Working adapter interface
- Supabase adapter with full CRUD operations
- Factory pattern for adapter selection

---

#### Phase 3: ETL Pipeline Integration (Days 4-5)
**Tasks**:
- [ ] Modify `backend/etl/utils/publication_tracker.py`:
  - Replace SQLAlchemy with DatabaseAdapter
  - Convert methods to async
  - Add manual upload handling
- [ ] Modify `backend/etl/orchestrator.py`:
  - Add single-publication processing method
  - Add selective stage execution
- [ ] Add webhook support for manual edits

**Key Changes**:
```python
# BEFORE: Direct SQLAlchemy usage
class PublicationTracker:
    def add_publication(self, pub, session=None):
        with self._get_session(session) as sess:
            stmt = select(PublicationTracking).where(...)
            # ...

# AFTER: Async adapter usage
class PublicationTracker:
    def __init__(self):
        self.adapter = get_database_adapter()

    async def add_publication(self, pub):
        existing = await self.adapter.get_publication(pub.paper_id)
        # ...
```

**Deliverables**:
- ETL pipeline uses adapter pattern
- All methods converted to async
- Manual upload workflow functional

---

#### Phase 4: FastAPI Service Update (Day 6)
**Tasks**:
- [ ] Modify `backend/service/database.py`:
  - Replace aiosqlite with adapter
  - Use factory pattern
- [ ] Modify `backend/service/routes.py`:
  - Ensure async operations
  - Use adapter methods
- [ ] Add `backend/service/webhooks.py`:
  - Webhook endpoint for Supabase notifications
  - Trigger ETL for manual edits

**Key Changes**:
```python
# BEFORE: aiosqlite
async def get_publications_status(...):
    conn = await aiosqlite.connect(DB_PATH)
    # raw SQL queries

# AFTER: Adapter pattern
async def get_publications_status(...):
    adapter = get_database_adapter()
    await adapter.initialize()
    publications = await adapter.get_publications_by_status(...)
```

**Deliverables**:
- FastAPI service queries Supabase
- Webhook endpoint for manual edits
- All PR#38 endpoints working

---

#### Phase 5: Configuration & Secrets (Day 7)
**Tasks**:
- [ ] Add Supabase dependency to `pyproject.toml`
- [ ] Update `.env.example` with new variables
- [ ] Store Supabase credentials in GCP Secret Manager
- [ ] Update `deployment/cloud-run/Dockerfile`
- [ ] Update `deployment/cloud-run/entrypoint.sh`
- [ ] Create local dev configuration

**Environment Variables**:
```bash
# Production
DATABASE_TYPE=supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=${SECRET_SUPABASE_SERVICE_KEY}  # From Secret Manager

# Local Dev
DATABASE_TYPE=supabase
SUPABASE_URL=https://test-xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # Test anon key
```

**Deliverables**:
- Dependencies updated
- Secrets stored securely
- Deployment config ready

---

#### Phase 6: Testing (Days 8-9)
**Tasks**:
- [ ] Write unit tests (mock Supabase client)
  - `backend/tests/storage/test_supabase_adapter.py`
- [ ] Write integration tests (real Supabase)
  - `backend/tests/storage/test_supabase_integration.py`
- [ ] Write E2E tests (full workflow)
  - Manual upload → ETL processing → status update
- [ ] Test concurrent access patterns
- [ ] Load test with 1,400 publications

**Test Coverage**:
- Unit: 70% (adapter methods with mocks)
- Integration: 20% (real Supabase operations)
- E2E: 10% (complete workflows)

**Deliverables**:
- Comprehensive test suite
- All tests passing
- Load test results documented

---

#### Phase 7: Deployment & Training (Days 10-11)
**Tasks**:
- [ ] Deploy to GCP Cloud Run
- [ ] Configure webhooks in Supabase
- [ ] Create training documentation for comms team
- [ ] Train 2-3 team members
- [ ] Create troubleshooting runbook

**Training Doc**: `docs/COMMS_TEAM_GUIDE.md`
- How to access Supabase Studio
- How to add publications
- How to upload PDFs
- How to correct metadata
- How to monitor processing status
- Troubleshooting common issues

**Deliverables**:
- Production deployment complete
- Team trained
- Documentation ready

---

#### Phase 8: Production Validation (Day 12)
**Tasks**:
- [ ] Run Phase 1 ETL test (10 publications)
- [ ] Manual upload test by comms team
- [ ] Verify webhooks trigger correctly
- [ ] Monitor Supabase usage/costs
- [ ] Check for any errors or performance issues
- [ ] Sign-off from stakeholders

**Success Criteria**:
- ETL pipeline processes publications successfully
- Status updates appear in Supabase Studio
- Team can add/edit publications manually
- Manual edits trigger selective reprocessing
- No errors in logs
- FREE tier limits not exceeded

**Deliverables**:
- Production validation complete
- Sign-off obtained
- System ready for full use

---

## Files to Create/Modify

### New Files (10)

#### Database Layer
1. `backend/storage/adapters/__init__.py` (empty)
2. `backend/storage/adapters/base.py` (~100 lines) - Abstract interface
3. `backend/storage/adapters/supabase_adapter.py` (~400 lines) - Implementation
4. `backend/storage/adapters/factory.py` (~50 lines) - Factory pattern
5. `backend/storage/supabase_migrations/001_initial_schema.sql` (~350 lines) - [DONE] ✅

#### API Layer
6. `backend/service/webhooks.py` (~100 lines) - Webhook endpoints

#### Testing
7. `backend/tests/storage/test_supabase_adapter.py` (~300 lines) - Unit tests
8. `backend/tests/storage/test_supabase_integration.py` (~200 lines) - Integration tests
9. `backend/tests/service/test_webhooks.py` (~150 lines) - Webhook tests

#### Documentation
10. `docs/COMMS_TEAM_GUIDE.md` (~200 lines) - Training guide

### Modified Files (9)

#### Core ETL
1. `backend/etl/utils/publication_tracker.py` - Use adapter, async conversion
2. `backend/etl/orchestrator.py` - Add single-pub processing

#### API Service (PR#38)
3. `backend/service/database.py` - Replace aiosqlite with adapter
4. `backend/service/routes.py` - Use adapter methods
5. `backend/service/main.py` - Add webhooks router

#### Configuration
6. `pyproject.toml` - Add supabase dependency
7. `.env.example` - Document new env vars

#### Deployment
8. `deployment/cloud-run/Dockerfile` - Add secret fetching
9. `deployment/cloud-run/entrypoint.sh` - Set Supabase env vars

---

## Testing Strategy

### Unit Tests (Mock Supabase Client)

**File**: `backend/tests/storage/test_supabase_adapter.py`

```python
@pytest.fixture
def mock_supabase():
    """Mock Supabase client for unit tests"""
    client = Mock()
    client.table = Mock(return_value=Mock())
    return client

@pytest.mark.asyncio
async def test_add_publication(mock_supabase):
    """Test adding publication via adapter"""
    adapter = SupabaseAdapter()
    adapter.client = mock_supabase

    # Setup mock response
    mock_response = Mock()
    mock_response.data = [{"publication_id": "test123", ...}]
    mock_supabase.table().upsert().execute = AsyncMock(return_value=mock_response)

    # Test
    pub = PublicationTracking(publication_id="test123", ...)
    result = await adapter.add_publication(pub)

    assert result.publication_id == "test123"
    mock_supabase.table.assert_called_with("publication_tracking")
```

### Integration Tests (Real Supabase)

**File**: `backend/tests/storage/test_supabase_integration.py`

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test complete workflow: add → update → query"""
    if not os.getenv("SUPABASE_TEST_URL"):
        pytest.skip("Supabase test credentials not configured")

    adapter = SupabaseAdapter()
    await adapter.initialize()

    # 1. Add publication
    pub = PublicationTracking(publication_id="integration_test_1", ...)
    await adapter.add_publication(pub)

    # 2. Update status
    await adapter.update_stage_status("integration_test_1", "download", "Downloaded")

    # 3. Query by status
    results = await adapter.get_publications_by_status(
        download_status=DownloadStatus.DOWNLOADED
    )

    assert len(results) >= 1
    assert any(p.publication_id == "integration_test_1" for p in results)
```

### E2E Tests (Complete Workflow)

**File**: `backend/tests/e2e/test_manual_upload_workflow.py`

```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_manual_upload_triggers_etl():
    """Simulate comms team uploading PDF → ETL processing"""

    # 1. Upload PDF to GCS
    pdf_url = await upload_pdf_to_gcs("test.pdf")

    # 2. Add publication via Supabase
    adapter = get_database_adapter()
    pub = PublicationTracking(
        publication_id="e2e_test_1",
        source_url="https://test.com",
        file_urls=[pdf_url]
    )
    await adapter.add_publication(pub)

    # 3. Trigger ETL
    tracker = PublicationTracker()
    plan = await tracker.process_manual_upload("e2e_test_1", pdf_url)

    orchestrator = ETLOrchestrator(...)
    await orchestrator.execute_processing_plan(plan)

    # 4. Verify status updated
    result = await adapter.get_publication("e2e_test_1")
    assert result.processing_status == ProcessingStatus.PROCESSED
    assert result.embedding_status == EmbeddingStatus.EMBEDDED
```

---

## Deployment Guide

### Prerequisites

1. **Supabase Account**: Create account at https://supabase.com
2. **GCP Project**: Existing project with Cloud Run, Secret Manager enabled
3. **GCS Bucket**: Existing bucket for PDF storage
4. **Docker**: For building container images
5. **gcloud CLI**: Authenticated with proper permissions

### Step 1: Supabase Project Setup

```bash
# 1. Go to https://supabase.com → New Project
# 2. Fill in:
#    - Name: gl-deep-search-manifest
#    - Database Password: [secure password]
#    - Region: us-east-1 (closest to GCP us-east4)
# 3. Wait for project to provision (~2 minutes)

# 4. Note credentials from Project Settings → API:
#    - Project URL: https://xxxxx.supabase.co
#    - anon key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... (for local dev)
#    - service_role key: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9... (for production)
```

### Step 2: Database Schema Setup

```bash
# 1. In Supabase Dashboard → SQL Editor → New Query
# 2. Copy contents of backend/storage/supabase_migrations/001_initial_schema.sql
# 3. Run the query
# 4. Verify in Database → Tables that publication_tracking exists
```

### Step 3: Store Credentials in GCP Secret Manager

```bash
# Store Supabase URL
gcloud secrets create supabase-url \
    --data-file=- <<< "https://xxxxx.supabase.co"

# Store Supabase service_role key (NOT anon key)
gcloud secrets create supabase-service-key \
    --data-file=- <<< "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Grant access to Cloud Run service account
SERVICE_ACCOUNT="etl-pipeline@cid-hks-1537286359734.iam.gserviceaccount.com"

gcloud secrets add-iam-policy-binding supabase-url \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding supabase-service-key \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor"
```

### Step 4: Update Dependencies

```bash
# Add Supabase to dependencies
uv add supabase --optional etl
uv add supabase --optional service

# Lock dependencies
uv lock
```

### Step 5: Build & Deploy

```bash
# Build Docker image with Cloud Build
./deployment/cloud-run/deploy.sh --cloud-build

# Deploy Cloud Run Job with new env vars
gcloud run jobs deploy etl-pipeline-job \
    --image us-east4-docker.pkg.dev/cid-hks-1537286359734/etl-pipeline/etl-pipeline:latest \
    --set-env-vars DATABASE_TYPE=supabase \
    --set-secrets SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-service-key:latest \
    --region us-east4

# Deploy FastAPI service (PR#38)
gcloud run deploy publication-status-api \
    --source backend/service \
    --set-env-vars DATABASE_TYPE=supabase \
    --set-secrets SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-service-key:latest \
    --region us-east4 \
    --allow-unauthenticated
```

### Step 6: Configure Webhooks (Optional - for manual edit triggers)

```bash
# In Supabase Dashboard → Database → Webhooks
# Create new webhook:
#   Name: publication-changed
#   Table: publication_tracking
#   Events: INSERT, UPDATE
#   Type: HTTP Request
#   URL: https://[your-fastapi-url]/api/v1/webhooks/supabase/publication-updated
#   Method: POST
```

### Step 7: Invite Team Members

```bash
# In Supabase Dashboard → Authentication → Users
# Click "Invite User"
# Enter team member emails
# They'll receive invite to set password
```

### Step 8: Validation

```bash
# Run Phase 1 test
python deployment/vm/test_batch_processing.py --limit 10 --phase 1

# Check Supabase Studio
# 1. Go to Table Editor → publication_tracking
# 2. Verify 10 publications appear
# 3. Check status fields are updating

# Test manual entry
# 1. Team member logs into Supabase Studio
# 2. Adds test publication manually
# 3. Uploads PDF to GCS, adds URL to file_urls
# 4. Verifies ETL processes it
```

---

## Success Criteria

### Technical Success

- [x] Supabase schema created successfully
- [ ] Database adapter pattern implemented
- [ ] ETL pipeline uses Supabase for status tracking
- [ ] FastAPI service queries Supabase
- [ ] All tests passing (unit, integration, E2E)
- [ ] Deployment successful to GCP Cloud Run
- [ ] Webhooks working for manual edits

### Functional Success

- [ ] Comms team can log into Supabase Studio
- [ ] Team can add publications manually
- [ ] Team can upload PDFs and update file_urls
- [ ] Team can correct metadata errors
- [ ] Team can view processing status in real-time
- [ ] Manual edits trigger selective ETL reprocessing
- [ ] ETL pipeline processes publications end-to-end
- [ ] Status updates visible in Supabase Studio

### Operational Success

- [ ] Staying within FREE tier limits ($0/month)
- [ ] Sub-second query response times
- [ ] No database connection errors
- [ ] Team trained and comfortable with system
- [ ] Documentation complete and clear
- [ ] Monitoring/alerting configured

---

## Risk Mitigation

### Risk 1: Exceeding FREE Tier Limits

**Risk**: Egress exceeds 5 GB/month, forcing upgrade to Pro ($25/month)

**Mitigation**:
- Monitor Supabase dashboard weekly
- Set up billing alerts
- Optimize queries to reduce data transfer
- Cache frequent queries in FastAPI

**Contingency**: Upgrade to Pro tier if needed (still cheaper than Cloud SQL)

### Risk 2: Concurrent Access Issues

**Risk**: Multiple ETL containers + team members cause conflicts

**Mitigation**:
- Use PostgreSQL transactions
- Implement optimistic locking with `last_updated` checks
- Row Level Security prevents unauthorized access
- Test with simulated concurrent load

**Contingency**: Add pessimistic locking if needed

### Risk 3: Webhook Reliability

**Risk**: Webhooks fail to trigger ETL reprocessing

**Mitigation**:
- Implement retry logic with exponential backoff
- Log all webhook calls
- Add manual "Reprocess" button in Supabase Studio
- Monitor webhook delivery in Supabase logs

**Contingency**: Fallback to periodic polling for changes

### Risk 4: Integration Complexity

**Risk**: Async conversion of ETL code introduces bugs

**Mitigation**:
- Comprehensive testing (unit + integration + E2E)
- Gradual rollout (test environment first)
- Keep both sync and async versions during transition
- Code reviews before merging

**Contingency**: Rollback to PR#37 original code if needed

---

## Next Steps (After Implementation)

### Short-Term (1-3 months)

1. **Monitor Usage**: Track database size, egress, query performance
2. **User Feedback**: Gather feedback from comms team, iterate on UX
3. **Optimize Queries**: Add indexes if slow queries identified
4. **Documentation**: Update training docs based on team feedback

### Medium-Term (3-6 months)

1. **Enhanced Search**: Add full-text search on title/abstract
2. **Batch Operations**: Enable bulk metadata updates
3. **Export Features**: CSV export of publication lists
4. **Analytics Dashboard**: Visualize ETL pipeline metrics

### Long-Term (6-12 months)

1. **Advanced Workflows**: Multi-stage approval process
2. **Automated Quality Checks**: Flag suspicious metadata
3. **Integration with Qdrant**: Direct link from Supabase to vector DB
4. **Public API**: Expose publication data via public API

---

## Appendix

### A. Supabase vs Firestore Comparison

| Feature | Supabase | Firestore |
|---------|----------|-----------|
| Database Type | PostgreSQL (SQL) | NoSQL Document Store |
| Query Language | SQL | Document queries |
| Admin UI | ✅ Built-in Studio | ❌ Need custom |
| Auth | ✅ Built-in | ⚠️ Firebase Auth (separate) |
| Webhooks | ✅ Native triggers | ⚠️ Cloud Functions |
| Joins | ✅ Full SQL joins | ❌ Client-side only |
| Transactions | ✅ ACID | ⚠️ Limited |
| Free Tier | 500 MB, 5 GB egress | 1 GB storage, 10 GB egress |
| Cost (low usage) | $0 | ~$2/month |
| GCP Native | ❌ External | ✅ Yes |
| Learning Curve | Low (SQL) | Medium (NoSQL) |

### B. Environment Variables Reference

```bash
# Required for Production
DATABASE_TYPE=supabase
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=${SECRET}  # service_role key from Secret Manager

# Optional
SUPABASE_MAX_CONNECTIONS=10
SUPABASE_TIMEOUT=30

# Existing ETL variables
GCS_BUCKET=gl-deep-search-data
GOOGLE_CLOUD_PROJECT=cid-hks-1537286359734
OPENAI_API_KEY=${SECRET}
```

### C. Troubleshooting Guide

#### Issue: "Database connection error"

**Symptoms**: ETL fails with connection timeout

**Causes**:
- Supabase credentials incorrect
- Service account lacks Secret Manager access
- Network connectivity issues

**Solutions**:
1. Verify credentials: `gcloud secrets versions access latest --secret=supabase-url`
2. Check service account permissions
3. Test connection manually: `curl https://xxxxx.supabase.co`

#### Issue: "Row Level Security (RLS) policy violation"

**Symptoms**: Queries fail with "permission denied"

**Causes**:
- Using anon key instead of service_role key
- RLS policies misconfigured

**Solutions**:
1. Verify using service_role key in production
2. Check RLS policies in Supabase Dashboard → Database → Policies
3. Add policy for service_role: `FOR ALL TO service_role USING (true)`

#### Issue: "Webhook not triggering"

**Symptoms**: Manual edits don't trigger ETL

**Causes**:
- Webhook URL incorrect
- FastAPI endpoint not deployed
- pg_notify not working

**Solutions**:
1. Test webhook manually: `curl -X POST [webhook-url] -d '{"test": true}'`
2. Check webhook logs in Supabase Dashboard → Database → Webhooks
3. Verify trigger exists: `SELECT * FROM pg_trigger WHERE tgname = 'trigger_notify_publication_change'`

---

## References

- **Supabase Documentation**: https://supabase.com/docs
- **Supabase Python Client**: https://github.com/supabase/supabase-py
- **PR #37**: Manifest branch implementation
- **PR #38**: FastAPI status endpoint
- **Deployment Architecture**: `MANIFEST_ARCHITECTURE_ANALYSIS.md`

---

**Document Status**: Living document - will be updated as implementation progresses

**Last Updated**: November 16, 2025

**Contributors**: Shreyas Gadgin Matha (Product Owner), Claude Code (Implementation)
