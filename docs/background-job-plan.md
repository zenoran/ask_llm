# Background Job Scheduler Framework - Implementation Plan

## Overview

Add a recurring job scheduler to run maintenance tasks on configurable intervals. The first job will be **profile summarization** - using an LLM to consolidate redundant/contradictory user profile attributes into clean summaries for system prompts.

## Architecture

```
JobScheduler (async watcher loop)
    ↓ checks scheduled_jobs table for due jobs
    ↓ creates Task via factory
    ↓ enqueues to TaskProcessor
TaskProcessor (existing)
    ↓ processes task
    ↓ calls ProfileMaintenanceService
JobScheduler
    ↓ records result in job_runs table
```

---

## Task 1: Add Configuration Settings

**File**: `src/ask_llm/utils/config.py`

**Add these fields to the `Config` class** (around line 50-100, near other settings):

```python
# Scheduler settings
SCHEDULER_ENABLED: bool = Field(default=True, description="Enable background job scheduler")
SCHEDULER_CHECK_INTERVAL_SECONDS: int = Field(default=30, description="How often to check for due jobs")
PROFILE_MAINTENANCE_INTERVAL_MINUTES: int = Field(default=60, description="Run profile summarization every N minutes")
PROFILE_MAINTENANCE_MODEL: str = Field(default="", description="Model to use for profile summarization (empty = default)")
```

---

## Task 2: Create Scheduler Database Models

**File**: `src/ask_llm/service/scheduler.py` (NEW FILE)

Create SQLModel tables for job definitions and run history:

```python
"""Background job scheduler with database persistence."""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel, Session, select, create_engine
from sqlalchemy import Column, DateTime, Text

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a job run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobType(str, Enum):
    """Types of scheduled jobs."""
    PROFILE_MAINTENANCE = "profile_maintenance"
    MEMORY_CONSOLIDATION = "memory_consolidation"
    MEMORY_DECAY = "memory_decay"


class ScheduledJob(SQLModel, table=True):
    """Definition of a recurring job."""
    __tablename__ = "scheduled_jobs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_type: JobType = Field(index=True)
    bot_id: str = Field(index=True, description="Bot this job runs for, or '*' for all")
    enabled: bool = Field(default=True)
    interval_minutes: int = Field(default=60)
    last_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    next_run_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    config_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Job-specific config as JSON")


class JobRun(SQLModel, table=True):
    """History of job executions."""
    __tablename__ = "job_runs"
    
    id: str = Field(default_factory=lambda: str(uuid4()), primary_key=True)
    job_id: str = Field(index=True, foreign_key="scheduled_jobs.id")
    bot_id: str = Field(index=True)
    status: JobStatus = Field(default=JobStatus.PENDING)
    started_at: datetime = Field(default_factory=datetime.utcnow, sa_column=Column(DateTime(timezone=True)))
    finished_at: Optional[datetime] = Field(default=None, sa_column=Column(DateTime(timezone=True)))
    duration_ms: Optional[int] = Field(default=None)
    result_json: Optional[str] = Field(default=None, sa_column=Column(Text), description="Success result as JSON")
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))


def create_scheduler_tables(engine) -> None:
    """Create scheduler tables if they don't exist."""
    SQLModel.metadata.create_all(engine, tables=[ScheduledJob.__table__, JobRun.__table__])
```

---

## Task 3: Implement Profile Maintenance Service

**File**: `src/ask_llm/memory/profile_maintenance.py` (NEW FILE)

This service uses LLM to consolidate redundant profile attributes:

```python
"""LLM-based profile attribute consolidation and summarization."""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from ask_llm.profiles import EntityProfileManager, ProfileCategory
from ask_llm.clients.base import LLMClient

logger = logging.getLogger(__name__)

CONSOLIDATION_PROMPT = '''You are a data cleanup assistant. Given a list of user profile attributes, consolidate them by:

1. MERGE duplicates (same meaning, different wording)
2. RESOLVE contradictions (keep the most recent/specific)
3. REMOVE noise (temporary states, greetings, one-time actions)
4. KEEP important facts (name, location, occupation, preferences, interests)

Input attributes (category: key = value):
{attributes}

Output a JSON object with consolidated attributes per category:
{{
  "preference": {{"key": "value", ...}},
  "fact": {{"key": "value", ...}},
  "interest": {{"key": "value", ...}},
  "communication": {{"key": "value", ...}},
  "health": {{"key": "value", ...}},
  "misc": {{"key": "value", ...}}
}}

Rules:
- Use snake_case keys (e.g., "home_location" not "location")
- Values should be concise statements
- Omit empty categories
- Prefer specific over vague (e.g., "lives in Ohio" over "has a location")
'''


@dataclass
class ProfileMaintenanceResult:
    """Result of profile maintenance run."""
    entity_id: str
    attributes_before: int
    attributes_after: int
    categories_updated: list[str]
    error: Optional[str] = None


class ProfileMaintenanceService:
    """Service to consolidate and clean up user profile attributes."""
    
    def __init__(self, profile_manager: EntityProfileManager, llm_client: LLMClient):
        self.profile_manager = profile_manager
        self.llm_client = llm_client
    
    def run(self, entity_id: str, entity_type: str = "user", dry_run: bool = False) -> ProfileMaintenanceResult:
        """
        Consolidate profile attributes for an entity.
        
        Args:
            entity_id: User or bot ID
            entity_type: "user" or "bot"
            dry_run: If True, don't save changes
            
        Returns:
            ProfileMaintenanceResult with before/after counts
        """
        # Get current attributes
        attributes = self.profile_manager.get_attributes(entity_id, entity_type)
        if not attributes:
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=0,
                attributes_after=0,
                categories_updated=[]
            )
        
        attributes_before = len(attributes)
        
        # Format for LLM
        attr_lines = []
        for attr in attributes:
            attr_lines.append(f"{attr.category}: {attr.key} = {attr.value}")
        
        prompt = CONSOLIDATION_PROMPT.format(attributes="\n".join(attr_lines))
        
        try:
            # Call LLM
            response = self.llm_client.query(prompt)
            
            # Parse JSON from response
            consolidated = self._parse_json_response(response)
            if not consolidated:
                return ProfileMaintenanceResult(
                    entity_id=entity_id,
                    attributes_before=attributes_before,
                    attributes_after=attributes_before,
                    categories_updated=[],
                    error="Failed to parse LLM response as JSON"
                )
            
            if dry_run:
                # Count what would be created
                after_count = sum(len(v) for v in consolidated.values() if isinstance(v, dict))
                return ProfileMaintenanceResult(
                    entity_id=entity_id,
                    attributes_before=attributes_before,
                    attributes_after=after_count,
                    categories_updated=list(consolidated.keys())
                )
            
            # Delete old attributes and create new ones
            categories_updated = []
            for category_name, attrs in consolidated.items():
                if not isinstance(attrs, dict) or not attrs:
                    continue
                    
                # Map string to enum
                try:
                    category = ProfileCategory(category_name)
                except ValueError:
                    category = ProfileCategory.MISC
                
                # Delete existing in this category
                self.profile_manager.delete_attributes_by_category(entity_id, entity_type, category)
                
                # Create consolidated attributes
                for key, value in attrs.items():
                    self.profile_manager.set_attribute(
                        entity_id=entity_id,
                        entity_type=entity_type,
                        category=category,
                        key=key,
                        value=value,
                        confidence=0.9,  # High confidence - LLM consolidated
                        source="maintenance"
                    )
                
                categories_updated.append(category_name)
            
            # Get new count
            new_attributes = self.profile_manager.get_attributes(entity_id, entity_type)
            
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=attributes_before,
                attributes_after=len(new_attributes),
                categories_updated=categories_updated
            )
            
        except Exception as e:
            logger.exception(f"Profile maintenance failed for {entity_id}")
            return ProfileMaintenanceResult(
                entity_id=entity_id,
                attributes_before=attributes_before,
                attributes_after=attributes_before,
                categories_updated=[],
                error=str(e)
            )
    
    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Extract JSON from LLM response."""
        import re
        
        # Look for ```json ... ``` block
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON object
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        return None
```

---

## Task 4: Add Task Type and Factory

**File**: `src/ask_llm/service/tasks.py`

**Add to `TaskType` enum**:
```python
PROFILE_MAINTENANCE = "profile_maintenance"
```

**Add factory function** (after existing `create_*_task` functions):
```python
def create_profile_maintenance_task(
    entity_id: str,
    entity_type: str = "user",
    bot_id: str = "nova",
    dry_run: bool = False,
    priority: int = 5,
) -> Task:
    """Create a profile maintenance task."""
    return Task(
        task_type=TaskType.PROFILE_MAINTENANCE,
        bot_id=bot_id,
        priority=priority,
        payload={
            "entity_id": entity_id,
            "entity_type": entity_type,
            "dry_run": dry_run,
        },
    )
```

---

## Task 5: Add Task Handler to TaskProcessor

**File**: `src/ask_llm/service/api.py`

**In the `process_task` method**, add a case for the new task type:

```python
# Add import at top
from ask_llm.memory.profile_maintenance import ProfileMaintenanceService

# In process_task method, add this case:
elif task.task_type == TaskType.PROFILE_MAINTENANCE:
    result = await self._process_profile_maintenance(task)
```

**Add the handler method** (in `TaskProcessor` class):

```python
async def _process_profile_maintenance(self, task: Task) -> dict:
    """Process profile maintenance task."""
    entity_id = task.payload.get("entity_id")
    entity_type = task.payload.get("entity_type", "user")
    dry_run = task.payload.get("dry_run", False)
    
    loop = asyncio.get_event_loop()
    
    def run_maintenance():
        from ask_llm.profiles import EntityProfileManager
        from ask_llm.utils.config import config
        
        profile_manager = EntityProfileManager(config.postgres_url)
        service = ProfileMaintenanceService(profile_manager, self.llm_client)
        return service.run(entity_id, entity_type, dry_run)
    
    result = await loop.run_in_executor(None, run_maintenance)
    
    return {
        "entity_id": result.entity_id,
        "attributes_before": result.attributes_before,
        "attributes_after": result.attributes_after,
        "categories_updated": result.categories_updated,
        "error": result.error,
    }
```

---

## Task 6: Implement Job Scheduler

**File**: `src/ask_llm/service/scheduler.py`

**Add the `JobScheduler` class** (append to the file created in Task 2):

```python
class JobScheduler:
    """
    Async scheduler that watches for due jobs and enqueues them.
    
    Usage:
        scheduler = JobScheduler(engine, task_processor)
        await scheduler.start()  # Runs until stopped
        await scheduler.stop()
    """
    
    def __init__(self, engine, task_processor, check_interval: int = 30):
        """
        Args:
            engine: SQLAlchemy engine for job tables
            task_processor: TaskProcessor instance to enqueue tasks
            check_interval: Seconds between checking for due jobs
        """
        self.engine = engine
        self.task_processor = task_processor
        self.check_interval = check_interval
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"JobScheduler started (check interval: {self.check_interval}s)")
    
    async def stop(self) -> None:
        """Stop the scheduler loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("JobScheduler stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run_due_jobs()
            except Exception as e:
                logger.exception(f"Scheduler loop error: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def _check_and_run_due_jobs(self) -> None:
        """Check for due jobs and enqueue them."""
        loop = asyncio.get_event_loop()
        
        def get_due_jobs():
            with Session(self.engine) as session:
                now = datetime.utcnow()
                statement = select(ScheduledJob).where(
                    ScheduledJob.enabled == True,
                    (ScheduledJob.next_run_at == None) | (ScheduledJob.next_run_at <= now)
                )
                return session.exec(statement).all()
        
        due_jobs = await loop.run_in_executor(None, get_due_jobs)
        
        for job in due_jobs:
            await self._execute_job(job)
    
    async def _execute_job(self, job: ScheduledJob) -> None:
        """Execute a single job."""
        logger.info(f"Running job: {job.job_type} for bot {job.bot_id}")
        
        loop = asyncio.get_event_loop()
        
        # Create job run record
        def create_run():
            with Session(self.engine) as session:
                run = JobRun(
                    job_id=job.id,
                    bot_id=job.bot_id,
                    status=JobStatus.RUNNING,
                )
                session.add(run)
                session.commit()
                session.refresh(run)
                return run.id
        
        run_id = await loop.run_in_executor(None, create_run)
        start_time = datetime.utcnow()
        
        try:
            # Create and execute task based on job type
            task = self._create_task_for_job(job)
            if task:
                result = await self.task_processor.process_task(task)
                status = JobStatus.SUCCESS if result.get("error") is None else JobStatus.FAILED
                result_json = json.dumps(result)
                error_message = result.get("error")
            else:
                status = JobStatus.SKIPPED
                result_json = None
                error_message = f"Unknown job type: {job.job_type}"
        except Exception as e:
            status = JobStatus.FAILED
            result_json = None
            error_message = str(e)
            logger.exception(f"Job {job.id} failed")
        
        # Update job run record
        finish_time = datetime.utcnow()
        duration_ms = int((finish_time - start_time).total_seconds() * 1000)
        
        def update_run():
            with Session(self.engine) as session:
                run = session.get(JobRun, run_id)
                if run:
                    run.status = status
                    run.finished_at = finish_time
                    run.duration_ms = duration_ms
                    run.result_json = result_json
                    run.error_message = error_message
                    session.add(run)
                
                # Update job's next_run_at
                db_job = session.get(ScheduledJob, job.id)
                if db_job:
                    db_job.last_run_at = start_time
                    db_job.next_run_at = start_time + timedelta(minutes=db_job.interval_minutes)
                    session.add(db_job)
                
                session.commit()
        
        await loop.run_in_executor(None, update_run)
        logger.info(f"Job {job.job_type} completed with status {status}")
    
    def _create_task_for_job(self, job: ScheduledJob):
        """Create a Task from a ScheduledJob."""
        from ask_llm.service.tasks import create_profile_maintenance_task, create_maintenance_task
        
        if job.job_type == JobType.PROFILE_MAINTENANCE:
            # For profile maintenance, entity_id comes from config or is the bot's user
            config = json.loads(job.config_json) if job.config_json else {}
            entity_id = config.get("entity_id", "nick")  # Default user
            return create_profile_maintenance_task(
                entity_id=entity_id,
                entity_type=config.get("entity_type", "user"),
                bot_id=job.bot_id,
            )
        elif job.job_type == JobType.MEMORY_CONSOLIDATION:
            return create_maintenance_task(
                bot_id=job.bot_id,
                run_consolidation=True,
                run_recurrence_detection=False,
                run_decay_pruning=False,
            )
        elif job.job_type == JobType.MEMORY_DECAY:
            return create_maintenance_task(
                bot_id=job.bot_id,
                run_consolidation=False,
                run_recurrence_detection=False,
                run_decay_pruning=True,
            )
        
        return None


def init_default_jobs(engine, config) -> None:
    """Initialize default scheduled jobs if they don't exist."""
    with Session(engine) as session:
        # Check if profile maintenance job exists
        existing = session.exec(
            select(ScheduledJob).where(ScheduledJob.job_type == JobType.PROFILE_MAINTENANCE)
        ).first()
        
        if not existing:
            job = ScheduledJob(
                job_type=JobType.PROFILE_MAINTENANCE,
                bot_id="*",  # All bots
                enabled=config.SCHEDULER_ENABLED,
                interval_minutes=config.PROFILE_MAINTENANCE_INTERVAL_MINUTES,
                config_json=json.dumps({"entity_id": "nick", "entity_type": "user"}),
            )
            session.add(job)
            session.commit()
            logger.info("Created default profile maintenance job")
```

---

## Task 7: Integrate Scheduler into FastAPI Service

**File**: `src/ask_llm/service/api.py`

**Add imports at top**:
```python
from ask_llm.service.scheduler import JobScheduler, create_scheduler_tables, init_default_jobs
```

**Modify the lifespan context manager** (or startup/shutdown events):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - start/stop background tasks."""
    from ask_llm.utils.config import config
    from sqlmodel import create_engine
    
    # Create scheduler tables
    engine = create_engine(config.postgres_url)
    create_scheduler_tables(engine)
    
    # Initialize default jobs
    init_default_jobs(engine, config)
    
    # Start task processor (existing)
    task_processor = TaskProcessor(...)
    await task_processor.start()
    
    # Start job scheduler
    scheduler = None
    if config.SCHEDULER_ENABLED:
        scheduler = JobScheduler(
            engine=engine,
            task_processor=task_processor,
            check_interval=config.SCHEDULER_CHECK_INTERVAL_SECONDS,
        )
        await scheduler.start()
    
    yield
    
    # Shutdown
    if scheduler:
        await scheduler.stop()
    await task_processor.stop()
```

---

## Task 8: Add Profile Manager Helper Method

**File**: `src/ask_llm/profiles.py`

**Add method to `EntityProfileManager`** for deleting attributes by category:

```python
def delete_attributes_by_category(
    self, 
    entity_id: str, 
    entity_type: str, 
    category: ProfileCategory
) -> int:
    """Delete all attributes in a category for an entity. Returns count deleted."""
    with Session(self.engine) as session:
        # Get profile
        profile = self._get_or_create_profile(session, entity_id, entity_type)
        
        # Find and delete attributes
        statement = select(EntityProfileAttribute).where(
            EntityProfileAttribute.profile_id == profile.id,
            EntityProfileAttribute.category == category,
        )
        attributes = session.exec(statement).all()
        count = len(attributes)
        
        for attr in attributes:
            session.delete(attr)
        
        session.commit()
        return count
```

---

## Testing

### Manual Test Commands

```bash
# 1. Verify tables are created
llm-service  # Start service, check logs for "Created default profile maintenance job"

# 2. Check job in database
psql -d ask_llm -c "SELECT * FROM scheduled_jobs;"

# 3. Trigger immediate run (set next_run_at to past)
psql -d ask_llm -c "UPDATE scheduled_jobs SET next_run_at = NOW() - INTERVAL '1 minute';"

# 4. Watch logs for job execution
# Look for "Running job: profile_maintenance" and "Job profile_maintenance completed"

# 5. Verify profile was consolidated
llm --bot nova "show my profile"
```

### Unit Test (Optional)

**File**: `tests/test_scheduler.py`

```python
import pytest
from datetime import datetime, timedelta
from ask_llm.service.scheduler import ScheduledJob, JobType, JobStatus

def test_scheduled_job_creation():
    job = ScheduledJob(
        job_type=JobType.PROFILE_MAINTENANCE,
        bot_id="test",
        interval_minutes=30,
    )
    assert job.enabled is True
    assert job.interval_minutes == 30

def test_job_due_calculation():
    job = ScheduledJob(
        job_type=JobType.PROFILE_MAINTENANCE,
        bot_id="test",
        interval_minutes=60,
        last_run_at=datetime.utcnow() - timedelta(hours=2),
    )
    # Job should be due (last run was 2 hours ago, interval is 1 hour)
    assert job.next_run_at is None or job.next_run_at <= datetime.utcnow()
```

---

## File Summary

| File | Action | Description |
|------|--------|-------------|
| `src/ask_llm/utils/config.py` | MODIFY | Add scheduler config settings |
| `src/ask_llm/service/scheduler.py` | CREATE | Job models + JobScheduler class |
| `src/ask_llm/memory/profile_maintenance.py` | CREATE | ProfileMaintenanceService with LLM consolidation |
| `src/ask_llm/service/tasks.py` | MODIFY | Add PROFILE_MAINTENANCE task type + factory |
| `src/ask_llm/service/api.py` | MODIFY | Add task handler + scheduler startup/shutdown |
| `src/ask_llm/profiles.py` | MODIFY | Add delete_attributes_by_category method |

---

## Implementation Order

1. **Task 1** - Config (no dependencies)
2. **Task 2** - Scheduler models (no dependencies)  
3. **Task 8** - Profile manager helper (no dependencies)
4. **Task 3** - Profile maintenance service (depends on Task 8)
5. **Task 4** - Task type/factory (no dependencies)
6. **Task 5** - Task handler (depends on Tasks 3, 4)
7. **Task 6** - JobScheduler class (depends on Tasks 2, 4)
8. **Task 7** - Integration (depends on all above)

---

## Notes for Implementer

- All database operations use **synchronous** SQLModel/SQLAlchemy - wrap in `run_in_executor` for async contexts
- Follow existing patterns in `memory/maintenance.py` for the service class structure
- The profile consolidation prompt may need tuning based on actual results
- Job runs are recorded for debugging - consider adding a cleanup job later to prune old runs
- The `entity_id` for profile maintenance is currently hardcoded to "nick" - this should eventually be configurable per bot or discovered from active users
