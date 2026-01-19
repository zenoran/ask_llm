## Summary: Comprehensive ask_llm Refactoring

### Completed Tasks

1. **Deleted stale build/ directory** - Removed build artifact that was committed before being added to .gitignore

2. **Deleted legacy user_profile.py** - Removed the 268-line deprecated module (superseded by profiles.py)

3. **Fixed dead code in cli.py** - Removed unreachable `return None` after `return True`, fixed duplicate `prelim_parser` definition

4. **Created `core/` package structure** with modular components:
   - __init__.py - Package exports
   - `core/prompt_builder.py` - Template-based system prompt assembly
   - pipeline.py - Modular request processing with clear stages
   - base.py - Shared `BaseAskLLM` class
   - client.py - CLI-mode `AskLLM` class

5. **Created PromptBuilder class** (`core/prompt_builder.py`):
   - Named, ordered sections with positions
   - `SectionPosition` constants (USER_CONTEXT=0, BOT_TRAITS=1, BASE_PROMPT=2, MEMORY_CONTEXT=3, TOOLS=4)
   - `get_verbose_summary()` for `--verbose` output
   - Method chaining support
   - Debug info via `build_with_debug()`

6. **Created RequestPipeline class** (pipeline.py):
   - 7 clear stages: PRE_PROCESS → CONTEXT_BUILD → MEMORY_RETRIEVAL → HISTORY_FILTER → MESSAGE_ASSEMBLY → EXECUTE → POST_PROCESS
   - `PipelineContext` dataclass for shared state
   - Hook system for extensibility
   - Decision point overrides
   - Timing instrumentation per stage

7. **Created BaseAskLLM class** (base.py):
   - Extracted ~70% shared logic from both AskLLM and ServiceAskLLM
   - Abstract `_initialize_client()` for subclass implementation
   - Integrated PromptBuilder for system prompt assembly
   - `verbose` and `debug` flags for logging control

8. **Refactored AskLLM** (client.py):
   - Now extends BaseAskLLM
   - Only implements `_initialize_client()` for OpenAI-compatible APIs
   - ~90 lines down from ~420 lines

9. **Refactored ServiceAskLLM** (core.py):
   - Now extends BaseAskLLM
   - Adds GGUF model support via `_initialize_client()`
   - ~250 lines down from ~539 lines

10. **Enhanced logging levels**:
    - `--verbose`: Rich INFO-level output with PromptBuilder structure
    - `--debug`: Full DEBUG spam with API payloads, timing, sections
    - `LogConfig.configure()` already supported both flags

11. **Split cli.py into package**:
    - __init__.py - Re-exports from cli_legacy.py during transition
    - `cli/parser.py` - Argument parsing module (new implementation)
    - main.py - Entry point wrapper
    - `cli/commands/` - Command subpackages (status, profile, models)
    - Original cli.py → cli_legacy.py (gradual migration)

### New File Structure

```
src/ask_llm/
├── core/                      # NEW: Core processing package
│   ├── __init__.py           # Exports AskLLM, PromptBuilder, RequestPipeline
│   ├── base.py               # BaseAskLLM shared logic
│   ├── client.py             # AskLLM for CLI (OpenAI-only)
│   ├── pipeline.py           # RequestPipeline with 7 stages
│   └── prompt_builder.py     # PromptBuilder with named sections
├── cli/                       # NEW: CLI package
│   ├── __init__.py           # Re-exports from cli_legacy
│   ├── main.py               # Entry point wrapper
│   ├── parser.py             # Argument parsing
│   └── commands/             # Command implementations
│       ├── __init__.py
│       ├── models.py
│       ├── profile.py
│       └── status.py
├── cli_legacy.py             # Renamed from cli.py (gradual migration)
├── service/
│   └── core.py               # ServiceAskLLM now extends BaseAskLLM
└── ... (other unchanged files)
```

### Key Improvements

1. **Pipeline Transparency**: The `--verbose` flag now shows prompt structure with section names and sizes
2. **Extensibility**: `RequestPipeline` hooks allow adding custom processing at any stage
3. **DRY**: ~350 lines of duplicate code eliminated between AskLLM and ServiceAskLLM
4. **Testability**: Each pipeline stage is independently testable
5. **Debugging**: `--debug` shows full API payloads, timing per stage, and decision logic

Made changes.