# Multi-Agent Code Review & Debugging Pipeline

> **Milestone 3:** Test Generation, Sandboxed Docker Execution & Verified Fix Proposals

An end-to-end, multi-agent Python code review and automated debugging pipeline built with **LangGraph**, **Ruff**, **Bandit**, and **Docker**.

The system combines deterministic static analysis with specialized AI agents to inspect code, model security threats, generate isolated unit tests, execute them safely in isolated Docker sandboxes, and synthesize verified, minimal patches for failing tests without ever modifying the user's primary source tree.

---

## 📋 Table of Contents

- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Key Features](#-key-features)
- [Milestone Progression](#-milestone-progression)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage Guide](#-usage-guide)
  - [CLI Flags & Options](#cli-flags--options)
  - [Example Invocations](#example-invocations)
  - [Running the Test Suite](#running-the-test-suite)
- [How It Works](#-how-it-works)
  - [1. Static Analysis (Milestone 1)](#1-static-analysis-milestone-1)
  - [2. Multi-Agent Review Workflow (Milestone 2)](#2-multi-agent-review-workflow-milestone-2)
  - [3. Test Validation & Storage (Milestone 3)](#3-test-validation--storage-milestone-3)
  - [4. Docker Sandbox Execution (Milestone 3)](#4-docker-sandbox-execution-milestone-3)
  - [5. Debugging Agent & Verified Patch Loop (Milestone 3)](#5-debugging-agent--verified-patch-loop-milestone-3)
- [Sample Demonstrations](#-sample-demonstrations)
- [Roadmap & Upcoming Milestones](#-roadmap--upcoming-milestones)

---

## 🏛 Project Architecture

```
                                  +------------------------------------+
                                  |            Target Source           |
                                  |        (File or Directory)         |
                                  +-----------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         |                                                     |
                         v                                                     v
            +------------------------+                             +------------------------+
            |      Ruff Linter       |                             |     Bandit Scanner     |
            |  (tools.py: run_ruff)  |                             | (tools.py: run_bandit) |
            +------------+-----------+                             +-----------+------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    |
                                      Deterministic Findings
                                                    |
                                                    v
                                  +------------------------------------+
                                  |       LangGraph Orchestrator       |
                                  |            (workflow.py)           |
                                  +-----------------+------------------+
                                                    |
                         +--------------------------+--------------------------+
                         | (Parallel Fan-Out)                                  | (Parallel Fan-Out)
                         v                                                     v
            +------------------------+                             +------------------------+
            |      Linter Agent      |                             |     Security Agent     |
            |     (agents.py)        |                             |      (agents.py)       |
            +------------+-----------+                             +-----------+------------+
                         |                                                     |
                         +--------------------------+--------------------------+
                                                    | (Fan-In Sync)
                                                    v
                                  +------------------------------------+
                                  |         Test-Writer Agent          |
                                  |     - Generates pytest suites      |
                                  |     - test_validator.py            |
                                  |     - Stores in temp workspace     |
                                  +-----------------+------------------+
                                                    |
                                      [enable_sandbox == True?]
                                       /                     \
                                    No/                       \Yes
                                     /                         v
                                    |             +------------------------+
                                    |             |  Docker Sandbox Runner |
                                    |             |   (sandbox_runner.py)  |
                                    |             |  - Read-only source    |
                                    |             |  - Isolated container  |
                                    |             +-----------+------------+
                                    |                         |
                                    |            [Tests Passed or Skipped?]
                                    |                    /         \
                                    |                Yes/           \No (Failed tests)
                                    |                  /             v
                                    |                 |  +------------------------+
                                    |                 |  |    Debugging Agent     |
                                    |                 |  |  - Analyzes fail log   |
                                    |                 |  |  - Minimal unified diff|
                                    |                 |  |  - Verified in copy    |
                                    |                 |  +-----------+------------+
                                    |                 |              |
                                    v                 v              v
                                  +------------------------------------+
                                  |        Orchestrator Report         |
                                  |    - Status & Comprehensive Summary|
                                  |    - Explanations & Test Artifacts |
                                  |    - Sandbox Results & Verified Fix|
                                  +------------------------------------+
```

---

## 📂 Project Structure

```
code review/
├── .gitignore                   # Git ignore patterns for Python, venv, caches, temp tests
├── .venv/                       # Project virtual environment
├── pytest.ini                   # Pytest configuration and pythonpath discovery
├── requirements.txt             # Pinned, compatible dependencies
├── README.md                    # Comprehensive architecture & usage documentation
├── state.py                     # ReviewState TypedDict definitions and reducers
├── tools.py                     # Subprocess execution and output parsing for Ruff & Bandit
├── agents.py                    # LinterAgent, SecurityAgent, TestWriterAgent, DebuggingAgent
├── test_validator.py            # Safety scanning, syntax validation, & temp storage for tests
├── sandbox_runner.py            # Docker-based test runner with timeouts and isolation flags
├── workflow.py                  # LangGraph StateGraph review workflow and state routing
├── main.py                      # Unified CLI entry point and human-readable reporting
├── sample_project/              # Demonstration codebase
│   └── app.py                   # Code containing deliberate lint and security issues
└── tests/                       # Automated test suite (134 unit tests)
    ├── test_tools.py            # Unit tests for Ruff and Bandit subprocess wrappers
    ├── test_agents.py           # Unit tests for Linter, Security, and Test-Writer agents
    ├── test_workflow.py         # Unit tests for Milestone 2 LangGraph workflow
    ├── test_test_validator.py   # Unit tests for test code validator and safe workspace storage
    ├── test_sandbox_runner.py   # Unit tests for Docker sandbox runner (mocked subprocess)
    ├── test_debugging_agent.py  # Unit tests for DebuggingAgent and patch verification
    ├── test_workflow_m3.py      # Unit tests for Milestone 3 orchestration and routing
    ├── test_sandbox_integration.py # Opt-in Docker live container integration tests
    └── test_cli.py              # Unit tests for CLI entry point and UI formatting
```

---

## ✨ Key Features

1. **Subprocess Isolation**: Linters run safely with `shell=False` and explicit argument lists to protect against command injection.
2. **Read-Only Source Integrity**: Target source code is never modified by the pipeline. Patches and tests are isolated in disposable temporary environments.
3. **Multi-Agent Orchestration**: LangGraph coordinates parallel Linter and Security agents, then sequences the Test-Writer, Sandbox Runner, and Debugging agents.
4. **AST & Static Safety Validation for Tests**: All LLM-generated test files undergo strict structural checks (must define test functions) and safety checks (blocks `subprocess`, `os.system`, `eval`, `exec`, write operations, etc.) before disk storage.
5. **Sandboxed Docker Execution**: Executes approved pytest test files against disposable, read-only copies of target code inside an ephemeral container (`--network none`, `--read-only`, capped memory and CPU).
6. **Graceful Degradation**: When Docker is not installed or the daemon is offline, sandbox execution is cleanly skipped with an informative notice; the pipeline continues without crashing.
7. **Automated Debugging Agent**: When tests fail, the Debugging Agent analyzes the failure trace and findings to generate a minimal unified diff patch, which is verified against a disposable copy in the sandbox before presentation.

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- Python 3.9+ (Python 3.11+ recommended)
- Windows, macOS, or Linux
- Docker (optional, required only for live sandbox execution)

### 2. Setup Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure API Credentials (Optional for AI agents)

Create a `.env` file or export your environment variable:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
```

*Note: If no API key is provided, the pipeline automatically falls back to deterministic static scanning with a clear advisory notice.*

### 4. Build Sandbox Docker Image (Required for Docker Sandbox Execution)

The sandboxed execution runner requires a Docker image with a compatible Python runtime and pre-installed `pytest` (`code-review-sandbox:3.11`). Build the pinned image from the project's Dockerfile:

```powershell
docker build -t code-review-sandbox:3.11 -t python:3.11-slim .
```

*Note: The image name can be customized via the `SANDBOX_IMAGE` environment variable.*

---

## 🚀 Usage Guide

### CLI Flags & Options

```
python main.py [target_path] [options]

Positional Arguments:
  target_path            Path to Python file or directory (defaults to sample_project/app.py)

Options:
  --no-ai                Run only deterministic static scanners (skips LLM agents)
  --no-sandbox           Skip Docker sandbox execution of generated tests
  --no-debug             Skip Debugging Agent patch proposal loop
  --max-debug-attempts N Maximum repair attempts for Debugging Agent (default: 2)
  --test-dir PATH        Custom directory for storing generated tests (default: temp dir)
  --model MODEL_NAME     Specify LLM model (overrides OPENAI_MODEL)
```

### Example Invocations

```powershell
# Full pipeline with Docker sandbox and debugging agent
python main.py sample_project/app.py

# Run static analysis and AI review without Docker sandbox
python main.py sample_project/app.py --no-sandbox

# Pure deterministic scan (no LLM, no network, no Docker)
python main.py sample_project/app.py --no-ai
```

### Running the Test Suite

All 134 automated unit tests run completely offline and use mocked runners (no external API calls or Docker required):

```powershell
pytest -v
```

To run the live Docker integration tests (when Docker daemon is running and sandbox image is built):

```powershell
docker build -t code-review-sandbox:3.11 -t python:3.11-slim .
$env:SANDBOX_INTEGRATION_TESTS="1"
pytest tests/test_sandbox_integration.py -v
```

---

## 🔍 How It Works

### 1. Static Analysis (Milestone 1)
- **Ruff**: Inspects Python code for PEP 8 compliance, unused imports (`F401`), unused variables (`F841`), and code quality issues.
- **Bandit**: Analyzes Python AST for security vulnerabilities (`B101` assert usage, `B105` hardcoded passwords, `B311` weak PRNGs).

### 2. Multi-Agent Review Workflow (Milestone 2)
- StateGraph runs **LinterAgent** and **SecurityAgent** concurrently using fan-out edges from `START`.
- Both agents independently evaluate findings and provide structured explanations, severity assessments, and threat-modeling context.
- Agents include explicit advisory disclaimers noting that AI explanations represent hypotheses and that static scans do not guarantee runtime security.

### 3. Test Validation & Storage (Milestone 3)
- **TestWriterAgent** generates targeted pytest unit tests for functions under review.
- **`test_validator.py`** inspects the test content:
  - Verifies presence of `test_` functions or `Test` classes.
  - Verifies syntactical validity via `ast.parse`.
  - Rejects dangerous calls (`subprocess`, `os.system`, `exec`, `eval`, arbitrary file writes).
  - Enforces size limits ($\le 50$ KB).
- Approved tests are written to a dedicated temporary test workspace (`.test_workspace/` or OS temp directory), never into the primary project folder.

### 4. Docker Sandbox Execution (Milestone 3)
- **`sandbox_runner.py`** executes tests inside a hardened container:
  - Read-only source mounting (`:ro`)
  - Container-isolated working directory
  - Strict resource constraints (`--memory 512m`, `--cpus 1.0`, `--pids-limit 100`)
  - Network isolation (`--network none`)
  - Configurable execution timeout (default: 60s)
- Returns structured results with return codes, pytest stdout/stderr, duration, and status.

### 5. Debugging Agent & Verified Patch Loop (Milestone 3)
- If sandbox tests fail, **DebuggingAgent** analyzes the source code, static findings, and pytest failure logs.
- Synthesizes a minimal unified diff patch.
- Verifies the patch against a disposable copy of the target in a scratch sandbox before presenting it to the user.
- Enforces bounded retry limits (`--max-debug-attempts`, default: 2) to prevent infinite loops.

---

## 🧪 Sample Demonstrations

The file [`sample_project/app.py`](sample_project/app.py) contains deliberate demonstrations:
- **Ruff Issues**: Unused imports (`os`, `math`), unused variables (`unused_debug_tag`), unsorted imports.
- **Bandit Warnings**: Use of `assert` in business logic (`B101`), hardcoded token string (`B105`), weak pseudo-random generator for tokens (`B311`).

---

## 🗺 Roadmap & Upcoming Milestones

- [x] **Milestone 1:** Automated Local Static Analysis & Security Review Pipeline (Ruff + Bandit)
- [x] **Milestone 2:** Multi-Agent AI Review using LangGraph (Linter, Security, Test-Writer Agents)
- [x] **Milestone 3:** Test Generation, Docker Sandboxed Execution & Verified Fix Proposals
- [ ] **Milestone 4:** GitHub Integration (PR review automation, inline GitHub suggestions, CI/CD webhooks)
