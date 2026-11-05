# Multi-Agent Code Review & Debugging Pipeline

> **Milestone 4:** GitHub Pull Request Review & Controlled Review Publishing

An end-to-end, multi-agent Python code review and automated debugging pipeline built with **LangGraph**, **Ruff**, **Bandit**, **Docker**, and the **GitHub REST API**.

The system combines deterministic static analysis with specialized AI agents to inspect code, model security threats, generate isolated unit tests, execute them safely in isolated Docker sandboxes, and synthesize verified, minimal patches for failing tests. In Milestone 4, the pipeline extends seamlessly to GitHub Pull Requests, offering read-only diff inspection, dry-run simulations, and human-gated review publishing with strict credential protection.

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
  - [6. GitHub PR Review & Controlled Publication (Milestone 4)](#6-github-pr-review--controlled-publication-milestone-4)
- [Sample Demonstrations](#-sample-demonstrations)
- [Roadmap & Completed Milestones](#-roadmap--completed-milestones)

---

## 🏛 Project Architecture

```
                                  +------------------------------------+
                                  |    Target Source or GitHub PR      |
                                  |   (Local Path or --pr owner/repo)  |
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
                                  |        Review Action Plan          |
                                  |   - Top-level Markdown summary     |
                                  |   - Validated inline comments      |
                                  |   - Verdict: COMMENT/APPROVE/REQ   |
                                  +-----------------+------------------+
                                                    |
                                        [--publish enabled?]
                                         /                \
                                      No/                  \Yes
                                       v                    v
                        +----------------------+    +--------------------------+
                        |  Read-Only Preview   |    |    Human Approval Gate   |
                        |  or --dry-run Output |    | - Plan-hash verification |
                        +----------------------+    | - PR head-SHA check      |
                                                    | - Interactive prompt     |
                                                    +-------------+------------+
                                                                  | [Approved?]
                                                                  v
                                                    +--------------------------+
                                                    |  GitHub REST Publication |
                                                    |  (github_client.py)      |
                                                    +--------------------------+
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
├── agents.py                    # LinterAgent, SecurityAgent, TestWriterAgent
├── debugging_agent.py           # DebuggingAgent: minimal patch synthesis & verification
├── test_validator.py            # Safety scanning, syntax validation, & temp storage for tests
├── sandbox_runner.py            # Hardened Docker runner with timeouts and cleanup guards
├── workflow.py                  # LangGraph StateGraph review workflow and state routing
├── github_client.py             # GitHub REST API client, diff parser, & credential defense
├── github_review_workflow.py    # PR review orchestration, action planning, & publishing
├── approval_gate.py             # Cryptographic plan-hash integrity & human approval gate
├── pr_workspace.py              # Isolated PR branch checkout & patch workspace manager
├── semantic_analyzer.py         # Interprocedural static None and call-graph analyzer
├── main.py                      # Unified CLI entry point and human-readable reporting
├── sample_project/              # Demonstration codebase
│   └── app.py                   # Code containing deliberate lint and security issues
└── tests/                       # Automated test suite (816 passed, 5 skipped)
    ├── test_tools.py            # Subprocess wrapper and option injection hardening tests
    ├── test_agents.py           # Linter, Security, and Test-Writer agent unit tests
    ├── test_workflow.py         # Local multi-agent LangGraph orchestration tests
    ├── test_test_validator.py   # Test syntax and defense-in-depth safety checks
    ├── test_sandbox_runner.py   # Docker sandbox execution and timeout cleanup tests
    ├── test_debugging_agent.py  # Patch synthesis, verification, & temp directory cleanup
    ├── test_github_client.py    # REST client, token resolution, & credential redaction
    ├── test_github_review_workflow.py # PR review planning & publication workflow tests
    ├── test_approval_gate.py    # Plan-hash integrity and interactive approval tests
    ├── test_controlled_publishing.py # Publication safeguards & dry-run validation tests
    ├── test_pr_workspace.py     # PR workspace isolation and checkout tests
    ├── test_semantic_analyzer.py# Static None-safety and call-graph analysis tests
    └── test_cli.py              # CLI entry point, flag handling, & deprecation tests
```

---

## ✨ Key Features

1. **Subprocess Isolation**: Linters run safely with `shell=False` and explicit argument lists separated by `--` to prevent option and command injection.
2. **Read-Only Source Integrity**: Target source code is never modified by the pipeline. Patches and tests are isolated in disposable temporary environments.
3. **Multi-Agent Orchestration**: LangGraph coordinates parallel Linter and Security agents, then sequences the Test-Writer, Sandbox Runner, and Debugging agents.
4. **AST & Static Safety Validation for Tests**: All LLM-generated test files undergo structural checks (must define test functions) and defense-in-depth safety screening (blocks `subprocess`, `socket`, `ctypes`, `os.system`, `eval`, `exec`, write operations, etc.) before storage.
5. **Sandboxed Docker Execution**: Executes approved pytest test files against disposable, read-only copies of target code inside an ephemeral container (`--network none`, `--read-only`, capped memory and CPU, bounded timeout, guaranteed cleanup).
6. **Graceful Degradation & Deterministic Mode**: When Docker is offline or disabled via `--no-sandbox`, sandbox execution is skipped cleanly. When API keys are absent or `--no-ai` is supplied, deterministic static analysis runs with full reporting.
7. **Automated Debugging Agent**: When tests fail, the Debugging Agent synthesizes a minimal unified diff patch, verified against a disposable copy in the sandbox before presentation. Can be disabled via `--no-debug`.
8. **GitHub Pull Request Review Automation**: Deep inspection of GitHub pull requests via `--pr` and `--repo`, with unified diff parsing that maps findings strictly to changed lines.
9. **Controlled Publication & Human Approval Gate**: Review publication (`--publish`) requires interactive human confirmation and enforces cryptographic plan-hash verification and PR head-SHA staleness checks before posting.
10. **Safe Dry-Run Simulations**: `--dry-run` previews complete top-level summaries and inline comment payloads in memory without making any mutating API calls.
11. **Comprehensive Credential Defense**: All tokens, whether loaded from environment variables, files, or instances, are globally scrubbed from logs, errors, and LLM prompts.

---

## ⚙️ Prerequisites & Installation

### 1. Requirements
- Python 3.9+ (Python 3.11+ recommended; developed with Python 3.13)
- Windows, macOS, or Linux
- Docker (optional, required only for live container sandbox execution)

### 2. Setup Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Configure Credentials

Create a `.env` file or export environment variables:

```env
# Optional: For AI Agent analysis, explanations, test generation, and patch proposals
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini

# Optional: For GitHub Pull Request reviews and publishing (Milestone 4)
GITHUB_TOKEN=ghp_your_github_token_here
```

> **Credential Safety & Token Precedence:**
> - The pipeline resolves GitHub credentials with the following precedence:
>   1. Explicit parameter / CLI `--token` (highest priority; deprecated for security).
>   2. `GITHUB_TOKEN` environment variable.
>   3. `GH_TOKEN` environment variable.
>   4. `GITHUB_TOKEN_FILE` environment variable.
> - All tokens are automatically redacted across all logs, exception traces, and terminal output.

### 4. Build Sandbox Docker Image (Optional)

The sandboxed execution runner uses a Docker image with Python and pre-installed `pytest` (`code-review-sandbox:3.11`). Build the pinned image from the project's Dockerfile:

```powershell
docker build -t code-review-sandbox:3.11 -t python:3.11-slim .
```

*Note: If Docker is unavailable or `--no-sandbox` is provided, sandbox execution is safely skipped without pipeline interruption.*

---

## 🚀 Usage Guide

### CLI Flags & Options

```
python main.py [target] [options]

Positional Arguments:
  target                 Path to Python file or directory to review (default: 'sample_project')

Options:
  -h, --help             Show this help message and exit
  --pr PR                GitHub PR to review in read-only preview mode (e.g., 'owner/repo#123' or PR URL)
  --repo REPO            GitHub repository name if --pr is only the PR number (e.g., 'owner/repo')
  --publish              Publish the review to GitHub PR after interactive human approval
  --dry-run              Simulate review publishing without performing any write requests
  --no-ai                Disable AI analysis and use only deterministic Ruff and Bandit scanners
  --no-sandbox           Disable Docker sandbox execution even when Docker is available
  --no-debug             Disable the Debugging Agent (no patch proposals)
  --token TOKEN          GitHub token (DEPRECATED for security: prefer GITHUB_TOKEN or GH_TOKEN env var)
```

### Example Invocations

#### 1. Local Code Review
```powershell
# Full pipeline with AI review, Docker sandbox, and Debugging Agent
python main.py sample_project/app.py

# Deterministic local scan (no LLM, no external calls)
python main.py sample_project/app.py --no-ai

# Local review skipping Docker sandbox and patch generation
python main.py sample_project/app.py --no-sandbox --no-debug
```

#### 2. GitHub Pull Request Review (Read-Only Preview)
```powershell
# Analyze a PR using full owner/repo#number format
python main.py --pr octocat/Hello-World#42

# Analyze a PR using separate --pr and --repo flags
python main.py --pr 42 --repo octocat/Hello-World

# Review PR using a direct URL
python main.py --pr https://github.com/octocat/Hello-World/pull/42

# Deterministic PR review (Ruff & Bandit on changed files, zero LLM calls)
python main.py --pr octocat/Hello-World#42 --no-ai
```

#### 3. Safe Dry-Run Publishing Simulation
```powershell
# Previews complete review summary and inline comments without making any write requests
python main.py --pr octocat/Hello-World#42 --publish --dry-run
```

#### 4. Controlled Review Publication (Human Approval Gate)
```powershell
# Runs PR analysis, verifies plan-hash integrity, and prompts for interactive approval
python main.py --pr octocat/Hello-World#42 --publish
```

### Running the Test Suite

All unit and workflow tests execute completely offline with mocked runners and zero external API or network calls:

```powershell
pytest -v
```

**Verified Test Suite Status:**
- **816 passed, 5 skipped** (integration tests requiring opt-in live Docker daemon)
- Fast execution (~25s total runtime)

To optionally run the live Docker integration tests when Docker is active:

```powershell
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
  - Rejects dangerous calls (`subprocess`, `socket`, `ctypes`, `os.system`, `exec`, `eval`, arbitrary file writes).
  - Enforces size limits (≤ 128 KB).
- Approved tests are written to a dedicated temporary test workspace, never into the primary project folder.

### 4. Docker Sandbox Execution (Milestone 3)
- **`sandbox_runner.py`** executes tests inside a hardened container:
  - Read-only source mounting (`:ro`)
  - Direct argument execution without shell interpretation
  - Strict resource constraints (`--memory 512m`, `--cpus 1.0`, `--pids-limit 100`)
  - Network isolation (`--network none`)
  - Bounded execution timeout with guaranteed container termination on timeout or failure.
- Returns structured results with return codes, pytest stdout/stderr, duration, and status.

### 5. Debugging Agent & Verified Patch Loop (Milestone 3)
- If sandbox tests fail, **DebuggingAgent** analyzes the source code, static findings, and pytest failure logs.
- Synthesizes a minimal unified diff patch.
- Verifies the patch against a disposable copy of the target in a scratch sandbox before presenting it to the user.
- Enforces bounded retry limits to prevent infinite loops.
- Automatically cleans up disposable directories on success, diff mismatch, and all exception paths.

### 6. GitHub PR Review & Controlled Publication (Milestone 4)
- **Read-Only Inspection**: Fetches PR metadata, commit status, and unified diffs via `github_client.py`.
- **Hunk & Line Mapping**: `parse_diff_hunks` maps changes to ensure that inline comments attach only to valid newly added or modified lines on the head commit.
- **PR Workspace Isolation**: `pr_workspace.py` manages an ephemeral workspace for running linters, tests, and patch checks on PR changes.
- **Action Planning**: The orchestrator produces a structured review action plan with overall verdict (`COMMENT`, `APPROVE`, or `REQUEST_CHANGES`), top-level summary, and file-anchored inline suggestions.
- **Safe Dry-Run Simulation (`--dry-run`)**: Fully prepares and formats the review payload, displaying all top-level commentary and inline line comments without sending HTTP write requests.
- **Cryptographic Approval Gate (`--publish`)**:
  - Calculates a SHA-256 plan hash over the finalized review payload.
  - Re-verifies PR head-SHA to ensure the PR branch was not modified during analysis (preventing stale reviews).
  - Prompts the human reviewer with an interactive approval gate requiring explicit confirmation (`yes`).
  - Publishes the review in a single atomic submission via GitHub REST API upon approval.

---

## 🧪 Sample Demonstrations

The file [`sample_project/app.py`](sample_project/app.py) contains deliberate demonstrations:
- **Ruff Issues**: Unused imports (`os`, `math`), unused variables (`unused_debug_tag`), unsorted imports.
- **Bandit Warnings**: Use of `assert` in business logic (`B101`), hardcoded token string (`B105`), weak pseudo-random generator for tokens (`B311`).

---

## 🗺 Roadmap & Completed Milestones

- [x] **Milestone 1:** Automated Local Static Analysis & Security Review Pipeline (Ruff + Bandit)
- [x] **Milestone 2:** Multi-Agent AI Review using LangGraph (Linter, Security, Test-Writer Agents)
- [x] **Milestone 3:** Test Generation, Docker Sandboxed Execution & Verified Fix Proposals
- [x] **Milestone 4:** GitHub Integration (PR review automation, inline GitHub suggestions, controlled publishing)
