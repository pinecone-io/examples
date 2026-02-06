# Linear Autonomous Implementation System

**Date:** 2026-02-06
**Status:** Design
**Owner:** jhamon

## Overview

An autonomous ticket implementation system that orchestrates a team of AI agents to implement Linear tickets in parallel. The system uses a team-lead for delegation, two implementers for parallel work, and a code reviewer to ensure quality before human approval.

### Key Features

- **Parallel Implementation:** Two agents work independently on different tickets
- **Smart Delegation:** Hybrid approach using Linear labels + content analysis
- **Git Worktree Isolation:** Each ticket gets its own workspace to avoid conflicts
- **Automated Status Tracking:** Updates Linear status (Todo → In Progress → In Review → Done)
- **Code Review Queue:** Reviewer iterates until CI green + all feedback addressed
- **Async Blocker Reporting:** No interactive prompts - blockers documented in Linear
- **Summary Updates:** Meaningful progress updates at natural checkpoints
- **Resumable Sessions:** Interrupt and resume work without losing progress

### Invocation

Slash command with user-provided criteria:
```
/linear-implement "criteria"
```

Examples:
- `/linear-implement "high priority tickets in current cycle"`
- `/linear-implement "tickets labeled 'python-sdk'"`
- `/linear-implement "unassigned bugs in SDK team"`

## Architecture

### Team Structure

**3-Agent Team:**
1. **Team Lead (1)** - Orchestration, delegation, status tracking, escalation
2. **Implementers (2)** - Parallel autonomous implementation
3. **Reviewer (1)** - Code review, iteration, CI monitoring

### File Structure

```
~/.claude/plugins/linear-autonomous/
├── plugin.json
├── skills/
│   └── autonomous-implementation/
│       ├── SKILL.md                     # Main skill
│       ├── references/
│       │   ├── delegation-logic.md      # Ticket categorization
│       │   ├── linear-status-mapping.md # Status field mappings
│       │   ├── worktree-management.md   # Git worktree patterns
│       │   └── error-escalation.md      # Escalation decision tree
│       └── examples/
│           ├── bug-batch.md
│           └── feature-batch.md
└── agents/
    ├── team-lead.md                     # Orchestrator
    ├── implementer.md                   # Implementation agent
    └── reviewer.md                      # Review agent
```

## Detailed Workflow

### Phase 1: Initialization & Ticket Selection

1. **User invokes:** `/linear-implement "criteria"`

2. **Team-lead parses criteria** into Linear API filters:
   - Priority levels (urgent, high, normal, low)
   - Cycle/sprint identifiers
   - Team names (default: "SDK")
   - Project names (default: "Notebook Examples")
   - Labels/tags
   - Assignee status
   - Current status (typically "Todo")

3. **Fetch matching tickets** using `mcp__plugin_linear_linear__list_issues`:
   - Apply parsed filters
   - Sort by priority (urgent first)
   - Retrieve: ID, title, description, labels, priority, status

4. **Display batch for confirmation:**
   - Show: ticket ID, title, priority, labels
   - Ask: "Found N tickets. Proceed? (y/n)"
   - Exit if declined

### Phase 2: Team Setup & Delegation

5. **Create team** using `TeamCreate`:
   - Team name: `linear-impl-{timestamp}`
   - Members: team-lead, impl-1, impl-2, reviewer

6. **Analyze tickets for delegation:**
   - **First:** Check Linear labels (bug, feature, refactor, etc.)
   - **Then:** Analyze description content for complexity/domain
   - **Assign category:** bug-fix, feature-dev, refactor, test, docs

7. **Create tasks** using `TaskCreate`:
   - One task per ticket: "Implement {TICKET-ID}: {title}"
   - Metadata: ticket URL, category, priority, worktree path
   - All start as "pending"

### Phase 3: Parallel Implementation

8. **Team-lead assigns initial work:**
   - Claim first 2 tasks from queue
   - For each task:
     - Create git worktree: `/tmp/linear-worktrees/{ticket-id}/`
     - Create branch: `linear/{ticket-id}-{slug}`
     - Update Linear status → "In Progress"
     - Add Linear comment: "🤖 Implementation started by {agent}"
   - Assign using `TaskUpdate` with `owner` field

9. **Implementers work autonomously:**
   - Read ticket via `mcp__plugin_linear_linear__get_issue`
   - Parse requirements and acceptance criteria
   - Read relevant codebase files
   - Implement changes in assigned worktree
   - Run tests
   - Commit with descriptive message
   - Push branch to origin

10. **Error handling during implementation:**
    - **Attempts 1-3:** Try to fix automatically (syntax, tests, linting)
    - **If still failing:** Send message to reviewer for help
    - **If reviewer fixes:** Continue normally
    - **If reviewer can't fix:** Escalate to team-lead
    - **Team-lead escalation:**
      - Update Linear status → "Blocked"
      - Add detailed comment with context
      - Mark task completed
      - Continue with other tickets

11. **On successful implementation:**
    - Create PR using `gh pr create`:
      - Title: "{TICKET-ID}: {title}"
      - Body: description + Linear ticket link
      - Labels from Linear
    - Update task status → "completed"
    - Team-lead assigns next pending task to now-idle implementer

### Phase 4: Code Review Queue

12. **Reviewer monitors completed work:**
    - Uses `TaskList` to find completed tasks
    - Claims review task using `TaskUpdate`

13. **Reviewer performs iterative review:**
    - Fetch PR: `gh pr view`
    - Read changed files
    - Check: correctness, tests, code quality, security
    - Run tests in worktree if needed

14. **Review iteration loop:**
    - **Check CI status:** `gh pr checks`
    - **Check PR comments:** `gh pr view --comments`
      - Parse unresolved comment threads
      - May include feedback from external AI agents
      - May include human reviewer feedback

    - **If CI failing OR unresolved comments exist:**
      - Address each piece of feedback
      - Make fixes in worktree
      - Commit with message referencing feedback
      - Push changes
      - Resolve/reply to PR comments
      - Wait for CI to re-run
      - Check for new comments
      - **Repeat until BOTH conditions met:**
        - ✅ CI green (all checks passing)
        - ✅ All feedback addressed (no unresolved comments)

    - **When ready:**
      - Update Linear status → "In Review"
      - Add Linear comment: "✅ PR #{number} ready for merge - all checks passing, all feedback addressed"
      - Mark review task as completed
      - **User merges when ready** (final approval)

### Phase 5: Progress Tracking & Status Updates

15. **Linear status transitions:**
    - `Todo` → `In Progress` (implementer starts)
    - `In Progress` → `In Review` (reviewer confirms ready)
    - `In Review` → `Done` (user merges)
    - `In Progress` → `Blocked` (escalation needed)

16. **Summary updates at checkpoints:**
    - **Implementation start:** "🤖 Implementation started by {agent} in branch {branch}"
    - **PR created:** "📝 PR #{number} opened: {url}"
    - **Review started:** "👀 Code review in progress"
    - **Feedback iteration:** "🔄 Addressing feedback: {summary}"
    - **Ready for merge:** "✅ All checks passing, all feedback addressed - ready for merge"
    - **Blocked:** "⚠️ Blocked: {explanation} - needs human input"

### Phase 6: Error Handling & Escalation

17. **Implementer error handling:**
    - **Attempt 1-3:** Auto-fix (syntax, tests, linting)
    - **If still failing:**
      - Send message to reviewer: "Need help with {issue}"
      - Reviewer examines and attempts fix
      - If fixed: continue
      - If not: escalate to team-lead

18. **Team-lead escalation:**
    - Update Linear status → "Blocked"
    - Add detailed comment:
      - What was attempted
      - Why it failed
      - What decision/info is needed
      - Relevant context (errors, logs)
    - Mark task completed (blocked)
    - Continue with other tickets
    - User unblocks via Linear comment with guidance

19. **External feedback handling:**
    - Reviewer polls PRs for new comments (every 5 minutes)
    - Addresses feedback as it arrives
    - Updates Linear with progress

### Phase 7: Completion & Cleanup

20. **When all tasks complete:**
    - Team-lead summarizes:
      - X tickets completed and ready for merge
      - Y tickets blocked (with links)
      - Z tickets in progress
    - Send summary to user

21. **Post-merge cleanup** (optional `/linear-cleanup` or automatic):
    - For each merged PR:
      - Update Linear → "Done"
      - Delete worktree
      - Delete local branch
    - Shutdown team using `TeamDelete`

### Phase 8: Resumability & State Management

22. **State persistence:**
    - Team config: `~/.claude/teams/linear-impl-{timestamp}/`
    - Task list: `~/.claude/tasks/linear-impl-{timestamp}/`
    - Each task metadata includes:
      - Linear ticket ID and URL
      - Worktree path
      - Branch name
      - PR number (once created)
      - Current status
      - Assigned agent

23. **Resuming work:**
    - **Automatic detection:** Check for existing team when running `/linear-implement`
    - If found: "Found existing session with X incomplete tickets. Resume or start fresh?"
    - **If resume:**
      - Rehydrate team from config
      - Resume agents with preserved context
      - Check current state:
        - Query Linear for updates
        - Check PR status
        - Verify worktrees exist
      - Continue from where left off
    - **If start fresh:**
      - Archive old team/tasks
      - Start new session

24. **Manual resume:**
    - `/linear-resume [team-name]` - explicitly resume specific session

25. **State reconciliation on resume:**
    - Sync Linear status with local task status
    - If PR merged externally: mark done and skip
    - If worktree deleted: recreate or skip
    - If ticket updated: re-fetch requirements

26. **Graceful shutdown:**
    - User interrupts anytime (Ctrl+C)
    - Agents finish current atomic operation
    - State saved automatically
    - Resume later without losing progress

## Agent Specifications

### Team Lead Agent

**Role:** Orchestration, delegation, status tracking, escalation

**Tools:** Full access
- Linear API (list, get, update issues, create comments)
- Task management (TaskCreate, TaskUpdate, TaskList)
- Team management (TeamCreate, SendMessage)
- Git operations (worktree management)

**Responsibilities:**
- Parse user criteria and fetch Linear tickets
- Analyze tickets and create task queue
- Assign tasks to implementers
- Monitor progress and reassign idle agents
- Handle escalations and update Linear with blockers
- Summarize results when complete
- Coordinate resume operations

**Key Behaviors:**
- Delegates but doesn't implement
- Maintains overall view of progress
- Handles all Linear status updates
- Makes escalation decisions

### Implementer Agent

**Role:** Autonomous implementation in isolated worktree

**Tools:**
- File operations (Read, Write, Edit)
- Bash (for tests, builds)
- Linear API (read-only: get issue)
- Communication (SendMessage)
- Task updates (TaskUpdate)

**Responsibilities:**
- Read assigned Linear ticket
- Understand requirements from ticket description
- Implement changes in assigned worktree
- Run tests and fix issues (up to 3 attempts)
- Create PR when done
- Escalate to reviewer if stuck
- Claim next task when idle

**Key Behaviors:**
- Works in isolation (own worktree)
- Makes 3 fix attempts before escalating
- Self-sufficient for common issues
- Proactive about claiming next work

### Reviewer Agent

**Role:** Code review, iteration, CI monitoring

**Tools:**
- File operations (Read, Write, Edit)
- Bash and gh CLI (PR operations, CI checks)
- Linear API (read/write)
- Communication (SendMessage)
- Task updates (TaskUpdate)

**Responsibilities:**
- Monitor for PRs needing review
- Review code quality, tests, security
- Iterate on fixes until CI green + all feedback addressed
- Reply to PR comments (from humans or AI agents)
- Update Linear when ready for merge
- Continue monitoring for new feedback

**Key Behaviors:**
- Iterates until BOTH CI green AND all comments resolved
- Polls for new feedback periodically
- Makes direct fixes for minor issues
- Reassigns major issues to original implementer
- Patient with external feedback loops

## Technical Patterns

### Criteria Parsing

User input → Linear API filters:
```
"high priority"     → priority: 2 (or 1 for urgent)
"current cycle"     → cycle: <current cycle ID>
"team SDK"          → team: "SDK" (or team ID)
"project Notebook Examples" → project: "Notebook Examples"
"label auth"        → labels: ["auth"]
"assigned to me"    → assignee: "me"
"unassigned"        → assignee: null
Multiple filters    → Combined with AND logic
```

### Worktree Management

```bash
# Create isolated worktree
git worktree add /tmp/linear-worktrees/SDK-123 -b linear/SDK-123-feature-name

# Work in isolation
cd /tmp/linear-worktrees/SDK-123
# make changes, commit, push

# Cleanup after merge
git worktree remove /tmp/linear-worktrees/SDK-123
git branch -d linear/SDK-123-feature-name
```

### CI Status Detection

```bash
# Check PR status
gh pr checks <pr-number>

# Wait for completion
while [[ $(gh pr checks --json state -q '.[].state' | grep -c "PENDING") -gt 0 ]]; do
  sleep 30
done

# Verify all green
gh pr checks --json conclusion -q '.[].conclusion' | grep -v "SUCCESS" && echo "Failed"
```

### PR Comment Handling

```bash
# Fetch all comments
gh pr view <pr-number> --comments --json comments

# Parse unresolved threads
# Look for comments without resolution markers

# Reply to comments
gh pr comment <pr-number> --body "Fixed in commit abc123"
```

## Configuration

**Default configuration** (`.claude/plugins/linear-autonomous.local.md`):

```yaml
---
# Team composition
team_size:
  implementers: 2
  reviewers: 1

# Worktree location
worktree_base: "/tmp/linear-worktrees"

# Linear configuration
linear:
  default_team: "SDK"
  default_project: "Notebook Examples"
  status_field: "State"
  status_values:
    todo: "Todo"
    in_progress: "In Progress"
    in_review: "In Review"
    done: "Done"
    blocked: "Blocked"

# Review settings
review:
  max_fix_attempts: 3
  ci_poll_interval: 30        # seconds
  comment_poll_interval: 300  # 5 minutes

# Delegation rules (optional overrides)
delegation:
  bug: "bug-fix"
  feature: "feature-dev"
  refactor: "refactor"
  test: "test-automation"
  docs: "documentation"
---
```

## Usage Examples

### Example 1: High Priority Tickets

```
User: /linear-implement "high priority tickets in current cycle"

Output:
→ Found 5 high-priority tickets in current cycle:
  - SDK-123: Fix Python client timeout issue [P1, bug]
  - SDK-124: Add async support to embeddings API [P1, feature]
  - SDK-125: Update quickstart notebook for v8 [P2, docs]
  - SDK-126: Refactor error handling [P2, refactor]
  - SDK-127: Add tests for sparse vectors [P2, test]

Proceed with implementation? (y/n)

User: y

Output:
→ Creating team 'linear-impl-20260206-1430'
→ Created 5 tasks
→ impl-1: SDK-123 (Fix Python client timeout)
→ impl-2: SDK-124 (Add async support)
→ Implementation started...
```

### Example 2: Label-Based Selection

```
User: /linear-implement "tickets labeled 'python-sdk' in Notebook Examples"

Output:
→ Found 12 tickets labeled 'python-sdk' in project 'Notebook Examples'
→ Creating team and assigning work...
→ impl-1: SDK-201 (Python client retry logic)
→ impl-2: SDK-202 (Add type hints to client)
→ Implementation started...
```

### Example 3: Resume Session

```
User: /linear-implement "high priority"

Output:
→ Found existing session 'linear-impl-20260206-1430' with 3 incomplete tickets
→ Resume or start fresh? (resume/fresh)

User: resume

Output:
→ Resuming session...
→ Syncing state:
  - SDK-123: PR #456 merged ✅
  - SDK-124: PR #457 in review (CI green, awaiting feedback)
  - SDK-125: Blocked on design decision
→ Continuing work on SDK-124...
```

### Example 4: Cleanup After Completion

```
User: /linear-cleanup

Output:
→ Found 5 merged PRs to clean up:
  - SDK-123: Removing worktree, deleting branch ✅
  - SDK-124: Removing worktree, deleting branch ✅
  - SDK-125: Removing worktree, deleting branch ✅
  - SDK-126: Removing worktree, deleting branch ✅
  - SDK-127: Removing worktree, deleting branch ✅
→ Shutting down team 'linear-impl-20260206-1430' ✅
```

## Benefits

1. **Parallel Throughput:** Two implementers work simultaneously on different tickets
2. **Quality Assurance:** Dedicated reviewer ensures CI green + feedback addressed
3. **Async Operation:** No blocking on user input - blockers documented in Linear
4. **Isolation:** Git worktrees prevent conflicts and enable true parallelism
5. **Resumability:** Interrupt and resume without losing progress
6. **Visibility:** Linear status and comments keep user informed
7. **Smart Delegation:** Hybrid approach uses labels + content analysis
8. **Autonomous:** Handles common issues without human intervention

## Success Criteria

- [ ] Can fetch and parse Linear tickets based on user criteria
- [ ] Can create team with 3 agents (lead, 2 implementers, 1 reviewer)
- [ ] Can create isolated worktrees for parallel work
- [ ] Implementers can autonomously implement tickets and create PRs
- [ ] Reviewer can iterate until CI green + all feedback addressed
- [ ] Linear status updates reflect current state (Todo → In Progress → In Review → Done)
- [ ] Blockers are documented in Linear without interactive prompts
- [ ] Can resume interrupted sessions without losing progress
- [ ] Can clean up merged PRs and worktrees
- [ ] User retains final merge approval

## Future Enhancements

- **Smart batching:** Group related tickets together for efficiency
- **Dynamic team sizing:** Add specialists on-demand (e.g., database expert)
- **Learning from feedback:** Track common issues and improve over time
- **Metrics dashboard:** Track implementation velocity, review time, block rate
- **Integration with CI/CD:** Trigger deployments after merge
- **Multi-repo support:** Handle tickets spanning multiple repositories

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Agents get stuck in loops | Max 3 fix attempts, then escalate |
| CI takes too long | Poll periodically, don't block other work |
| External feedback delays | Reviewer polls for comments, continues other work |
| Worktree conflicts | Each agent gets dedicated worktree path |
| State corruption on crash | Persist state after each operation, reconcile on resume |
| User loses control | User retains final merge approval, can halt anytime |
