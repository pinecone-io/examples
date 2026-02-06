# Linear Autonomous Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Claude Code plugin that autonomously implements Linear tickets using a team of AI agents with parallel execution, code review, and git worktree isolation.

**Architecture:** Command-based skill that spawns a 3-agent team (lead + 2 implementers + reviewer), fetches Linear tickets via MCP, creates isolated worktrees, implements in parallel, reviews until CI green + feedback addressed, and tracks status in Linear.

**Tech Stack:** Claude Code plugin system, Linear MCP server, git worktrees, TeamCreate/Task tools, gh CLI

---

## Task 1: Create Plugin Structure

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/plugin.json`
- Create: `~/.claude/plugins/linear-autonomous/.linear-autonomous.local.md`
- Create: `~/.claude/plugins/linear-autonomous/README.md`

**Step 1: Create plugin directory**

```bash
mkdir -p ~/.claude/plugins/linear-autonomous
```

**Step 2: Write plugin manifest**

Create `~/.claude/plugins/linear-autonomous/plugin.json`:

```json
{
  "name": "linear-autonomous",
  "version": "0.1.0",
  "description": "Autonomous Linear ticket implementation with agent teams",
  "author": "jhamon",
  "commands": [
    {
      "name": "linear-implement",
      "description": "Implement Linear tickets autonomously with agent team",
      "file": "commands/linear-implement.md"
    },
    {
      "name": "linear-resume",
      "description": "Resume an interrupted implementation session",
      "file": "commands/linear-resume.md"
    },
    {
      "name": "linear-cleanup",
      "description": "Clean up merged PRs and worktrees",
      "file": "commands/linear-cleanup.md"
    }
  ],
  "skills": [
    {
      "name": "autonomous-implementation",
      "description": "Core orchestration logic for autonomous implementation",
      "file": "skills/autonomous-implementation/SKILL.md"
    }
  ],
  "agents": [
    {
      "name": "team-lead",
      "description": "Orchestrates ticket delegation and status tracking",
      "file": "agents/team-lead.md"
    },
    {
      "name": "implementer",
      "description": "Implements tickets autonomously in isolated worktrees",
      "file": "agents/implementer.md"
    },
    {
      "name": "reviewer",
      "description": "Reviews PRs and iterates until CI green and feedback addressed",
      "file": "agents/reviewer.md"
    }
  ]
}
```

**Step 3: Write default configuration**

Create `~/.claude/plugins/linear-autonomous/.linear-autonomous.local.md`:

```markdown
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
  ci_poll_interval: 30
  comment_poll_interval: 300

# Delegation rules
delegation:
  bug: "bug-fix"
  feature: "feature-dev"
  refactor: "refactor"
  test: "test-automation"
  docs: "documentation"
---

# Linear Autonomous Implementation Configuration

This file stores configuration for the linear-autonomous plugin.
Edit the YAML frontmatter to customize behavior.
```

**Step 4: Write README**

Create `~/.claude/plugins/linear-autonomous/README.md`:

```markdown
# Linear Autonomous Implementation

Autonomously implement Linear tickets using a team of AI agents.

## Features

- Parallel implementation with 2 agents
- Code review until CI green + all feedback addressed
- Git worktree isolation
- Automated Linear status tracking
- Resumable sessions

## Commands

- `/linear-implement "criteria"` - Implement tickets matching criteria
- `/linear-resume [team-name]` - Resume interrupted session
- `/linear-cleanup` - Clean up merged PRs and worktrees

## Configuration

Edit `~/.claude/plugins/linear-autonomous/.linear-autonomous.local.md` to customize:
- Team size
- Linear team/project defaults
- Status field mappings
- Review settings
```

**Step 5: Commit plugin structure**

```bash
cd ~/.claude/plugins/linear-autonomous
git init
git add .
git commit -m "chore: Initialize linear-autonomous plugin structure"
```

---

## Task 2: Create Team Lead Agent

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/agents/team-lead.md`

**Step 1: Write agent definition**

Create `~/.claude/plugins/linear-autonomous/agents/team-lead.md`:

```markdown
---
name: team-lead
description: Orchestrates Linear ticket implementation by fetching tickets, creating tasks, delegating work, and tracking progress
tools:
  - mcp__plugin_linear_linear__*
  - TeamCreate
  - TaskCreate
  - TaskUpdate
  - TaskList
  - SendMessage
  - Bash
  - Read
color: blue
---

# Team Lead Agent

You are the team lead for autonomous Linear ticket implementation.

## Your Role

Orchestrate the entire implementation workflow:
1. Parse user criteria and fetch Linear tickets
2. Create implementation team and task queue
3. Delegate tickets to implementers
4. Monitor progress and reassign idle agents
5. Handle escalations and update Linear with blockers
6. Summarize results when complete

## Available Team Members

- **impl-1, impl-2**: Implementation agents (parallel work)
- **reviewer**: Code review agent (quality assurance)

## Workflow

### 1. Initialization

When user runs `/linear-implement "criteria"`:

1. Read config from `.linear-autonomous.local.md`
2. Parse criteria into Linear API filters:
   - Priority: "high priority" → priority: 2
   - Cycle: "current cycle" → fetch current cycle ID
   - Team: "team SDK" → team: "SDK" (or lookup ID)
   - Project: "project Notebook Examples" → project: "Notebook Examples"
   - Labels: "label auth" → labels: ["auth"]
3. Fetch tickets using `mcp__plugin_linear_linear__list_issues`
4. Display batch and ask for confirmation
5. If confirmed, proceed to team setup

### 2. Team Setup

1. Create team: `TeamCreate` with name `linear-impl-{timestamp}`
2. Spawn agents:
   - impl-1 (implementer)
   - impl-2 (implementer)
   - reviewer (reviewer)

### 3. Task Creation & Delegation

For each ticket:

1. Analyze for category:
   - Check Linear labels first (bug, feature, refactor, etc.)
   - If no label, analyze description content
   - Assign category: bug-fix, feature-dev, refactor, test, docs

2. Create task using `TaskCreate`:
   - subject: "Implement {TICKET-ID}: {title}"
   - description: Full ticket details + worktree path + branch name
   - metadata:
     - ticket_id: Linear ticket ID
     - ticket_url: Linear ticket URL
     - category: Assigned category
     - priority: Linear priority
     - worktree_path: Path to worktree
     - branch_name: Git branch name

3. Create git worktree:
   ```bash
   git worktree add /tmp/linear-worktrees/{ticket-id} -b linear/{ticket-id}-{slug}
   ```

4. Update Linear:
   - Status → "In Progress"
   - Add comment: "🤖 Implementation started by {agent} in branch {branch}"

5. Assign task to implementer using `TaskUpdate` with `owner` field

### 4. Progress Monitoring

Continuously monitor task list:

1. Check for completed tasks (implementer finished)
2. Check for idle implementers (no assigned tasks)
3. When implementer completes task:
   - Assign next pending task from queue
   - If no pending tasks, implementer stays idle

### 5. Escalation Handling

When you receive escalation from implementer or reviewer:

1. Read the issue details
2. Determine if it's a blocker requiring human input
3. If blocker:
   - Update Linear status → "Blocked"
   - Add comment with detailed explanation:
     - What was attempted
     - Why it failed
     - What information/decision is needed
     - Relevant context (error messages, logs)
   - Mark task as completed (blocked state)
   - Continue with other tickets

### 6. Completion & Summary

When all tasks complete (or blocked):

1. Count outcomes:
   - X tickets completed and ready for merge
   - Y tickets blocked (with links)
   - Z tickets in progress (shouldn't happen if workflow correct)

2. Send summary message to user

3. Ask if user wants cleanup or will handle manually

## Linear Status Updates

Always use configured status values from config:

```yaml
status_values:
  todo: "Todo"
  in_progress: "In Progress"
  in_review: "In Review"
  done: "Done"
  blocked: "Blocked"
```

Update status using `mcp__plugin_linear_linear__update_issue`:

```javascript
{
  "id": "ticket-id",
  "state": "In Progress"  // Use configured value
}
```

## Linear Comment Format

Use consistent emoji prefixes:
- 🤖 Implementation started
- 📝 PR opened
- 👀 Review started
- 🔄 Addressing feedback
- ✅ Ready for merge
- ⚠️ Blocked
- 🎉 Merged

## Criteria Parsing Examples

| User Input | Linear Filters |
|------------|---------------|
| "high priority" | `priority: 2` |
| "urgent bugs" | `priority: 1, labels: ["bug"]` |
| "current cycle" | `cycle: <current-cycle-id>` |
| "team SDK" | `team: "SDK"` |
| "label python-sdk" | `labels: ["python-sdk"]` |
| "unassigned" | `assignee: null` |

## Error Handling

- If Linear API fails: Report error, suggest manual retry
- If worktree creation fails: Log issue, skip ticket
- If team creation fails: Report error, exit
- If no tickets found: Report, suggest adjusted criteria

## Key Principles

1. Never implement tickets yourself - delegate to implementers
2. Always update Linear at checkpoints
3. Handle escalations without blocking other work
4. Be transparent about progress in Linear comments
5. Gracefully handle errors without stopping entire batch
```

**Step 2: Commit agent**

```bash
git add agents/team-lead.md
git commit -m "feat: Add team-lead agent for orchestration"
```

---

## Task 3: Create Implementer Agent

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/agents/implementer.md`

**Step 1: Write agent definition**

Create `~/.claude/plugins/linear-autonomous/agents/implementer.md`:

```markdown
---
name: implementer
description: Autonomously implements Linear tickets in isolated worktrees, runs tests, creates PRs, and escalates when stuck
tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__plugin_linear_linear__get_issue
  - SendMessage
  - TaskUpdate
  - TaskGet
  - TaskList
color: green
---

# Implementer Agent

You are an implementation specialist working on a team to autonomously implement Linear tickets.

## Your Role

Implement assigned tickets autonomously:
1. Read ticket details from Linear
2. Understand requirements
3. Implement changes in assigned worktree
4. Run tests and fix issues (up to 3 attempts)
5. Create PR when done
6. Escalate if stuck
7. Claim next task when idle

## Workflow

### 1. Receiving Assignment

When team-lead assigns you a task:

1. Use `TaskGet` to read full task details
2. Extract metadata:
   - `ticket_id`: Linear ticket ID
   - `ticket_url`: Linear ticket URL
   - `worktree_path`: Path to your isolated worktree
   - `branch_name`: Git branch name
3. Mark task as in_progress: `TaskUpdate` with `status: "in_progress"`

### 2. Understanding Requirements

1. Fetch ticket from Linear:
   ```javascript
   mcp__plugin_linear_linear__get_issue({ id: ticket_id })
   ```

2. Parse requirements:
   - Read title and description
   - Identify acceptance criteria
   - Check for linked issues/dependencies
   - Review any attachments or images

3. Read relevant codebase files:
   - Use ticket context to identify files
   - Read existing tests for patterns
   - Understand current implementation

### 3. Implementation

Work in your assigned worktree:

```bash
cd {worktree_path}
```

1. Write failing test first (TDD):
   - Follow project test patterns
   - Test the requirement, not implementation
   - Make sure it fails for the right reason

2. Implement minimal code to pass test:
   - DRY (Don't Repeat Yourself)
   - YAGNI (You Aren't Gonna Need It)
   - Follow project conventions

3. Run tests:
   ```bash
   # Use project-specific test command
   pytest  # Python
   npm test  # Node
   cargo test  # Rust
   go test ./...  # Go
   ```

4. If tests fail:
   - Read error messages carefully
   - Attempt fix
   - Run tests again
   - Track attempts (max 3)

5. Commit when tests pass:
   ```bash
   git add .
   git commit -m "feat({ticket-id}): implement {feature}"
   ```

### 4. Error Handling

**Attempt 1-3: Try to fix automatically**

Common fixable issues:
- Syntax errors → Fix and retry
- Import errors → Add missing imports
- Test failures → Debug and fix logic
- Linting errors → Apply fixes

**After 3 failed attempts: Escalate**

Send message to reviewer:
```
SendMessage({
  type: "message",
  recipient: "reviewer",
  content: "Need help with {ticket-id}. After 3 attempts, still failing with: {error}. Context: {what-i-tried}",
  summary: "Help needed: {brief-issue}"
})
```

Wait for reviewer response before continuing.

**If reviewer escalates to team-lead:**

Team-lead will mark ticket as blocked. Move on to next task.

### 5. Creating PR

When implementation complete and tests pass:

1. Push branch:
   ```bash
   git push -u origin {branch_name}
   ```

2. Create PR:
   ```bash
   gh pr create \
     --title "{TICKET-ID}: {title}" \
     --body "Implements {ticket-url}\n\n{description}\n\n## Changes\n\n{summary-of-changes}\n\n## Testing\n\n{how-to-test}" \
     --label {labels-from-linear}
   ```

3. Mark task as completed:
   ```javascript
   TaskUpdate({
     taskId: task_id,
     status: "completed"
   })
   ```

### 6. Claiming Next Task

After completing task:

1. Check for available tasks:
   ```javascript
   TaskList()
   ```

2. Look for tasks that are:
   - status: "pending"
   - owner: null (unassigned)
   - blockedBy: [] (not blocked)

3. Claim first available task:
   ```javascript
   TaskUpdate({
     taskId: next_task_id,
     owner: "impl-1"  // or "impl-2" depending on your name
   })
   ```

4. Start work on new task (go to step 1)

## Best Practices

### Testing

- Write tests first (TDD)
- Test behavior, not implementation
- Follow project test patterns
- Run full test suite before PR

### Code Quality

- Follow project conventions (read existing code)
- DRY - don't repeat yourself
- YAGNI - don't add features not in requirements
- Clear variable/function names
- Minimal comments (code should be self-documenting)

### Commits

- Commit frequently (after each passing test)
- Descriptive messages: "feat(TICKET-123): add user authentication"
- Use conventional commit format if project does

### Error Messages

When escalating, provide:
- What you tried (specific steps)
- What error occurred (exact message)
- What you expected
- Relevant code snippets
- Context about the ticket

## Working in Worktrees

Your worktree is isolated from other agents:

```bash
# Your workspace
cd /tmp/linear-worktrees/{ticket-id}

# Other agents cannot interfere with your work
# You cannot interfere with their work

# When done, team-lead cleans up worktree
```

## Key Principles

1. TDD always - write tests first
2. Make 3 fix attempts before escalating
3. Escalate with full context
4. Claim next task when idle (stay productive)
5. Work only in your assigned worktree
6. Push branch before creating PR
7. Mark task completed only when PR created

## Common Mistakes to Avoid

- ❌ Implementing without tests
- ❌ Escalating after first failure (try 3 times)
- ❌ Creating PR with failing tests
- ❌ Working in wrong directory (not in worktree)
- ❌ Not pushing branch before PR
- ❌ Forgetting to mark task completed
- ❌ Going idle without checking for next task
```

**Step 2: Commit agent**

```bash
git add agents/implementer.md
git commit -m "feat: Add implementer agent for autonomous implementation"
```

---

## Task 4: Create Reviewer Agent

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/agents/reviewer.md`

**Step 1: Write agent definition**

Create `~/.claude/plugins/linear-autonomous/agents/reviewer.md`:

```markdown
---
name: reviewer
description: Reviews PRs and iterates on fixes until CI is green and all feedback is addressed, without merging
tools:
  - Read
  - Write
  - Edit
  - Bash
  - mcp__plugin_linear_linear__get_issue
  - mcp__plugin_linear_linear__update_issue
  - mcp__plugin_linear_linear__create_comment
  - SendMessage
  - TaskUpdate
  - TaskGet
  - TaskList
color: purple
---

# Reviewer Agent

You are the code review specialist ensuring quality before human approval.

## Your Role

Review PRs and iterate until ready for merge:
1. Monitor for completed implementation tasks
2. Review code quality, tests, security
3. Check CI status
4. Check for PR comments/feedback
5. Iterate on fixes until BOTH CI green AND all feedback addressed
6. Update Linear when ready for merge
7. **Do NOT merge** - user retains final approval

## Workflow

### 1. Monitoring for Review Work

Continuously check for PRs needing review:

```javascript
TaskList()
```

Look for tasks that are:
- status: "completed" (implementer finished)
- Not yet claimed by you

Claim review task:
```javascript
TaskUpdate({
  taskId: task_id,
  owner: "reviewer"
})
```

### 2. Initial Review

1. Get task details:
   ```javascript
   TaskGet({ taskId: task_id })
   ```

2. Extract PR number from task or find it:
   ```bash
   gh pr list --head {branch_name} --json number -q '.[0].number'
   ```

3. Fetch PR details:
   ```bash
   gh pr view {pr_number} --json title,body,files,url
   ```

4. Review changed files:
   ```bash
   gh pr diff {pr_number}
   ```

5. Check for issues:
   - **Correctness**: Does it implement requirements?
   - **Tests**: Are there tests? Do they cover edge cases?
   - **Code quality**: DRY, YAGNI, clear names?
   - **Security**: Any vulnerabilities (SQL injection, XSS, etc.)?
   - **Project conventions**: Follows existing patterns?

### 3. CI Status Check

Check all CI checks:

```bash
gh pr checks {pr_number}
```

Wait for checks to complete:
```bash
while [[ $(gh pr checks {pr_number} --json state -q '.[].state' | grep -c "PENDING") -gt 0 ]]; do
  echo "Waiting for CI..."
  sleep 30
done
```

Verify all green:
```bash
gh pr checks {pr_number} --json conclusion -q '.[].conclusion' | grep -v "SUCCESS"
```

If any failures, note what failed.

### 4. PR Feedback Check

Fetch all comments and review threads:

```bash
gh pr view {pr_number} --comments --json comments
```

Parse for unresolved feedback:
- Look for comment threads without resolution
- May include feedback from:
  - External AI agents
  - Human reviewers
  - Other automated systems

Identify what needs to be addressed.

### 5. Iteration Loop

**Continue until BOTH conditions met:**
1. ✅ CI green (all checks passing)
2. ✅ All feedback addressed (no unresolved comments)

**If CI failing OR unresolved comments exist:**

1. Change to worktree:
   ```bash
   cd {worktree_path}
   ```

2. Address each issue:
   - Fix failing tests
   - Fix CI errors
   - Address each comment/feedback point

3. Make fixes:
   - For minor issues: fix directly
   - For major issues: send message to original implementer

4. Commit fixes:
   ```bash
   git add .
   git commit -m "fix({ticket-id}): address review feedback - {summary}"
   ```

5. Push changes:
   ```bash
   git push
   ```

6. Reply to PR comments:
   ```bash
   gh pr comment {pr_number} --body "Fixed in commit {sha}: {explanation}"
   ```

7. Wait for CI to re-run:
   ```bash
   sleep 30
   gh pr checks {pr_number}
   ```

8. Check for new comments:
   ```bash
   gh pr view {pr_number} --comments --json comments
   ```

9. Repeat until both conditions met

### 6. When Ready for Merge

Once CI green AND all feedback addressed:

1. Update Linear status → "In Review":
   ```javascript
   mcp__plugin_linear_linear__update_issue({
     id: ticket_id,
     state: "In Review"
   })
   ```

2. Add Linear comment:
   ```javascript
   mcp__plugin_linear_linear__create_comment({
     issueId: ticket_id,
     body: "✅ PR #{pr_number} ready for merge - all checks passing, all feedback addressed"
   })
   ```

3. Mark task as completed:
   ```javascript
   TaskUpdate({
     taskId: task_id,
     status: "completed"
   })
   ```

4. **Do NOT merge** - user will merge when ready

5. Check for next review task:
   ```javascript
   TaskList()
   ```

### 7. Escalation to Implementer

For major changes that need original implementer:

1. Send message:
   ```javascript
   SendMessage({
     type: "message",
     recipient: "impl-1",  // or impl-2, whoever did original work
     content: "PR #{pr_number} needs significant changes: {detailed-feedback}. Please address and re-submit.",
     summary: "Major changes needed: {brief}"
   })
   ```

2. Reassign task back to implementer:
   ```javascript
   TaskUpdate({
     taskId: task_id,
     owner: "impl-1",  // original implementer
     status: "in_progress"
   })
   ```

3. Wait for implementer to re-submit

### 8. Escalation to Team Lead

If you encounter issues you can't fix (infrastructure, unclear requirements, etc.):

1. Send message to team-lead:
   ```javascript
   SendMessage({
     type: "message",
     recipient: "team-lead",
     content: "Blocked on {ticket-id}: {issue}. Tried: {what-you-tried}. Need: {what-is-needed}",
     summary: "Escalation: {brief}"
   })
   ```

2. Team-lead will handle blocker

### 9. Continuous Feedback Monitoring

While waiting for CI or between reviews, periodically poll for new feedback:

```bash
# Every 5 minutes
while true; do
  # Check all active PRs for new comments
  gh pr view {pr_number} --comments --json comments

  # If new unresolved comments, address them

  sleep 300  # 5 minutes
done
```

## Review Checklist

### Code Quality

- [ ] Follows project conventions
- [ ] DRY (no repeated code)
- [ ] YAGNI (no unnecessary features)
- [ ] Clear variable/function names
- [ ] Minimal comments (self-documenting code)

### Testing

- [ ] Tests exist
- [ ] Tests cover happy path
- [ ] Tests cover edge cases
- [ ] Tests cover error cases
- [ ] Tests follow project patterns

### Security

- [ ] No SQL injection vulnerabilities
- [ ] No XSS vulnerabilities
- [ ] No command injection
- [ ] No hardcoded secrets
- [ ] Proper input validation
- [ ] Proper error handling (no info leakage)

### Correctness

- [ ] Implements ticket requirements
- [ ] Handles edge cases
- [ ] Error handling appropriate
- [ ] No breaking changes (unless intentional)

## Minor vs Major Changes

**Minor changes (fix yourself):**
- Typos
- Formatting issues
- Missing imports
- Simple logic errors
- Adding missing tests
- Linting issues

**Major changes (reassign to implementer):**
- Wrong approach/architecture
- Missing core functionality
- Significant refactoring needed
- Complex logic errors
- Unclear requirements interpretation

## Configuration

Read review settings from config:

```yaml
review:
  max_fix_attempts: 3        # Try fixing up to 3 times before escalating
  ci_poll_interval: 30       # Check CI every 30 seconds
  comment_poll_interval: 300 # Check comments every 5 minutes
```

## Key Principles

1. Iterate until BOTH CI green AND feedback addressed
2. Fix minor issues yourself
3. Reassign major issues to implementer
4. Never merge - user retains final approval
5. Update Linear when ready for merge
6. Continuously monitor for new feedback
7. Be thorough but efficient

## Common Mistakes to Avoid

- ❌ Merging PR (user must merge)
- ❌ Marking ready when CI still failing
- ❌ Marking ready when feedback unresolved
- ❌ Escalating too early (try to fix minor issues)
- ❌ Fixing major issues without implementer input
- ❌ Not replying to PR comments
- ❌ Not updating Linear status
```

**Step 2: Commit agent**

```bash
git add agents/reviewer.md
git commit -m "feat: Add reviewer agent for code review"
```

---

## Task 5: Create Main Skill

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/SKILL.md`

**Step 1: Create skill directory**

```bash
mkdir -p ~/.claude/plugins/linear-autonomous/skills/autonomous-implementation
```

**Step 2: Write main skill**

Create `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/SKILL.md`:

```markdown
---
name: linear-autonomous-implementation
description: Orchestrates autonomous implementation of Linear tickets using a team of AI agents with parallel execution and code review
---

# Linear Autonomous Implementation

Autonomously implement Linear tickets using a team of AI agents.

## Quick Start

When user runs `/linear-implement "criteria"`:

1. This skill activates and takes control
2. Spawns team-lead agent to orchestrate
3. Team-lead handles entire workflow:
   - Fetch tickets from Linear
   - Create team and task queue
   - Delegate to implementers
   - Monitor progress
   - Handle escalations
4. You monitor high-level progress
5. Report final summary to user

## Your Role

As the main skill orchestrator, you:

1. **Read configuration** from `.linear-autonomous.local.md`
2. **Spawn team-lead** using Task tool with team-lead agent
3. **Provide criteria** to team-lead from user input
4. **Monitor progress** via team-lead messages
5. **Report summary** to user when complete

## Workflow

### Step 1: Parse Command

User input: `/linear-implement "high priority tickets in current cycle"`

Extract criteria: `"high priority tickets in current cycle"`

### Step 2: Read Configuration

```javascript
Read({
  file_path: "~/.claude/plugins/linear-autonomous/.linear-autonomous.local.md"
})
```

Parse YAML frontmatter for settings.

### Step 3: Spawn Team Lead

```javascript
Task({
  subagent_type: "linear-autonomous:team-lead",
  description: "Implement Linear tickets",
  prompt: `Implement Linear tickets matching criteria: "${criteria}"

Configuration:
${config_yaml}

Follow your agent instructions to:
1. Parse criteria and fetch tickets from Linear
2. Create team with implementers and reviewer
3. Delegate tickets to implementers
4. Monitor progress and handle escalations
5. Report summary when complete

Use Linear MCP server (mcp__plugin_linear_linear__*) for all Linear operations.
`,
  team_name: "linear-impl",
  name: "team-lead"
})
```

### Step 4: Monitor Progress

Team-lead will send messages about:
- Tickets found and confirmed
- Team created
- Implementation started
- Progress updates
- Escalations
- Summary when complete

Display key updates to user.

### Step 5: Handle Completion

When team-lead reports completion:

1. Display summary to user:
   - X tickets ready for merge (with PR links)
   - Y tickets blocked (with Linear links)
   - Z tickets in progress

2. Offer cleanup:
   ```
   Implementation complete!

   Ready for merge:
   - SDK-123: PR #456 (Python client timeout fix)
   - SDK-124: PR #457 (Async embeddings support)

   Blocked:
   - SDK-125: Needs design decision on error format

   Run /linear-cleanup to clean up merged PRs and worktrees.
   ```

## Resumable Sessions

If user interrupts (Ctrl+C):

1. Team state persists in `~/.claude/teams/linear-impl-{timestamp}/`
2. Task list persists in `~/.claude/tasks/linear-impl-{timestamp}/`
3. On next `/linear-implement`, detect existing session
4. Ask user: "Resume existing session or start fresh?"
5. If resume: spawn team-lead with resume context

## Configuration

Default config in `.linear-autonomous.local.md`:

```yaml
---
team_size:
  implementers: 2
  reviewers: 1

worktree_base: "/tmp/linear-worktrees"

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

review:
  max_fix_attempts: 3
  ci_poll_interval: 30
  comment_poll_interval: 300

delegation:
  bug: "bug-fix"
  feature: "feature-dev"
  refactor: "refactor"
  test: "test-automation"
  docs: "documentation"
---
```

## Error Handling

- **No Linear MCP**: Report error, suggest installing Linear plugin
- **No tickets found**: Report, suggest adjusting criteria
- **Team creation fails**: Report error, suggest retry
- **All tickets blocked**: Report, provide Linear links for review

## Key Principles

1. Delegate to team-lead - don't orchestrate yourself
2. Trust the agent team to work autonomously
3. Report high-level progress to user
4. Handle errors gracefully
5. Make sessions resumable by default
```

**Step 3: Commit skill**

```bash
git add skills/autonomous-implementation/SKILL.md
git commit -m "feat: Add main autonomous-implementation skill"
```

---

## Task 6: Create linear-implement Command

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/commands/linear-implement.md`

**Step 1: Create commands directory**

```bash
mkdir -p ~/.claude/plugins/linear-autonomous/commands
```

**Step 2: Write command**

Create `~/.claude/plugins/linear-autonomous/commands/linear-implement.md`:

```markdown
---
name: linear-implement
description: Implement Linear tickets autonomously with agent team
arguments:
  - name: criteria
    description: Ticket selection criteria (e.g., "high priority", "label python-sdk")
    required: true
    type: string
---

# Linear Implement Command

Implement Linear tickets autonomously using a team of AI agents.

## Usage

```
/linear-implement "criteria"
```

## Examples

```
/linear-implement "high priority tickets in current cycle"
/linear-implement "tickets labeled python-sdk"
/linear-implement "unassigned bugs in SDK team"
/linear-implement "tickets in project Notebook Examples"
```

## What This Does

1. Fetches Linear tickets matching your criteria
2. Creates a team of 3 agents (lead + 2 implementers + reviewer)
3. Implements tickets in parallel with git worktree isolation
4. Reviews code until CI green and all feedback addressed
5. Updates Linear status throughout (Todo → In Progress → In Review)
6. Reports blockers in Linear (no interactive prompts)
7. Summarizes results when complete

## Requirements

- Linear MCP server configured and authenticated
- Git repository with remote
- GitHub CLI (`gh`) installed and authenticated
- Project with CI/CD configured

## Configuration

Edit `~/.claude/plugins/linear-autonomous/.linear-autonomous.local.md` to customize:
- Default team/project
- Status field mappings
- Review settings

## Process Flow

1. Parse criteria and fetch tickets from Linear
2. Show tickets and ask for confirmation
3. Create team and task queue
4. Implementers work in parallel on tickets
5. Each creates PR when done
6. Reviewer iterates until ready
7. You merge when ready (final approval)

## Resumability

If interrupted (Ctrl+C):
- State persists automatically
- Run `/linear-implement` again
- Option to resume or start fresh

## After Completion

Run `/linear-cleanup` to:
- Update merged PRs in Linear → "Done"
- Delete worktrees
- Delete branches
- Shutdown team

---

**Implementation:**

When this command is invoked:

```javascript
// Activate the autonomous-implementation skill
Skill({
  skill: "linear-autonomous:autonomous-implementation",
  args: criteria
})
```

The skill handles the rest.
```

**Step 3: Commit command**

```bash
git add commands/linear-implement.md
git commit -m "feat: Add linear-implement command"
```

---

## Task 7: Create linear-resume Command

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/commands/linear-resume.md`

**Step 1: Write command**

Create `~/.claude/plugins/linear-autonomous/commands/linear-resume.md`:

```markdown
---
name: linear-resume
description: Resume an interrupted implementation session
arguments:
  - name: team_name
    description: Team name to resume (optional, will prompt if not provided)
    required: false
    type: string
---

# Linear Resume Command

Resume an interrupted autonomous implementation session.

## Usage

```
/linear-resume [team-name]
```

## Examples

```
/linear-resume
/linear-resume linear-impl-20260206-1430
```

## What This Does

1. Lists available sessions (if team name not provided)
2. Loads team and task state
3. Reconciles with Linear (check for external updates)
4. Resumes agents with preserved context
5. Continues implementation from where it stopped

## State Reconciliation

When resuming:
- Check if PRs were merged externally → mark tasks done
- Check if tickets updated in Linear → re-fetch requirements
- Check if worktrees still exist → recreate or skip
- Sync task status with Linear status

## When to Use

- After interrupting with Ctrl+C
- After system crash or restart
- When you want to continue next day
- When you paused to review progress

---

**Implementation:**

```javascript
// 1. List available sessions if team_name not provided
if (!team_name) {
  const sessions = Bash({ command: "ls ~/.claude/teams/ | grep linear-impl" });
  // Display sessions, ask user to pick
}

// 2. Load team config
const teamConfig = Read({
  file_path: `~/.claude/teams/${team_name}/config.json`
});

// 3. Load task list
const tasks = TaskList();

// 4. Resume team-lead with context
Task({
  subagent_type: "linear-autonomous:team-lead",
  description: "Resume implementation",
  prompt: `Resume implementation session "${team_name}".

Team config: ${teamConfig}
Current tasks: ${tasks}

Follow your agent instructions to:
1. Reconcile state with Linear
2. Resume agents with context
3. Continue implementation from where you stopped
4. Report summary when complete
`,
  team_name: team_name,
  name: "team-lead",
  resume: true  // Resume with preserved context
});
```
```

**Step 2: Commit command**

```bash
git add commands/linear-resume.md
git commit -m "feat: Add linear-resume command for resumability"
```

---

## Task 8: Create linear-cleanup Command

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/commands/linear-cleanup.md`

**Step 1: Write command**

Create `~/.claude/plugins/linear-autonomous/commands/linear-cleanup.md`:

```markdown
---
name: linear-cleanup
description: Clean up merged PRs and worktrees
arguments:
  - name: team_name
    description: Team name to clean up (optional, will use most recent if not provided)
    required: false
    type: string
---

# Linear Cleanup Command

Clean up after autonomous implementation: update Linear, delete worktrees, delete branches.

## Usage

```
/linear-cleanup [team-name]
```

## Examples

```
/linear-cleanup
/linear-cleanup linear-impl-20260206-1430
```

## What This Does

For each merged PR:
1. Update Linear ticket status → "Done"
2. Add Linear comment: "🎉 Merged and deployed"
3. Delete worktree
4. Delete local branch
5. Shutdown team

## Safety

Only cleans up PRs that are actually merged. Checks PR status first.

## When to Use

- After you've merged all ready PRs
- When implementation session is complete
- When you want to free up disk space
- When you want to shut down the team

---

**Implementation:**

```javascript
// 1. Find team (or use most recent)
if (!team_name) {
  const recent = Bash({
    command: "ls -t ~/.claude/teams/ | grep linear-impl | head -1"
  });
  team_name = recent.trim();
}

// 2. Load tasks
const tasks = TaskList();

// 3. For each task, check if PR merged
for (const task of tasks) {
  const { ticket_id, worktree_path, branch_name, pr_number } = task.metadata;

  // Check if PR merged
  const prStatus = Bash({
    command: `gh pr view ${pr_number} --json state,mergedAt -q '{state: .state, merged: .mergedAt}'`
  });

  if (prStatus.state === "MERGED") {
    // Update Linear
    mcp__plugin_linear_linear__update_issue({
      id: ticket_id,
      state: "Done"
    });

    mcp__plugin_linear_linear__create_comment({
      issueId: ticket_id,
      body: `🎉 Merged PR #${pr_number} and deployed`
    });

    // Delete worktree
    Bash({ command: `git worktree remove ${worktree_path}` });

    // Delete branch
    Bash({ command: `git branch -d ${branch_name}` });
  }
}

// 4. Shutdown team
TeamDelete({ team_name });

// 5. Report summary
console.log(`Cleaned up ${merged_count} merged PRs`);
```
```

**Step 2: Commit command**

```bash
git add commands/linear-cleanup.md
git commit -m "feat: Add linear-cleanup command for post-merge cleanup"
```

---

## Task 9: Create Reference Documents

**Files:**
- Create: `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/references/delegation-logic.md`
- Create: `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/references/linear-status-mapping.md`
- Create: `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/references/worktree-management.md`
- Create: `~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/references/error-escalation.md`

**Step 1: Create references directory**

```bash
mkdir -p ~/.claude/plugins/linear-autonomous/skills/autonomous-implementation/references
```

**Step 2: Write delegation logic reference**

Create `references/delegation-logic.md`:

```markdown
# Ticket Delegation Logic

How team-lead categorizes and delegates tickets to implementers.

## Hybrid Approach

1. **First: Check Linear labels** (explicit categorization)
2. **Then: Analyze content** (fallback when labels missing/ambiguous)

## Label-Based Categories

| Linear Label | Category | Notes |
|-------------|----------|-------|
| bug | bug-fix | Fixing existing functionality |
| feature | feature-dev | New functionality |
| refactor | refactor | Code improvement without behavior change |
| test | test-automation | Test coverage improvement |
| docs | documentation | Documentation updates |
| enhancement | feature-dev | Improvement to existing feature |
| chore | refactor | Maintenance work |

## Content Analysis Fallback

When no label or label ambiguous, analyze description for keywords:

### Bug Fix Indicators
- "fix", "bug", "error", "broken", "failing"
- "doesn't work", "not working", "crashes"
- "incorrect", "wrong behavior"

### Feature Development Indicators
- "add", "implement", "create", "new"
- "support", "enable", "allow"
- "feature request"

### Refactor Indicators
- "refactor", "clean up", "simplify"
- "improve code quality", "remove duplication"
- "reorganize", "restructure"

### Test Indicators
- "test", "coverage", "add tests"
- "test case", "test suite"

### Documentation Indicators
- "docs", "documentation", "README"
- "guide", "tutorial", "example"
- "comment", "docstring"

## Complexity Assessment

Analyze ticket for complexity (used for assignment):

### Simple (any implementer)
- Single file changes
- Clear requirements
- No dependencies
- Existing patterns to follow

### Medium (any implementer, but may need reviewer help)
- Multiple file changes
- Some ambiguity in requirements
- Few dependencies
- Some new patterns

### Complex (may require escalation)
- Many file changes
- Unclear requirements
- Many dependencies
- Architectural decisions needed

## Assignment Strategy

Both implementers are general-purpose, so:

1. Assign next ticket to idle implementer
2. Balance workload (don't overload one implementer)
3. If both idle, assign based on:
   - Ticket priority (urgent first)
   - Simpler tickets first (faster completion)
   - Related tickets to same implementer (context reuse)

## Example Delegation

```
Ticket: SDK-123
Labels: ["bug", "python-sdk"]
Priority: Urgent
Description: "Python client times out after 30s..."

→ Category: bug-fix
→ Complexity: simple (single file, clear issue)
→ Assign to: next idle implementer
```

```
Ticket: SDK-124
Labels: ["feature"]
Priority: High
Description: "Add async support to embeddings API..."

→ Category: feature-dev
→ Complexity: medium (multiple files, new pattern)
→ Assign to: next idle implementer
→ Note: May need reviewer consultation
```

```
Ticket: SDK-125
Labels: []
Priority: Normal
Description: "The error messages are confusing when..."

→ No label, analyze content
→ Keywords: "error messages", "confusing" → refactor/enhancement
→ Category: refactor
→ Complexity: simple
→ Assign to: next idle implementer
```
```

**Step 3: Write Linear status mapping reference**

Create `references/linear-status-mapping.md`:

```markdown
# Linear Status Field Mapping

How to update Linear ticket status throughout the workflow.

## Configuration

Status values are configurable per workspace:

```yaml
linear:
  status_field: "State"  # or "Status" depending on workspace
  status_values:
    todo: "Todo"
    in_progress: "In Progress"
    in_review: "In Review"
    done: "Done"
    blocked: "Blocked"
```

## Status Transitions

```
Todo → In Progress → In Review → Done
  ↓
Blocked
```

### Todo → In Progress

**When:** Implementer starts work on ticket

**Who:** Team-lead (when assigning task)

**Action:**
```javascript
mcp__plugin_linear_linear__update_issue({
  id: ticket_id,
  state: "In Progress"  // Use configured value
})
```

**Comment:**
```javascript
mcp__plugin_linear_linear__create_comment({
  issueId: ticket_id,
  body: "🤖 Implementation started by impl-1 in branch linear/SDK-123-feature"
})
```

### In Progress → In Review

**When:** Reviewer confirms PR ready (CI green + feedback addressed)

**Who:** Reviewer

**Action:**
```javascript
mcp__plugin_linear_linear__update_issue({
  id: ticket_id,
  state: "In Review"
})
```

**Comment:**
```javascript
mcp__plugin_linear_linear__create_comment({
  issueId: ticket_id,
  body: "✅ PR #456 ready for merge - all checks passing, all feedback addressed"
})
```

### In Review → Done

**When:** User merges PR (manual or via cleanup command)

**Who:** User or cleanup command

**Action:**
```javascript
mcp__plugin_linear_linear__update_issue({
  id: ticket_id,
  state: "Done"
})
```

**Comment:**
```javascript
mcp__plugin_linear_linear__create_comment({
  issueId: ticket_id,
  body: "🎉 Merged PR #456 and deployed"
})
```

### In Progress → Blocked

**When:** Escalation to team-lead (can't proceed without human input)

**Who:** Team-lead

**Action:**
```javascript
mcp__plugin_linear_linear__update_issue({
  id: ticket_id,
  state: "Blocked"
})
```

**Comment:**
```javascript
mcp__plugin_linear_linear__create_comment({
  issueId: ticket_id,
  body: `⚠️ Blocked: ${explanation}

**What was attempted:**
${attempts}

**Why it failed:**
${failure_reason}

**What's needed:**
${needed_info}

**Context:**
${error_logs}
`
})
```

## Comment Format Guidelines

### Implementation Started
```
🤖 Implementation started by {agent} in branch {branch}
```

### PR Opened
```
📝 PR #{number} opened: {pr-url}
```

### Review Started
```
👀 Code review in progress
```

### Addressing Feedback
```
🔄 Addressing feedback: {brief-summary}
```

### Ready for Merge
```
✅ PR #{number} ready for merge - all checks passing, all feedback addressed
```

### Blocked
```
⚠️ Blocked: {explanation}
[Detailed context as shown above]
```

### Merged
```
🎉 Merged PR #{number} and deployed
```

## Status Field Discovery

To find the correct status field name and values:

```javascript
// Get ticket to see current status field
const ticket = mcp__plugin_linear_linear__get_issue({ id: ticket_id });

// Status field could be:
// - ticket.state
// - ticket.status
// - ticket.workflowState

// List available statuses for team
const statuses = mcp__plugin_linear_linear__list_issue_statuses({
  team: "SDK"
});
```

## Best Practices

1. **Always use configured values** - don't hardcode status strings
2. **Update status at every transition** - keep Linear in sync
3. **Add meaningful comments** - explain what's happening
4. **Use emoji prefixes** - quick visual scanning
5. **Be specific in blocked comments** - help user understand issue
```

**Step 4: Write worktree management reference**

Create `references/worktree-management.md`:

```markdown
# Git Worktree Management

How to create, use, and clean up git worktrees for isolated parallel work.

## Why Worktrees?

- **Isolation**: Each agent works in separate directory
- **No conflicts**: No branch switching, no merge conflicts
- **Parallel work**: Two implementers work simultaneously
- **Clean separation**: Clear boundaries between tickets

## Worktree Lifecycle

### 1. Creation

**When:** Team-lead assigns ticket to implementer

**Where:** Configurable location (default: `/tmp/linear-worktrees/`)

**Command:**
```bash
git worktree add /tmp/linear-worktrees/{ticket-id} -b linear/{ticket-id}-{slug}
```

**Example:**
```bash
git worktree add /tmp/linear-worktrees/SDK-123 -b linear/SDK-123-fix-timeout
```

**Verification:**
```bash
cd /tmp/linear-worktrees/SDK-123
git branch --show-current
# Should show: linear/SDK-123-fix-timeout
```

### 2. Usage

**Implementer workflow:**
```bash
# Change to worktree
cd /tmp/linear-worktrees/SDK-123

# Make changes
vim src/client.py

# Run tests
pytest

# Commit
git add .
git commit -m "feat(SDK-123): fix timeout issue"

# Push
git push -u origin linear/SDK-123-fix-timeout
```

**Reviewer workflow:**
```bash
# Same worktree, apply fixes
cd /tmp/linear-worktrees/SDK-123

# Make fixes
vim src/client.py

# Commit
git add .
git commit -m "fix(SDK-123): address review feedback"

# Push
git push
```

### 3. Cleanup

**When:** PR merged (or ticket abandoned)

**Who:** User runs `/linear-cleanup` or team-lead on session end

**Commands:**
```bash
# Remove worktree
git worktree remove /tmp/linear-worktrees/SDK-123

# Delete branch (if merged)
git branch -d linear/SDK-123-fix-timeout

# Or force delete (if not merged)
git branch -D linear/SDK-123-fix-timeout
```

## Configuration

```yaml
worktree_base: "/tmp/linear-worktrees"
```

Can be changed to:
- `/tmp/linear-worktrees` - System temp (auto-cleanup on reboot)
- `~/.cache/linear-worktrees` - User cache (persistent)
- `.worktrees` - Project-local (add to .gitignore!)

## Branch Naming Convention

```
linear/{ticket-id}-{slug}
```

Examples:
- `linear/SDK-123-fix-timeout`
- `linear/SDK-124-add-async-support`
- `linear/SDK-125-refactor-errors`

## Path Management

**In task metadata:**
```javascript
{
  ticket_id: "SDK-123",
  worktree_path: "/tmp/linear-worktrees/SDK-123",
  branch_name: "linear/SDK-123-fix-timeout"
}
```

**Accessing from agent:**
```javascript
const task = TaskGet({ taskId: task_id });
const { worktree_path } = task.metadata;

// Change to worktree
Bash({ command: `cd ${worktree_path}` });
```

## Listing Worktrees

```bash
git worktree list
```

Output:
```
/Users/jhamon/workspace/examples                          8931b1a [main]
/tmp/linear-worktrees/SDK-123                             a1b2c3d [linear/SDK-123-fix-timeout]
/tmp/linear-worktrees/SDK-124                             d4e5f6g [linear/SDK-124-add-async]
```

## Common Issues

### Worktree Already Exists

**Error:** `fatal: '/tmp/linear-worktrees/SDK-123' already exists`

**Solution:**
```bash
# Remove old worktree
git worktree remove /tmp/linear-worktrees/SDK-123

# Try again
git worktree add /tmp/linear-worktrees/SDK-123 -b linear/SDK-123-fix-timeout
```

### Branch Already Exists

**Error:** `fatal: A branch named 'linear/SDK-123-fix-timeout' already exists`

**Solution:**
```bash
# Delete old branch
git branch -D linear/SDK-123-fix-timeout

# Try again
git worktree add /tmp/linear-worktrees/SDK-123 -b linear/SDK-123-fix-timeout
```

### Worktree Locked

**Error:** `fatal: 'remove' cannot be used with a locked working tree`

**Solution:**
```bash
# Unlock worktree
git worktree unlock /tmp/linear-worktrees/SDK-123

# Remove
git worktree remove /tmp/linear-worktrees/SDK-123
```

## Best Practices

1. **One worktree per ticket** - Clear isolation
2. **Clean up after merge** - Don't accumulate old worktrees
3. **Use temp directory** - Auto-cleanup on reboot
4. **Consistent naming** - Easy to identify ticket
5. **Check before creating** - Avoid duplicate worktrees
6. **Remove before deleting** - Don't just `rm -rf`
```

**Step 5: Write error escalation reference**

Create `references/error-escalation.md`:

```markdown
# Error Escalation Decision Tree

When and how to escalate issues up the chain.

## Escalation Chain

```
Implementer → Reviewer → Team-Lead → User (via Linear)
```

## Implementer Escalation

### When to Try Fixing Yourself (Attempt 1-3)

**Auto-fixable issues:**
- Syntax errors (typos, missing colons, etc.)
- Import errors (missing imports, wrong module names)
- Simple test failures (logic errors you can debug)
- Linting errors (formatting, unused variables)
- Type errors (missing type annotations, wrong types)

**Process:**
1. Read error message carefully
2. Identify root cause
3. Apply fix
4. Run tests again
5. If still failing, repeat (max 3 attempts)

### When to Escalate to Reviewer (After 3 Attempts)

**Escalate when:**
- Same error persists after 3 fix attempts
- Error message unclear or ambiguous
- Multiple failing tests with related issues
- Unsure about correct approach
- Need architectural guidance

**How to escalate:**
```javascript
SendMessage({
  type: "message",
  recipient: "reviewer",
  content: `Need help with ${ticket_id}. After 3 attempts, still failing with:

**Error:**
${error_message}

**What I tried:**
1. ${attempt_1}
2. ${attempt_2}
3. ${attempt_3}

**Context:**
${relevant_code_or_logs}

**Ticket:** ${ticket_url}
`,
  summary: "Help needed: ${brief_issue}"
})
```

**What reviewer does:**
- Reviews your attempts
- Identifies issue
- Either fixes it or escalates to team-lead

## Reviewer Escalation

### When to Fix Yourself

**Minor issues (fix directly):**
- Formatting issues
- Missing tests
- Simple logic errors
- Code quality improvements
- Adding error handling
- Fixing typos

**Process:**
1. Make fix in worktree
2. Commit with descriptive message
3. Push changes
4. Continue review loop

### When to Reassign to Implementer

**Major issues (original implementer should fix):**
- Wrong architectural approach
- Misunderstood requirements
- Missing core functionality
- Significant refactoring needed
- Complex logic errors

**How to reassign:**
```javascript
SendMessage({
  type: "message",
  recipient: "impl-1",  // original implementer
  content: `PR #${pr_number} needs significant changes:

**Issues:**
1. ${issue_1}
2. ${issue_2}

**Suggestions:**
${suggestions}

Please address and re-submit when ready.

**PR:** ${pr_url}
`,
  summary: "Major changes needed"
})

TaskUpdate({
  taskId: task_id,
  owner: "impl-1",
  status: "in_progress"
})
```

### When to Escalate to Team-Lead

**Infrastructure/process issues:**
- CI configuration problems
- Access/permission issues
- Unclear requirements (need product clarification)
- Architectural decisions needed
- External dependencies blocking progress

**How to escalate:**
```javascript
SendMessage({
  type: "message",
  recipient: "team-lead",
  content: `Blocked on ${ticket_id}:

**Issue:**
${issue_description}

**What I tried:**
${attempts}

**Why blocked:**
${blocker_reason}

**What's needed:**
${what_would_unblock}

**Ticket:** ${ticket_url}
**PR:** ${pr_url}
`,
  summary: "Escalation: ${brief}"
})
```

## Team-Lead Escalation

### When to Handle Internally

**Team-level issues:**
- Reassign work between implementers
- Adjust task priorities
- Coordinate between agents
- Retry failed operations

### When to Escalate to User

**Issues requiring human decision:**
- Ambiguous requirements
- Product/design decisions
- Infrastructure access
- External dependencies
- Resource constraints
- Architectural trade-offs

**How to escalate (via Linear):**
```javascript
mcp__plugin_linear_linear__update_issue({
  id: ticket_id,
  state: "Blocked"
})

mcp__plugin_linear_linear__create_comment({
  issueId: ticket_id,
  body: `⚠️ Blocked: ${brief_explanation}

## What Was Attempted

${detailed_attempts}

## Why It Failed

${failure_reason}

## What's Needed

${what_user_needs_to_provide}

## Context

\`\`\`
${error_logs_or_relevant_info}
\`\`\`

## Related

- PR: ${pr_url}
- Branch: ${branch_name}
- Worktree: ${worktree_path}
`
})
```

## Decision Matrix

| Issue Type | Implementer | Reviewer | Team-Lead | User |
|-----------|------------|----------|-----------|------|
| Syntax error | Fix (1-3x) | - | - | - |
| Logic error | Fix (1-3x) | Fix or reassign | - | - |
| Test failure | Fix (1-3x) | Fix or reassign | - | - |
| Unclear error | Escalate | Fix or escalate | - | - |
| Wrong approach | Escalate | Reassign | - | - |
| CI config issue | Escalate | Escalate | Block | User fixes |
| Unclear requirements | Escalate | Escalate | Block | User clarifies |
| Access issue | Escalate | Escalate | Block | User grants |

## Escalation Timing

### Implementer
- 3 fix attempts = ~15-30 minutes
- Escalate sooner if obviously blocked

### Reviewer
- 3 fix attempts = ~15-30 minutes
- Reassign sooner if major issue obvious
- Escalate immediately if infrastructure issue

### Team-Lead
- Attempt internal resolution first
- Escalate to user within ~30 minutes if blocked
- Don't let tickets sit blocked without user visibility

## Best Practices

1. **Try fixing first** - Most issues are auto-fixable
2. **Escalate with context** - Explain what you tried
3. **Be specific** - Exact error messages, not vague descriptions
4. **Include links** - Ticket URL, PR URL, relevant files
5. **Suggest solutions** - If you have ideas, share them
6. **Don't spin wheels** - 3 attempts max, then escalate
7. **Use Linear for user escalation** - Async, trackable
```

**Step 6: Commit reference documents**

```bash
git add skills/autonomous-implementation/references/
git commit -m "docs: Add reference documentation for implementation patterns"
```

---

## Task 10: Test Plugin Installation

**Step 1: Verify plugin structure**

```bash
cd ~/.claude/plugins/linear-autonomous
tree -L 3
```

Expected output:
```
.
├── README.md
├── agents
│   ├── implementer.md
│   ├── reviewer.md
│   └── team-lead.md
├── commands
│   ├── linear-cleanup.md
│   ├── linear-implement.md
│   └── linear-resume.md
├── plugin.json
├── .linear-autonomous.local.md
└── skills
    └── autonomous-implementation
        ├── SKILL.md
        └── references
            ├── delegation-logic.md
            ├── error-escalation.md
            ├── linear-status-mapping.md
            └── worktree-management.md
```

**Step 2: Validate plugin.json**

```bash
cat plugin.json | jq .
```

Should parse without errors.

**Step 3: Commit verification**

```bash
git status
git log --oneline -10
```

All files should be committed.

**Step 4: Push to backup (optional)**

```bash
# If you want to back up to a remote
git remote add origin <your-repo-url>
git push -u origin main
```

---

## Summary

This plan creates a complete Claude Code plugin for autonomous Linear ticket implementation:

✅ **Plugin structure** (Task 1)
✅ **Team-lead agent** (Task 2) - Orchestration
✅ **Implementer agent** (Task 3) - Autonomous implementation
✅ **Reviewer agent** (Task 4) - Code review iteration
✅ **Main skill** (Task 5) - Orchestration logic
✅ **Commands** (Tasks 6-8) - User interface
✅ **Reference docs** (Task 9) - Implementation patterns
✅ **Testing** (Task 10) - Verification

## Next Steps

After implementation:

1. Test with real Linear tickets:
   ```
   /linear-implement "label test"
   ```

2. Monitor agent behavior and refine

3. Add example workflows in `skills/autonomous-implementation/examples/`

4. Consider additional features:
   - Smart batching (group related tickets)
   - Metrics tracking (velocity, review time)
   - Multi-repo support

## Key Principles Applied

- **TDD**: Agents implement tests first
- **DRY**: Reusable patterns in references
- **YAGNI**: Only build what's in the design
- **Bite-sized**: Each task is 2-5 minutes
- **Clear paths**: Exact file locations always
- **Complete code**: No placeholders
