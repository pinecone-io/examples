# tb-cleanup-orphaned

Find tickets marked "In Progress" that have no associated PR and move them back to Backlog.

**IMPORTANT**: This is a cleanup operation. Do not create PRs or do implementation work.

## Find In Progress Tickets

Use Linear MCP to query for tickets in project "Notebook Examples" with label "docs:examples" 
that have status "In Progress".

## Check Each Ticket for Associated PR

For each ticket found:

1. **Extract the ticket identifier** (e.g., "EXA-123")

2. **Search GitHub for associated PRs** using:
   ```bash
   gh pr list --state all --search "<ticket_identifier>"
   ```
   
   Also check for PRs that might reference the ticket in the branch name:
   ```bash
   gh pr list --state open --json number,title,headRefName | grep -i "<ticket_identifier>"
   ```

3. **Determine if the ticket is orphaned**:
   - If NO open PR is found that references this ticket → ticket is ORPHANED
   - If an open PR exists → ticket is NOT orphaned (leave it alone)
   - If only closed/merged PRs exist → ticket may need status update to "Done"

## Handle Orphaned Tickets

For each orphaned ticket (In Progress with no open PR):

1. **Update the ticket status** in Linear to "Backlog"

2. **Add a comment** to the ticket explaining:
   ```
   Automated cleanup: This ticket was marked "In Progress" but no associated GitHub PR was found. 
   Moving back to Backlog so it can be picked up again.
   ```

3. **Log the action**: "Moved [TICKET-ID] back to Backlog (no PR found)"

## Handle Tickets with Merged PRs

If a ticket is "In Progress" but has a merged PR (not open):

1. **Update the ticket status** in Linear to "Done"

2. **Add a comment**:
   ```
   Automated cleanup: Found merged PR #[NUMBER]. Updating status to Done.
   ```

3. **Log the action**: "Marked [TICKET-ID] as Done (PR #[NUMBER] was merged)"

## Summary

At the end, report:
- Total tickets checked
- Tickets moved to Backlog (orphaned)
- Tickets marked Done (merged PR found)
- Tickets left unchanged (open PR exists)
