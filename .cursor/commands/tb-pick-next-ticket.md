# tb-pick-next-ticket

Pick up one ticket from the backlog and implement it.

## Check Capacity

Query Linear for tickets in project "Notebook Examples" with label "docs:examples" 
that have status "In Progress".

If the count is 6 or more, log "At capacity - skipping" and exit immediately.

## Checkout Master

Checkout the master branch and pull to get the latest changes.

## Choose a Ticket

Find the next Linear ticket in project "Notebook Examples" with label "docs:examples" 
that has status "Backlog", prioritized by priority then creation date.

Mark the ticket as started in Linear (set status to "In Progress").

## Review and Plan

Review the ticket description:
- If it contains a detailed implementation plan, validate the plan and proceed 
  with implementation if sensible.
- If no plan exists, draft a reasonable implementation plan based on the ticket 
  description and proceed with implementation.

## Implement 

- Create a new feature branch
- Iterate by assessing the code against the criteria in .github/NOTEBOOK_REVIEW_TEMPLATE.md
- Ensure the notebook has valid syntax
- Format the notebook changed using uv to run ruff format on the notebook

## Open a PR

**Title:**
- Use a title that follows Conventional Commits format

**Description:**
- Include a clear summary of what the example or documentation accomplishes
- Describe the intended audience and use case
- List any prerequisites or dependencies
- Highlight key concepts demonstrated
- Link to any related GitHub issues, PRs, or Linear issues
- Link to related Pinecone documentation where relevant
- Avoid long lists of files changed

**Metadata:**
- Apply a GitHub label to the PR (helpful for categorizing changes)

Update the Linear ticket:
- Add a link to the GitHub PR
- Set status to "In Review"

## Done

Exit after opening the PR. Review iteration is handled by a separate worker.
