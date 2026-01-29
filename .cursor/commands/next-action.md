# next-action

Checkout the master branch and pull to get the latest changes.

## Choose a ticket

Find the next Linear ticket in project "Notebook Examples" with label "docs:examples" 
that has status "Backlog", prioritized by priority then creation date.

Mark the ticket as started in Linear.

## Review and Plan

Review the ticket description:
- If it contains a detailed implementation plan, validate the plan and proceed 
  with implementation if sensible.
- If no plan exists, draft one and get my approval before continuing.

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

Update the Linear ticket with a link to the GitHub PR.

## Review feedback and iterate

Check the PR to see if there are any CI failures, review comments, or inline comments.

For each error or piece of feedback:
- Verify the problem exists. If the problem does not exist, reply on github with relevant context.
- Plan and implement a fix for the problem
- Make a commit that describes the change
- Push the commit
- If the feedback is an inline comment, reply saying the feedback was addressed. Then mark the conversation as resolved.
- If the feedback was from an overall review comment, reply saying the feedback was addressed.

Next, check the PR to see if there are any inline comments that are not part of a formal review. Follow the same procedure for confirming, fixing, and pushing changes. Mark the conversation as resolved when the fix has been pushed.

## Merge

Verify all of these conditions:
- The PR has an appropriate GitHub label
- The PR has a completed Bugbot review with all feedback comments addressed
- All CI checks are passing (if not, run `/pr-iterate`)
- There is no unresolved review feedback (if there is, run `/pr-iterate`)

If all conditions are met, merge the PR.