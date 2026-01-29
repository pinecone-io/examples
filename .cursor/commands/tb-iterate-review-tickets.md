# tb-iterate-review-tickets

Find one ticket in review that matches this worker's shard, then actively work on its PR.

**IMPORTANT**: Do not just list PRs or tickets. Pick ONE and actively work on it.

## Parse Shard Info

This command is invoked with shard information in the prompt:
- Worker shard index (e.g., 0, 1, 2)
- Total workers (e.g., 3)

## Find a Ticket

Use Linear MCP to query for tickets in project "Notebook Examples" with label "docs:examples" 
that have status "In Review".

Filter the results to only tickets where:
`(numeric_portion_of_ticket_id % total_workers) == worker_index`

For example, if the ticket ID is "EXA-123" and this is worker 1 of 3:
- Extract 123 from the ID
- Check: 123 % 3 = 0, which does not equal 1
- Skip this ticket

**Pick the first matching ticket.** If no matching tickets are found, say "No tickets for this shard" and exit.

## Find the Associated PR

From the selected ticket, find the associated GitHub PR. Check:
- The ticket description for a PR link
- Comments on the ticket
- Or use `gh pr list` to find PRs mentioning the ticket ID

**If no PR is found, say "No PR found for ticket [ID]" and exit.**

## Checkout the PR Branch

Run these commands to checkout the PR branch:
```bash
git fetch origin
gh pr checkout <PR_NUMBER>
```

**Important:** After checkout, restore ticketbot files to avoid including them in commits:
```bash
git checkout origin/master -- scripts/ticketbot.py .cursor/commands/tb-*.md .cursor/commands/process-*.md 2>/dev/null || true
git reset HEAD scripts/ .cursor/commands/ 2>/dev/null || true
```

## Run PR Iteration

**Now actively work on this PR.** Check for CI failures, review comments, or inline comments using:
```bash
gh pr checks <PR_NUMBER>
gh pr view <PR_NUMBER> --comments
```

For each error or piece of feedback, first assess the scope:

**Small/easy fixes (address now):**
- Typos, formatting issues, minor code improvements
- Bug fixes directly related to the PR's purpose
- Clarifications or documentation improvements within scope

For these:
- Plan and implement a fix for the problem
- Stage ONLY the files you modified (do NOT use `git add -A` or `git add .`):
  ```bash
  git add <specific-files-you-changed>
  ```
- Make a commit that describes the change
- Push the commit
- Reply saying the feedback was addressed and mark the conversation as resolved

**IMPORTANT:** Never stage `scripts/ticketbot.py` or `.cursor/commands/tb-*.md` files - these are automation files that should not be part of PR changes.

**Large or out-of-scope changes (defer to follow-up):**
- Significant refactoring beyond the PR's original purpose
- New features or enhancements not in the original ticket
- Changes that would require substantial additional testing or refactoring
- Suggestions that are good ideas but not blocking for this PR

For these, create a follow-up ticket:

1. **Create a Linear issue** in the "Notebook Examples" project with:
   - **Title**: A clear, actionable summary of the feedback
   - **Labels**: `docs:examples` and `Improvement`
   - **Description** that includes:
     - The original feedback/suggestion
     - Link to the PR where this was raised
     - Link to the original ticket this is a follow-up to
     - Any relevant context about why this was deferred
     - The standard implementation instructions:

```
This ticket needs to be implemented in the https://github.com/pinecone-io/examples repository.

Pull master to ensure you have the latest changes before beginning any implementation work.

When the fix is complete, the agent should:

1. **Quality Review**
   - Verify the notebook executes successfully from top to bottom
   - Ensure code follows Python best practices and is well-commented
   - Check that all markdown cells render correctly
   - Score this notebook according to the criteria in .github/NOTEBOOK_REVIEW_TEMPLATE.md and implement fixes for any issues uncovered.

2. **Create Pull Request**
   - Create a PR with a title that follows Conventional Commits format
   - Include a clear description of what was improved and why
   - Reference the related GitHub issue and Linear issue number
   - Summarize the value of the improvement
```

2. **Reply on GitHub** with: "This is a good suggestion but outside the scope of this PR. Created follow-up ticket [TICKET-ID] to address this separately."

3. **Mark the conversation as resolved**

**Invalid feedback:**
- If the problem does not exist, reply on GitHub with relevant context explaining why. Then resolve the conversation.

Next, check the PR to see if there are any inline comments that are not part of a formal review. Follow the same procedure for confirming, fixing, and pushing changes. Mark the conversation as resolved when the fix has been pushed.

## Check if Ready to Merge

After addressing all feedback, check if this PR is now ready to merge:

```bash
gh pr checks <PR_NUMBER>
gh pr view <PR_NUMBER> --json reviews,state,mergeable
```

**Merge conditions:**
1. All CI checks are passing
2. No unresolved review feedback (comments with follow-up tickets count as resolved)
3. Has an appropriate GitHub label
4. PR is in a mergeable state

**If all conditions are met:**

1. Merge the PR:
   ```bash
   gh pr merge <PR_NUMBER> --squash --delete-branch
   ```

2. Update the Linear ticket status to "Done"

3. Say "Merged PR #[NUMBER] and marked ticket [ID] as Done"

**If NOT ready to merge:**

Say "PR #[NUMBER] is not ready to merge yet" and list what's blocking:
- Failing CI checks
- Unresolved feedback
- Missing labels
- etc.

The next iteration will pick it up again.

## Summary

After completing work on this PR:
- All addressable feedback should be fixed and pushed
- All conversations should be resolved (either fixed or deferred with follow-up ticket)
- Any follow-up tickets should be created in Linear
- If ready, the PR should be merged and ticket marked Done

**Remember**: The goal is to make progress on the PR, not just report its status.
