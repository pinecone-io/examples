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
- Make a commit that describes the change
- Push the commit
- Reply saying the feedback was addressed and mark the conversation as resolved

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

## Done

After completing work on this PR:
- All addressable feedback should be fixed and pushed
- All conversations should be resolved (either fixed or deferred with follow-up ticket)
- Any follow-up tickets should be created in Linear

Exit after processing this one PR. The next iteration will pick up the next ticket.

**Remember**: The goal is to make progress on the PR, not just report its status.
