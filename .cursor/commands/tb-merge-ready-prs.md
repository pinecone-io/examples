# tb-merge-ready-prs

Find one PR that is ready to merge and merge it.

## Find Merge-Ready PRs

Use `gh pr list` to find open PRs in this repository. For each PR, check if it meets 
all merge conditions:

1. All CI checks are passing
2. Has an appropriate GitHub label
3. Has no unresolved review feedback (comments with follow-up tickets linked count as resolved)
4. Has a completed Bugbot review with all comments addressed or deferred to follow-up tickets

## Select a Candidate

If no PRs meet all conditions, log "No PRs ready to merge" and exit.

If multiple PRs are ready, pick one (preferably the oldest).

## Checkout the PR Branch

Fetch and checkout the branch associated with the selected PR.

## Verify and Merge

Verify all conditions one more time:
- The PR has an appropriate GitHub label
- The PR has a completed Bugbot review with all comments addressed or deferred to follow-up tickets
- All CI checks are passing
- There is no unresolved review feedback (comments with follow-up tickets linked count as resolved)

If any condition is not met, log the issue and exit without merging.

If all conditions are met, merge the PR.

## Update Linear

After successful merge, find the associated Linear ticket and update its status to "Done".

## Done

Exit after processing this one PR. The next iteration will pick up the next one.
