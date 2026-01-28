# pr-merge

Check the status of the pull request.

Verify all of these conditions:
- The PR has an appropriate GitHub label
- The PR has a completed Bugbot review with all comments addressed
- All CI checks are passing (if not, run `/pr-iterate`)
- There is no unresolved review feedback (if there is, run `/pr-iterate`)

If all conditions are met, merge the PR.
