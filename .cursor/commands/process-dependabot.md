# process-dependabot

List open PRs on GitHub to find open pull requests from dependabot.

For each dependabot PR, execute `/pr-iterate` and `/pr-merge` workflows. Request a dependabot rebase if needed. When complete (PR merged or blocked), automatically move on to the next open dependabot PR and repeat.

Continue until no dependabot PRs remain open.