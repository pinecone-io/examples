# github-list

List GitHub issues for batch triage or review.

## Usage

Fetch and display open GitHub issues to process with `/github-triage`.

## Output Format

For each issue, show:
- Issue number and URL
- Current labels
- Last updated date
- Issue title

## Filtering Options

Filter by:
- **State**: `open`, `closed`
- **Labels**: `bug`, `enhancement`, `question`, etc.
- **Sort order**: `created`, `updated`, `comments`

## Suggested Workflow

1. **List untriaged issues**: Focus on issues without status labels
2. **Prioritize**: Start with unlabeled or high-priority issues
3. **Process iteratively**: For each issue, run `/github-triage`
