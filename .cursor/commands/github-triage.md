# github-triage

Quick triage and routing of GitHub issues for example notebooks.

**IMPORTANT: This command must be idempotent.** Before taking any action, check:
- If a Linear issue already exists for this GitHub issue
- If information requests have already been posted
- If the issue has already been closed
- Existing labels before adding or removing any

## Triage Process

1. **Categorize the issue**
   - Determine if it's a bug, improvement request, question, or other
   - Assess priority based on impact
   - Check if it's a duplicate
   - Apply appropriate labels: `bug`, `enhancement`, `question`, `documentation`

2. **Extract key information**
   - Summarize the problem or request
   - Identify which notebook(s) are affected
   - Note relevant context (Python version, package versions, etc.)

3. **Request additional information (if needed)**
   - Check if similar requests have already been made
   - If missing: error messages, steps to reproduce, environment details
   - Be polite and specific

4. **Route to next step**
   - **Clear bug**: Create Linear issue using `/create-bug`
   - **Clear improvement**: Create Linear issue using `/create-improvement`
   - **Question**: Provide a helpful answer or point to documentation
   - **Missing information**: Wait for user response

5. **Link tracking**
   - Include the GitHub issue URL when creating Linear issues
   - Maintain bidirectional references between GitHub and Linear
