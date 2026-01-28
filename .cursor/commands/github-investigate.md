# github-investigate

Conduct a thorough investigation of a GitHub issue to understand the problem and produce actionable next steps.

**When to use:**
- After `/github-triage` determines the issue needs investigation
- When an issue is complex or unclear
- When you need to understand the scope before prioritizing

## Investigation Process

### 1. Understand the Issue
- Read the issue title, description, and all comments
- Extract: problem description, error messages, steps to reproduce, environment details
- Identify which notebook(s) and sections are affected

### 2. Analyze the Content
- Review the affected notebook(s)
- Identify the problematic code or documentation
- Check for similar patterns in other notebooks
- Look for related issues or PRs

### 3. Assess Impact
- How many notebooks are affected?
- Is this a common workflow or edge case?
- What's the impact on users trying to learn from the example?

### 4. Document Findings

**Summary**
- Brief overview of the issue
- Type (bug, improvement, etc.)
- Severity assessment

**Analysis**
- Root cause or hypothesis
- Relevant code or documentation references
- Related issues found

**Next Steps**
- Recommended action: create bug issue, improvement issue, or provide answer
- Suggested priority
- Estimated effort

### 5. Update GitHub Issue (if appropriate)
- Share key findings
- Suggest next steps
- Request additional information if needed
