# create-bug

Create a Linear issue for this work in the "Notebook Examples" project.

Give the issue these labels:
- docs:examples
- Bug

Include the plan and any relevant context in the issue description.

If the plan was made to address a GitHub issue, include a link to the original GitHub issue in the description.

Every Linear issue created should include these additional instructions at the end of the description.

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
