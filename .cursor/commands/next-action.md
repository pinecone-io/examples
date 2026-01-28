# next-action

Checkout the master branch and pull to get the latest changes.

Find the next Linear ticket in project "Notebook Examples" with label "docs:examples" 
that has status "Todo" or "Backlog", prioritized by priority then creation date.

Show me the ticket title, description, and priority. Ask for confirmation before 
marking it as started.

Once confirmed, mark the ticket as started.

Review the ticket description:
- If it contains a detailed implementation plan, validate the plan and proceed 
  with implementation if sensible.
- If no plan exists, draft one and get my approval before continuing.

For implementation: create a new feature branch, make changes, verify the notebook 
runs successfully, and run /pr-open when complete.
