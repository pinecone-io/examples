# process-queue

Execute `/next-action` workflow. When complete (PR merged or blocked), 
automatically pick up the next available ticket and repeat.

Continue until no "Backlog" tickets remain with label "docs:examples".

## Autonomous Operation

This is a fully autonomous workflow. Assume permission to continue through the 
entire queue. No confirmation is needed between items.

- Do NOT stop to ask for confirmation between tickets
- Do NOT pause after context summarization—continue working
- If context is summarized mid-task, resume from where you left off
- Only stop when the queue is empty or you encounter an unrecoverable error
- Treat each ticket completion as a trigger to immediately start the next one