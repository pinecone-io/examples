# process-queue

Execute `/next-action` workflow. When complete (PR merged or blocked), 
automatically pick up the next available ticket and repeat.

Continue until no "Backlog" tickets remain with label "docs:examples".