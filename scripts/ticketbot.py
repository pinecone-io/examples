#!/usr/bin/env python3
"""
Ticketbot - Automated ticket processing with independent worker pools.

This script manages ticket lifecycle automation by spawning Cursor agents
to handle different stages: picking new work, iterating on PRs, and merging.
"""

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_draining = False


def _handle_drain_signal(signum, frame):
    """Handle SIGTERM by setting drain flag."""
    global _draining
    _draining = True
    logger.info("Received shutdown signal, will exit after current iteration completes...")

WORKSPACE = Path(os.environ.get("TICKETBOT_WORKSPACE", os.getcwd()))
WORKTREE_DIR = WORKSPACE.parent / f"{WORKSPACE.name}-worktrees"
LOG_DIR = WORKSPACE.parent / f"{WORKSPACE.name}-logs"


def get_worktree(job: str, worker_index: int) -> Path:
    """Get or create a worktree for this worker."""
    worktree_path = WORKTREE_DIR / f"{job}-{worker_index}"

    if not worktree_path.exists():
        WORKTREE_DIR.mkdir(exist_ok=True)
        # Use --detach to avoid conflicts with branches checked out elsewhere
        # The Cursor commands will handle checking out the appropriate branch
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree_path), "HEAD"],
            cwd=WORKSPACE,
            check=True,
        )
        logger.info(f"Created worktree: {worktree_path}")

    return worktree_path


MAX_RETRIES = 3
INITIAL_BACKOFF = 10  # seconds


def invoke_cursor(prompt: str, worktree: Path, worker_id: str) -> subprocess.CompletedProcess:
    """Invoke Cursor CLI with the given prompt in the specified worktree.
    
    Retries with exponential backoff on transient failures.
    """
    logger.info(f"[{worker_id}] Invoking Cursor in {worktree}")
    
    # Ensure log directory exists
    LOG_DIR.mkdir(exist_ok=True)
    
    last_error = None
    
    for attempt in range(MAX_RETRIES):
        # Create timestamped log file for this run
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = LOG_DIR / f"{worker_id}-{timestamp}.log"
        
        result = subprocess.run(
            [
                "agent",
                "--print",
                "--force",
                "--approve-mcps",
                "--workspace", str(worktree),
                prompt,
            ],
            capture_output=True,
            text=True,
        )
        
        # Write output to log file
        with open(log_file, "w") as f:
            f.write(f"=== Ticketbot Agent Log ===\n")
            f.write(f"Worker: {worker_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Attempt: {attempt + 1}/{MAX_RETRIES}\n")
            f.write(f"Worktree: {worktree}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Exit Code: {result.returncode}\n")
            f.write(f"\n=== STDOUT ===\n")
            f.write(result.stdout or "(empty)")
            f.write(f"\n\n=== STDERR ===\n")
            f.write(result.stderr or "(empty)")
        
        logger.info(f"[{worker_id}] Agent output logged to: {log_file}")
        
        # Success
        if result.returncode == 0:
            return result
        
        # Check for transient errors that should be retried
        stderr_lower = (result.stderr or "").lower()
        is_transient = any(err in stderr_lower for err in [
            "connection stalled",
            "connection reset",
            "connection refused",
            "timeout",
            "rate limit",
            "503",
            "502",
            "504",
        ])
        
        if not is_transient:
            # Non-transient error, don't retry
            raise subprocess.CalledProcessError(
                result.returncode, result.args, result.stdout, result.stderr
            )
        
        # Transient error, retry with backoff
        last_error = subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
        
        if attempt < MAX_RETRIES - 1:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            logger.warning(
                f"[{worker_id}] Transient error (attempt {attempt + 1}/{MAX_RETRIES}), "
                f"retrying in {backoff}s: {result.stderr[:100] if result.stderr else 'unknown error'}"
            )
            time.sleep(backoff)
    
    # All retries exhausted
    logger.error(f"[{worker_id}] All {MAX_RETRIES} attempts failed")
    raise last_error


def pick_work(worker_index: int) -> None:
    """Pick up new work from the backlog."""
    worker_id = f"tb-pick-work-{worker_index}"
    logger.info(f"[{worker_id}] Starting pick_work iteration")

    worktree = get_worktree("tb-pick-work", worker_index)
    prompt = "Run /tb-pick-next-ticket"

    try:
        invoke_cursor(prompt, worktree, worker_id)
        logger.info(f"[{worker_id}] Completed pick_work iteration")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{worker_id}] Cursor failed: {e.stderr}")
        raise


def iterate_prs(worker_index: int, total_workers: int) -> None:
    """Iterate on PRs in review, filtered by shard."""
    worker_id = f"tb-iterate-prs-{worker_index}"
    logger.info(f"[{worker_id}] Starting iterate_prs iteration (shard {worker_index}/{total_workers})")

    worktree = get_worktree("tb-iterate-prs", worker_index)
    prompt = f"""Run /tb-iterate-review-tickets
Worker shard: {worker_index} of {total_workers}
Only process tickets where (ticket_number % {total_workers}) == {worker_index}"""

    try:
        invoke_cursor(prompt, worktree, worker_id)
        logger.info(f"[{worker_id}] Completed iterate_prs iteration")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{worker_id}] Cursor failed: {e.stderr}")
        raise


def cleanup_orphaned() -> None:
    """Find orphaned In Progress tickets and move them back to Backlog."""
    worker_id = "tb-cleanup-orphaned"
    logger.info(f"[{worker_id}] Starting cleanup of orphaned tickets")

    worktree = get_worktree("tb-cleanup", 0)
    prompt = "Run /tb-cleanup-orphaned"

    try:
        invoke_cursor(prompt, worktree, worker_id)
        logger.info(f"[{worker_id}] Completed cleanup")
    except subprocess.CalledProcessError as e:
        logger.error(f"[{worker_id}] Cursor failed: {e.stderr}")
        raise


def spawn_worker(
    job: str,
    interval: int,
    worker_index: int = 0,
    total_workers: int = 1,
) -> subprocess.Popen:
    """Spawn a worker subprocess for the given job type."""
    cmd = [
        sys.executable,
        __file__,
        "run",
        "--job", job,
        "--interval", str(interval),
        "--worker-index", str(worker_index),
        "--total-workers", str(total_workers),
    ]
    logger.info(f"Spawning worker: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


@click.group()
def cli():
    """Ticketbot - Automated ticket processing with independent worker pools."""
    pass


@cli.command()
@click.option("--pick-workers", default=3, help="Number of pick-work workers")
@click.option("--iterate-workers", default=3, help="Number of iterate-prs workers")
@click.option("--interval", default=45, help="Seconds between job iterations")
@click.option("--stagger", default=30, help="Seconds between worker starts within a pool")
def start_all(
    pick_workers: int,
    iterate_workers: int,
    interval: int,
    stagger: int,
):
    """Start all worker pools."""
    processes = []

    # Spawn pick workers
    for i in range(pick_workers):
        if i > 0:
            time.sleep(stagger)
        processes.append(spawn_worker("tb-pick-work", interval, worker_index=i))

    # Spawn iterate workers (sharded by index)
    # These also handle merging when PR is ready
    for i in range(iterate_workers):
        if i > 0:
            time.sleep(stagger)
        processes.append(
            spawn_worker(
                "tb-iterate-prs",
                interval,
                worker_index=i,
                total_workers=iterate_workers,
            )
        )

    logger.info(f"Started {len(processes)} workers. Press Ctrl+C to drain and stop.")

    try:
        # Wait for all processes (they run forever, so this blocks until interrupt)
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        logger.info("Draining workers (waiting for current iterations to complete)...")
        
        # Send SIGTERM to all workers to trigger drain mode
        for p in processes:
            if p.poll() is None:  # Still running
                p.send_signal(signal.SIGTERM)
        
        # Wait for workers to finish gracefully (with timeout)
        drain_timeout = 600  # 10 minutes max wait
        start_time = time.time()
        
        while any(p.poll() is None for p in processes):
            elapsed = time.time() - start_time
            if elapsed > drain_timeout:
                logger.warning(f"Drain timeout ({drain_timeout}s) exceeded, force killing...")
                for p in processes:
                    if p.poll() is None:
                        p.kill()
                break
            
            remaining = sum(1 for p in processes if p.poll() is None)
            logger.info(f"Waiting for {remaining} workers to finish (elapsed: {int(elapsed)}s)...")
            time.sleep(5)
        
        logger.info("All workers stopped.")


@cli.command()
@click.option(
    "--job",
    type=click.Choice(["tb-pick-work", "tb-iterate-prs"]),
    required=True,
    help="Job type to run",
)
@click.option("--interval", default=45, help="Seconds between iterations")
@click.option("--worker-index", default=0, help="This worker's index (for sharding)")
@click.option("--total-workers", default=1, help="Total workers of this job type (for sharding)")
def run(job: str, interval: int, worker_index: int, total_workers: int):
    """Run a single job type in a loop."""
    global _draining
    
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGTERM, _handle_drain_signal)
    
    worker_id = f"{job}-{worker_index}"
    logger.info(f"[{worker_id}] Starting worker loop (interval={interval}s)")

    while not _draining:
        try:
            if job == "tb-pick-work":
                pick_work(worker_index)
            elif job == "tb-iterate-prs":
                iterate_prs(worker_index, total_workers)
        except Exception as e:
            logger.error(f"[{worker_id}] Error: {e}")

        # Check drain flag before sleeping
        if _draining:
            break

        # Pick workers sleep between iterations to avoid creating too many tickets
        # Iterate workers have a brief pause to avoid rate limits
        if job == "tb-pick-work":
            logger.info(f"[{worker_id}] Sleeping {interval}s...")
            # Sleep in small increments to respond to drain signal faster
            for _ in range(interval):
                if _draining:
                    break
                time.sleep(1)
        else:
            # Brief pause to avoid rate limits
            time.sleep(10)
    
    logger.info(f"[{worker_id}] Draining complete, exiting gracefully.")


@cli.command("tb-pick-work")
@click.option("--worker-index", default=0, help="Worker index for worktree")
def tb_pick_work_cmd(worker_index: int):
    """Run pick-work once (for testing)."""
    pick_work(worker_index)


@cli.command("tb-iterate-prs")
@click.option("--worker-index", default=0, help="Worker index for sharding and worktree")
@click.option("--total-workers", default=1, help="Total workers for sharding")
def tb_iterate_prs_cmd(worker_index: int, total_workers: int):
    """Run iterate-prs once (for testing)."""
    iterate_prs(worker_index, total_workers)


@cli.command("tb-cleanup-orphaned")
def tb_cleanup_orphaned_cmd():
    """Find orphaned In Progress tickets and move them back to Backlog."""
    cleanup_orphaned()


if __name__ == "__main__":
    cli()
