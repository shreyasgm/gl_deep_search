#!/usr/bin/env python3
"""
Test Batch Processing Orchestrator for GCP ETL Pipeline.

This script orchestrates a test run of the ETL pipeline with a limited number
of publications, monitors execution, tracks costs in real-time, and generates
a comprehensive test report.

The pipeline runs in Docker containers on VM instances, using pre-built images
from Artifact Registry. This ensures consistency with Cloud Run deployments.

Safety Features:
- Active cost monitoring during execution
- Automatic VM termination if cost thresholds exceeded
- Hard stops at Phase 1 ($1.00) and Phase 2 ($10.00) limits
- Periodic cost checks every 2 minutes

Usage:
    python deployment/vm/test_batch_processing.py --limit 10 --phase 1
    python deployment/vm/test_batch_processing.py --limit 100 --phase 2

Prerequisites:
    - Docker image must be built and pushed: ./deployment/cloud-run/deploy.sh
    - GCP configuration must include IMAGE_NAME
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Colors:
    """ANSI color codes for terminal output."""

    BLUE = "\033[0;34m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    RED = "\033[0;31m"
    NC = "\033[0m"  # No Color


def log_info(message: str) -> None:
    """Log an info message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {timestamp} - {message}", file=sys.stderr)


def log_success(message: str) -> None:
    """Log a success message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{Colors.GREEN}[SUCCESS]{Colors.NC} {timestamp} - {message}", file=sys.stderr
    )


def log_warning(message: str) -> None:
    """Log a warning message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"{Colors.YELLOW}[WARNING]{Colors.NC} {timestamp} - {message}", file=sys.stderr
    )


def log_error(message: str) -> None:
    """Log an error message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{Colors.RED}[ERROR]{Colors.NC} {timestamp} - {message}", file=sys.stderr)


def log_step(message: str) -> None:
    """Log a step header."""
    print("\n" + "━" * 60, file=sys.stderr)
    print(f"▶ {message}", file=sys.stderr)
    print("━" * 60 + "\n", file=sys.stderr)


def run_command(
    cmd: list[str], check: bool = True, capture_output: bool = False
) -> subprocess.CompletedProcess:
    """
    Run a shell command and return the result.

    Args:
        cmd: Command to run as a list of strings
        check: If True, raise exception on non-zero exit code
        capture_output: If True, capture stdout/stderr

    Returns:
        CompletedProcess object
    """
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        return result
    except subprocess.CalledProcessError as e:
        log_error(f"Command failed: {' '.join(cmd)}")
        log_error(f"Error: {e.stderr if e.stderr else str(e)}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {cmd[0]}")
        raise


def load_gcp_config() -> dict[str, Any]:
    """
    Load GCP configuration from gcp-config.sh.

    Returns:
        Dictionary of configuration values
    """
    script_dir = Path(__file__).parent
    config_file = script_dir.parent / "config" / "gcp-config.sh"

    if not config_file.exists():
        log_error(f"GCP configuration file not found: {config_file}")
        log_error(
            "Please copy deployment/config/gcp-config.sh.template to "
            "deployment/config/gcp-config.sh"
        )
        sys.exit(1)

    # Parse bash config file (simple approach - just grep for export statements)
    config = {}
    with open(config_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export ") and "=" in line:
                # Extract key=value pairs
                parts = line.replace("export ", "").split("=", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().strip('"').strip("'")
                    config[key] = value

    # Perform bash variable expansion (simple implementation)
    # This handles cases like ZONE="${REGION}-a"
    def expand_vars(value: str, config_dict: dict[str, str]) -> str:
        """Expand bash variables in a value string."""
        # Find all ${VAR} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            if var_name in config_dict:
                value = value.replace(f"${{{var_name}}}", config_dict[var_name])
        return value

    # Expand variables in config values (may need multiple passes)
    for _ in range(3):  # Max 3 levels of variable nesting
        for key, value in config.items():
            config[key] = expand_vars(value, config)

    # Debug: Log expanded zone value
    if "ZONE" in config and "REGION" in config:
        log_info(f"Expanded ZONE: {config['ZONE']} (from REGION: {config['REGION']})")

    # Validate required config
    required_keys = ["PROJECT_ID", "BUCKET_NAME", "ZONE", "BILLING_ACCOUNT_ID"]
    missing = [k for k in required_keys if k not in config]
    if missing:
        log_error(f"Missing required configuration: {', '.join(missing)}")
        sys.exit(1)

    return config


def check_gcp_auth() -> None:
    """Check if gcloud is authenticated."""
    try:
        run_command(
            [
                "gcloud",
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ],
            check=False,
        )
    except Exception:
        log_error("GCP authentication not found. Please run 'gcloud auth login' first.")
        sys.exit(1)


def check_docker_image_exists(config: dict[str, Any]) -> bool:
    """
    Check if Docker image exists in Artifact Registry.

    Args:
        config: GCP configuration

    Returns:
        True if image exists, False otherwise
    """
    project_id = config.get("PROJECT_ID")
    region = config.get("REGION", "us-central1")
    repo_name = config.get("ARTIFACT_REGISTRY_REPO", "etl-pipeline")
    image_name = config.get("IMAGE_NAME", "")

    if not image_name:
        # Construct image name from config
        image_name = (
            f"{region}-docker.pkg.dev/{project_id}/{repo_name}/etl-pipeline:latest"
        )

    log_info(f"Checking if Docker image exists: {image_name}")

    try:
        # List images in the repository
        result = run_command(
            [
                "gcloud",
                "artifacts",
                "docker",
                "images",
                "list",
                f"{region}-docker.pkg.dev/{project_id}/{repo_name}",
                "--format=value(package)",
            ],
            check=False,
            capture_output=True,
        )

        if result.returncode != 0:
            log_warning(f"Failed to list images in Artifact Registry: {result.stderr}")
            return False

        # Check if our image is in the list
        images = result.stdout.strip().split("\n")
        image_package = f"{region}-docker.pkg.dev/{project_id}/{repo_name}/etl-pipeline"

        if image_package in images:
            log_info(f"Found image: {image_package}")
            return True
        else:
            log_warning(f"Image not found in registry. Available images: {images}")
            return False

    except Exception as e:
        log_warning(f"Exception checking Docker image existence: {e}")
        return False


def get_cost_baseline(project_id: str) -> float:
    """
    Get current cost baseline by running calculate-costs.sh.

    Args:
        project_id: GCP project ID

    Returns:
        Current cost baseline in USD
    """
    script_dir = Path(__file__).parent
    cost_script = script_dir.parent / "scripts" / "calculate-costs.sh"

    try:
        result = run_command(
            [str(cost_script), "--project", project_id, "--days", "1"],
            check=False,
            capture_output=True,
        )
        # Read cost from temp file
        cost_file = Path("/tmp/etl_cost.txt")
        if cost_file.exists():
            try:
                return float(cost_file.read_text().strip())
            except ValueError:
                return 0.0
    except Exception as e:
        log_warning(f"Could not get cost baseline: {e}")
    return 0.0


def check_current_cost(project_id: str, baseline: float) -> float:
    """
    Check current cost and return delta from baseline.

    Args:
        project_id: GCP project ID
        baseline: Cost baseline

    Returns:
        Cost delta in USD
    """
    script_dir = Path(__file__).parent
    cost_script = script_dir.parent / "scripts" / "calculate-costs.sh"

    try:
        run_command(
            [str(cost_script), "--project", project_id, "--days", "1"],
            check=False,
            capture_output=True,
        )
        cost_file = Path("/tmp/etl_cost.txt")
        if cost_file.exists():
            try:
                current_cost = float(cost_file.read_text().strip())
                return current_cost - baseline
            except ValueError:
                return 0.0
    except Exception as e:
        log_warning(f"Could not check current cost: {e}")
    return 0.0


def create_test_vm(
    config: dict[str, Any],
    vm_name: str,
    publication_limit: int,
    *,
    scraper_limit: int | None = None,
    download_limit: int | None = None,
) -> None:
    """
    Create test VM instance using create-vm.sh script.

    Args:
        config: GCP configuration
        vm_name: Name for the VM instance
        publication_limit: Number of publications to process (legacy, used as
            scraper_limit fallback)
        scraper_limit: Limit number of publications to scrape
        download_limit: Limit number of publications to download/process
    """
    script_dir = Path(__file__).parent
    create_vm_script = script_dir / "create-vm.sh"

    cmd = [
        str(create_vm_script),
        "--vm-name",
        vm_name,
        "--on-demand",  # Use on-demand for testing (not spot)
    ]

    # Use explicit limits if provided; None means no limit
    effective_scraper_limit = scraper_limit
    effective_download_limit = download_limit

    if effective_scraper_limit is not None:
        cmd.extend(["--scraper-limit", str(effective_scraper_limit)])
        log_info(f"Scraper limit: {effective_scraper_limit}")

    if effective_download_limit is not None:
        cmd.extend(["--download-limit", str(effective_download_limit)])
        log_info(f"Download limit: {effective_download_limit}")

    log_info(f"Creating VM '{vm_name}'...")
    log_info("Using on-demand instance for predictable test execution")
    run_command(cmd)
    log_success("VM created successfully")


def get_vm_status(config: dict[str, Any], vm_name: str) -> str:
    """
    Get current VM status.

    Args:
        config: GCP configuration
        vm_name: VM instance name

    Returns:
        VM status (RUNNING, TERMINATED, STOPPED, etc.)
    """
    try:
        result = run_command(
            [
                "gcloud",
                "compute",
                "instances",
                "describe",
                vm_name,
                "--zone",
                config["ZONE"],
                "--format",
                "value(status)",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Log error details for debugging
            if result.stderr:
                log_warning(f"Failed to get VM status: {result.stderr.strip()}")
            return "NOT_FOUND"
    except Exception as e:
        log_warning(f"Exception getting VM status: {e}")
        return "NOT_FOUND"


def stop_vm(config: dict[str, Any], vm_name: str) -> None:
    """
    Stop/delete VM instance to prevent further costs.

    Args:
        config: GCP configuration
        vm_name: VM instance name
    """
    log_warning(f"Stopping VM '{vm_name}' to prevent cost overrun...")
    try:
        run_command(
            [
                "gcloud",
                "compute",
                "instances",
                "delete",
                vm_name,
                "--zone",
                config["ZONE"],
                "--quiet",
            ],
            check=False,
        )
        log_success("VM stopped successfully")
    except Exception as e:
        log_error(f"Failed to stop VM: {e}")


def get_serial_port_output(config: dict[str, Any], vm_name: str) -> str:
    """
    Get serial port output from VM.

    Args:
        config: GCP configuration
        vm_name: VM instance name

    Returns:
        Serial port output as string
    """
    try:
        result = run_command(
            [
                "gcloud",
                "compute",
                "instances",
                "get-serial-port-output",
                vm_name,
                "--zone",
                config["ZONE"],
                "--port",
                "1",
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout
        return ""
    except Exception:
        return ""


def analyze_serial_output(
    serial_output: str, last_position: int, startup_script_started: bool
) -> tuple[bool, bool, list[str], int, bool]:
    """
    Analyze serial port output for errors and progress.

    Args:
        serial_output: Complete serial port output
        last_position: Last character position analyzed
        startup_script_started: Whether startup script has begun execution

    Returns:
        Tuple of (has_errors, is_complete, interesting_lines,
                  new_position, startup_script_started)
    """
    # Get only new content since last check
    new_content = serial_output[last_position:]
    new_position = len(serial_output)

    if not new_content:
        return False, False, [], new_position, startup_script_started

    lines = new_content.split("\n")
    interesting_lines = []
    has_errors = False
    is_complete = False

    # Markers that indicate startup script has started
    startup_markers = [
        "startup-script:",  # GCP startup script marker
        "ETL Pipeline",
        "Installing Docker",
        "Installing Google Cloud SDK",
    ]

    # Error patterns to detect (only after startup script starts)
    # Must be specific to avoid false positives from normal output
    error_patterns = [
        "Failed to pull container",
        "Failed to fetch",
        "Failed to install",
        "docker: Error",
        "Docker: Error",
        "Container failed",
        "ERROR: Failed",
        "startup-script: Error",
        "startup-script: error",
        "startup-script exit",
        'Script "startup-script" failed',
        "gcloud: command not found",
        "docker: command not found",
        'Permission "artifactregistry',
    ]

    # Success/completion patterns
    completion_patterns = [
        "ETL pipeline completed successfully",
        "Pipeline execution complete",
        "Shutting down VM",
        "VM will shut down",
    ]

    # Progress patterns to show
    progress_patterns = [
        "startup-script:",
        "ETL Pipeline Container Starting",
        "Starting ETL Pipeline",
        "Installing Docker",
        "Installing Google Cloud SDK",
        "Pulling container image",
        "Running ETL pipeline",
        "Scraping publications",
        "Processing",
        "Uploading",
        "SUCCESS",
        "✓",
        "Configuring Docker",
        "Authenticating",
    ]

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if startup script has started
        if not startup_script_started:
            for marker in startup_markers:
                if marker in line:
                    startup_script_started = True
                    interesting_lines.append(f"[STARTUP] {line}")
                    break

        # Only check for errors after startup script has started
        if startup_script_started:
            # Check for errors
            for pattern in error_patterns:
                if pattern in line:
                    has_errors = True
                    interesting_lines.append(f"[ERROR] {line}")
                    break

        # Check for completion
        for pattern in completion_patterns:
            if pattern in line:
                is_complete = True
                interesting_lines.append(f"[COMPLETE] {line}")
                break

        # Check for progress
        for pattern in progress_patterns:
            if pattern in line:
                interesting_lines.append(f"[PROGRESS] {line}")
                break

    return (
        has_errors,
        is_complete,
        interesting_lines,
        new_position,
        startup_script_started,
    )


def monitor_vm_execution(
    config: dict[str, Any],
    vm_name: str,
    phase: int,
    cost_baseline: float,
    max_cost: float,
    max_wait_time: int = 7200,
) -> tuple[bool, float, int]:
    """
    Monitor VM execution with active cost monitoring and error detection.

    Args:
        config: GCP configuration
        vm_name: VM instance name
        phase: Test phase (1 or 2)
        cost_baseline: Cost baseline
        max_cost: Maximum allowed cost (hard stop)
        max_wait_time: Maximum wait time in seconds

    Returns:
        Tuple of (success, final_cost_delta, execution_time)
    """
    log_step("Monitoring VM execution with active cost monitoring")
    log_info(f"Cost threshold: ${max_cost:.2f} (hard stop)")
    log_info("Checking costs every 2 minutes...")
    log_info("Monitoring serial port output for errors and progress...")

    vm_running = True
    check_interval = 30  # Check VM status every 30 seconds
    cost_check_interval = 120  # Check costs every 2 minutes
    serial_check_interval = 30  # Check serial output every 30 seconds
    elapsed_time = 0
    last_cost_check = 0
    last_serial_check = 0
    serial_position = 0
    cost_exceeded = False
    error_detected = False
    startup_script_started = False

    while vm_running and elapsed_time < max_wait_time:
        # Check VM status
        vm_status = get_vm_status(config, vm_name)

        if vm_status not in ["RUNNING", "STAGING", "PROVISIONING"]:
            log_info(f"VM status: {vm_status}")
            if vm_status in ["TERMINATED", "STOPPED"]:
                vm_running = False
                if not error_detected:
                    log_success("VM has completed execution")
                break

        # Check serial output for errors and progress
        if elapsed_time - last_serial_check >= serial_check_interval:
            serial_output = get_serial_port_output(config, vm_name)
            (
                has_errors,
                is_complete,
                interesting_lines,
                serial_position,
                startup_script_started,
            ) = analyze_serial_output(
                serial_output, serial_position, startup_script_started
            )
            last_serial_check = elapsed_time

            # Display interesting lines
            for line in interesting_lines:
                if "[ERROR]" in line:
                    log_error(line.replace("[ERROR] ", ""))
                elif "[COMPLETE]" in line:
                    log_success(line.replace("[COMPLETE] ", ""))
                elif "[STARTUP]" in line:
                    log_success(line.replace("[STARTUP] ", "Startup script began: "))
                elif "[PROGRESS]" in line:
                    log_info(line.replace("[PROGRESS] ", ""))

            # Handle errors
            if has_errors and not error_detected:
                error_detected = True
                log_error("ERROR DETECTED in VM execution!")
                log_error("Showing last 50 lines of serial output:")
                lines = serial_output.split("\n")
                for line in lines[-50:]:
                    if line.strip():
                        print(f"  {line}", file=sys.stderr)
                log_error("Stopping VM due to detected errors...")
                stop_vm(config, vm_name)
                vm_running = False
                break

            # Handle completion
            if is_complete:
                log_success("ETL pipeline completed successfully (detected from logs)")
                # Give VM time to shutdown gracefully
                time.sleep(30)
                break

        # Check costs periodically
        if elapsed_time - last_cost_check >= cost_check_interval:
            cost_delta = check_current_cost(config["PROJECT_ID"], cost_baseline)
            last_cost_check = elapsed_time

            log_info(
                f"Current cost delta: ${cost_delta:.4f} (threshold: ${max_cost:.2f})"
            )

            if cost_delta > max_cost:
                log_error(
                    f"COST THRESHOLD EXCEEDED: ${cost_delta:.4f} > ${max_cost:.2f}"
                )
                log_error("Stopping VM immediately to prevent further costs!")
                stop_vm(config, vm_name)
                cost_exceeded = True
                vm_running = False
                break

            # Warn if approaching threshold
            if cost_delta > max_cost * 0.8:
                log_warning(
                    f"Cost approaching threshold: ${cost_delta:.4f} "
                    f"(80% of ${max_cost:.2f})"
                )

        # Show progress every minute
        if elapsed_time > 0 and elapsed_time % 60 == 0:
            minutes = elapsed_time // 60
            log_info(f"Still running... ({minutes}m elapsed)")

        time.sleep(check_interval)
        elapsed_time += check_interval

    if elapsed_time >= max_wait_time:
        log_warning("Maximum wait time reached")
        # Show final serial output before stopping
        serial_output = get_serial_port_output(config, vm_name)
        log_warning("Final serial output (last 50 lines):")
        lines = serial_output.split("\n")
        for line in lines[-50:]:
            if line.strip():
                print(f"  {line}", file=sys.stderr)
        stop_vm(config, vm_name)
        vm_running = False

    # Final cost check
    final_cost_delta = check_current_cost(config["PROJECT_ID"], cost_baseline)

    # Return success only if no errors or cost exceeded
    success = not (cost_exceeded or error_detected)
    return success, final_cost_delta, elapsed_time


def retrieve_execution_report(
    config: dict[str, Any], phase: int
) -> dict[str, Any] | None:
    """
    Retrieve execution report from GCS.

    Args:
        config: GCP configuration
        phase: Test phase

    Returns:
        Report data as dictionary or None
    """
    log_step("Retrieving execution report")
    bucket_name = config["BUCKET_NAME"]

    try:
        # List reports
        result = run_command(
            [
                "gcloud",
                "storage",
                "ls",
                f"gs://{bucket_name}/reports/etl-execution-*.json",
            ],
            check=False,
            capture_output=True,
        )

        if result.returncode != 0 or not result.stdout.strip():
            log_warning("No execution report found")
            return None

        # Get most recent report
        reports = sorted(result.stdout.strip().split("\n"), reverse=True)
        if not reports:
            return None

        latest_report = reports[0]
        local_report = Path(f"/tmp/etl_execution_report_phase{phase}.json")

        log_info(f"Downloading report: {latest_report}")
        run_command(
            [
                "gcloud",
                "storage",
                "cp",
                latest_report,
                str(local_report),
            ],
            check=False,
        )

        if local_report.exists():
            with open(local_report) as f:
                return json.load(f)
    except Exception as e:
        log_warning(f"Failed to retrieve execution report: {e}")

    return None


def verify_outputs(config: dict[str, Any]) -> dict[str, int]:
    """
    Verify pipeline outputs in GCS.

    Args:
        config: GCP configuration

    Returns:
        Dictionary with output counts
    """
    log_step("Verifying outputs")
    bucket_name = config["BUCKET_NAME"]
    results = {}

    paths = {
        "documents": f"gs://{bucket_name}/processed/documents/growthlab/",
        "chunks": f"gs://{bucket_name}/processed/chunks/documents/growthlab/",
        "embeddings": f"gs://{bucket_name}/processed/embeddings/documents/growthlab/",
    }

    for key, gcs_path in paths.items():
        try:
            result = run_command(
                ["gcloud", "storage", "ls", "-r", gcs_path],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0:
                count = len(
                    [line for line in result.stdout.strip().split("\n") if line.strip()]
                )
                results[key] = count
                log_info(f"{key.capitalize()}: {count} files")
            else:
                results[key] = 0
        except Exception:
            results[key] = 0

    return results


def generate_test_report(
    phase: int,
    publication_limit: int,
    vm_name: str,
    start_time: datetime,
    end_time: datetime,
    execution_time: int,
    cost_baseline: float,
    cost_delta: float,
    success: bool,
    outputs: dict[str, int],
    report_data: dict[str, Any] | None,
) -> Path:
    """
    Generate test report markdown file.

    Args:
        phase: Test phase
        publication_limit: Number of publications processed
        vm_name: VM instance name
        start_time: Test start time
        end_time: Test end time
        execution_time: Execution time in seconds
        cost_baseline: Cost baseline
        cost_delta: Cost delta
        success: Whether test succeeded
        outputs: Output verification results
        report_data: Execution report data

    Returns:
        Path to generated report file
    """
    script_dir = Path(__file__).parent
    report_file = script_dir.parent / f"TEST_RESULTS_PHASE{phase}.md"

    execution_minutes = execution_time // 60
    execution_seconds = execution_time % 60
    cost_per_pub = cost_delta / publication_limit if publication_limit > 0 else 0.0

    # Determine success criteria based on phase
    if phase == 1:
        cost_threshold = 1.00
        time_threshold = 600  # 10 minutes
    else:
        cost_threshold = 10.00
        time_threshold = 7200  # 2 hours

    cost_passed = cost_delta < cost_threshold
    time_passed = execution_time < time_threshold

    report_content = f"""# Test Phase {phase} Results

**Test Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")}
**Publication Limit:** {publication_limit}
**VM Name:** {vm_name}

## Execution Summary

- **Start Time:** {start_time.isoformat()}
- **End Time:** {end_time.isoformat()}
- **Duration:** {execution_minutes}m {execution_seconds}s ({execution_time} seconds)
- **Cost Delta:** ${cost_delta:.4f}
- **Cost Baseline:** ${cost_baseline:.4f}
- **Cost per Publication:** ${cost_per_pub:.4f}

## Cost Analysis

### Cost Breakdown
- **Compute (VM):** Calculated based on VM runtime
- **Storage (GCS):** Based on data stored
- **API Calls (OpenAI):** Based on embeddings generated
- **Network:** Minimal for GCS transfers

### Cost Safety
- **Threshold:** ${cost_threshold:.2f}
- **Actual Cost:** ${cost_delta:.4f}
- **Status:** {"✅ PASS" if cost_passed else "❌ FAIL"}

## Output Verification

- **Documents:** {outputs.get("documents", 0)}
- **Chunks:** {outputs.get("chunks", 0)}
- **Embeddings:** {outputs.get("embeddings", 0)}

## Execution Report

```json
{json.dumps(report_data, indent=2) if report_data else "No report data available"}
```

## Success Criteria

### Phase {phase} Criteria
- [{"✅" if cost_passed else "❌"}] Total cost < ${cost_threshold:.2f} \
(Actual: ${cost_delta:.4f})
- [{"✅" if time_passed else "❌"}] Execution time < {time_threshold // 60} minutes \
(Actual: {execution_minutes}m {execution_seconds}s)
- [{"✅" if success else "❌"}] Pipeline completed without critical errors
- [{"✅" if outputs.get("documents", 0) > 0 else "❌"}] Documents processed
- [{"✅" if outputs.get("chunks", 0) > 0 else "❌"}] Chunks created
- [{"✅" if outputs.get("embeddings", 0) > 0 else "❌"}] Embeddings generated

## Recommendations
{
        "⚠️ **WARNING:** Cost exceeds threshold for Phase " + str(phase)
        if not cost_passed
        else ""
    }
{
        "⚠️ **WARNING:** Execution time exceeds threshold for Phase " + str(phase)
        if not time_passed
        else ""
    }

## Next Steps
{
        "1. Review cost and timing results\\n"
        "2. If Phase 1 passes criteria, proceed to Phase 2 (100 publications)\\n"
        "3. If Phase 1 fails, investigate issues before proceeding"
        if phase == 1
        else "1. Extrapolate costs for full batch (1,400 publications)\\n"
        "2. Estimate total runtime\\n"
        "3. Compare against deployment guide estimates\\n"
        "4. Make go/no-go decision for full batch processing"
    }

---
*Report generated automatically by test_batch_processing.py*
"""

    report_file.write_text(report_content)
    log_success(f"Test report generated: {report_file}")
    return report_file


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test batch processing orchestration for GCP ETL pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--limit", type=int, required=True, help="Number of publications to process"
    )
    parser.add_argument(
        "--phase", type=int, required=True, choices=[1, 2], help="Test phase (1 or 2)"
    )
    parser.add_argument(
        "--vm-name", type=str, help="Custom VM name (default: auto-generated)"
    )
    parser.add_argument(
        "--download-limit",
        type=int,
        help=(
            "Limit number of publications to download/process"
            " (default: same as --limit)"
        ),
    )
    parser.add_argument(
        "--scraper-limit",
        type=int,
        help=(
            "Limit number of publications to scrape"
            " (default: same as --limit for Phase 1,"
            " unlimited for Phase 2)"
        ),
    )

    args = parser.parse_args()

    # Validate limit
    if args.limit <= 0:
        log_error("Limit must be a positive integer")
        sys.exit(1)

    # Set cost thresholds based on phase
    if args.phase == 1:
        max_cost = 1.00  # $1.00 for Phase 1
        max_time = 600  # 10 minutes
    else:
        max_cost = 10.00  # $10.00 for Phase 2
        max_time = 7200  # 2 hours (OCR for 100 PDFs can take 30-90 min)

    log_step(f"Starting Test Phase {args.phase}")
    log_info(f"Publication limit: {args.limit}")
    log_info(f"Cost threshold: ${max_cost:.2f} (hard stop)")
    log_info(f"Time threshold: {max_time // 60} minutes")

    # Load configuration
    config = load_gcp_config()
    check_gcp_auth()

    # Validate Docker image exists in Artifact Registry
    log_step("Validating Docker image exists")
    if not check_docker_image_exists(config):
        log_error("Docker image does not exist in Artifact Registry")
        log_error("Please build and push the image first:")
        log_error("  ./deployment/cloud-run/deploy.sh")
        sys.exit(1)
    log_success("Docker image validated successfully")

    # Generate VM name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    vm_name = args.vm_name or f"etl-pipeline-vm-test-phase{args.phase}-{timestamp}"

    # Record start time and cost baseline
    start_time = datetime.now(UTC)
    log_step("Recording cost baseline")
    cost_baseline = get_cost_baseline(config["PROJECT_ID"])
    log_info(f"Cost baseline: ${cost_baseline:.4f}")

    # Determine scraper and download limits based on phase
    if args.phase == 1:
        # Phase 1: scraper-limit controls everything (small test)
        if args.scraper_limit is not None:
            effective_scraper_limit = args.scraper_limit
        else:
            effective_scraper_limit = args.limit
        effective_download_limit = args.download_limit
    else:
        # Phase 2: scrape ALL, only download/process --limit
        # Override with explicit --scraper-limit/--download-limit
        effective_scraper_limit = args.scraper_limit  # None = all
        if args.download_limit is not None:
            effective_download_limit = args.download_limit
        else:
            effective_download_limit = args.limit

    # Create VM
    try:
        log_info(f"Creating VM '{vm_name}' for Phase {args.phase}")
        log_info(f"Zone: {config['ZONE']}, Project: {config['PROJECT_ID']}")
        log_info(
            f"Scraper limit: {effective_scraper_limit or 'unlimited'}, "
            f"Download limit: {effective_download_limit or 'unlimited'}"
        )
        create_test_vm(
            config,
            vm_name,
            args.limit,
            scraper_limit=effective_scraper_limit,
            download_limit=effective_download_limit,
        )
    except Exception as e:
        log_error(f"Failed to create VM: {e}")
        sys.exit(1)

    # Give VM a moment to register with API before checking status
    log_info("Waiting 5 seconds for VM to register with GCP API...")
    time.sleep(5)

    # Wait for VM to start
    log_step("Waiting for VM to start")
    max_startup_wait = 300  # 5 minutes for on-demand instances (more generous timeout)
    startup_interval = 10  # Check every 10 seconds (less API pressure)
    log_info(f"Checking VM status every {startup_interval} seconds...")
    status = "UNKNOWN"

    for i in range(max_startup_wait // startup_interval):
        status = get_vm_status(config, vm_name)

        if status == "RUNNING":
            elapsed = i * startup_interval
            log_success(f"VM is running (started in {elapsed}s)")
            break

        # Log progress every 30 seconds
        if i > 0 and i % 3 == 0:  # Every 30 seconds (3 * 10s intervals)
            elapsed = i * startup_interval
            log_info(
                f"Still waiting for VM to start... "
                f"({elapsed}s elapsed, status: {status})"
            )

        time.sleep(startup_interval)
    else:
        elapsed = max_startup_wait
        log_error(f"VM failed to start within timeout ({elapsed}s)")
        log_error(f"Last known status: {status}")
        log_info(
            f"Check VM logs: gcloud compute instances get-serial-port-output "
            f"{vm_name} --zone={config['ZONE']}"
        )
        log_info(
            "Or check in console: "
            f"https://console.cloud.google.com/compute/instances?project="
            f"{config['PROJECT_ID']}"
        )
        sys.exit(1)

    # Monitor execution with cost checks
    try:
        success, final_cost_delta, execution_time = monitor_vm_execution(
            config, vm_name, args.phase, cost_baseline, max_cost, max_time
        )
    except KeyboardInterrupt:
        log_warning("Interrupted by user")
        stop_vm(config, vm_name)
        sys.exit(1)

    # Record end time
    end_time = datetime.now(UTC)

    # Retrieve execution report
    report_data = retrieve_execution_report(config, args.phase)

    # Verify outputs
    outputs = verify_outputs(config)

    # Generate report
    report_file = generate_test_report(
        args.phase,
        args.limit,
        vm_name,
        start_time,
        end_time,
        execution_time,
        cost_baseline,
        final_cost_delta,
        success,
        outputs,
        report_data,
    )

    # Display summary
    log_step("Test Summary")
    print("\n" + "━" * 60)
    print(f"Phase {args.phase} Test Results")
    print("━" * 60)
    print(f"\nPublications Processed: {args.limit}")
    print(f"Execution Time: {execution_time // 60}m {execution_time % 60}s")
    print(f"Cost Delta: ${final_cost_delta:.4f}")
    print(f"Cost per Publication: ${final_cost_delta / args.limit:.4f}")
    print("\nOutputs:")
    for key, value in outputs.items():
        print(f"  • {key.capitalize()}: {value}")
    print(f"\nFull report: {report_file}")
    print("━" * 60 + "\n")

    if success and final_cost_delta < max_cost:
        log_success(f"Test Phase {args.phase} completed successfully!")
        sys.exit(0)
    else:
        log_error(f"Test Phase {args.phase} failed or exceeded thresholds")
        sys.exit(1)


if __name__ == "__main__":
    main()
