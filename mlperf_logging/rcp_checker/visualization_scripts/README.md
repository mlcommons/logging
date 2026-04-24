# RCP Interactive Viewer

A browser-based tool for exploring MLPerf Reference Convergence Points (RCPs) interactively. Built with [Dash](https://dash.plotly.com/) and [Plotly](https://plotly.com/python/).

## Requirements

```
pip install dash plotly
```

The script also depends on the `mlperf_logging` package, which must be importable from the repository root (this is handled automatically when running from within the repo).

## Usage

```bash
cd mlperf_logging/rcp_checker/visualization_scripts
python rcp_dash_app.py
```

Then open **http://localhost:8050** in your browser.

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `127.0.0.1` | Host address to bind to |
| `--port` | `8050` | Port to listen on |
| `--debug` | off | Enable Dash hot-reload and debug overlay |

Example — expose on all interfaces with debug mode:

```bash
python rcp_dash_app.py --host 0.0.0.0 --port 8080 --debug
```

## Interface

### RCP controls

| Control | Description |
|---------|-------------|
| **Usage** | Working group that produced the benchmark: `training` or `hpc` |
| **Version** | MLPerf ruleset version (1.0.0 – 6.0.0). The Benchmark list updates automatically to show only benchmarks that have RCP data for the selected usage + version. |
| **Benchmark** | The benchmark to visualise. |
| **RCP type** | `Pruned` (default) shows only the RCP points used for compliance checks; `Full` shows all recorded reference points. |

### Graph

The line graph plots convergence data against **Batch Size (BS)**:

| Trace | Description |
|-------|-------------|
| **RCP Mean** | Mean epochs to converge across all reference runs |
| **Min Epochs** | Minimum acceptable epochs (used as the pass/fail threshold) |
| **Mean + Stdev / Mean − Stdev** | Dotted boundary lines one standard deviation above and below the mean |
| **Mean ± Stdev band** | Semi-transparent fill between the two boundary lines |
| **Submission** ★ | Red star placed at the submission's (BS, epochs) coordinates (visible after loading a result file — see below) |

Hovering over any point shows its exact values. Hovering over the Submission star additionally shows the filename and the computed **scaling factor**.

### Data table

Below the graph, a table lists every RCP record for the current selection with columns: Batch Size, RCP Mean, RCP Stdev, and Min Epochs.

## Loading a result file

The **Result file** upload zone (drag-and-drop or click to browse) accepts a single MLPerf result log (`result_*.txt`).

On upload the app will:

1. Parse the file with `read_submission_file` from the RCP checker to extract the **benchmark**, **global batch size**, and **epochs to converge**.
2. Auto-select the detected benchmark in the Benchmark dropdown.
3. Look up the matching RCP record for that batch size (exact match, or linear interpolation between the two surrounding records; falls back to the smallest available RCP when the submission BS is below the RCP range).
4. Compute the **scaling factor**:

   ```
   scaling_factor = max(1.0, rcp_mean / submission_epochs)
   ```

   A value of `1.0` means the submission is at or slower than the RCP mean and no normalisation is needed. A value above `1.0` means the submission converged faster than the mean and scores should be normalised by this factor.

5. Plot the submission as a red star on the graph and display a summary status line (benchmark, BS, epochs, scaling factor) below the upload zone.

> **Note:** the scaling factor computed here is based on a single run. The full RCP compliance check averages multiple runs and applies olympic scoring (dropping the fastest and slowest); the single-run value shown here is indicative only.

### Error cases

| Message | Likely cause |
|---------|-------------|
| `Could not detect benchmark from file` | The log does not contain a `submission_benchmark` key |
| `Could not detect global_batch_size from file` | The log does not contain a `global_batch_size` key |
| `Run did not converge` | The log contains a `run_stop` with `status != success` |
| `Benchmark "X" has no RCP data for <usage> v<version>` | The detected benchmark is not present for the selected usage/version; try a different version |
