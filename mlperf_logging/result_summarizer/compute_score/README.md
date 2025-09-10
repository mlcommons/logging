# MLPerf compute standalone score

MLPerf compute standalone score

## Usage

To compute the scores of a single benchmark. All the results files are assumed to be in the same folder:

```sh
python3 -m mlperf_logging.result_summarizer.compute_score --benchmark BENCHMARK \
    --system SYSTEM_NAME --benchmark_folder BENCHMARK_FOLDER --usage USAGE --ruleset RULESET \
    [--is_weak_scaling] [--scale] [--has_power]
```


**BENCHMARK:** Name of the benchmark to compute the score such as rgat, llama31_8b, etc.
**SYSTEM_NAME:** Optional system name.
**BENCHMARK_FOLDER:** Folder containing all the results files of the benchmark.
**USAGE:** Either "training" or "hpc",
**RULESET:** Version of the rules that applies one of "1.0.0", "1.1.0", "2.0.0", "2.1.0", "3.0.0", "3.1.0", "4.0.0", "4.1.0", "5.0.0", "5.1.0".
**[--is_weak_scaling]:** Is the benchmark weak scaling (only applies to HPC).
**[--scale]:** Compute the scaling.json file (only if the folder does not contain it already).
**[--has_power]:** Have the results power measurements .



## Tested software versions
Tested and confirmed working using the following software versions:

Python 3.9.18
