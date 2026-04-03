from .. import result_summarizer
from ...rcp_checker import rcp_checker
from ...compliance_checker.mlp_compliance import usage_choices, rule_choices
from ...compliance_checker.mlp_parser import parse_file
from ...benchmark_meta import get_result_file_counts
import argparse
import glob
import json
import os


def get_compute_args():
    parser = argparse.ArgumentParser(
        prog="mlperf_logging.result_summarizer.compute_score",
        description="Compute the score of a single benchmark",
    )
    parser.add_argument("--system", type=str, help="System name", default=None)
    parser.add_argument(
        "--has_power", action="store_true", help="Compute power score as well"
    )
    parser.add_argument(
        "--benchmark_folder",
        type=str,
        help="Folder containing all the result files",
        required=True,
    )
    parser.add_argument(
        "--usage",
        type=str,
        default="training",
        choices=usage_choices(),
        help="the usage such as training, hpc, inference_edge, inference_server",
        required=True,
    )
    parser.add_argument(
        "--ruleset",
        type=str,
        choices=rule_choices(),
        help="the ruleset such as 0.6.0, 0.7.0, or 1.0.0",
        required=True,
    )
    parser.add_argument(
        "--is_weak_scaling", action="store_true", help="Compute weak scaling score"
    )
    parser.add_argument(
        "--scale", action="store_true", help="Compute the scaling factor"
    )

    return parser.parse_args()


def print_benchmark_info(args, benchmark):
    print("INFO -------------------------------------------------------")
    print(f"MLPerf {args.usage}")
    print(f"Folder: {args.benchmark_folder}")
    print(f"Version: {args.ruleset}")
    print(f"System: {args.system}")
    print(f"Benchmark: {benchmark}")
    print("-------------------------------------------------------------")


def _reset_scaling(results_dir):
    filepath = results_dir + "/scaling.json"
    if os.path.exists(filepath):
        os.remove(filepath)


def _get_scaling_factor(results_dir):
    scaling_factor = 1.0
    scaling_file = results_dir + "/scaling.json"
    if os.path.exists(scaling_file):
        with open(scaling_file, "r") as f:
            contents = json.load(f)
        scaling_factor = contents["scaling_factor"]
    return scaling_factor


def _find_benchmark(result_file, ruleset):
    loglines, _ = parse_file(result_file, ruleset)
    benchmark = None
    for logline in loglines:
        if logline.key == "submission_benchmark":
            benchmark = logline.value["value"]
            break
    if benchmark is None:
        raise ValueError("Benchmark not specified in result file")
    return benchmark


def _epochs_samples_to_converge(result_file, ruleset):
    loglines, _ = parse_file(result_file, ruleset)
    epoch_num = None
    samples_count = None
    for logline in loglines:
        if logline.key == "eval_accuracy":
            if "epoch_num" in logline.value["metadata"]:
                epoch_num = logline.value["metadata"]["epoch_num"]
            if "samples_count" in logline.value["metadata"]:
                samples_count = logline.value["metadata"]["samples_count"]
    if samples_count is not None:
        return samples_count
    if epoch_num is not None:
        return epoch_num
    raise ValueError(
        "Not enough values specified in result file. One of ('samples_count')"
        "or ('epoch_num') is needed"
    )


args = get_compute_args()
_reset_scaling(args.benchmark_folder)
pattern = "{folder}/result_*.txt".format(folder=args.benchmark_folder)
result_files = glob.glob(pattern, recursive=True)
benchmark = _find_benchmark(result_files[0], args.ruleset)
required_runs = get_result_file_counts(args.usage)[benchmark]
if required_runs > len(result_files):
    print(
        f"WARNING: Not enough runs found for an official submission."
        f" Found: {len(result_files)}, required: {required_runs}"
    )

if args.scale:
    rcp_checker.check_directory(
        args.benchmark_folder,
        args.usage,
        args.ruleset,
        False,
        False,
        rcp_file=None,
        rcp_pass="pruned_rcps",
        rcp_bypass=False,
        set_scaling=True,
    )

scaling_factor = _get_scaling_factor(args.benchmark_folder)

if args.is_weak_scaling:
    scores, power_scores = result_summarizer._compute_weak_score_standalone(
        benchmark,
        args.system,
        args.has_power,
        args.benchmark_folder,
        args.usage,
        args.ruleset,
    )
    print_benchmark_info(args, benchmark)
    print(f"Scores: {scores}")
    if power_scores:
        print(f"Power Scores - Energy (kJ): {power_scores}")
else:
    scores_track, power_scores_track, score, power_score = (
        result_summarizer._compute_strong_score_standalone(
            benchmark,
            args.system,
            args.has_power,
            args.benchmark_folder,
            args.usage,
            args.ruleset,
            return_full_scores=True,
        )
    )
    print_benchmark_info(args, benchmark)
    mean_score = 0
    for file, s in scores_track.items():
        epochs_samples_to_converge = _epochs_samples_to_converge(file, args.ruleset)
        print(
            f"Score - Time to Train (minutes) for {file}: {s}. Samples/Epochs to converge: {epochs_samples_to_converge}"
        )
        mean_score += s
    mean_score /= len(result_files)
    mean_score *= scaling_factor
    if required_runs > len(result_files):
        print("WARNING: Olympic scoring skipped")
        print(f"Final score - Time to Train (minutes): {mean_score}")
    else:
        print(f"Final score - Time to Train (minutes): {score}")
    if power_score:
        mean_power = 0
        for file, ps in power_scores_track.items():
            print(f"Power Score - Energy (kJ) for {file}: {ps}")
            mean_power += ps
        mean_power /= len(result_files)
        mean_power *= scaling_factor
        if required_runs > len(result_files):
            print("WARNING: Olympic scoring skipped")
            print(f"Final score - Time to Train (minutes): {mean_power}")
        else:
            print(f"Power Score - Energy (kJ): {power_score}")
