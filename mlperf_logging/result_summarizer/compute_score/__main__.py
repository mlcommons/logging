from .. import result_summarizer
from ...compliance_checker.mlp_compliance import usage_choices, rule_choices
import argparse


def get_compute_args():
    parser = argparse.ArgumentParser(
        prog="mlperf_logging.result_summarizer.compute_score",
        description="Compute the score of a single benchmark",
    )
    parser.add_argument("benchmark", type=str, help="TODO:", required=True)
    parser.add_argument("system", type=str, help="System name", default=None)
    parser.add_argument(
        "has_power", action="store_true", help="Compute power score as well"
    )
    parser.add_argument(
        "benchmark_folder", type=str, help="Folder containing all the result files", required=True
    )
    parser.add_argument(
        "usage",
        type=str,
        default="training",
        choices=usage_choices(),
        help="the usage such as training, hpc, inference_edge, inference_server",
        required=True,
    )
    parser.add_argument(
        "ruleset",
        type=str,
        choices=rule_choices(),
        help="the ruleset such as 0.6.0, 0.7.0, or 1.0.0",
        required=True,
    )

    parser.add_argument(
        "weak_scaling", action="store_true", help="Compute weak scaling score"
    )

    return parser.parse_args()


args = get_compute_args()

if args.weak_scaling:
    scores, power_scores = result_summarizer._compute_weak_score_standalone(
        args.benchmark,
        args.system,
        args.has_power,
        args.benchmark_folder,
        args.usage,
        args.ruleset,
    )
    print(f"Scores: {scores}")
    if power_scores:
        print(f"Power Scores: {power_scores}")
else:
    score, power_score = result_summarizer._compute_strong_score_standalone(
        args.benchmark,
        args.system,
        args.has_power,
        args.benchmark_folder,
        args.usage,
        args.ruleset,
    )
    print(f"Score: {score}")
    if power_score:
        print(f"Power Score: {power_score}")
