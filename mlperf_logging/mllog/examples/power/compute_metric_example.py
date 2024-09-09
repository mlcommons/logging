import argparse
from mlperf_logging.compliance_checker.mlp_parser import parse_file
from mlperf_logging.result_summarizer.result_summarizer import _compute_power_node, _compute_power_sw

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-log", type=str, default=None)
    parser.add_argument("--hardware-type", type=str, choices=["node", "sw"], default="node")
    parser.add_argument("--ruleset", type=str, choices=["0.6.0", "0.7.0", "1.0.0", "1.1.0", "2.0.0", "2.1.0", "3.0.0", "3.1.0", "4.0.0", "4.1.0"], default="4.1.0")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    loglines, _ = parse_file(args.input_log, args.ruleset)
    # Calculate time to train
    run_start = None
    run_stop = None
    for logline in loglines:
        if logline.key == 'power_measurement_start':
            run_start = logline.timestamp
        if logline.key == 'power_measurement_stop':
            run_stop = logline.timestamp
        if run_start is not None and run_stop is not None:
            break
    time_to_train = run_stop - run_start

    if args.hardware_type == "node":
        ans = _compute_power_node(loglines, time_to_train)
    elif args.hardware_type == "sw":
        ans = _compute_power_sw(loglines, time_to_train)
    print(f"Power consumed: {ans}")
