'''
Runs a set of checks on an organization's submission package.
'''
from __future__ import print_function

import argparse
import glob
import os
import sys

from ..compliance_checker import mlp_compliance

_ALLOWED_BENCHMARKS = [
    'bert',
    'dlrm',
    'gnmt',
    'maskrcnn',
    'minigo',
    'resnet',
    'ssd',
    'transformer',
]

_EXPECTED_RESULT_FILE_COUNTS = {
    'bert': 10,
    'dlrm': 10,
    'gnmt': 10,
    'maskrcnn': 5,
    'minigo': 10,
    'resnet': 5,
    'ssd': 5,
    'transformer': 10,
}


def _get_sub_folders(folder):
    sub_folders = [os.path.join(folder, sub_folder)
                   for sub_folder in os.listdir(folder)]
    return [sub_folder
            for sub_folder in sub_folders
            if os.path.isdir(sub_folder)]


def _print_divider_bar():
    print('------------------------------')


def check_training_result_files(folder, ruleset, quiet, werror):
    """Checks all result files for compliance.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0 or 0.7.0.
    """
    checker = mlp_compliance.make_checker(
        ruleset=ruleset,
        quiet=quiet,
        werror=werror)

    result_folder = os.path.join(folder, 'results')
    for system_folder in _get_sub_folders(result_folder):
        for benchmark_folder in _get_sub_folders(system_folder):
            folder_parts = benchmark_folder.split('/')
            benchmark = folder_parts[-1]
            system = folder_parts[-2]

            # If it is not a recognized benchmark, skip further checks.
            if benchmark not in _ALLOWED_BENCHMARKS:
                continue

            # Find all result files for this benchmark.
            pattern = '{folder}/result_*.txt'.format(folder=benchmark_folder)
            result_files = glob.glob(pattern, recursive=True)

            # No result files were found. That is okay, because the organization
            # may not have submitted any results for this benchmark.
            if not result_files:
                continue

            _print_divider_bar()
            print('System {}'.format(system))
            print('Benchmark {}'.format(benchmark))

            # If the organization did submit results for this benchmark, the number
            # of result files must be an exact number.
            if len(result_files) != _EXPECTED_RESULT_FILE_COUNTS[benchmark]:
                print('Expected {} runs, but detected {} runs.'.format(
                    _EXPECTED_RESULT_FILE_COUNTS[benchmark],
                    len(result_files)))

            result_files.sort()
            for result_file in result_files:
                result_basename = os.path.basename(result_file)
                result_name, _ = os.path.splitext(result_basename)
                run = result_name.split('_')[-1]

                # For each result file, run the benchmark's compliance checks.
                _print_divider_bar()
                print('Run {}'.format(run))
                config_file = '{ruleset}/{benchmark}.yaml'.format(
                    ruleset=ruleset,
                    benchmark=benchmark)
                mlp_compliance.main(result_file, config_file, checker)

            _print_divider_bar()


def check_training_package(folder, ruleset, quiet, werror):
    """Checks a training package for compliance.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0 or 0.7.0.
    """
    check_training_result_files(folder, ruleset, quiet, werror)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.package_checker',
        description='Lint MLPerf submission packages.',
    )

    parser.add_argument('folder', type=str,
                    help='the folder for a submission package')
    parser.add_argument('usage', type=str,
                    help='the usage such as training, inference_edge, inference_server')
    parser.add_argument('ruleset', type=str,
                    help='the ruleset such as 0.6.0, 0.7.0')
    parser.add_argument('--werror', action='store_true',
                    help='Treat warnings as errors')
    parser.add_argument('--quiet', action='store_true',
                    help='Suppress warnings. Does nothing if --werror is set')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.usage != 'training':
        print('Usage {} is not yet supported.'.format(args.usage))
        sys.exit(1)
    if args.ruleset not in ['0.6.0', '0.7.0']:
        print('Ruleset {} is not yet supported.'.format(args.ruleset))
        sys.exit(1)

    check_training_package(args.folder, args.ruleset, args.quiet, args.werror)


if __name__ == '__main__':
    main()
