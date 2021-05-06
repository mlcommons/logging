'''
Runs a set of checks on an organization's submission package.
'''
from __future__ import print_function

import argparse
import glob
import json
import os
import sys

from ..compliance_checker import mlp_compliance
from ..rcp_checker import rcp_checker
from .seed_checker import find_source_files_under, SeedChecker
from ..system_desc_checker import system_desc_checker

_ALLOWED_BENCHMARKS_V06 = [
    'resnet',
    'ssd',
    'maskrcnn',
    'gnmt',
    'transformer',
    'ncf',
    'minigo',
]

_ALLOWED_BENCHMARKS_V07 = [
    'bert',
    'dlrm',
    'gnmt',
    'maskrcnn',
    'minigo',
    'resnet',
    'ssd',
    'transformer',
]

_ALLOWED_BENCHMARKS_V10 = [
    'bert',
    'dlrm',
    'maskrcnn',
    'minigo',
    'resnet',
    'ssd',
    'rnnt'
    'unet3d',
]

_EXPECTED_RESULT_FILE_COUNTS = {
    'bert': 10,
    'dlrm': 5,
    'gnmt': 10,
    'maskrcnn': 5,
    'minigo': 10,
    'resnet': 5,
    'ssd': 5,
    'transformer': 10,
    'ncf': 10,
    'rnnt': 10,
    'unet3d': 40,
}


def _get_sub_folders(folder):
    sub_folders = [
        os.path.join(folder, sub_folder) for sub_folder in os.listdir(folder)
    ]
    return [
        sub_folder for sub_folder in sub_folders if os.path.isdir(sub_folder)
    ]


def _print_divider_bar():
    print('------------------------------')


def check_training_result_files(folder, ruleset, quiet, werror, rcp_bypass):
    """Checks all result files for compliance.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0
    """

    if ruleset == '0.6.0':
       allowed_benchmarks = _ALLOWED_BENCHMARKS_V06
    elif ruleset == '0.7.0':
       allowed_benchmarks = _ALLOWED_BENCHMARKS_V07
    elif ruleset == '1.0.0':
       allowed_benchmarks = _ALLOWED_BENCHMARKS_V10
    seed_checker = SeedChecker(ruleset)
    too_many_errors = False
    result_folder = os.path.join(folder, 'results')
    for system_folder in _get_sub_folders(result_folder):
        for benchmark_folder in _get_sub_folders(system_folder):
            folder_parts = benchmark_folder.split('/')
            benchmark = folder_parts[-1]
            system = folder_parts[-2]

            # Find whether submission is closed and only then run seed and RCP checkers
            system_desc_file = os.path.join(folder, 'systems/') + system + '.json'
            division = ''
            with open(system_desc_file, 'r') as f:
                contents = json.load(f)
                if contents['division'] == 'closed':
                    division = 'closed'

            # If it is not a recognized benchmark, skip further checks.
            if benchmark not in allowed_benchmarks:
                print('Skipping benchmark: {}'.format(benchmark))
                continue

            # Find all result files for this benchmark.
            pattern = '{folder}/result_*.txt'.format(folder=benchmark_folder)
            result_files = glob.glob(pattern, recursive=True)

            # No result files were found. That is okay, because the organization
            # may not have submitted any results for this benchmark.
            if not result_files:
                print('No Result Files!')
                continue

            # Find all source codes for this benchmark.
            source_files = find_source_files_under(
                os.path.join(folder, 'benchmarks', benchmark))

            _print_divider_bar()
            print('System {}'.format(system))
            print('Benchmark {}'.format(benchmark))

            # If the organization did submit results for this benchmark, the
            # number of result files must be an exact number.
            if len(result_files) != _EXPECTED_RESULT_FILE_COUNTS[benchmark]:
                print('Expected {} runs, but detected {} runs.'.format(
                    _EXPECTED_RESULT_FILE_COUNTS[benchmark],
                    len(result_files),
                ))
                too_many_errors = True

            errors_found = 0
            result_files.sort()
            for result_file in result_files:
                result_basename = os.path.basename(result_file)
                result_name, _ = os.path.splitext(result_basename)
                run = result_name.split('_')[-1]

                # For each result file, run the benchmark's compliance checks.
                _print_divider_bar()
                print('Run {}'.format(run))
                config_file = '{ruleset}/common.yaml'.format(
                    ruleset=ruleset,
                    benchmark=benchmark,
                )
                checker = mlp_compliance.make_checker(
                    ruleset=ruleset,
                    quiet=quiet,
                    werror=werror,
                )
                valid, _, _, _ = mlp_compliance.main(
                    result_file,
                    config_file,
                    checker,
                )
                if not valid:
                    errors_found += 1
            if errors_found == 1 and benchmark != 'unet3d':
                print('WARNING: One file does not comply.')
                print('WARNING: Allowing this failure under olympic scoring '
                      'rules.')
            elif errors_found > 0 and errors_found <= 4 and benchmark == 'unet3d':
                print('WARNING: {errors} file does not comply.'.format(errors=errors_found))
                print('WARNING: Allowing this failure for unet3d under olympic scoring '
                      'rules.')
            elif errors_found > 0:
                too_many_errors = True

            # Check if each run use unique seeds.
            if ruleset == '1.0.0' and division == 'closed':
                if not seed_checker.check_seeds(result_files, source_files):
                    too_many_errors = True

            # Run RCP checker for 1.0.0
            if ruleset == '1.0.0' and division == 'closed' and benchmark != 'minigo':
                rcp_chk = rcp_checker.make_checker(ruleset, verbose=False)
                rcp_chk._compute_rcp_stats()

                # Now go again through result files to do RCP checks
                rcp_pass, rcp_msg = rcp_chk._check_directory(benchmark_folder, rcp_bypass)
                if not rcp_pass:
                    print('ERROR: RCP Test Failed: {}.'.format(rcp_msg))
                    too_many_errors = True

            _print_divider_bar()
    if too_many_errors:
        raise Exception(
            'Found too many errors in logging, see log above for details.')


def check_systems(folder, ruleset):
    """Checks the system decription files

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0.
    """
    system_folder = os.path.join(folder,'systems')
    pattern = '{folder}/*.json'.format(folder=system_folder)
    json_files = glob.glob(pattern)
    too_many_errors = False

    for json_file in json_files:
        valid, _, _, _ = system_desc_checker.check_training_system_desc(json_file, ruleset)
        if not valid:
            too_many_errors = True

    if too_many_errors:
        raise Exception(
            'Found too many errors in system checking, see log above for details.')


def check_training_package(folder, ruleset, quiet, werror, rcp_bypass):
    """Checks a training package for compliance.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0.
    """
    check_training_result_files(folder, ruleset, quiet, werror, rcp_bypass)
    if ruleset == '1.0.0':
        check_systems(folder, ruleset)

def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.package_checker',
        description='Lint MLPerf submission packages.',
    )

    parser.add_argument(
        'folder',
        type=str,
        help='the folder for a submission package',
    )
    parser.add_argument(
        'usage',
        type=str,
        help='the usage such as training, inference_edge, inference_server',
    )
    parser.add_argument(
        'ruleset',
        type=str,
        help='the ruleset such as 0.6.0, 0.7.0, or 1.0.0'
    )
    parser.add_argument(
        '--werror',
        action='store_true',
        help='Treat warnings as errors',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress warnings. Does nothing if --werror is set',
    )
    parser.add_argument(
        '--rcp_bypass',
        action='store_true',
        help='Bypass failed RCP checks so that submission uploads'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.usage != 'training':
        print('Usage {} is not yet supported.'.format(args.usage))
        sys.exit(1)
    if args.ruleset not in ['0.6.0', '0.7.0', '1.0.0']:
        print('Ruleset {} is not yet supported.'.format(args.ruleset))
        sys.exit(1)

    check_training_package(args.folder, args.ruleset, args.quiet, args.werror, args.rcp_bypass)


if __name__ == '__main__':
    main()
