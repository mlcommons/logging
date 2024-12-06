import argparse
import logging
import os
import subprocess


def list_files_recursively(*path):
    path = os.path.join(*path)
    return [
        os.path.join(dirpath, file)
        for dirpath, dirs, files in os.walk(path)
        for file in files
    ]


def list_dirs_recursively(*path):
    path = os.path.join(*path)
    return [dirpath for dirpath, dirs, files in os.walk(path)]


def _check_bad_filenames(files, dirs):
    """Checks for filename errors.
    Git does not like filenames with spaces or that start with ., or /. .
    """
    logging.info("Running git-unfriendly file name checks.")
    dir_names = [(dir_, os.path.basename(dir_)) for dir_ in dirs]
    file_names = [(file_, os.path.basename(file_)) for file_ in files]
    git_error_names = [
        name[0]
        for name in dir_names
        if name[1].startswith(".") or " " in name[1]
    ] + [
        name[0]
        for name in file_names
        if name[1].startswith(".") or " " in name[1]
    ]
    if len(git_error_names) > 0:
        error = "\n".join(git_error_names)
        logging.error("Files with git-unfriendly name: %s ", error)
        logging.error(
            'Please remove spaces from filenamed and make sure they do not start with ".", or "/."'
        )
        return False
    return True


def _check_file_sizes(files, file_size_limit_mb=50):
    """Checks for large file sizes.
    Git does not like file sizes > 50MB.
    """
    logging.info('Running large file checks.')
    MB_TO_BYTES = 1024 * 1024
    files_over_size_limit = [
        f
        for f in files
        if not os.path.islink(f)
        and os.path.getsize(f) > file_size_limit_mb * MB_TO_BYTES
    ]
    if len(files_over_size_limit) > 0:
        error = '\n'.join(files_over_size_limit)
        logging.error('Files > 50MB: %s', error)
        logging.error('Please remove or reduce the size of these files.')
        return False
    return True


def _check_symbolic_links(submission_dir, files):
    """Check folder for broken symbolic links
    """
    broken_symbolic_links = [
        f
        for f in files
        if os.path.islink(f) and not os.path.exists(os.readlink(f))
    ]
    if len(broken_symbolic_links) > 0:
        error = '\n'.join(broken_symbolic_links)
        logging.error(
            "%s contains broken symbolic links: %s",
            submission_dir,
            error,
        )
        return False
    return True


def run_checks(submission_dir):
    """Top-level checker function.
    Call individual checkers from this function.
    """
    logging.info('Running repository checks.')

    # Get files and directories
    files = list_files_recursively(submission_dir)
    dirs = list_dirs_recursively(submission_dir)

    # Execute checks
    bad_filename_error = _check_bad_filenames(files, dirs)
    large_file_error = _check_file_sizes(files)
    symlinks_error = _check_symbolic_links(submission_dir, files)

    if not (bad_filename_error and large_file_error and symlinks_error):
        logging.info('CHECKS FAILED.')
        return False

    logging.info('ALL CHECKS PASSED.')
    return False


def get_parser():
    """Parse commandline."""
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.repo_checker',
        description='Sanity checks to make sure that package is github compliant.',
    )

    parser.add_argument(
        'folder',
        type=str,
        help='the folder for a submission package.',
    )
    parser.add_argument(
        'usage',
        type=str,
        choices=['training', 'hpc'],
        help='the usage -- only training is currently supported.',
    )
    parser.add_argument(
        'ruleset',
        type=str,
        choices=['2.0.0', '2.1.0', '3.0.0', '3.1.0', '4.0.0', '4.1.0', '5.0.0'],
        help='the ruleset. 2.0.0, 2.1.0, 3.0.0, 3.1.0, 4.0.0, 4.1.0 and 5.0.0 are currently supported.'
    )
    parser.add_argument(
        '--log_output',
        type=str,
        default='repo_checker.log',
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_output, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    logging.getLogger().handlers[0].setFormatter(formatter)
    logging.getLogger().handlers[1].setFormatter(formatter)

    valid = run_checks(args.folder)
    return valid

if __name__ == '__main__':
    main()
