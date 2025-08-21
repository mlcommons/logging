#!/usr/bin/env python3

'''
RCP viewer: show the RCP means and mins after pruning
'''

import sys
import os
import argparse

#Add the project root directory (assumed to be 3 levels up) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from  mlperf_logging.rcp_checker.rcp_checker import RCP_Checker

def print_rcp_record(record):
    print(f"{record['BS']},{record['RCP Mean']},{record['Min Epochs']}")

# this should be a method of rcp_checker.RCP_Checker, but it's missing.
# Instead we derived it from _find_min_rcp()
def find_max_rcp(checker, rcp_pass_arg='pruned_rcps'):
    '''Find RCP with the smallest batch size for a benchmark'''
    max_bs = -1
    max_record = None
    rcp_data = checker._get_rcp_data(rcp_pass_arg)
    for _, record_contents in rcp_data.items():
        if record_contents['BS'] > max_bs:
            max_record = record_contents
            max_bs = record_contents['BS']
    return max_record
    
# this should be a method of rcp_checker.RCP_Checker, but it's missing.
# Instead we derived it by extracting parts of rcp_checker.check_directory()
def get_rcp_record_for_bs(bs, checker, rcp_pass_arg='pruned_rcps'):
    rcp_record = checker._find_rcp(bs, rcp_pass_arg)
    if rcp_record is None:
        # bs is not one of the generated sizes, so need to interpolate:
        rcp_max = checker._find_bottom_max_rcp(bs, rcp_pass_arg)
        if rcp_max is None:
            raise RuntimeError("Error: no sufficiently large RCP bs found")
        rcp_min = checker._find_top_min_rcp(bs, rcp_pass_arg)
        if rcp_min is None:
            # bs is smaller than the smallest rcp, so just use smallest rcp
            rcp_record = checker._find_min_rcp(rcp_pass_arg)
        else:
            # interpolate
            interp_record_name, interp_record = checker._create_interp_rcp(bs, rcp_min, rcp_max)
            rcp_record = interp_record
    return rcp_record
    
def main():
    parser = argparse.ArgumentParser(
        description='Parse rcps_.json file, prune, and print out rcp means and mins'
    )

    parser.add_argument('benchmark', type=str, help="name of benchmark")
    parser.add_argument('--usage', type=str, default='training',
                        choices=['training', 'hpc'],
                        help="the WG that produced the benchmark")
    parser.add_argument('--version', type=str, default='5.1.0',
                        help='what version of the ruleset')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--unpruned', action='store_true',
                        help='print the unpruned rcps instead of the pruned')
    parser.add_argument('--no-header', action='store_true',
                        help='do not print the header line')
    parser.add_argument('--custom_rcps', type=argparse.FileType('r'),
                    help='specify an RCP json file to use')
    parser.add_argument('--interpolate', action='store_true',
                        help='generate interpolated rcp min/mean for all batch sizes')
    

    args = parser.parse_args()
    rcp_pass_arg='pruned_rcps'
    if (args.unpruned):
        rcp_pass_arg='full_rcps'

    checker=RCP_Checker(args.usage, args.version, args.benchmark, args.verbose, args.custom_rcps)

    if not args.no_header:
        print("BS,Mean,Min")

    if not args.interpolate:
        data=checker._get_rcp_data(rcp_pass_arg)
        for key, record in data.items():
            print_rcp_record(record)
    else:
        for bs in range(checker._find_min_rcp(rcp_pass_arg)['BS'], find_max_rcp(checker, rcp_pass_arg)['BS']+1):
            record = get_rcp_record_for_bs(bs, checker, rcp_pass_arg)
            print_rcp_record(record)

if __name__ == '__main__':
    main()
