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

def main():
    parser = argparse.ArgumentParser(
        description='Parse rcps_.json file, prune, and print out rcp means and mins'
    )

    parser.add_argument('benchmark', type=str, help="name of benchmark")
    parser.add_argument('--usage', type=str, default='training',
                        choices=['training', 'hpc'],
                        help="the WG that produced the benchmark")
    parser.add_argument('--version', type=str, default='5.0.0',
                        help='what version of the ruleset')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--unpruned', action='store_true',
                        help='print the unpruned rcps instead of the pruned')
    parser.add_argument('--custom_rcps', type=argparse.FileType('r'),
                    help='specify an RCP json file to use')

    args = parser.parse_args()
    checker=RCP_Checker(args.usage, args.version, args.benchmark, args.verbose, args.custom_rcps)
    data=checker.pruned_rcp_data
    if (args.unpruned):
        data=checker.rcp_data

    print("BS,Mean,Min")
    for key, record in data.items():
        print(f"{record['BS']},{record['RCP Mean']},{record['Min Epochs']}")

if __name__ == '__main__':
    main()
