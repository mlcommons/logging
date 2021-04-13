'''
RCP checker: Verifies convergence points of submissions by comparing them against RCPs (Reference Convergence Points)
'''

import argparse
import glob
import json
import os
import numpy as np
import re
import scipy.stats

# Number of submission runs for each benchmark
# References need 2x of these runs
# We use olympic scoring for statistics, so we reject
# both the top and bottom reference and submission numbers
submission_runs = {
    'bert': 10,
    'dlrm': 5,
    'maskrcnn' : 5,
    'resnet' : 5,
    'ssd' : 5,
    'unet3d' : 20,
    'rnnt': 10,
}

MLLOG_TOKEN = ':::MLLOG '


def get_submission_epochs(result_files):
    '''
    Extract convergence epochs from a list of submission files
    Returns the batch size and the list of epochs to converge
    -1 means run did not converge. Return None if > 1 files
    fail to converge
    '''
    not_converged = 0
    subm_epochs = []
    bs = -1
    for result_file in result_files:
        with open(result_file, 'r') as f:
            file_contents = f.readlines()
            for line in file_contents:
                if line.startswith(MLLOG_TOKEN):
                    str = line[len(MLLOG_TOKEN):]
                    if "global_batch_size" in str:
                        # Do we need to make sure global_batch_size is the same
                        # in all files? If so, this is obviously a bad submission
                        bs = json.loads(str)["value"]
                    if "eval_accuracy" in str:
                        eval_accuracy_str = str
                    if "run_stop" in str:
                        # Epochs to converge is the the last epochs value on
                        # eval_accuracy line before run_stop
                        conv_result = json.loads(str)["metadata"]["status"]
                        conv_epoch = json.loads(eval_accuracy_str)["metadata"]["epoch_num"]
                        if conv_result == "success":
                            subm_epochs.append(conv_epoch)
                        else:
                            subm_epochs.append(-1)
                            not_converged = not_converged + 1
    subm_epochs.sort()
    if not_converged > 1:
        return None
    return bs, subm_epochs


def eval_submission_record(rcp_record, subm_epochs):
    '''Compare reference and submission convergence.'''
    mean_subm_epochs = np.mean(subm_epochs[1:len(subm_epochs)-1])
    if mean_subm_epochs >= (rcp_record["RCP Mean"] / rcp_record["Max Speedup"]):
        # TODO: Print some useful debug info
        # print('Pass', mean_subm_epochs, rcp_record["RCP Mean"], rcp_record["Max Speedup"])
        return(True)
    else:
        # TODO: Print some useful debug info
        # print('Fail', mean_subm_epochs, rcp_record["RCP Mean"], rcp_record["Max Speedup"])
        return(False)


class RCP_Checker:

    def __init__(self, ruleset):
        self.p_value = 0.05
        self.tolerance = 0.0001
        raw_rcp_data = self._consume_json_file(ruleset)
        self.rcp_data = self._process_raw_rcp_data(raw_rcp_data)


    def _consume_json_file(self, ruleset):
        '''Read json file'''
        json_file = os.getcwd() + '/mlperf_logging/rcp_checker/' + ruleset + '/rcps.json'
        with open(json_file, 'r') as f:
            return json.load(f)


    def _process_raw_rcp_data(self, raw_rcp_data):
        '''
        Load the raw json file data into a dictionary
        that also contains mean, stdev, and max speedup for each record
        '''
        processed_rcps = {}
        for record, record_contents in raw_rcp_data.items():
            processed_record = {'Benchmark': record_contents['Benchmark'],
                                'BS': record_contents['BS'],
                                'Hyperparams': record_contents['Hyperparams'],
                                'Epochs to converge': record_contents['Epochs to converge'],
                                'RCP Mean': 0.0,
                                'RCP Stdev': 0.0,
                                'Max Speedup': 0.0}
            processed_rcps[record] = processed_record
            # TBD: Sanity check RCPs, eg number of runs, duplicate RCPs, RCPs or unknown benchmark
            # numbers out of bounds, etc.
        return processed_rcps


    def _compute_rcp_stats(self):
        '''Compute RCP mean, stdev and min acceptable epochs for RCPs'''
        for record, record_contents in self.rcp_data.items():
            epoch_list = record_contents['Epochs to converge']
            # Use olympic mean
            epoch_list.sort()
            record_contents['RCP Mean'] = np.mean(epoch_list[1:len(epoch_list)-1])
            record_contents['RCP Stdev'] = np.std(epoch_list[1:len(epoch_list)-1])
            min_epochs = self._find_min_acceptable_mean(
                              record_contents['Benchmark'],
                              record_contents['RCP Mean'],
                              record_contents['RCP Stdev'],
                              len(epoch_list)-2)
            record_contents['Max Speedup'] = record_contents['RCP Mean'] / min_epochs
            #print(record, record_contents)
            #print("\n")

    def _find_rcp(self, benchmark, bs):
        '''Find RCP based on benchmark and batch size'''
        for _, record_contents in self.rcp_data.items():
            if record_contents['Benchmark'] == benchmark and record_contents['BS'] == bs:
                return record_contents


    def _find_min_rcp(self, benchmark):
        '''Find RCP with the smallest batch size for a benchmark'''
        min_bs = 1e9
        min_record = None
        for _, record_contents in self.rcp_data.items():
            if record_contents['BS'] < min_bs and record_contents['Benchmark'] == benchmark:
                min_record = record_contents
                min_bs = record_contents['BS']
        return min_record

    def _find_top_min_rcp(self, benchmark, bs):
        '''
        Find top RCP to serve as min in interpolation.
        For example, if bs = 100 and reference has bs = 10, 20, 110, 120
        this will return the RCP with bs = 20.
        '''
        min_bs = 0
        min_record = None
        for _, record_contents in self.rcp_data.items():
            if record_contents['Benchmark'] == benchmark:
                if record_contents['BS'] < bs and record_contents['BS'] > min_bs:
                    min_bs = record_contents['BS']
                    min_record = record_contents
        return min_record


    def _find_bottom_max_rcp(self, benchmark, bs):
        '''
        Find bottom RCP to serve as max in interpolation.
        For example, if bs = 100 and reference has bs = 10, 20, 110, 120
        this will return the RCP with bs = 110.
        '''
        max_bs = 1e9
        max_record = None
        for _, record_contents in self.rcp_data.items():
            if record_contents['Benchmark'] == benchmark:
                if record_contents['BS'] > bs and record_contents['BS'] < max_bs:
                    max_bs = record_contents['BS']
                    max_record = record_contents
        return max_record


    def _find_p_value(self, subm_mean, subm_stdev, subm_num_samples,
                      ref_mean, ref_stdev, ref_num_samples,
                      p_value_lim=0.05):
        '''
        Do t-test between submission and reference and return p-value and
        whether it is larger than the limit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
        '''
        t_stat, p_value = scipy.stats.ttest_ind_from_stats(
                              subm_mean, subm_stdev, subm_num_samples,
                              ref_mean, ref_stdev, ref_num_samples,
                              equal_var=True)

        # convert from 2-sided test to 1-sided test
        if t_stat < 0:
            p_value = p_value / 2
        else:
            p_value = 1 - (p_value / 2)
        return p_value > p_value_lim, p_value


    def _find_min_acceptable_mean(self, benchmark, mean, stdev, num_samples_ref):
        '''
        Do a binary search to find the min acceptable epoch mean to converge
        The limits are 0 and the reference mean, anything above reference is acceptable
        '''

        if stdev == 0:
            return mean
        num_samples_subm = submission_runs[benchmark] - 2
        mean_max = mean
        mean_min = 0.0
        mean_mid = (mean_min + mean_max) / 2
        while mean_max - mean_min > self.tolerance:
            # We assume similar stdev between submission and reference
            # Samples and means are different for p-value function
            _, p_value = self._find_p_value(
                             mean_mid, stdev, num_samples_ref,
                             mean, stdev, num_samples_subm,
                             self.p_value)
            if p_value > self.p_value:
                mean_max = mean_mid
            else:
                mean_min = mean_mid
            mean_mid = (mean_min + mean_max) / 2

        return mean_mid


    def _create_interp_rcp(self, benchmark, target_bs, low_rcp, high_rcp):
        '''
        Create an interpolation RCP for batch size target_bs by interpolating
        low_rcp and high_rcp. Add the RCP into rcp_data.
        This RCP is marked as _interp_ in its name so it does not have epochs or hparams
        '''
        mean = np.interp(
                   target_bs,
                   [low_rcp['BS'], high_rcp['BS']],
                   [low_rcp['RCP Mean'], high_rcp['RCP Mean']])
        stdev = np.interp(
                    target_bs,
                    [low_rcp['BS'], high_rcp['BS']],
                    [low_rcp['RCP Stdev'], high_rcp['RCP Stdev']])

        min_epochs = self._find_min_acceptable_mean(
                         benchmark,
                         mean,
                         stdev,
                         submission_runs[benchmark]*2)
        interp_record_name = benchmark + '_interp_' + str(target_bs)
        interp_record = {'Benchmark': benchmark,
                         'BS': target_bs,
                         'Hyperparams': {},
                         'Epochs to converge': [],
                         'RCP Mean': mean,
                         'RCP Stdev': stdev,
                         'Max Speedup': mean / min_epochs}

        self.rcp_data[interp_record_name] = interp_record

    def check_directory(self, dir):
        '''
        Check directory for RCP compliance.
        Returns (Pass/Fail, string with explanation)
        Possible cases, the top 3 fail before RCP check.
        - Fail / did not find global_batch_size in log
        - Fail / run failed to converge. This will not happen when called
          from the results summarizer, only standalone
        - Fail / Benchmark w/o RCP records
        - Pass / RCP found
        - Pass / RCP interpolated
        - Pass / RCP missing but submission converges slower on smaller batch size
        - Fail / RCP found
        - Fail / RCP interpolated
        - Missing RCP / Submit missing RCP
        '''
        dir = dir.rstrip("/")
        pattern = '{folder}/result_*.txt'.format(folder=dir)
        benchmark = os.path.split(dir)[1]
        result_files = glob.glob(pattern, recursive=True)
        bs, subm_epochs = get_submission_epochs(result_files)

        if bs == -1:
            return 'Fail', 'Could not detect global_batch_size'
        if subm_epochs is None:
            return 'Fail', 'Insufficient convergence'

        rcp_record = self._find_rcp(benchmark, bs)
        rcp_msg = ''
        if rcp_record is not None:
            rcp_msg = 'RCP found'
            rcp_check = eval_submission_record(rcp_record, subm_epochs)
        else:
            rcp_min = self._find_top_min_rcp(benchmark, bs)
            rcp_max = self._find_bottom_max_rcp(benchmark, bs)
            if rcp_min is not None and rcp_max is not None:
                rcp_msg = "RCP Interpolation"
                self._create_interp_rcp(benchmark,bs,rcp_min,rcp_max)
                interp_rcp_record = self._find_rcp(benchmark, bs)
                rcp_check = eval_submission_record(interp_rcp_record, subm_epochs)
            elif rcp_min is not None and rcp_max is None:
                rcp_msg = 'Missing RCP, please submit RCP with BS = {b}'.format(b=bs)
                rcp_check = False
            elif rcp_min is None and rcp_max is not None:
                rcp_min_record = self._find_min_rcp(benchmark)
                rcp_check = eval_submission_record(rcp_min_record, subm_epochs)
                mean_subm_epochs = np.mean(subm_epochs[1:len(subm_epochs)-1])
                if rcp_check == False:
                   rcp_msg = 'Missing RCP, please submit RCP with BS = {b}'.format(b=bs)
                else:
                   rcp_msg = 'RCP not found but slower convergence on smaller batch size'
            else:
                rcp_check = False
                rcp_msg = 'Cannot find any RCPs'

        return rcp_check, rcp_msg


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.rcp_checker',
        description='Run RCP Checks on logs.',
    )

    parser.add_argument('dir', type=str,
                    help='the directory to check for compliance')
    parser.add_argument('--rcp_version', type=str, default='1.0.0',
                    help='what version of rules to check the log against')
    return parser


def make_checker(ruleset):
  return RCP_Checker(ruleset)


def main(checker, dir):
    return checker.check_directory(dir)
