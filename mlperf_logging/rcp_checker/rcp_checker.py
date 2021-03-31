'''
RCP checker: Verifies convergence points of submissions by comparing them against RCPs (Reference Convergence Points)
'''

import json
import os
import numpy as np
import scipy.stats

# Number of submission runs for each benchmark
# References need 2x of these runs
# We use olympic scoring for statistics, so
# we use 8 or 3 runs from submissions.
submission_runs = {
    'bert': 10,
    'dlrm': 5,
    'maskrcnn' : 5,
    'resnet' : 5,
    'ssd' : 5,
    'unet3d' : 5,
    'rnnt': 10,
}


class RCP_Checker:

    def __init__(self, ruleset):
        self.p_value = 0.005
        self.tolerance = 0.0001
        raw_rcp_data = self._consume_json_file(ruleset)
        self.rcp_data = self._process_raw_rcp_data(raw_rcp_data)


    def _consume_json_file(self, ruleset):
        '''Read json file'''
        json_file = os.getcwd() + '/rcp_checker/' + ruleset + '/rcps.json'
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


    def _find_rcp(self, benchmark, bs):
        '''Find RCP based on benchmark and batch size'''

        for _, record_contents in self.rcp_data.items():
            if record_contents['Benchmark'] == benchmark and record_contents['BS'] == bs:
                return record_contents


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
                             0.05)
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
                         'RCP Max speedup': mean / min_epochs}

        self.rcp_data[interp_record_name] = interp_record

    def check_directory(self, dir):
        '''Check directory for RCP compliance. WIP, currently empty'''
        return True

def make_checker(ruleset):
  return RCP_Checker(ruleset)


def main(checker, dir):
    return checker.check_directory(dir)



