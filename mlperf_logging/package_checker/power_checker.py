import warnings
import os
import logging


class PowerChecker:
    """ Check for errors in the MLPerf Power submissions.
    Current checks are:

    1. Check there is a power folder for each result
    2. Check the power file names
    3. Check there are the same number of nodes and switches in each run
       (No file is missing)

    Unsatisfying any of the above checks results in failure.
    """
    def __init__(self, ruleset):
        self._ruleset = ruleset

    def check_range(self, l, n):
        seen = set({})
        errors = []
        for e in l:
            if e < 0 or e > (n-1) or e in seen:
                return False

        return True

    def check_equals(self, l):
        counter = {}
        errors = []
        for e in l:
            if e in counter:
                counter[e] += 1
            else:
                counter[e] = 1
        max_equals = max(counter, key = counter.get)
        for i, e in enumerate(l):
            if e != max_equals:
                errors.append(i)
        
        return len(errors) == 0, errors

    def check_power(self, power_folder, result_files):
        system, benchmark = os.path.normpath(power_folder).split(os.sep)[-3:-1]
        errors_found = 0
        errors_set = set()

        node_lens = []
        sw_lens = []
        for result_file in result_files:
            result_name, _ = os.path.splitext(os.path.basename(result_file))
            if os.path.exists(os.path.join(power_folder, result_name)):
                power_result_folder = os.path.join(power_folder, result_name)
                power_files = os.listdir(power_result_folder)
                node_results  = [file for file in power_files if file.startswith("node")]
                sw_results  = [file for file in power_files if file.startswith("sw")]
                node_idx  = [int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) for file in node_results]
                sw_idx  = [int(os.path.splitext(os.path.basename(file))[0].split('_')[-1]) for file in sw_results]

                if len(power_files) > len(node_results) + len(sw_results):
                    logging.warning("Detected %d total files in directory %s, but some do not conform", len(power_files), power_result_folder)

                if not self.check_range(node_idx, len(node_results)):
                    logging.warning("Bad naming of node power files in directory %s, expected to be node_x with x in range [0, %d]", power_result_folder, len(node_results)-1)
                    errors_found += 1
                    errors_set.add(result_name)
                if not self.check_range(sw_idx, len(sw_results)):
                    logging.warning("Bad naming of sw power files in directory %s, expected to be sw_x with x in range [0, %d]", power_result_folder, len(sw_results)-1)
                    errors_found += 1
                    errors_set.add(result_name)
                
                node_lens.append(len(node_results))
                sw_lens.append(len(sw_results))
                pass
            else:
                logging.warning("Package does not contain power result for %s/%s: %s", system, benchmark, result_name)
                errors_found += 1
                errors_set.add(result_name)

        result_names = [os.path.splitext(os.path.basename(result_file))[0] for result_file in result_files]

        valid, errors = self.check_equals(node_lens)
        node_errors = set([result_names[error] for error in errors])
        for error_result in [result_names[error] for error in errors]:
            logging.warning("Inconsistent number of nodes in directory %s/%s", power_folder, error_result)
            logging.warning("Directory %s/%s does not comply", power_folder, error_result)
        
        valid, errors = self.check_equals(sw_lens)
        sw_errors = set([result_names[error] for error in errors])
        for error_result in [result_names[error] for error in errors]:
            logging.warning("Inconsistent number of sw in directory %s/%s", power_folder, error_result)
            logging.warning("Directory %s/%s does not comply", power_folder, error_result)
        
        errors_set = errors_set | node_errors | sw_errors
        return errors_found == 0, errors_set