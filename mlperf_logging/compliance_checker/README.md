# MLPerf training logging compliance checker

### Requirements
The checker works with both python2 and python3, requires PyYaml package.
[See exact versions tested](#tested-software-versions)

### Usage

To check a log file for compliance:

    python -m mlperf_logging.compliance_checker [--config YAML] [--usage training/hpc] [--ruleset MLPERF_EDITION] FILENAME

By default, 5.1.0 training edition rules are used and the default config is set to `5.1.0/common.yaml`.
This config will check all common keys and enqueue benchmark specific config to be checked as well.
Old training editions, still supported are 5.0.0, 4.1.0, 4.0.0, 3.1.0, 3.0.0, 2.1.0, 2.0.0, 1.1.0, 1.0.0, 0.7.0 and 0.6.0

To check hpc compliance rules (only 1.0.0 ruleset is supported), set --usage hpc --ruleset 1.0.0.

Prints `SUCCESS` when no issues were found. Otherwise will print error details.

As log examples use [NVIDIA's training logs](https://github.com/mlperf/training_results_v{0.6,0,7,1.0,1.1}/tree/master/NVIDIA/results).

### Existing config files for training submissions

    5.1.0/common.yaml          - currently the default config file, checks common fields complience and equeues benchmark-specific config file
    5.1.0/closed_common.yaml   - the common rules file for closed submissions. These rules apply to all benchmarks
    5.1.0/open_common.yaml     - the common rules file for open submissions. These rules apply to all benchmarks
    5.1.0/closed_retinanet.yaml      - Per-benchmark rules, closed submissions.    
    5.1.0/closed_llama31_8b.yaml
    5.1.0/closed_llama31_405b.yaml
    5.1.0/closed_dlrm_dcnv2.yaml
    5.1.0/closed_rgat.yaml
    5.1.0/closed_llama2_70b_lora.yaml
    5.1.0/closed_flux1.yaml
    5.1.0/open_retinanet.yaml        - Per-benchmark rules, open submissions.    
    5.1.0/open_llama31_8b.yaml
    5.1.0/open_llama31_405b.yaml
    5.1.0/open_dlrm_dcnv2.yaml
    5.1.0/open_rgat.yaml
    5.1.0/open_llama2_70b_lora.yaml
    5.1.0/open_flux1.yaml

### Existing config files for HPC submissions

### Implementation details
Compliance checking is done following below algorithm.

1. Parser converts the log into a list of records, each record corresponds to MLLOG
   line and contains all relevant extracted information
2. Set of rules to be checked in loaded from provided config yaml file
3. Process optional `BEGIN` rule if present by executing provided `CODE` section
3. Remove messages for rules that are overridden
4. Loop through the records of the log
   1. If the key in the record is defined in rules process the rule:
      1. If present, execute `PRE` section
      2. If present, evaluate `CHECK` section, and store a warning message if the result is false
      3. If present, execute `POST` section
   2. Increment occurrences counter
5. Store a warning message if any occurrences requirements were violated
6. Process optional `END` rule if present:
   1. If present, execute `PRE`
   2. If present, evaluate `CHECK` section, and raise an exception if the result is false
7. Print all warning messages

Possible side effects of yaml sections execution can be [printing output](#other-operations), or [enqueueing
additional yaml files to be verified](#enqueuing-additional-config-files).

### Config file syntax
Rules to be checked are provided in yaml (config) file. A config file contains the following records:

#### `BEGIN` record
Defines `CODE` to be executed before any other rules defined in the current file. This record is optional
and there can be up to a single `BEGIN` record per config file.

Example:

    - BEGIN:
        CODE: " s.update({'run_start':None}) "


#### `KEY` record
Defines the actions to be triggered while processing a specific `KEY`. The name of the `KEY` is specified in field `NAME`.

The following fields are optional:
- `REQ` - specifies the requirement regarding occurrence. Possible values :
    - `EXACTLY_ONE` - current key has to appear exactly once
    - `AT_LEAST_ONE` - current key has to appear at least once
    - `AT_LEAST(n)` - current key has to appear at least n times
    - `AT_LEAST_ONE_OR(alternatives)` - current key or one of the alternative has to appear at least once;
            alternatives is a comma separated list of keys
- `PRE` - code to be executed before performing checks
- `CHECK` - expression to be evaluated as part of checking this key. False result would mean a failure.
- `POST` - code to be executed after performing checks

Example:

    - KEY:
        NAME:  epoch_start
        REQ:   AT_LEAST_ONE
        CHECK: " s['run_started'] and not s['in_epoch'] and ( v['metadata']['epoch_num'] == (s['last_epoch']+1) ) and not s['run_stopped']"
        POST:  " s['in_epoch'] = True; s['last_epoch'] = v['metadata']['epoch_num'] "


#### `END` record
Specifies actions to be taken after processing all the lines in log file. This record is optional and
there can be up to a single `END` record per config file.

The following fields are optional:
- `PRE` - code to be executed before performing checks
- `CHECK` - expression to be evaluated as part of checking this key. False result would mean a failure.

#### Global and local state access

During processing of the records there is a global state `s` maintained, accessible from
code provided in yaml. In addition, rules can access the information fields (values) `v`
of the record, as well as timestamp and the original line string as part of the record `ll`.

Global state `s` can be used to enforce any cross keys rules, by updating the global state
in `POST` (or `PRE`) of one `KEY` and using that information for `CHECK` of another `KEY`.
For each config file, `s` starts as an empty dictionary, so in order to track global state
it would require adding an entry to `s`.

Example:

    - BEGIN:
        CODE: " s.update({'run_start':None}) "

`ll` is a structure representing current log line that triggered `KEY` record. `ll` has the following fields
that can be accessed:
- `full_string` - the complete line as a string
- `timestamp` - milliseconds as an integer
- `key` - the string key
- `value` - the parsed value associated with the key, or None if no value
- `lineno` - line number in the original file of the current key

`v` is a shortcut for `ll.value`

Example:

    - KEY:
        NAME:  run_stop
        CHECK: " ( v['metadata']['status'] == 'success' )"
        POST:  " print('score [sec]:' , ll.timestamp - s['run_start']) "



#### Enqueuing additional config files

To enqueue additional rule config files to be verified use `enqueue_config(YAML)` function.
Config files in the queue are processed independently, meaning that they do not share state or any rules.

Each config file may define it's `BEGIN` and `END` records, as well as any other `KEY` rules.

Example:

    - KEY:
        NAME:  submission_benchmark
        REQ:   EXACTLY_ONE
        CHECK: " v['value'] in ['resnet', 'ssd', 'maskrcnn', 'transformer', 'gnmt'] "
        POST:  " enqueue_config('1.0.0/{}.yaml'.format(v['value'])) "


#### Other operations

`CODE`, `REQ`, and `POST` fields are executed using python's `exec` function. `CHECK` is performed
using `eval` call. As such, any legal python code would be suitable for use.

For instance, can define rules that would print out information as shown in the [example above](#global-and-local-state-access).


### Tested software versions
Tested and confirmed working using the following software versions:
- Python 2.7.12 + PyYAML 3.11
- Python 3.6.8  + PyYAML 5.1
- Python 2.9.2 + PyYAML 5.3.1
- Python 3.9.10 + PyYAML 5.5.0

### How to install PyYaML

    pip install pyyaml
