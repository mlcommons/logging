# Power measurement

## Table of Contents <!-- omit in toc -->
1. [power_measurement.py](#power_measurementpy)
2. [Result Summarizer](#result-summarizer---computing-power-metric)
3. [Step by step example](#step-by-step-example)


## [power_measurement.py](power_measurement.py)
This script outputs the power logs in the mlperf format. It has two modes. First, you can provide a file containing the power logs in their native format (currently, only 'IPMI' and 'Bios' (csv) format are supported), and it will translate them into the mlperf format. Alternatively, you can run it in the debug mode and it will simulate the power reading in the specified format, and the translate the simulated power readings.

### Usage
```
python power_measurement.py [--power-log <path_to_input_power_log>]
                            [--output-log <output_file>]
                            [--output-folder <output_folder>][--node <name_of_node>]
                            [--skip-lines <number_of_lines_to_skip_in_input>]
                            [--start-with-readings]
                            [--time-range <time_to_measure>]
                            [--time-freq <time_between_measures>]
                            [--convertion-coef <convertion-coef>]
                            [--measurement-type <AC_or_DC>]
                            [--debug]
                            [--log-type <IPMI_or_Bios>]
```

#### Arguments
- **power-log:** Path to original power readings. This file that will be translated to the mlperf format. Only used if it is not in debug mode.
- **output-log:** Output log file.
- **output-folder:** Ouput folder.
- **skip-lines:** Number of lines to skip in the original power log. This is useful to skip headers in the csv files. Defaults to 0.
- **start-with-readings:** Sycronize the power measurement start and power measurement stop with the power reading in the power log. If this flag is not given, the current timestamp are used.
- **time-range:** Total time duration of power measurements. E.g: 600, 
- **time-freq:** Time frequency of power measurements. E.g: 1, 
- **convertion-coef:** Convertion coeffiecient, only applicable for AC.
- **measurement-type:** Power measurement type. Either AC or DC.
- **debug:** Flag to use debug mode. In debug mode, power readings will be simulated. The `power-log` argument is not needed in this mode.
- **log-type:** Format type of the original power logs. Either IPMI or Bios.


## Result Summarizer - Computing power metric
Several functions have been added to the [result_summarizer.py](../../../result_summarizer/result_summarizer.py) file in order to compute the final power metric. The script assumes the following folder structure:
```
├── ...
├── <submitter>
│   ├── benchmarks
│   ├── results
│   │   └── <system>
│   │       └── <benchmark>
│   │           ├── result_0.txt
│   │           ├── result_1.txt
│   │           ├── ...
│   │           ├── result_n.txt
│   │           └── power
│   │               ├── result_0
│   │               ├── result_1
│   │                   ├── node_0.txt
│   │                   ├── ...
│   │                   ├── node_k.txt
│   │                   ├── sw_0.txt
│   │                   ├── ...
│   │                   └── sw_m.txt
│   │               ├── ...
│   │               └── result_n
│   └── systems                   
└── ...
```

### [_compute_power_node](../../../result_summarizer/result_summarizer.py##L470-L488)
Computes the power metric of a single log file representing the energy comsumption of a single node in a single run. This function is called once for every node.
#### Arguments
- **loglines:** Parsed power log of a single node
- **time_to_train:** Time to train in milliseconds obtained from the performance logs

### [_compute_power_sw](../../../result_summarizer/result_summarizer.py##L494-L503)
Computes the power metric of a single log file representing the energy comsumption of a single switch (interconnect) in a single run. This function is called once for every node.
#### Arguments
- **loglines:** Parsed power log of a single node
- **time_to_train:** Time to train in milliseconds obtained from the performance logs

### [_compute_total_power](../../../result_summarizer/result_summarizer.py##L453-L568)
Computes the total energy used by the system in a single run using the previous two functions for each of the nodes and switches. 
#### Arguments
- **benchmark_folder:** Path to folder containing the benchmark
- **result_file:** Path to perf results. This is used to identify the correspoding power logs
- **time_to_train:** Time to train in milliseconds obtained from the performance logs
- **ruleset:** Version of the benchmark. E.g: "3.1.0", "3.0.0", "2.1.0"

### Computing the power result
Similar to the performance results, we perform and olympic average to compute the power result of the benchmark. This means the best and worst scores are dropped and the result is the average of the other ones.

## Step by step examples

### Producing a MLPerf power log
1. Clone the `logging` repository, install it as a pip package:
```
git clone https://github.com/mlcommons/logging.git mlperf-logging --branch power_support
pip install -e mlperf-logging
```
Then go into the `power` folder:
```
cd mlperf-logging/mlperf_logging/mllog/examples/power
```
2. Download the sample IPMI power readings from the MLCommons shared drive and place it inside the current folder. [Link](https://drive.google.com/file/d/1292hHqZqwjfPBFBaFIsC5hJRqcbB7Mas/view?usp=drive_link). Alternatively, you can download the 'Bios' power readings and change the IMPI for 'Bios'. [Alternative Link](https://drive.google.com/file/d/17_F-pJJkbsUMmMvZ_parFynn-Wf5yjHQ/view?usp=drive_link)

3. Run the `power_measurement.py` python script using the following command:
```
python power_measurement.py --power-log IPMIPower.txt --output-log node_0.txt --log-type IPMI --start-with-readings
```
4. Check the output file `./output/power/node_0.txt`

### Computing the metric for a single power log file
It is assumed that you completed the previous example and you were able to produce a MLPerf power log.

1. Run the `compute_metric_example.py` python script using the following command:
```
python compute_metric_example.py --input-log ./output/power/node_0.txt
```
2. The output of this should be 
```
Power consumed: 6514419.0 J
```

### Constructing a sample power submission
For this example we will add power logs to a previous training submission. This will simulate how a power submission should look like. **Note that the power logs we will generate do not correspond to this performance results, so this wouldn't be a valid submission.**

0. Setup. Go to a clean working directory and create a folder called example_power and set a variable with the path of the logging repo.
```
cd <CLEAN_WORKING_DIRECTORY>
mkdir example_power
export MLPERF_LOGGING_PATH=<PATH_TO_LOGGING_REPO>
```

1. Clone the training v3.0 repository
```
git clone https://github.com/mlcommons/training_results_v3.0.git
```

2. We will use the previous results from `Quanta_Cloud_Technology` and add power logs to it. Move the results to the `example_power`.
```
cp -r training_results_v3.0/Quanta_Cloud_Technology example_power
```

3. Create the necessary power folders
```
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_0
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_1
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_2
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_3
mkdir example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_4
```

4. Follow the steps from the [first example](#producing-a-mlperf-power-log) to create power logs. You can use any of the source logs provided. We recommend to create logs for several nodes, e.g node_0.txt, node_1.txt. Place them into `$MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/`

5. Move the power logs files to each of the results folder we created.

```
cp -r $MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/* example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_0/
cp -r $MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/* example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_1/
cp -r $MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/* example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_2/
cp -r $MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/* example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_3/
cp -r $MLPERF_LOGGING_PATH/mlperf_logging/mllog/examples/power/output/power/* example_power/Quanta_Cloud_Technology/results/D54Q-2U/resnet/power/result_4/
```
The folder structure should look like this:
```
├── Quanta_Cloud_Technology
│   ├── benchmarks
│   ├── results
│   │   └── D54Q-2U
│   │       └── resnet
│   │           ├── result_0.txt
│   │           ├── result_1.txt
│   │           ├── ...
│   │           ├── result_n.txt
│   │           └── power
│   │               ├── result_0
│   │                   ├── node_0.txt
│   │                   ├── node_1.txt
│   │                   ├── ...
│   │               ├── result_1
│   │                   ├── node_0.txt
│   │                   ├── node_1.txt
│   │                   ├── ...
│   │               ├── result_2
│   │                   ├── node_0.txt
│   │                   ├── node_1.txt
│   │                   ├── ...
│   │               ├── result_3
│   │                   ├── node_0.txt
│   │                   ├── node_1.txt
│   │                   ├── ...
│   │               ├── result_4
│   │                   ├── node_0.txt
│   │                   ├── node_1.txt
│   │                   ├── ...
```

6. Run the `result_sumarizer.py` script on this folder.
```
python3 -m mlperf_logging.result_summarizer example_power/Quanta_Cloud_Technology/ training 3.1.0
```
