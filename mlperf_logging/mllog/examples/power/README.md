# Power measurement

## [power_measurement.py](power_measurement.py)
This script outputs the power logs in the mlperf format. It has two modes. First, you can provide a file containing the power logs in their native format (currently, only 'IMPI' and 'Bios' (csv) format are supported), and it will translate them into the mlperf format. Alternatively, you can run it in the debug mode and it will simulate the power reading in the specified format, and the translate the simulated power readings.

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
                            [--log-type <IMPI_or_Bios>]
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
- **log-type:** Format type of the original power logs. Either IMPI or Bios.


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
│   │               ├── result_0.txt
│   │               ├── result_1.txt
│   │                   ├── node_0.txt
│   │                   ├── ...
│   │                   ├── node_k.txt
│   │                   ├── sw_0.txt
│   │                   ├── ...
│   │                   └── sw_m.txt
│   │               ├── ...
│   │               └── result_n.txt
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
- **ruleset:** Version of the benchmark. E.g: "3.0.0", "2.1.0"

### Computing the power result
Similar to the performance results, we perform and olympic average to compute the power result of the benchmark. This means the best and worst scores are dropped and the result is the average of the other ones.
