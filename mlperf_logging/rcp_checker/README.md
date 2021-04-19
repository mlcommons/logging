# MLPerf RCP checker

MLPerf reference convergence point (RCP) checker

## Description

Currently this ingests RCP points from json file and provides API
to run convergence checks using t-tests, find convergence points
and generate interpolated convergence points.

## Usage

From the base directory of the repo (by default `logging`):

python3 -m mlperf_logging.rcp_checker

Currently this loads RCPs from 1.0.0/rcps.json and runs a basic test

## Requirements

You need numpy and scipy
python3 -m pip install numpy scipy

## Tested software versions

TBD


