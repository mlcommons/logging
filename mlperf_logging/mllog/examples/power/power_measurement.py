# Copyright 2019 MLBenchmark Group. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
from datetime import datetime
from abc import ABC, abstractmethod
import argparse
from mlperf_logging import mllog
import mlperf_logging.mllog.constants as mllog_constants
from mlperf_logging.mllog.examples.power.reader import DebugPowerReader, FilePowerReader, PowerReader


MLLOGGER = mllog.get_mllogger()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--power-log", type=str, default=None)
    parser.add_argument("--output-log", type=str, default="power_0.txt")
    parser.add_argument("--output-folder", type=str, default="output/power")
    parser.add_argument("--node", type=str, default="node_0")
    parser.add_argument("--skip-lines", type=int, default=1)
    parser.add_argument("--start-with-readings", action="store_true")

    parser.add_argument("--time-range", type=int, default=600, help="")
    parser.add_argument("--time-freq", type=int, default=1, help="")

    parser.add_argument("--convertion-coef", type=float, default=1.0)
    parser.add_argument(
        "--measurement-type", type=str, choices=["AC", "DC"], default="AC"
    )

    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-type", choices=["IPMI", "Bios"], default="IPMI")

    args = parser.parse_args()
    return args


def read_power_log(path):
    pass



class LogParser(ABC):
    def __init__(self, args) -> None:
        super().__init__()
        self.date_format = ""
        self.node = args.get("node", "node_0")

    def date_to_ms(self, date, format):
        return int(datetime.strptime(date, format).timestamp() * 1e3)

    @abstractmethod
    def extract_date(self, s):
        pass

    @abstractmethod
    def extract_power(self, s):
        pass

    def convert_power_units(self, power, units, target_units="W"):
        if units == "kW":
            return power * 1000
        elif units == "mW":
            return power / 1000
        else:
            return power

    def convert2mlperf(self, s):
        date = self.extract_date(s)
        power, units = self.extract_power(s)
        power = self.convert_power_units(power, units)
        time_ms = self.date_to_ms(date, self.date_format)
        # If it is a direct power reading
        MLLOGGER.event(
            key=mllog_constants.POWER_READING,
            value=power,
            time_ms=time_ms,
            metadata=dict(host=self.node),
        )
        # If this is a switch measurement the next line applies
        # MLLOGGER.event(key=mllog_constants.INTERCONNECT_POWER_EST, value=power, time_ms=time_ms, metadata=dict(host = "sw_0"))


class IPMIParser(LogParser):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.date_format = "%a %b %d %H:%M:%S %Y"

    def extract_date(self, s: str):
        s = s.replace("\n", "")
        splits = s.split("  ")
        target = splits[-1]
        idx = 0
        while not target[idx].isalnum():
            idx += 1
        return splits[-1][idx:]

    def extract_power(self, s: str):
        s = s.replace("\n", "")
        s = s.replace("\t", "")
        splits = s.split("  ")
        splits = [s for s in splits if s != ""]
        target = splits[1].split()[0]
        return float(target), "W"


class BiosParser(LogParser):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.date_format = "Y-%m-%d %H:%M:%S.%f+%z"

    def extract_date(self, s: str):
        splits = s.split(",")
        date = splits[0][:-3] + splits[0][-2:]
        return date

    def extract_power(self, s: str):
        return s.split(",")[-1], "W"


def run():
    args = get_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    mllog.config(filename=f"{args.output_folder}/{args.output_log}")
    debug = args.debug
    log_type = args.log_type
    start_with_readings = args.start_with_readings

    # Initilize power reader
    args = vars(args)
    if debug:
        power_reader = DebugPowerReader(args)
    else:
        assert args.get("power_log", None) is not None, "Argument power log is None"
        power_reader = FilePowerReader(args)

    # Initilize log parser
    if log_type == "IPMI":
        log_parser = IPMIParser(args)
    elif log_type == "Bios":
        log_parser = BiosParser(args)
    else:
        raise Exception("Log type currently not supported")

    if not start_with_readings:
        MLLOGGER.start(key=mllog_constants.POWER_MEASUREMENT_START)
        MLLOGGER.event(
            key=mllog_constants.CONVERTION_EFF,
            value=args.get("convertion_coef"),
            metadata=dict(measurement_type=args.get("measurement_type")),
        )
    else:
        s = power_reader.lines[0]
        date = log_parser.extract_date(s)
        time_ms = log_parser.date_to_ms(date, log_parser.date_format)
        MLLOGGER.start(key=mllog_constants.POWER_MEASUREMENT_START, time_ms=time_ms)
        MLLOGGER.event(
            key=mllog_constants.CONVERTION_EFF,
            value=args.get("convertion_coef"),
            metadata=dict(measurement_type=args.get("measurement_type")),
            time_ms=time_ms,
        )

    while power_reader.has_next():
        s = power_reader.read_power()
        log_parser.convert2mlperf(s)

    if not start_with_readings:
        MLLOGGER.end(key=mllog_constants.POWER_MEASUREMENT_STOP)
    else:
        assert debug == False
        s = power_reader.lines[-1]
        date = log_parser.extract_date(s)
        time_ms = log_parser.date_to_ms(date, log_parser.date_format)
        MLLOGGER.end(key=mllog_constants.POWER_MEASUREMENT_STOP, time_ms=time_ms)


if __name__ == "__main__":
    run()
