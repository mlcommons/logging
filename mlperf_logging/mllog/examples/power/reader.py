from abc import ABC, abstractmethod
from datetime import datetime
import time
import random

class PowerReader(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def read_power(self):
        raise NotImplementedError()

    @abstractmethod
    def has_next(self):
        raise NotImplementedError()


class DebugPowerReader(PowerReader):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.format = args.get("log_type", "IPMI")
        if self.format == "IPMI":
            self.s = """Instantaneous power reading:                   {} Watts	    IPMI timestamp:                           {}"""
            self.date_format = "%a %b %d %H:%M:%S %Y"
        if self.format == "bios":
            self.s = """{},{},{}"""
            self.date_format = "%Y-%m-%d %H:%M:%S.%f+"
        self.l = args.get("min_power", 10)
        self.r = args.get("max_power", 12)
        self.freq = args.get("time_freq", 1)
        self.time_range = args.get("time_range", 600)
        self.readings = 0

    def read_power(self):
        self.readings += 1
        if self.format == "IPMI":
            date = datetime.strftime(datetime.utcnow(), self.date_format)
            time.sleep(self.freq)
            return self.s.format(random.uniform(self.l, self.r), date)
        elif self.format == "bios":
            now = datetime.utcnow()
            timezone = datetime.strftime(now, "%z")
            timezone = timezone[:2] + ":" + timezone[2:]
            start_date = datetime.strftime(datetime.utcnow(), self.date_format)
            time.sleep(self.freq)
            end_date = datetime.strftime(datetime.utcnow(), self.date_format)
            return self.s.format(
                start_date + timezone,
                end_date + timezone,
                random.uniform(self.l, self.r),
            )
        raise NotImplementedError()

    def has_next(self):
        return self.readings < (self.time_range / self.freq)


class FilePowerReader(PowerReader):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.power_log = args.get("power_log")
        with open(self.power_log) as f:
            self.lines = f.readlines()
        self.next = args.get("skip_lines", 0)

    def read_power(self):
        line = self.lines[self.next]
        self.next += 1
        return line

    def has_next(self):
        return self.next < len(self.lines)
