'''
Parses a text MLPerf log into a structured format.
'''

from __future__ import print_function

import collections
import json
import re
import sys
from dataclasses import dataclass

from io import open

@dataclass
class LogLine:
    """Class for keeping track of an item in inventory."""
    full_string: str
    timestamp: float
    key: str
    value: str
    lineno: int

TOKEN = ':::MLLOG '


def parse_line(line):
    if not line.startswith(TOKEN):
        return None

    return json.loads(line[len(TOKEN):])


def string_to_logline(lineno, string):
    ''' Returns a LogLine or raises a ValueError '''
    m = parse_line(string)

    if m is None:
        raise ValueError('does not match regex')

    args = []
    args.append(string) # full string

    ts = float(m['time_ms']) # may raise error, e.g. "1.2.3"
    # TODO check for weird values
    args.append(ts)

    args.append(m['key']) # key

    j = { 'value': m['value'], 'metadata': m['metadata'] }
    args.append(j)

    args.append(lineno)
    return LogLine(*args)


def parse_file(filename):
    ''' Reads a file by name and returns list of loglines and list of errors'''
    with open(filename, encoding='latin-1') as f:
        return parse_generator(f)


def strip_and_dedup(gen):
    lines = []
    for l in gen:
        if TOKEN not in l:
            continue
        lines.append(re.sub(".*"+TOKEN, TOKEN, l))
    return lines



def parse_generator(gen):
    ''' Reads a generator of lines and returns (loglines, errors)
    The list of errors are any parsing issues as a tuple (str_line, error_msg)
    '''
    loglines = []
    failed = []
    for lineno, line in enumerate(strip_and_dedup(gen)):
        line = line.strip()
        try:
            ll = string_to_logline(lineno, line)
            loglines.append(ll)
        except ValueError as e:
            failed.append((line, str(e)))
    return loglines, failed


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('usage: mlp_parser.py FILENAME')
        print('       tests parsing on the file.')
        sys.exit(1)

    filename = sys.argv[1]
    lines, errors = parse_file(filename)

    print('Parsed {} log lines with {} errors.'.format(len(lines), len(errors)))

    if len(errors) > 0:
        print('Lines which failed to parse:')
        for line, error in errors:
            print('  Following line failed: {}'.format(error))
            print(line)

