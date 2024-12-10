from .ruleset_060 import parse_file as parse_file_060
from .ruleset_070 import parse_file as parse_file_070
from .ruleset_100 import parse_file as parse_file_100
from .ruleset_110 import parse_file as parse_file_110
from .ruleset_200 import parse_file as parse_file_200
from .ruleset_210 import parse_file as parse_file_210
from .ruleset_300 import parse_file as parse_file_300
from .ruleset_310 import parse_file as parse_file_310
from .ruleset_400 import parse_file as parse_file_400
from .ruleset_410 import parse_file as parse_file_410
from .ruleset_500 import parse_file as parse_file_500


def parse_file(filename, ruleset='0.6.0'):
    if ruleset == '0.6.0':
        return parse_file_060(filename)
    elif ruleset == '0.7.0':
        return parse_file_070(filename)
    elif ruleset == '1.0.0':
        return parse_file_100(filename)
    elif ruleset == '1.1.0':
        return parse_file_110(filename)
    elif ruleset == '2.0.0':
        return parse_file_200(filename)
    elif ruleset == '2.1.0':
        return parse_file_210(filename)
    elif ruleset == '3.0.0':
        return parse_file_300(filename)
    elif ruleset == '3.1.0':
        return parse_file_310(filename)
    elif ruleset == '4.0.0':
        return parse_file_400(filename)    
    elif ruleset == '4.1.0':
        return parse_file_410(filename)
    elif ruleset == '5.0.0':
        return parse_file_500(filename)
    else:
        raise Exception(f'Ruleset "{ruleset}" is not supported')
