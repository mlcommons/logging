import sys

from . import rcp_checker

parser = rcp_checker.get_parser()
args = parser.parse_args()

# Results summarizer makes these 3 calls to invoke RCP test
checker = rcp_checker.make_checker(args.rcp_version)
checker._compute_rcp_stats()
result = checker.check_directory(args.dir)

print(result)
