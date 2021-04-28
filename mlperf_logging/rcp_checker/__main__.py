import sys

from . import rcp_checker

parser = rcp_checker.get_parser()
args = parser.parse_args()

# Results summarizer makes these 3 calls to invoke RCP test
checker = rcp_checker.make_checker(args.rcp_version, args.verbose)
checker._compute_rcp_stats()
test, msg = checker._check_directory(args.dir)

if test:
    print(msg, ",RCP test passed")
else:
    print(msg, ",RCP test failed")
    sys.exit(1)
