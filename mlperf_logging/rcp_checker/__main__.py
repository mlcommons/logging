import sys

from . import rcp_checker

checker = rcp_checker.make_checker('1.0.0')

checker._compute_rcp_stats()

rcp = checker._find_rcp('maskrcnn', 96)
print(rcp,"\n")

rcp = checker._find_rcp('maskrcnn', 112)

if rcp == None:
    rcp_min = checker._find_top_min_rcp('maskrcnn', 112)
    rcp_max = checker._find_bottom_max_rcp('maskrcnn', 112)
    print("MIN",rcp_min,"\n")
    print("MAX",rcp_max,"\n")
    checker._create_interp_rcp('maskrcnn',112,rcp_min,rcp_max)
    print(checker._find_rcp('maskrcnn', 112))

