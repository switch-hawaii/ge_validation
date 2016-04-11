import sys, os
curdir = os.path.dirname(__file__)
switch_dir = os.path.join(curdir, 'switch')
if os.path.isdir(os.path.join(switch_dir, 'switch_mod')):
    # insert main switch directory into system path
    sys.path = [switch_dir] + sys.path
