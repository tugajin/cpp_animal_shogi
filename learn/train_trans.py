import subprocess
from merge_count import *
from merge_resolved import *
from train_network_trans import *
import sys

args = sys.argv
num = int(args[1])

#merge_count()
#merge_resolved()
train_network()

