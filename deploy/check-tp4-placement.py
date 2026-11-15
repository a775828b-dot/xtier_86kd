"""Verify that XTier TP4 ranks use disjoint CPU0 core groups."""

import os
import sys

from xtier.engine import placement, worker_cpus

rank = int(sys.argv[1])
placement.apply(12, rank, 4)
print(
    f"rank={rank} group={os.environ['XTIER_CORE_GROUP']} "
    f"workers={worker_cpus(12)} "
    f"dispatcher={os.environ['XTIER_PROBE_DISPATCH_CPU']} "
    f"main={sorted(os.sched_getaffinity(0))}",
    flush=True,
)
