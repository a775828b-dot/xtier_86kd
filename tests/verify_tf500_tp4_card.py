"""Read-only check of the tf500 console's TP4 card against the live service."""

import importlib
import os
import sys

sys.path.insert(0, "/data/src/xtier_86kd/deploy")
module = importlib.import_module(os.environ.get("TF500_CONSOLE_MODULE", "tf500_console"))
console = module.Tf500Console()

assert set(console.models) == {module.MODEL_ID, module.TP4_MODEL_ID}
options = console.defaults(module.TP4_MODEL_ID)
assert options == {
    "engine": "on", "spec": "on", "verify": "static", "engram": "cpu",
    "context": "524288", "kvpool": "20000000", "running": "32",
    "batch": "49152", "interleave": "72", "graph": "32",
    "memory": "0.85", "reserve": "2",
}
env, argv = console.build(module.TP4_MODEL_ID, options)
assert env["PORT"] == "30001" and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
assert env["KV_POOL"] == "20000000" and env["BATCH"] == "49152"
assert argv[-1].endswith("start-xtier-tp4.sh")
assert "加速包" not in console.describe(module.TP4_MODEL_ID, options)
state = console.read_state()
assert state["model"] == module.TP4_MODEL_ID and state["runner"] == "adopted"
assert state["port"] == 30001 and console.active(state)
status = console.status()
assert status["state"] == "online" and status["model"] == module.TP4_MODEL_ID
assert status["gpu"]["count"] == 4
page = module.tf500_nav.decorate(
    module.ui.page_html({**console.console_data(), "status": status}, title="XTier 86KD Console"),
    "console",
)
assert "ds41-xtier-tp4" in page and "XTier TP4" in page
print("TP8 card retained; TP4 card defaults, live adoption, endpoint and GPU scope OK")
