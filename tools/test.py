"""Run numerical tests and make Vulkan validation diagnostics fail the gate."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--vulkan-validation", action="store_true")
parser.add_argument("--swarm", action="store_true", help="require the native swarm worker integration tests")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
logs = root / "artifacts" / "validation"
logs.mkdir(parents=True, exist_ok=True)
env = os.environ.copy()
if args.swarm:
    env["MUNET_TEST_SWARM"] = "1"
if args.vulkan_validation:
    env.update(MUNET_TEST_VULKAN="1", VK_INSTANCE_LAYERS="VK_LAYER_KHRONOS_validation",
               VK_LAYER_ENABLES="VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT")
proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "--junitxml=" + str(logs / "tests.xml")],
                      cwd=root, env=env, capture_output=True, text=True)
output = proc.stdout + "\n" + proc.stderr
(logs / "tests.log").write_text(output)
print(output)
errors = re.findall(r"(?:Validation Error|SYNC-HAZARD|VUID-)", output)
print(f"Vulkan validation error markers: {len(errors)}")
raise SystemExit(proc.returncode or (1 if errors else 0))
