from pathlib import Path
import os
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    command = [sys.executable, "-m", "core.pipeline", "--config",
               str(root / "configs/train_4gpu.yaml"), "--nproc-per-node", "4",
               *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=root))
