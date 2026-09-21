"""Create a local five-setting RVC audition without retraining a model."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import load_settings
from core.rvc_comparison import run_comparison
from core.task_queue import TASK_GATE


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--guide', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--transpose', type=int, default=0)
    args = parser.parse_args()
    folder = TASK_GATE.run('comparaison RVC', run_comparison, load_settings(), args.model, args.guide,
                           args.reference, args.transpose, lambda message: print(message, flush=True))
    print(folder / 'ecouter.html', flush=True)
