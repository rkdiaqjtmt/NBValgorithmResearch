"""크기·장애물 확률마다 서로 다른 시드로 3회 실행하고 원본 결과를 저장한다."""
from pathlib import Path
import secrets
import subprocess
import sys


def main():
    runner = Path(__file__).resolve().parent / "run.py"
    failed = False
    for size in range(100, 550, 100):
        for obstacle_probability in (0.1, 0.3, 0.5):
            used_seeds = set()
            for repeat in range(1, 4):
                seed = secrets.randbelow(2**32)
                while seed in used_seeds:
                    seed = secrets.randbelow(2**32)
                used_seeds.add(seed)

                print(
                    f"\n반복={repeat}/3, 크기={size}, "
                    f"장애물 확률={obstacle_probability}, 시드={seed}",
                    flush=True,
                )
                completed = subprocess.run(
                    [
                        sys.executable, str(runner),
                        "--size", str(size),
                        "--obstacle_probability", str(obstacle_probability),
                        "--seed", str(seed),
                        "--sensor_range", "4",
                        "--algorithm", "ALL",
                    ],
                    check=False,
                )
                if completed.returncode != 0:
                    failed = True
                    print("이 조합에서 오류 또는 결과 불일치가 발생했습니다.", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
