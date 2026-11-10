from pathlib import Path
import secrets
import subprocess
import sys

runner = Path(__file__).resolve().parent / "R.py"
for qwer in range(3):
    for size in range(100, 550, 100):
        for obstacle_probability in (0.1, 0.3, 0.5):
            seed = secrets.randbelow(2**32)
    
            print(
                f"\n크기={size}, 장애물 확률={obstacle_probability}, "
                f"시드={seed}",
                flush=True,
            )
    
            completed = subprocess.run(
                [
                    sys.executable,
                    str(runner),
                    "--size", str(size),
                    "--obstacle_probability", str(obstacle_probability),
                    "--seed", str(seed),
                    "--sensor_range", "4",
                    "--algorithm", "ALL",
                ],
                check=False,
            )
    
            if completed.returncode != 0:
                print("이 조합에서 오류 또는 결과 불일치가 발생했습니다.", flush=True)
