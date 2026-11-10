"""Run the original basiccode algorithms with simple command-line inputs."""
import argparse
import csv
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# 저장소 루트 또는 basiccode 폴더에서 실행할 수 있도록 설정
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "basiccode"))
import nbv_common as common

MODULES = {
    "A": "A_wjstnNBV",
    "B": "B_stalelazygain",
    "C": "C_cstarNBV",
}

# CSV에 저장할 열 이름과 순서
CSV_FIELDS = [
    "size",
    "obstacle_probability",
    "actual_density",
    "seed",
    "start_row",
    "start_col",
    "sensor_range",
    "algorithm",
    "map_seconds",
    "exploration_seconds",
    "total_cost",
    "rounds",
    "status",
    "remaining_unknown",
    "matches_first_algorithm",
    "exact_gain_calls",
]


def main():
    parser = argparse.ArgumentParser(
        description="NBV experiment: python run.py --size 20 --density 0.1"
    )
    parser.add_argument(
        "--size", type=int, required=True,
        help="Square map side length"
    )
    parser.add_argument(
        "--density", type=float, required=True,
        help="Obstacle probability, from 0 (inclusive) to 1 (exclusive)"
    )
    parser.add_argument(
        "--algorithm", type=str.upper,
        choices=["ALL", "A", "B", "C"], default="ALL",
        help="Default: run A, B and C on the same map"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Map seed (default: 42)"
    )
    parser.add_argument(
        "--max-attempts", type=int, default=1000,
        help="Maximum map generation attempts (default: 1000)"
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results.csv",
        help="Append results to this CSV (default: basiccode/results.csv)"
    )
    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be at least 1")
    if not 0 <= args.density < 1:
        parser.error("--density must be at least 0 and less than 1")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    # 이전 형식의 CSV에 새 형식의 행이 섞이지 않도록 검사
    if output.exists() and output.stat().st_size > 0:
        with output.open("r", newline="", encoding="utf-8") as stream:
            existing_header = next(csv.reader(stream), None)

        if existing_header != CSV_FIELDS:
            parser.error(
                "기존 CSV의 열 구성이 현재 코드와 다릅니다. "
                "--output results_with_counts.csv처럼 "
                "새 파일명을 지정하세요. 기존 파일은 변경하지 않았습니다."
            )

    def save_failure(stage, error, algorithm=None):
        failure = dict(
            timestamp=datetime.now(timezone.utc).isoformat(),
            stage=stage,
            size=args.size,
            density=args.density,
            seed=args.seed,
            max_attempts=args.max_attempts,
            sensor_range=4.0,
            algorithm=algorithm,
            error=str(error),
        )
        failure_path = output.with_suffix(".failures.jsonl")
        with failure_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        print(f"Failure saved: {failure_path}", flush=True)

    print(
        f"Map: {args.size}x{args.size}, "
        f"obstacle probability={args.density}, "
        f"seed={args.seed}, sensor range=4.0",
        flush=True,
    )
    print("Generating map...", flush=True)

    started = perf_counter()
    try:
        truth, initial_known, start = common.generate_map(
            args.size,
            args.size,
            args.density,
            args.seed,
            args.max_attempts,
        )
    except RuntimeError as exc:
        save_failure("MAP_GENERATION_FAILED", exc)
        parser.exit(
            1,
            f"Map generation failed: {exc}\n"
            "The original generator requires connected free cells and a free "
            "neighbor for every obstacle. Large/dense maps may fail. "
            "No algorithm was run; the requested density was not changed.\n"
        )

    generation_seconds = perf_counter() - started
    actual_density = (
        sum(row.count(common.WALL) for row in truth) / args.size**2
    )
    print(
        f"Actual density={actual_density:.4f}, start={start}, "
        f"map generation={generation_seconds:.3f}s",
        flush=True,
    )

    algorithms = MODULES if args.algorithm == "ALL" else [args.algorithm]
    reference = None
    failed = False

    for name in algorithms:
        module = importlib.import_module(MODULES[name])
        module.HEIGHT = module.WIDTH = args.size
        module.OBSTACLE_RATE = args.density
        module.SEED = args.seed
        module.SENSOR_RANGE = 4.0
        module.truth = [row[:] for row in truth]
        module.known = [row[:] for row in initial_known]

        print(f"Running {name}...", flush=True)

        started = perf_counter()
        try:
            result = module.explore(module.truth, module.known, start)
        except Exception as exc:
            save_failure("ALGORITHM_ERROR", exc, name)
            raise
        elapsed = perf_counter() - started

        # 各アルゴリズム内で集計した正確な情報利得の計算回数
        exact_gain_calls = module.exact_gain_calls

        if reference is None:
            reference = result
        matches = result == reference
        failed |= result.status != common.COMPLETE or not matches

        record = dict(
            size=args.size,
            obstacle_probability=args.density,
            actual_density=actual_density,
            seed=args.seed,
            start_row=start[0],
            start_col=start[1],
            sensor_range=4.0,
            algorithm=name,
            map_seconds=generation_seconds,
            exploration_seconds=elapsed,
            total_cost=result.total_cost,
            rounds=result.rounds,
            status=result.status,
            remaining_unknown=result.remaining_unknown,
            matches_first_algorithm=matches,
            exact_gain_calls=exact_gain_calls,
        )

        needs_header = not output.exists() or output.stat().st_size == 0
        with output.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(record)

        print(
            f"{name}: {result.status}, time={elapsed:.3f}s, "
            f"cost={result.total_cost}, rounds={result.rounds}, "
            f"unknown={result.remaining_unknown}, "
            f"exact_gain_calls={exact_gain_calls}",
            flush=True,
        )

        if not matches:
            print(
                "WARNING: exploration result differs from the first algorithm.",
                flush=True,
            )

    print(f"Saved: {output}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
