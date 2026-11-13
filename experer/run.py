"""같은 지도에서 NBV 알고리즘을 실행하고 결과를 CSV에 누적 저장한다.

예: python experrewp/run.py --size 20 --obstacle_probability 0.2 \
        --seed 42 --sensor_range 4 --algorithm ALL
관측은 초기 위치와 목표 도착 시 수행한다. 자세한 사용법은 저장소 README.md 참고.
"""
import argparse
import csv
import importlib
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nbv_common as common


MODULES = {"A": "A_wjstnNBV", "B": "B_stalelazygain", "C": "C_cstarNBV"}
MAX_MAP_ATTEMPTS = 1000
RESULT_PATH = Path(__file__).resolve().parent / "result.csv"
FAILURE_PATH = RESULT_PATH.with_suffix(".failures.csv")
CSV_FIELDS = [
    "size", "obstacle_probability", "actual_density", "seed",
    "start_row", "start_col", "sensor_range", "algorithm", "map_seconds",
    "exploration_seconds", "total_cost", "moved_cells", "rounds", "remaining_unknown",
    "impossible_unknown", "matches_first_algorithm", "exact_gain_calls",
]
FAILURE_FIELDS = [
    "timestamp_utc", "stage", "size", "obstacle_probability", "seed",
    "sensor_range", "algorithm", "actual_density", "start_row", "start_col",
    "map_seconds", "exploration_seconds", "error",
]


def prepare_csv(path, fields):
    """기존 데이터의 열 이름과 순서가 현재 저장 형식에 맞는지 확인한다."""
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), None)
        if header != fields:
            raise ValueError(
                f"CSV 열 구성이 다릅니다: {path}\n"
                "기존 파일을 다른 이름으로 보관한 뒤 다시 실행하세요."
            )


def append_csv(path, fields, record):
    """실행 결과마다 파일을 닫아 앞서 저장한 행을 유지한다."""
    needs_header = not path.exists() or path.stat().st_size == 0
    encoding = "utf-8-sig" if needs_header else "utf-8"
    with path.open("a", newline="", encoding=encoding) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(record)


def parse_args():
    parser = argparse.ArgumentParser(
        description="같은 지도에서 A/B/Cstar를 실행하고 result.csv에 저장합니다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--size", type=int, required=True,
                        help="정사각형 지도의 한 변 길이 (1 이상)")
    parser.add_argument("--obstacle_probability", "--obstacle-probability",
                        type=float, required=True,
                        help="장애물 생성 확률 (0 이상, 1 미만)")
    parser.add_argument("--seed", type=int, default=42, help="지도 생성 시드")
    parser.add_argument("--sensor_range", "--sensor-range", type=float,
                        default=4.0, help="센서 반경 (1 이상)")
    parser.add_argument("--algorithm", type=str.upper,
                        choices=["ALL", "A", "B", "C", "CSTAR"], default="ALL",
                        help="실행 알고리즘. C와 CSTAR는 같은 알고리즘")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("size는 1 이상이어야 합니다.")
    if not math.isfinite(args.obstacle_probability) or not 0 <= args.obstacle_probability < 1:
        parser.error("obstacle_probability는 0 이상 1 미만의 유한한 수여야 합니다.")
    if not math.isfinite(args.sensor_range) or args.sensor_range < 1:
        parser.error("sensor_range는 1 이상의 유한한 수여야 합니다.")
    if args.algorithm == "CSTAR":
        args.algorithm = "C"
    return args


def main():
    args = parse_args()
    # 실행 전에 두 파일의 형식을 확인한다. 기존 파일은 덮어쓰지 않는다.
    try:
        prepare_csv(RESULT_PATH, CSV_FIELDS)
        prepare_csv(FAILURE_PATH, FAILURE_FIELDS)
    except (OSError, ValueError, csv.Error) as exc:
        print(f"CSV 준비 실패: {exc}", flush=True)
        return 1

    base_record = dict(
        size=args.size, obstacle_probability=args.obstacle_probability,
        seed=args.seed, sensor_range=args.sensor_range,
    )

    def save_failure(stage, error, algorithm, **details):
        record = dict(
            base_record, timestamp_utc=datetime.now(timezone.utc).isoformat(),
            stage=stage, algorithm=algorithm, error=str(error), **details,
        )
        append_csv(FAILURE_PATH, FAILURE_FIELDS, record)
        print(f"실패 기록 저장: {FAILURE_PATH}\n{error}", flush=True)

    print(
        f"지도={args.size}x{args.size}, 장애물 확률={args.obstacle_probability}, "
        f"시드={args.seed}, 센서 반경={args.sensor_range}", flush=True,
    )
    print("지도 생성 중...", flush=True)
    started = perf_counter()
    try:
        truth, initial_known, start = common.generate_map(
            args.size, args.size, args.obstacle_probability,
            args.seed, MAX_MAP_ATTEMPTS,
        )
    except Exception as exc:
        save_failure("MAP_GENERATION_FAILED", exc, args.algorithm,
                     map_seconds=perf_counter() - started)
        return 1

    map_seconds = perf_counter() - started
    actual_density = sum(row.count(common.WALL) for row in truth) / args.size**2
    map_record = dict(actual_density=actual_density, start_row=start[0],
                      start_col=start[1], map_seconds=map_seconds)
    print(
        f"실제 장애물 비율={actual_density:.6f}, 시작 좌표={start}, "
        f"지도 생성 시간={map_seconds:.6f}초", flush=True,
    )

    # 정답 지도를 이용한 집계 준비는 한 번만 하며, 알고리즘 탐사 시간에서 제외한다.
    print("시작 FREE 영역과 실제 센서의 관측 가능 영역 계산 중...", flush=True)
    started = perf_counter()
    try:
        context = common.prepare_exploration(truth, start, args.sensor_range)
    except Exception as exc:
        save_failure("OBSERVABILITY_PREPARATION_FAILED", exc, args.algorithm,
                     **map_record)
        return 1
    preparation_seconds = perf_counter() - started
    print(
        f"시작 FREE 영역={context.target_free_mask.bit_count()}칸, "
        f"관측 가능 영역={context.observable_mask.bit_count()}칸, "
        f"집계 준비 시간={preparation_seconds:.6f}초 (탐사 시간에서 제외)", flush=True,
    )

    algorithms = list(MODULES) if args.algorithm == "ALL" else [args.algorithm]
    first_algorithm = algorithms[0]
    reference = None
    failed = False
    for name in algorithms:
        print(f"{name} 실행 중...", flush=True)
        started = None
        try:
            module = importlib.import_module(MODULES[name])
            module.HEIGHT = module.WIDTH = args.size
            module.OBSTACLE_RATE = args.obstacle_probability
            module.SEED = args.seed
            module.SENSOR_RANGE = args.sensor_range
            # 알고리즘별 초기 지도만 복사한다. 관측마다 복사하지 않는다.
            module.truth = [row[:] for row in truth]
            module.known = [row[:] for row in initial_known]
            started = perf_counter()
            result = module.explore(module.truth, module.known, start, context=context)
            elapsed = perf_counter() - started
            exact_gain_calls = module.exact_gain_calls
        except Exception as exc:
            elapsed = "" if started is None else perf_counter() - started
            save_failure("ALGORITHM_ERROR", exc, name,
                         **map_record, exploration_seconds=elapsed)
            failed = True
            # 다른 알고리즘의 결과도 수집한다.
            continue

        if name == first_algorithm:
            reference = result
        # 첫 알고리즘이 예외로 실패했다면 비교할 기준이 없으므로 빈칸이다.
        matches = result == reference if reference is not None else ""
        # 실제 경로에는 시작점이 한 번 포함된다. 재방문도 각각 이동으로 센다.
        moved_cells = len(result.actual_path) - 1
        record = dict(
            base_record, **map_record, algorithm=name,
            exploration_seconds=elapsed, total_cost=result.total_cost,
            moved_cells=moved_cells, rounds=result.rounds,
            remaining_unknown=result.remaining_unknown,
            impossible_unknown=result.impossible_unknown,
            matches_first_algorithm=matches, exact_gain_calls=exact_gain_calls,
        )
        append_csv(RESULT_PATH, CSV_FIELDS, record)
        print(
            f"{name}: 탐사 시간={elapsed:.6f}초, 이동 비용={result.total_cost}, "
            f"실제 이동 칸 수={moved_cells}, "
            f"목표 이동 횟수={result.rounds}, "
            f"관측 가능 미관측={result.remaining_unknown}, "
            f"관측 불가능 미관측={result.impossible_unknown}, "
            f"결과 일치={matches if matches != '' else '비교 불가'}, "
            f"정확 평가 횟수={exact_gain_calls}", flush=True,
        )

        problems = []
        if result.status != common.COMPLETE:
            problems.append("시작 FREE 영역의 FREE 관측이 완료되지 않았습니다.")
        if matches is False:
            problems.append(f"첫 알고리즘 {first_algorithm}과 탐사 결과가 다릅니다.")
        if problems:
            failed = True
            save_failure("RESULT_CHECK_FAILED", " ".join(problems), name,
                         **map_record, exploration_seconds=elapsed)

    if reference is not None or not failed:
        print(f"결과 저장 위치: {RESULT_PATH}", flush=True)
    else:
        print(f"결과/실패 기록 위치: {RESULT_PATH.parent}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
