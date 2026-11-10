"""NBV 공통 규칙: 4방향 이동, 가시성, 점수, 실제 경로와 종료 출력.

센서 모델은 기존 코드와 같다. 계획 ray는 UNKNOWN을 통과하며,
실제 관측은 광선의 첫 벽까지 공개한다. 이동 중 매 칸에서 관측한다.
"""
from collections import deque
from dataclasses import dataclass
from math import ceil
from random import Random

UNKNOWN, FREE, WALL = -1, 0, 1
MOVES = ((-1, 0), (0, -1), (0, 1), (1, 0))
STEP_COST = 1
SCORE_EPSILON = 1e-9
COMPLETE = "COMPLETE"
NO_OBSERVABLE_CANDIDATE = "NO_OBSERVABLE_CANDIDATE"


def free_neighbors(grid, position):
    """현재 지도에서 상하좌우 known-FREE 이웃만 반환한다."""
    height, width = len(grid), len(grid[0])
    r, c = position
    for dr, dc in MOVES:
        nr, nc = r + dr, c + dc
        if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] == FREE:
            yield nr, nc


def shortest_paths(known, robot):
    """단위 비용 BFS. 이웃 순서까지 고정하여 동일 비용 경로를 결정한다."""
    distances, parents = {robot: 0}, {}
    queue = deque([robot])
    while queue:
        current = queue.popleft()
        for neighbor in free_neighbors(known, current):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + STEP_COST
                parents[neighbor] = current
                queue.append(neighbor)
    return distances, parents


def ray_cells(start, end):
    """모서리 양옆 셀을 포함하는 기존 corner-inclusive supercover."""
    r, c = start
    dr, dc = abs(end[0] - r), abs(end[1] - c)
    sr = (end[0] > r) - (end[0] < r)
    sc = (end[1] > c) - (end[1] < c)
    row_crossing = col_crossing = 1
    cells = [start]
    while (r, c) != end:
        row_event = row_crossing * dc
        col_event = col_crossing * dr
        if row_event == col_event:
            cells.extend(sorted(((r + sr, c), (r, c + sc))))
        if row_event <= col_event:
            r += sr
            row_crossing += 2
        if col_event <= row_event:
            c += sc
            col_crossing += 2
        cells.append((r, c))
    return cells


def sensor_targets(origin, height, width, sensor_range):
    """셀 중심의 유클리드 거리가 센서 반경 이하인 셀을 반환한다."""
    r, c = origin
    radius = ceil(sensor_range)
    for nr in range(max(0, r - radius), min(height, r + radius + 1)):
        for nc in range(max(0, c - radius), min(width, c + radius + 1)):
            if (nr - r) ** 2 + (nc - c) ** 2 <= sensor_range ** 2:
                yield nr, nc


def visible_unknown_cells(known, candidate, sensor_range):
    """정확한 optimistic 가시 UNKNOWN 셀. truth는 읽지 않는다.

    전수/B는 원소 수를 gain으로, C*는 같은 셀들을 비트셋으로 사용할 수 있다.
    """
    for target in sensor_targets(candidate, len(known), len(known[0]), sensor_range):
        if known[target[0]][target[1]] != UNKNOWN:
            continue
        if all(known[r][c] != WALL for r, c in ray_cells(candidate, target)[1:-1]):
            yield target


def information_gain(known, candidate, sensor_range):
    return sum(1 for _ in visible_unknown_cells(known, candidate, sensor_range))


def nbv_score(gain, distance):
    """B/C와 동일한 점수. 큰 값이 좋다."""
    return gain / (distance + SCORE_EPSILON)


def candidate_rank(candidate, evaluation):
    """작은 키가 우선: 큰 score, 큰 gain, 짧은 거리, 작은 행, 작은 열."""
    score, gain, distance = evaluation
    return (-score, -gain, distance, candidate[0], candidate[1])


def choose_best_candidate(evaluations):
    if not evaluations:
        return None
    return min(evaluations, key=lambda v: candidate_rank(v, evaluations[v]))


def generate_map(height, width, obstacle_rate, seed, max_attempts):
    """기존과 같은 난수 생성 순서 및 지도 수용 조건을 사용한다."""
    if height < 1 or width < 1 or not 0 <= obstacle_rate < 1:
        raise ValueError("맵 크기는 양수, 장애물 확률은 0 이상 1 미만이어야 합니다.")
    rng = Random(seed)
    for _ in range(max_attempts):
        truth = [[int(rng.random() < obstacle_rate) for _ in range(width)]
                 for _ in range(height)]
        free_cells = [(r, c) for r in range(height) for c in range(width)
                      if truth[r][c] == FREE]
        if not free_cells:
            continue
        robot = rng.choice(free_cells)
        distances, _ = shortest_paths(truth, robot)
        if len(distances) != len(free_cells):
            continue
        if any(truth[r][c] == WALL and not any(free_neighbors(truth, (r, c)))
               for r in range(height) for c in range(width)):
            continue
        return truth, [[UNKNOWN] * width for _ in range(height)], robot
    raise RuntimeError("전체를 관측할 수 있는 맵 생성 실패: 장애물 확률을 낮춰주세요.")


def observe(truth, known, robot, sensor_range):
    """각 광선의 첫 실제 벽까지 공개한다. 첫 벽 자체도 관측한다."""
    for target in sensor_targets(robot, len(known), len(known[0]), sensor_range):
        for r, c in ray_cells(robot, target):
            known[r][c] = truth[r][c]
            if truth[r][c] == WALL:
                break


def unknown_count(known):
    return sum(row.count(UNKNOWN) for row in known)


def reconstruct_path(parents, robot, target):
    path = [target]
    while path[-1] != robot:
        path.append(parents[path[-1]])
    return tuple(reversed(path))


def move_to_nbv(truth, known, robot, target, parents, total_cost, sensor_range):
    """실제로 밟은 셀만 기록한다. 전체 관측 완료 시 기존처럼 중간 종료한다."""
    planned_path = reconstruct_path(parents, robot, target)
    actual_path = [robot]
    for robot in planned_path[1:]:
        total_cost += STEP_COST
        actual_path.append(robot)
        observe(truth, known, robot, sensor_range)
        if unknown_count(known) == 0:
            break
    return robot, total_cost, planned_path, tuple(actual_path)


@dataclass(frozen=True)
class PlanRecord:
    target: tuple
    score: float
    gain: int
    distance: int
    planned_path: tuple
    actual_path: tuple

    @property
    def target_reached(self):
        return self.actual_path[-1] == self.target


@dataclass(frozen=True)
class ExplorationResult:
    start: tuple
    end: tuple
    total_cost: int
    status: str
    remaining_unknown: int
    actual_path: tuple
    decisions: tuple

    @property
    def rounds(self):
        return len(self.decisions)

    @property
    def targets(self):
        return tuple(record.target for record in self.decisions)


def run_exploration(truth, known, robot, sensor_range, select_candidate):
    """공통 탐사 진행. 선택 함수는 (target, exact_evaluations, parents)를 반환한다.

    각 알고리즘은 선택 함수만 제공한다. 관측/이동/경로 기록은 여기서 공유한다.
    관측 가능한 후보가 없는 종료도 결과로 반환하여 콘솔에서 구분해 출력한다.
    """
    if sensor_range < 1:
        raise ValueError("전체 지도 관측을 위해 시야 반경은 1 이상이어야 합니다.")
    start = robot
    total_cost = 0
    actual_path = [robot]
    decisions = []
    status = COMPLETE
    observe(truth, known, robot, sensor_range)
    while unknown_count(known):
        target, exact, parents = select_candidate(known, robot)
        if target is None:
            status = NO_OBSERVABLE_CANDIDATE
            break
        score, gain, distance = exact[target]
        robot, total_cost, planned, moved = move_to_nbv(
            truth, known, robot, target, parents, total_cost, sensor_range)
        decisions.append(PlanRecord(target, score, gain, distance, planned, moved))
        actual_path.extend(moved[1:])
    return ExplorationResult(start, robot, total_cost, status, unknown_count(known),
                             tuple(actual_path), tuple(decisions))


def print_result(result, known):
    """선택 기준, 계획/실제 경로, 종료 상태를 콘솔에 실제로 출력한다."""
    for number, record in enumerate(result.decisions, 1):
        print(f"plan={number}, target={record.target}, gain={record.gain}, "
              f"distance={record.distance}, score={record.score:.17g}, "
              f"target_reached={record.target_reached}")
        print(f"  planned_path={list(record.planned_path)}")
        print(f"  actual_path={list(record.actual_path)}")
    print(f"start={result.start}, end={result.end}, "
          f"NBV selections={result.rounds}, total cost={result.total_cost}")
    print(f"NBV_targets={list(result.targets)}")
    print(f"actual_path={list(result.actual_path)}")
    print(f"termination_status={result.status}, remaining_unknown={result.remaining_unknown}")
    if result.status == COMPLETE:
        print("종료 사유: 모든 셀의 관측을 완료했습니다.")
    else:
        print("종료 사유: 미확인 셀이 남았지만 양의 정보 이득을 가진 도달 가능한 후보가 없습니다.")
    symbols = {UNKNOWN: "?", FREE: ".", WALL: "#"}
    for r, row in enumerate(known):
        print("".join("A" if (r, c) == result.end else symbols[value]
                      for c, value in enumerate(row)))
