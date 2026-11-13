"""NBV 공통 규칙: 4방향 이동, 가시성, 점수, 실제 경로와 종료 출력.

센서 모델은 기존 코드와 같다. 계획 ray는 UNKNOWN을 통과하며,
실제 관측은 광선의 첫 벽까지 공개한다. 초기 위치와 목표 도착 시에만 관측한다.
"""
from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite
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


def free_component(grid, start):
    """상하좌우 플러드필로 start가 속한 FREE 영역을 반환한다."""
    if grid[start[0]][start[1]] != FREE:
        raise ValueError("시작 위치는 FREE여야 합니다.")
    cells = [start]
    visited = {start}
    queue = deque([start])
    while queue:
        for neighbor in free_neighbors(grid, queue.popleft()):
            if neighbor not in visited:
                visited.add(neighbor)
                cells.append(neighbor)
                queue.append(neighbor)
    return cells


def generate_map(height, width, obstacle_rate, seed, max_attempts):
    """최대 FREE 영역이 전체 FREE의 2/3 이상인 지도를 채택한다.

    셀별 장애물 생성 확률은 obstacle_rate이다. 조건 미달 지도는 재생성하며,
    채택한 지도의 가장 큰 FREE 영역에서 시작 위치를 무작위로 고른다.
    WALL의 두께나 모든 FREE의 연결 여부는 별도로 제한하지 않는다.
    """
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
        visited = set()
        largest = []
        for cell in free_cells:
            if cell in visited:
                continue
            component = free_component(truth, cell)
            visited.update(component)
            if len(component) > len(largest):
                largest = component
        # 정수 비교로 경계값(정확히 2/3)도 포함한다.
        if 3 * len(largest) < 2 * len(free_cells):
            continue
        robot = rng.choice(largest)
        return truth, [[UNKNOWN] * width for _ in range(height)], robot
    raise RuntimeError(
        f"{max_attempts}회 안에 최대 FREE 영역이 전체 FREE의 2/3 이상인 "
        "지도를 생성하지 못했습니다.")


def observation_cells(truth, robot, sensor_range):
    """실제 센서가 공개하는 셀을 반환한다. 첫 WALL 포함, 중복 가능."""
    for target in sensor_targets(robot, len(truth), len(truth[0]), sensor_range):
        for r, c in ray_cells(robot, target):
            yield r, c
            if truth[r][c] == WALL:
                break


@dataclass(frozen=True)
class ExplorationContext:
    """시뮬레이터의 종료/집계용 정답 정보. 후보 평가에는 전달하지 않는다."""
    start: tuple
    shape: tuple
    sensor_range: float
    target_free_mask: int
    observable_mask: int


def prepare_exploration(truth, start, sensor_range):
    """시작 FREE 영역과 그 영역 어디서든 관측 가능한 셀의 합집합을 구한다.

    실제 observe와 같은 광선/첫 벽 규칙을 사용한다. known을 만들거나 바꾸지
    않으며, run.py는 같은 지도·센서의 A/B/C에 이 결과를 공유한다.
    """
    if not isfinite(sensor_range) or sensor_range < 1:
        raise ValueError("시야 반경은 1 이상의 유한한 수여야 합니다.")
    height, width = len(truth), len(truth[0])
    component = free_component(truth, start)
    target_bits = bytearray((height * width + 7) // 8)
    observable_bits = bytearray(len(target_bits))
    for r, c in component:
        index = r * width + c
        target_bits[index // 8] |= 1 << (index % 8)
        for nr, nc in observation_cells(truth, (r, c), sensor_range):
            index = nr * width + nc
            observable_bits[index // 8] |= 1 << (index % 8)
    return ExplorationContext(
        start, (height, width), sensor_range,
        int.from_bytes(target_bits, "little"),
        int.from_bytes(observable_bits, "little"))


def build_unknown_mask(known):
    """시작 시 UNKNOWN을 한 번 읽어 row-major 마스크로 만든다."""
    width = len(known[0])
    bits = bytearray((len(known) * width + 7) // 8)
    for r, row in enumerate(known):
        for c, value in enumerate(row):
            if value == UNKNOWN:
                index = r * width + c
                bits[index // 8] |= 1 << (index % 8)
    return int.from_bytes(bits, "little")


def observe(truth, known, robot, sensor_range):
    """첫 실제 벽까지 공개하고 새로 알려진 셀의 row-major 비트 마스크를 반환한다.

    UNKNOWN에서 FREE/WALL로 바뀐 셀만 포함한다. 첫 벽 자체도 관측하며,
    여러 광선이 같은 셀을 지나도 한 번만 기록한다. 변경이 없으면 0이다.
    """
    width = len(known[0])
    newly_known_mask = 0
    for r, c in observation_cells(truth, robot, sensor_range):
        if known[r][c] == UNKNOWN and truth[r][c] != UNKNOWN:
            newly_known_mask |= 1 << (r * width + c)
        known[r][c] = truth[r][c]
    return newly_known_mask


def unknown_count(known):
    return sum(row.count(UNKNOWN) for row in known)


def reconstruct_path(parents, robot, target):
    path = [target]
    while path[-1] != robot:
        path.append(parents[path[-1]])
    return tuple(reversed(path))


def move_to_nbv(truth, known, robot, target, parents, total_cost, sensor_range,
                *, on_observation=None):
    """전체 경로를 이동한 뒤 목표에서 한 번 관측한다. 이동 중에는 종료하지 않는다.

    on_observation이 있으면 관측 후 변경 마스크를 전달한다.
    반환값은 기존의 (끝점, 누적 비용, 계획 경로, 실제 경로)를 유지한다.
    """
    planned_path = reconstruct_path(parents, robot, target)
    actual_path = [robot]
    for robot in planned_path[1:]:
        total_cost += STEP_COST
        actual_path.append(robot)
    newly_known_mask = observe(truth, known, robot, sensor_range)
    if on_observation is not None:
        on_observation(newly_known_mask)
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
    """remaining_unknown과 impossible_unknown의 합이 전체 미관측 셀 수이다.

    remaining: 시작 FREE 영역에서 관측 가능하지만 미관측인 셀.
    impossible: 시작 FREE 영역 어디에서도 관측할 수 없는 미관측 셀.
    """
    start: tuple
    end: tuple
    total_cost: int
    status: str
    remaining_unknown: int
    impossible_unknown: int
    actual_path: tuple
    decisions: tuple

    @property
    def rounds(self):
        return len(self.decisions)

    @property
    def targets(self):
        return tuple(record.target for record in self.decisions)


def run_exploration(truth, known, robot, sensor_range, select_candidate,
                    *, on_observation=None, context=None):
    """공통 탐사 진행. 선택 함수는 (target, exact_evaluations, parents)를 반환한다.

    각 알고리즘은 선택 함수만 제공한다. 관측/이동/경로 기록은 여기서 공유한다.
    관측 가능한 후보가 없는 종료도 결과로 반환하여 콘솔에서 구분해 출력한다.
    초기 위치와 각 목표 도착 시에만 관측하고, 시작 FREE 영역의 모든 FREE가
    관측되면 완료한다. 전체 지도 UNKNOWN이 0일 필요는 없다.
    on_observation(mask)는 지도 갱신 후, 다음 후보 선택 전에 호출한다.
    초기 관측과 마지막 관측도 전달하며, 새로 알려진 셀이 없으면 0을 전달한다.
    """
    if context is None:
        context = prepare_exploration(truth, robot, sensor_range)
    if (context.start != robot or context.shape != (len(truth), len(truth[0]))
            or context.sensor_range != sensor_range):
        raise ValueError("종료/집계용 context의 시작 위치·크기·센서가 다릅니다.")
    # 처음에만 지도를 읽는다. 이후 종료 판정과 결과 집계는 관측 증분을 쓴다.
    unknown_mask = build_unknown_mask(known)

    def record_observation(newly_known_mask):
        nonlocal unknown_mask
        unknown_mask &= ~newly_known_mask
        if on_observation is not None:
            on_observation(newly_known_mask)

    start = robot
    total_cost = 0
    actual_path = [robot]
    decisions = []
    status = COMPLETE
    newly_known_mask = observe(truth, known, robot, sensor_range)
    record_observation(newly_known_mask)
    while unknown_mask & context.target_free_mask:
        target, exact, parents = select_candidate(known, robot)
        if target is None:
            status = NO_OBSERVABLE_CANDIDATE
            break
        score, gain, distance = exact[target]
        robot, total_cost, planned, moved = move_to_nbv(
            truth, known, robot, target, parents, total_cost, sensor_range,
            on_observation=record_observation)
        decisions.append(PlanRecord(target, score, gain, distance, planned, moved))
        actual_path.extend(moved[1:])
    remaining = (unknown_mask & context.observable_mask).bit_count()
    impossible = (unknown_mask & ~context.observable_mask).bit_count()
    return ExplorationResult(
        start=start, end=robot, total_cost=total_cost, status=status,
        remaining_unknown=remaining, impossible_unknown=impossible,
        actual_path=tuple(actual_path), decisions=tuple(decisions))


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
    print(f"termination_status={result.status}, "
          f"remaining_unknown={result.remaining_unknown}, "
          f"impossible_unknown={result.impossible_unknown}")
    if result.status == COMPLETE:
        print("종료 사유: 시작 FREE 영역의 모든 FREE 셀을 관측했습니다.")
    else:
        print("종료 사유: 미확인 셀이 남았지만 양의 정보 이득을 가진 도달 가능한 후보가 없습니다.")
    symbols = {UNKNOWN: "?", FREE: ".", WALL: "#"}
    for r, row in enumerate(known):
        print("".join("A" if (r, c) == result.end else symbols[value]
                      for c, value in enumerate(row)))
