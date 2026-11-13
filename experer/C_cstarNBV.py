"""비트셋 상한과 힙을 이용하는 C* NBV. 실행: python C_cstarNBV.py

같은 폴더의 nbv_common.py가 필요하다.
기존 C*의 range 상한, 가시 집합 캐시, 지연 교집합 갱신을 유지한다.
이동/가시성/점수/관측/출력은 A 및 B와 같은 공통 함수를 사용한다.
UNKNOWN 마스크는 탐사 시작 시 한 번 만들고, 초기·목표 도착 관측의 변경 마스크로 갱신한다.
"""
from heapq import heapify, heappop, heappush
import nbv_common as common
from nbv_common import (
    UNKNOWN, FREE, WALL, MOVES, STEP_COST, SCORE_EPSILON,
    free_neighbors, shortest_paths, ray_cells,
    nbv_score, candidate_rank,
    ExplorationResult, print_result,
)

HEIGHT = 20
WIDTH = 20
OBSTACLE_RATE = 0.20
SENSOR_RANGE = 4.0
SEED = 42
MAX_MAP_ATTEMPTS = 1000

# 이번 탐사에서 정확한 정보 이득을 계산한 횟수
exact_gain_calls = 0

truth = None
known = None
NBVpath = []


def create_map():
    global truth, known
    NBVpath.clear()
    truth, known, robot = common.generate_map(
        HEIGHT, WIDTH, OBSTACLE_RATE, SEED, MAX_MAP_ATTEMPTS)
    return truth, known, robot


def sensor_targets(origin):
    return common.sensor_targets(origin, HEIGHT, WIDTH, SENSOR_RANGE)


def cell_bit(coord, width=None):
    """row-major 비트 번호. 내부 계산에는 실제 입력 지도의 너비를 전달한다."""
    if width is None:
        width = WIDTH
    return 1 << (coord[0] * width + coord[1])


def build_unknown_mask(known):
    """탐사 시작 시 입력 지도의 UNKNOWN 마스크를 한 번 구성한다."""
    width = len(known[0])
    mask = 0
    for row, values in enumerate(known):
        for col, value in enumerate(values):
            if value == UNKNOWN:
                mask |= cell_bit((row, col), width)
    return mask


def sensor_range_mask(candidate, range_masks, shape=None):
    """고정된 센서의 범위를 처음 필요할 때만 비트셋으로 저장한다.

    같은 cache/range_masks를 다른 지도 크기나 센서 반경에 재사용하지 않는다.
    """
    height, width = shape if shape is not None else (HEIGHT, WIDTH)
    if candidate not in range_masks:
        mask = 0
        for target in common.sensor_targets(candidate, height, width, SENSOR_RANGE):
            mask |= cell_bit(target, width)
        range_masks[candidate] = mask
    return range_masks[candidate]


def visible_unknown_mask(known, candidate):
    """A/B와 같은 가시 셀들을 비트셋으로 표현한다."""
    width = len(known[0])
    mask = 0
    for target in common.visible_unknown_cells(known, candidate, SENSOR_RANGE):
        mask |= cell_bit(target, width)
    return mask


def exact_evaluate(known, candidate, distance, cache, upper_gain):
    """현재 정확 가시 집합과 gain으로 캐시를 교체한다."""
    global exact_gain_calls
    exact_gain_calls += 1

    mask = visible_unknown_mask(known, candidate)
    gain = mask.bit_count()
    if gain > upper_gain:
        raise RuntimeError(
            f"후보 {candidate}의 현재 이득 {gain}이 상한 {upper_gain}보다 큽니다. "
            "환경·센서 변경 또는 다른 탐사의 캐시 재사용 여부를 확인하세요.")
    cache[candidate] = {"visible_mask": mask, "upper_gain": gain}
    return nbv_score(gain, distance), gain, distance


def evaluate_candidates(known, robot, cache, range_masks, unknown_mask):
    """(목표 또는 None, 이번 정확 평가 사전, BFS 부모 사전)을 반환한다.

    정적 환경·단조 공개·고정 센서가 전제이다. 기존 가시 집합을 현재 UNKNOWN과
    교차하는 작업은 후보가 힙에서 꺼내질 때만 수행한다.
    unknown_mask는 최신 관측까지 반영한 마스크를 전달받으며, 선택 중에는 고정한다.
    """
    distances, parents = shortest_paths(known, robot)
    candidates = tuple(sorted(v for v in distances if v != robot))
    exact = {}
    if not candidates:
        return None, exact, parents

    refreshed = set()
    range_bounds = {}
    queue = []
    shape = (len(known), len(known[0]))
    for candidate in candidates:
        range_mask = sensor_range_mask(candidate, range_masks, shape)
        range_upper = (range_mask & unknown_mask).bit_count()
        range_bounds[candidate] = range_upper
        entry = cache.get(candidate)
        upper_gain = range_upper if entry is None else min(entry["upper_gain"], range_upper)
        distance = distances[candidate]
        queue.append(candidate_rank(
            candidate,
            (nbv_score(upper_gain, distance), upper_gain, distance)))
    heapify(queue)
    best_candidate = best_key = None

    while queue:
        top_key = queue[0]
        top_upper_gain = -top_key[1]
        if top_upper_gain == 0 and (
            best_candidate is None or exact[best_candidate][1] == 0
        ):
            return None, exact, parents

        # 현재 exact 최선이 모든 미평가 후보의 상한을 이기는 경우만 확정한다.
        if best_key is not None and best_key < top_key:
            return best_candidate, exact, parents

        popped_key = heappop(queue)
        candidate = (popped_key[3], popped_key[4])
        distance = distances[candidate]
        entry = cache.get(candidate)
        if entry is not None and candidate not in refreshed:
            # visible_mask는 변경하지 않고 상한 숫자만 줄인다.
            change_upper = (entry["visible_mask"] & unknown_mask).bit_count()
            if change_upper > entry["upper_gain"] or change_upper > range_bounds[candidate]:
                raise RuntimeError(
                    f"후보 {candidate}의 상한이 증가했습니다. "
                    "지도·센서 조건과 캐시 재사용 범위를 확인하세요.")
            entry["upper_gain"] = change_upper
            refreshed.add(candidate)
            upper_gain = min(change_upper, range_bounds[candidate])
            heappush(queue, candidate_rank(
                candidate,
                (nbv_score(upper_gain, distance), upper_gain, distance)))
            continue

        evaluation = exact_evaluate(
            known, candidate, distance, cache, -popped_key[1])
        exact[candidate] = evaluation
        rank = candidate_rank(candidate, evaluation)
        if best_key is None or rank < best_key:
            best_candidate, best_key = candidate, rank
        # 정확 평가한 후보는 큐에 다시 넣지 않는다.

    if best_candidate is not None and exact[best_candidate][1] > 0:
        return best_candidate, exact, parents
    return None, exact, parents


def observe(truth, known, robot):
    common.observe(truth, known, robot, SENSOR_RANGE)


def move_to_nbv(truth, known, robot, target, parents, total_cost):
    """(실제 끝점, 누적 비용, 계획 경로, 실제 경로)를 반환한다."""
    return common.move_to_nbv(
        truth, known, robot, target, parents, total_cost, SENSOR_RANGE)


def explore(truth, known, robot, *, context=None):
    """탐사마다 두 캐시와 UNKNOWN 마스크를 초기화한다.

    초기 관측 전에 마스크를 구성하고, 이후 각 관측의 변경 비트만 제거한다.
    마스크는 이번 탐사의 지역 상태로 보관하며 다음 계획에 전달한다.
    """
    global exact_gain_calls
    exact_gain_calls = 0

    NBVpath.clear()
    cache, range_masks = {}, {}
    unknown_mask = build_unknown_mask(known)

    def on_observation(newly_known_mask):
        nonlocal unknown_mask
        if newly_known_mask:
            unknown_mask &= ~newly_known_mask

    def select_candidate(grid, position):
        return evaluate_candidates(grid, position, cache, range_masks, unknown_mask)

    result = common.run_exploration(
        truth, known, robot, SENSOR_RANGE, select_candidate,
        on_observation=on_observation, context=context)
    NBVpath.extend(result.targets)
    return result


if __name__ == "__main__":
    truth, known, start = create_map()
    result = explore(truth, known, start)
    print_result(result, known)
    print(f"정확한 정보 이득 계산 횟수: {exact_gain_calls}")
