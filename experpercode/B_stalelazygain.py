"""Stale-Gain Lazy NBV. 실행: python B_stalelazygain.py

같은 폴더의 nbv_common.py가 필요하다.
기존 B의 scalar 캐시와 선형 상한 검색을 유지한다.
이동/가시성/점수/관측/출력은 A 및 C*와 같은 공통 함수를 사용한다.
"""
import nbv_common as common
from nbv_common import (
    UNKNOWN, FREE, WALL, MOVES, STEP_COST, SCORE_EPSILON,
    free_neighbors, shortest_paths, ray_cells,
    nbv_score, candidate_rank, choose_best_candidate,
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


def Information_Gain(x, y, grid=None):
    """A와 같은 정확한 optimistic gain. 계획에서는 grid를 명시한다."""
    if grid is None:
        grid = known
    if grid is None:
        raise RuntimeError("known 지도가 아직 생성되지 않았습니다.")
    return common.information_gain(grid, (x, y), SENSOR_RANGE)


def exact_evaluate(known, candidate, distance, cached_gains):
    """현재 gain을 정확히 계산하고 마지막 정확 gain 숫자만 저장한다."""
    global exact_gain_calls
    exact_gain_calls += 1

    gain = Information_Gain(*candidate, grid=known)
    old_gain = cached_gains.get(candidate)
    if old_gain is not None and gain > old_gain:
        raise RuntimeError(
            f"후보 {candidate}의 현재 이득 {gain}이 캐시 상한 {old_gain}보다 큽니다. "
            "환경·센서 변경 또는 다른 탐사의 캐시 재사용 여부를 확인하세요.")
    cached_gains[candidate] = gain
    return nbv_score(gain, distance), gain, distance


def evaluate_candidates(known, robot, cached_gains):
    """(목표 또는 None, 이번 정확 평가 사전, BFS 부모 사전)을 반환한다.

    캐시는 같은 정적 세계·센서에서 단조롭게 공개되는 지도에만 재사용한다.
    과거 점수/거리를 재사용하지 않고 현재 거리로 상한 점수를 계산한다.
    """
    distances, parents = shortest_paths(known, robot)
    candidates = tuple(sorted(v for v in distances if v != robot))
    exact = {}
    if not candidates:
        return None, exact, parents

    # 신규 후보는 과거 gain 상한이 없으므로 모두 정확히 평가한다.
    for candidate in candidates:
        if candidate not in cached_gains:
            exact[candidate] = exact_evaluate(
                known, candidate, distances[candidate], cached_gains)
    remaining = set(candidates) - set(exact)

    def upper_rank(candidate):
        upper_gain = cached_gains[candidate]
        distance = distances[candidate]
        return candidate_rank(
            candidate,
            (nbv_score(upper_gain, distance), upper_gain, distance))

    while True:
        # 상한이 모두 0이면 현재 양의 gain을 가진 후보도 없다.
        if all(cached_gains[v] == 0 for v in candidates):
            return None, exact, parents

        best_exact = choose_best_candidate(exact)
        if not remaining:
            return best_exact, exact, parents

        # 기존 B와 같은 선형 검색. 전체 동점 규칙까지 비교한다.
        next_candidate = min(remaining, key=upper_rank)
        if best_exact is not None:
            if candidate_rank(best_exact, exact[best_exact]) < upper_rank(next_candidate):
                return best_exact, exact, parents

        remaining.remove(next_candidate)
        exact[next_candidate] = exact_evaluate(
            known, next_candidate, distances[next_candidate], cached_gains)


def observe(truth, known, robot):
    common.observe(truth, known, robot, SENSOR_RANGE)


def move_to_nbv(truth, known, robot, target, parents, total_cost):
    """(실제 끝점, 누적 비용, 계획 경로, 실제 경로)를 반환한다."""
    return common.move_to_nbv(
        truth, known, robot, target, parents, total_cost, SENSOR_RANGE)


def explore(truth, known, robot):
    """탐사마다 캐시를 초기화하고 A와 같은 경로·종료 결과를 반환한다."""
    global exact_gain_calls
    exact_gain_calls = 0

    NBVpath.clear()
    cached_gains = {}

    def select_candidate(grid, position):
        return evaluate_candidates(grid, position, cached_gains)

    result = common.run_exploration(
        truth, known, robot, SENSOR_RANGE, select_candidate)
    NBVpath.extend(result.targets)
    return result


if __name__ == "__main__":
    truth, known, start = create_map()
    result = explore(truth, known, start)
    print_result(result, known)
    print(f"정확한 정보 이득 계산 횟수: {exact_gain_calls}")
