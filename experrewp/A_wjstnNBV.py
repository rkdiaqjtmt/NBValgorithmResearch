"""상하좌우 전수 NBV. 실행: python A_wjstnNBV.py

같은 폴더의 nbv_common.py가 필요하다.
모든 후보를 정확히 평가하며, 점수식과 동점 규칙은 기존 B/C와 같다.
상하좌우 4방향으로 이동하며, 한 칸의 이동 비용은 1이고 센서 반경은 4이다.
초기 위치와 목표 도착 시에만 관측하며, 이동 중에는 관측하거나 중간 종료하지 않는다.
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
NBVpath = []  # 선택 목표 목록. 실제 이동 경로는 result.actual_path에 따로 보관한다.


def create_map():
    """기존과 같은 지도를 생성하고 전역 known 및 목표 목록을 초기화한다."""
    global truth, known
    NBVpath.clear()
    truth, known, robot = common.generate_map(
        HEIGHT, WIDTH, OBSTACLE_RATE, SEED, MAX_MAP_ATTEMPTS)
    return truth, known, robot


def sensor_targets(origin):
    """기존 호출 형태를 유지한 센서 범위 함수."""
    return common.sensor_targets(origin, HEIGHT, WIDTH, SENSOR_RANGE)


def Information_Gain(x, y, grid=None):
    """정확한 optimistic gain. 후보 평가에서는 grid를 명시적으로 전달한다."""
    if grid is None:
        grid = known
    if grid is None:
        raise RuntimeError("known 지도가 아직 생성되지 않았습니다.")
    return common.information_gain(grid, (x, y), SENSOR_RANGE)


def exact_evaluate(known, candidate, distance):
    """gain을 한 번 계산하여 (score, gain, distance)를 반환한다. 캐시는 없다."""
    global exact_gain_calls
    exact_gain_calls += 1

    gain = Information_Gain(*candidate, grid=known)
    return nbv_score(gain, distance), gain, distance


def evaluate_candidates(known, robot):
    """현재 위치를 제외한 모든 도달 가능한 FREE 후보를 빠짐없이 정확 평가한다.

    반환: (선택 목표 또는 None, 전체 정확 평가 사전, BFS 부모 사전).
    점수/gain이 모두 0이면 None이며, 전체 지도 관측 완료와는 구분한다.
    """
    distances, parents = shortest_paths(known, robot)
    exact = {
        candidate: exact_evaluate(known, candidate, distances[candidate])
        for candidate in sorted(distances) if candidate != robot
    }
    target = choose_best_candidate(exact)
    if target is not None and exact[target][1] == 0:
        target = None
    return target, exact, parents


def observe(truth, known, robot):
    """기존과 같은 first-hit 실제 관측."""
    common.observe(truth, known, robot, SENSOR_RANGE)


def move_to_nbv(truth, known, robot, target, parents, total_cost):
    """목표까지 전체 경로를 이동한 뒤 한 번 관측한다.

    반환: (실제 끝점, 누적 비용, 계획 경로, 실제 경로).
    """
    return common.move_to_nbv(
        truth, known, robot, target, parents, total_cost, SENSOR_RANGE)


def explore(truth, known, robot):
    """ExplorationResult를 반환한다. known을 갱신하고 NBVpath에 목표를 기록한다.

    초기 위치와 각 목표 도착 시에 관측하고, 관측 후 전체 지도 공개 여부를 판단한다.
    이동 중 관측과 목표 도착 전 종료는 하지 않는다.
    result.actual_path는 시작점과 실제 밟은 모든 셀을 순서대로 포함한다.
    result.status는 COMPLETE 또는 NO_OBSERVABLE_CANDIDATE이다.
    """
    global exact_gain_calls
    exact_gain_calls = 0

    NBVpath.clear()
    result = common.run_exploration(
        truth, known, robot, SENSOR_RANGE, evaluate_candidates)
    NBVpath.extend(result.targets)
    return result


if __name__ == "__main__":
    truth, known, start = create_map()
    result = explore(truth, known, start)
    print_result(result, known)
    print(f"정확한 정보 이득 계산 횟수: {exact_gain_calls}")
