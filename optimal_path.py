import heapq
from pedestrian_path_env import PedestrianPaths, env


def dijkstra_optimal_path(env: PedestrianPaths, start_state: str, goal_state: str):
    """
    Finds the optimal path from start_state to goal_state in the PedestrianPaths environment
    using Dijkstra's algorithm.

    Args:
        env: PedestrianPaths environment instance.
        start_state: Starting state (e.g., "S1").
        goal_state: Goal state (e.g., "S14").

    Returns:
        tuple: (optimal_path_cost, optimal_path_states)
               optimal_path_cost: Total cost of the optimal path (sum of edge weights).
               optimal_path_states: List of states representing the optimal path.
                      Returns (float('inf'), []) if no path exists.
    """
    distances = {state: float("inf") for state in env.possible_actions}  # Initialize distances to infinity
    distances[start_state] = 0  # Distance from start state to itself is 0
    priority_queue = [(0, start_state)]  # Priority queue to store (distance, state) pairs
    previous_states = {state: None for state in env.possible_actions}  # To reconstruct the path

    while priority_queue:
        current_distance, current_state = heapq.heappop(priority_queue)

        if current_distance > distances[current_state]:
            continue  # Already found a shorter path

        if current_state == goal_state:
            break  # Goal reached

        for next_state in env.possible_actions[current_state]:
            cost = -env.calculate_reward(current_state, next_state)  # Cost is negative reward
            distance = current_distance + cost

            if distance < distances[next_state]:
                distances[next_state] = distance
                previous_states[next_state] = current_state  # Keep track of path
                heapq.heappush(priority_queue, (distance, next_state))

    # Reconstruct path from goal to start
    optimal_path_states = []
    current = goal_state
    if previous_states[goal_state] is not None:  # Path exists
        while current is not None:
            optimal_path_states.insert(0, current)  # Prepend to build path from start to goal
            current = previous_states[current]
        return distances[goal_state], optimal_path_states
    else:
        return float("inf"), []  # No path found


# Find optimal path
optimal_cost, optimal_path = dijkstra_optimal_path(env, env.start_state, env.goal)

if optimal_cost != float("inf"):
    print("Optimal Path Cost:", optimal_cost)
    print("Optimal Path:", optimal_path)
else:
    print("No path found from", env.start_state, "to", env.goal)
