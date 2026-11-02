"""
    Cooperative DMA-RRT: Algorithms 6-8 in the DMA-RRT paper (step 7)

    The best path can go through another robot. When it does, the token holder compares its own path cut short
    with asking that robot to stop early at one of its emergency stop nodes, and takes what is the best for the
    two robots together.

    A robot can be asked to stop at any of its emergency stop nodes, so it has to be safe to stop at all of them.
    The others therefore treat each of them as blocked from the time the robot arrives there.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from dma_rrt.agent import Agent, PlanMessage, StopNode
from dma_rrt.collision import MovingObstacle
from dma_rrt.tree import Node

TIME_TOLERANCE = 1e-6  # [s]


class Decision(NamedTuple):
    end_node: Node  # end of the plan
    stop_node: Optional[StopNode]  # where the blocker has to stop, None if it can keep its plan
    blocker: Optional[int]  # the robot blocking the best path
    improvement: float  # reduction of the cost of the two robots [s]


class PathMasks:
    """
    What another robot allows on a path
    edge_free[i]: the trajectory to node i + 1 is safe. stop_safe[i]: the robot can stay at node i for all future time
    """

    def __init__(self, edge_free: np.ndarray, stop_safe: np.ndarray) -> None:
        self.edge_free = edge_free
        self.stop_safe = stop_safe

    def __and__(self, other: PathMasks) -> PathMasks:
        return PathMasks(self.edge_free & other.edge_free, self.stop_safe & other.stop_safe)

    def last_safe_node(self) -> Optional[int]:
        """
        :return: index of the last node on the path to stop at, None if the robot cannot even stop at the first one
        """
        num_free_edges = len(self.edge_free) if self.edge_free.all() else int(np.argmin(self.edge_free))
        indices = np.flatnonzero(self.stop_safe[: num_free_edges + 1])
        return int(indices[-1]) if len(indices) > 0 else None


class CooperativeAgent(Agent):
    def __init__(
        self, *args, stop_interval: float = 4.0, as_in_paper: bool = False, **kwargs
    ) -> None:
        """
        :param stop_interval: time between the emergency stop nodes on a path, 4 s in Sec. 7.3.3 of the paper [s]
        :param as_in_paper: the tree ignores the others and the plan always comes from the best path. Otherwise
            the tree avoids the others and the plan can come from any branch, see _best_feasible_node
        """
        super().__init__(*args, select_ignoring_others=True, grow_ignoring_others=as_in_paper, **kwargs)
        self._stop_interval = stop_interval
        self._use_other_paths = not as_in_paper
        self._check_estops = True
        self._checked_edges: Dict[Node, bool] = {}
        self._estop_time = 0.0
        self.own_stop_nodes: List[StopNode] = []
        self._stop_nodes: Dict[int, List[StopNode]] = {}
        self._end_costs: Dict[int, float] = {}
        self._root_times: Dict[int, float] = {}
        self.executor.plan_selector = self._select_plan

    def add_other_robot(self, agent_id: int, start_state: np.ndarray) -> None:
        super().add_other_robot(agent_id, start_state)
        self._stop_nodes[agent_id] = []
        self._end_costs[agent_id] = 0.0
        self._root_times[agent_id] = 0.0

    def receive_bid(self, agent_id: int, bid: float, root_time: float = 0.0) -> None:
        """
        The bid also has the time of the sender's root: it cannot stop before that anymore. The paper only uses
        stop nodes far enough ahead (Sec. 6.3), and this is the exact limit of our commit
        """
        super().receive_bid(agent_id, bid)
        self._root_times[agent_id] = root_time

    def receive_root_time(self, agent_id: int, root_time: float) -> None:
        self._root_times[agent_id] = root_time

    def make_plan_message(self, winner: int) -> PlanMessage:
        """
        Line 19 of Algorithm 6, with the emergency stop nodes. Ours also broadcasts the estop message, which the
        paper sends only to the stopped robot: the others then do not believe its old plan for one more cycle
        """
        message = super().make_plan_message(winner)
        return message._replace(
            stop_nodes=self.own_stop_nodes,
            end_cost=self.planner.finish_time_estimate(self.executor.plan_nodes[-1], self.goal_pos),
            estop_target=self.estop_target,
            estop_time=self._estop_time,
        )

    def receive_plan(self, message: PlanMessage) -> None:
        """
        Line 4 and lines 12-15 of Algorithm 8
        """
        if message.estop_target == self.agent_id:
            self._stop_at(message.estop_time)
            self._check_estops = False
        elif message.estop_target is not None:
            self._apply_estop(message.estop_target, message.estop_time)
        super().receive_plan(message)
        self._stop_nodes[message.sender] = message.stop_nodes
        self._end_costs[message.sender] = message.end_cost

    def _stop_at(self, time: float) -> None:
        """
        Prune the plan after the node of the given time. The new plan is shared when this robot gets the token,
        which is right after this message
        """
        executor = self.executor
        for i, node in enumerate(executor.plan_nodes):
            if abs(self.planner.node_time(node) - time) < TIME_TOLERANCE:
                executor.plan_nodes = executor.plan_nodes[: i + 1]
                return
        raise ValueError(f"robot {self.agent_id} has no node at time {time}")

    def _apply_estop(self, agent_id: int, time: float) -> None:
        self._other_robots[agent_id].positions = self._other_robots[agent_id].stopped_at(time).positions
        stop_nodes = self._stop_nodes[agent_id]
        self._end_costs[agent_id] = next(node.cost for node in stop_nodes if abs(node.time - time) < TIME_TOLERANCE)
        self._stop_nodes[agent_id] = [node for node in stop_nodes if node.time < time]

    def _select_plan(self) -> Tuple[Node, int]:
        """
        Lines 4-13 of Algorithm 6
        :return: end of the plan, and no paths removed since the tree does not depend on the others
        """
        decision = self._decide()
        self.estop_target = None
        if decision.stop_node is not None:
            self.estop_target = decision.blocker
            self._estop_time = decision.stop_node.time
            self._apply_estop(decision.blocker, decision.stop_node.time)
        self._check_estops = True
        self._remove_blocked_nodes(decision.end_node)
        self.own_stop_nodes = self._identify_stop_nodes(self.planner.tree.nodes_to(decision.end_node))
        return decision.end_node, 0

    def _remove_blocked_nodes(self, end_node: Node) -> None:
        """
        Like the lazy check, remove the nodes found to be blocked while selecting the plan, or new samples keep
        growing from them
        """
        plan_nodes = set(self.planner.tree.nodes_to(end_node))
        for node, is_free in self._checked_edges.items():
            if not is_free and self._checked_edges[node.parent] and node not in plan_nodes:
                self.planner.tree.remove_subtree(node)
        self._checked_edges = {}

    def _decide(self) -> Decision:
        """
        Algorithm 7. Keeping the current plan is an option here, which is not in the paper: the best path
        cut short can be much worse than a plan the robot already has
        """
        path = self.planner.tree.nodes_to(self.planner.best_node(self.goal_pos))
        current_end = self.executor.plan_nodes[-1]
        points, times = self._path_points(path)
        masks = {agent_id: self._masks(path, points, times, agent_id) for agent_id in self._other_robots}
        combined = self._combine(path, masks.values())
        if combined.last_safe_node() == len(path) - 1:
            return Decision(path[-1], None, None, self._cost(current_end) - self._cost(path[-1]))

        def rank(agent_id: int) -> int:
            last_safe = masks[agent_id].last_safe_node()
            return -1 if last_safe is None else last_safe

        blocker = min(masks, key=rank)
        blocker_cost = self._end_costs[blocker]
        current_cost = self._cost(current_end) + blocker_cost
        best = Decision(current_end, None, blocker, 0.0)

        def consider(end_node: Node, other_cost: float, stop_node: Optional[StopNode]) -> Decision:
            improvement = current_cost - self._cost(end_node) - other_cost
            return Decision(end_node, stop_node, blocker, improvement) if improvement > best.improvement else best

        last_safe = combined.last_safe_node()
        if last_safe is not None:
            best = consider(path[last_safe], blocker_cost, None)
        if self._use_other_paths:
            other_end = self._best_feasible_node(self._cost(best.end_node))
            if other_end is not None:
                best = consider(other_end, blocker_cost, None)

        if self._check_estops:
            others = self._combine(path, (mask for agent_id, mask in masks.items() if agent_id != blocker))
            for stop_node in self._viable_stop_nodes(blocker):
                stopped = self._other_robots[blocker].stopped_at(stop_node.time)
                last_safe = (others & self._trajectory_masks(path, points, times, stopped)).last_safe_node()
                if last_safe is not None:
                    best = consider(path[last_safe], stop_node.cost, stop_node)
        return best

    def _best_feasible_node(self, max_cost: float) -> Optional[Node]:
        """
        Not in the paper. The node with the lowest cost that the others allow, which is what DMA-RRT selects.
        The best path may stay blocked for long while another branch of the tree is free. Without this and the
        removal of blocked nodes, a tree stops growing around the other robots, and even without any emergency
        stops the team reaches fewer goals than DMA-RRT.
        :param max_cost: only look for nodes better than this
        :return: node, None if there is no such node
        """
        nodes = self.planner.tree.nodes
        costs = np.array([self._cost(node) for node in nodes])
        is_free = {self.planner.tree.root: True}
        self._checked_edges = is_free
        for index in np.argsort(costs):
            if costs[index] >= max_cost:
                return None
            node = nodes[index]
            if self._is_path_free(node, is_free) and self._is_safe_stop(node, self.planner.node_time(node)):
                return node
        return None

    def _is_path_free(self, node: Node, is_free: Dict[Node, bool]) -> bool:
        """
        Check the path from the root, keeping the results for the other paths through the same nodes
        """
        for path_node in self.planner.tree.nodes_to(node)[1:]:
            if path_node not in is_free:
                is_free[path_node] = is_free[path_node.parent] and self._is_edge_free(path_node)
            if not is_free[path_node]:
                return False
        return True

    def _is_edge_free(self, node: Node) -> bool:
        points = node.trajectory[:, :2]
        times = self.planner.trajectory_times(self.planner.node_time(node.parent), len(points))
        if self.checker.moving_obstacle_dists(points, times).min() < self.checker.safe_dist:
            return False
        min_dist = self.checker.safe_dist + self._robot_radius
        for agent_id in self._stop_nodes:
            for stop_node in self._viable_stop_nodes(agent_id):
                is_late = times >= stop_node.time - TIME_TOLERANCE
                if (np.linalg.norm(points - stop_node.position, axis=1)[is_late] < min_dist).any():
                    return False
        return True

    def _viable_stop_nodes(self, agent_id: int) -> List[StopNode]:
        """
        Only the stop nodes the robot has not passed yet can be used, and only these block the others
        """
        return [
            node for node in self._stop_nodes[agent_id] if node.time >= self._root_times[agent_id] - TIME_TOLERANCE
        ]

    def _cost(self, node: Node) -> float:
        return self.planner.finish_time_estimate(node, self.goal_pos)

    def _path_points(self, path: List[Node]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        :return: positions and times of the trajectory to each node after the first
        """
        points, times = [], []
        for node in path[1:]:
            points.append(node.trajectory[:, :2])
            times.append(self.planner.trajectory_times(self.planner.node_time(node.parent), len(node.trajectory)))
        return points, times

    def _combine(self, path: List[Node], masks) -> PathMasks:
        combined = PathMasks(np.ones(len(path) - 1, dtype=bool), np.ones(len(path), dtype=bool))
        for mask in masks:
            combined = combined & mask
        return combined

    def _masks(self, path, points, times, agent_id: int) -> PathMasks:
        """
        Everything another robot blocks: its trajectory, and its emergency stop nodes after it arrives there
        """
        masks = self._trajectory_masks(path, points, times, self._other_robots[agent_id])
        min_dist = self.checker.safe_dist + self._robot_radius
        for stop_node in self._viable_stop_nodes(agent_id):
            edge_free = np.array(
                [
                    (np.linalg.norm(p - stop_node.position, axis=1)[t >= stop_node.time - TIME_TOLERANCE] >= min_dist).all()
                    for p, t in zip(points, times)
                ],
                dtype=bool,
            )
            positions = np.array([node.state[:2] for node in path])
            stop_safe = np.linalg.norm(positions - stop_node.position, axis=1) >= min_dist
            masks = masks & PathMasks(edge_free, stop_safe)
        return masks

    def _trajectory_masks(self, path, points, times, obstacle: MovingObstacle) -> PathMasks:
        min_dist = self.checker.safe_dist + obstacle.radius
        edge_free = np.array(
            [(np.linalg.norm(p - obstacle.positions_at(t), axis=1) >= min_dist).all() for p, t in zip(points, times)],
            dtype=bool,
        )
        stop_safe = np.array(
            [
                np.linalg.norm(obstacle.positions_from(self.planner.node_time(node)) - node.state[:2], axis=1).min()
                >= min_dist
                for node in path
            ],
            dtype=bool,
        )
        return PathMasks(edge_free, stop_safe)

    def _identify_stop_nodes(self, plan_nodes: List[Node]) -> List[StopNode]:
        """
        Line 12 of Algorithm 6: the next node that is safe to stop at, every stop_interval along the plan.
        The end of the plan is not one: the others already know the robot stops there
        """
        stop_nodes = []
        next_time = -np.inf
        for node in plan_nodes[:-1]:
            time = self.planner.node_time(node)
            if time < next_time or not self._is_safe_stop(node, time):
                continue
            stop_nodes.append(StopNode(time, node.state[:2], self._cost(node)))
            next_time = time + self._stop_interval
        return stop_nodes

    def _is_safe_stop(self, node: Node, time: float) -> bool:
        if not self.checker.is_stop_safe(node.state[:2], time):
            return False
        min_dist = self.checker.safe_dist + self._robot_radius
        for agent_id in self._stop_nodes:
            for stop_node in self._viable_stop_nodes(agent_id):
                if np.linalg.norm(node.state[:2] - stop_node.position) < min_dist:
                    return False
        return True
