"""
    A team of robots coordinated with a token

    "merit" is DMA-RRT (step 6): only the token holder changes its plan, and it passes the token to the robot with
    the best bid, its potential path improvement (Sec. 4.1 of the paper).
    "merit_negative" is what we first implemented: the same, but a stopped robot can bid less than 0, and starve.
    "merit_wait" is not in the paper: the same as "merit", but a stopped robot adds the time it has waited to its bid.
    "round_robin" is the baseline of Sec. 7.2.1: the same, but the token goes around in a fixed order (step 5).
    "cooperative_paper" is Cooperative DMA-RRT (step 7) as in Algorithms 6-8: bids, the tree ignores the others,
    and the token holder can ask another robot to stop early.
    "cooperative" is what works in our scenes: the same, but the tree avoids the others and any branch can be used.
    "cooperative_negative" and "cooperative_wait" are that, with the bids of "merit_negative" and "merit_wait".
    "none" has no coordination: all robots change their plans at the same time and learn about the others' new
    plans afterwards. This is the failure of Fig. 2 of the paper, kept to compare.

    The network is lossless and has no delay, as assumed in Sec. 5.1, so messages are simply delivered in order.
"""

from __future__ import annotations

from typing import List
import numpy as np

from dma_rrt.agent import Agent
from dma_rrt.collision import SIXTEEN_PILLAR_MAP
from dma_rrt.cooperative import CooperativeAgent
from dma_rrt.dynamics import PurePursuit
from dma_rrt.execution import CycleResult

COORDINATIONS = (
    "merit", "merit_negative", "merit_wait", "cooperative_paper", "cooperative", "cooperative_negative", "cooperative_wait", "round_robin", "none",
)
WITH_BIDS = ("merit", "merit_negative", "merit_wait", "cooperative_paper", "cooperative", "cooperative_negative", "cooperative_wait")
WITH_WAITING_TIME = ("merit_wait", "cooperative_wait")
WITH_NEGATIVE_BIDS = ("merit_negative", "cooperative_negative")


class Team:
    def __init__(
        self,
        start_states: List[np.ndarray],
        goals: List[List[np.ndarray]],
        controller: PurePursuit,
        coordination: str = "merit",
        map_path: str = SIXTEEN_PILLAR_MAP,
        seed: int = 0,
        stop_interval: float = 4.0,
        paper_tree: bool = False,
        can_turn: bool = True,
    ) -> None:
        """
        :param start_states: start state of each robot
        :param goals: goals to visit of each robot
        :param controller: closed loop model of all robots
        :param coordination: one of COORDINATIONS
        :param map_path: map yaml file, relative to the repo root
        :param seed: the robots use different seeds based on this
        :param stop_interval: time between the emergency stop nodes of Cooperative DMA-RRT [s]
        :param paper_tree: grow the trees as in Algorithm 1 (PaperCLRRT), instead of our CLRRT
        :param can_turn: False never turns in place, as in the paper
        """
        if coordination not in COORDINATIONS:
            raise ValueError(f"coordination must be one of {COORDINATIONS}")
        self._coordination = coordination
        self._stop_interval = stop_interval
        self._paper_tree = paper_tree
        self._can_turn = can_turn
        self.agents = [
            self._make_agent(i, start_states[i], goals[i], controller, map_path, seed * 1000 + i)
            for i in range(len(start_states))
        ]
        for agent in self.agents:
            for other_id, start_state in enumerate(start_states):
                if other_id != agent.agent_id:
                    agent.add_other_robot(other_id, start_state)
        self.token_holder = 0  # the paper selects it randomly, this keeps runs easy to compare
        self.token_counts = [0] * len(self.agents)
        self.num_estops = 0
        self.estop_target = None  # robot asked to stop in the last cycle
        self.time = 0.0

    def _make_agent(self, agent_id, start_state, goals, controller, map_path, seed) -> Agent:
        if not self._coordination.startswith("cooperative"):
            return Agent(
                agent_id, start_state, goals, controller, map_path=map_path, seed=seed,
                paper_tree=self._paper_tree, can_turn=self._can_turn,
            )
        return CooperativeAgent(
            agent_id,
            start_state,
            goals,
            controller,
            map_path=map_path,
            seed=seed,
            paper_tree=self._paper_tree,
            can_turn=self._can_turn,
            stop_interval=self._stop_interval,
            as_in_paper=self._coordination == "cooperative_paper",
        )

    def step(self) -> List[CycleResult]:
        """
        One cycle of all robots
        :return: result of each robot
        """
        if self._coordination == "none":
            results = self._step_without_coordination()
        else:
            results = self._step_with_token()
        self.time = self.agents[0].executor.time
        return results

    def _step_with_token(self) -> List[CycleResult]:
        """
        The token holder goes first, so that the others grow their trees with its new plan in the same cycle
        The winner is selected with the bids of the last cycle, as in Algorithm 4
        """
        holder = self.agents[self.token_holder]
        self.token_counts[holder.agent_id] += 1
        if self._coordination in WITH_BIDS:
            winner = holder.select_winner()
        else:
            winner = (self.token_holder + 1) % len(self.agents)
        results: List[CycleResult] = [None] * len(self.agents)
        self.estop_target = None

        results[holder.agent_id] = holder.update(has_token=True)
        if holder.estop_target is not None:
            # line 15 of Algorithm 6: the stopped robot updates its plan next
            winner = holder.estop_target
            self.estop_target = holder.estop_target
            self.num_estops += 1
        message = holder.make_plan_message(winner)
        for agent in self.agents:
            if agent is not holder:
                agent.receive_plan(message)
                results[agent.agent_id] = agent.update(has_token=False)

        if self._coordination in WITH_BIDS:
            # The token holder bids as well. Otherwise the others keep the high bid that won it the token,
            # and two robots can keep passing the token to each other
            for agent in self.agents:
                bid = agent.compute_bid(
                    add_waiting_time=self._coordination in WITH_WAITING_TIME,
                    allow_negative=self._coordination in WITH_NEGATIVE_BIDS,
                )
                for other in self.agents:
                    if other is not agent:
                        other.receive_bid(agent.agent_id, bid, agent.planner.root_time)

        self.token_holder = winner
        return results

    def _step_without_coordination(self) -> List[CycleResult]:
        results = [agent.update(has_token=True) for agent in self.agents]
        messages = [agent.make_plan_message(agent.agent_id) for agent in self.agents]
        for message in messages:
            for agent in self.agents:
                if agent.agent_id != message.sender:
                    agent.receive_plan(message)
        return results
