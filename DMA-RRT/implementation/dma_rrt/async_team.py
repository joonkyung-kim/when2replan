"""
    A team whose robots run their own cycles (step 9): Algorithms 4-5 and Sec. 5.3 of the paper

    Team runs all cycles at the same time and in order. Here each robot starts its cycles at its own time, grows its
    tree during the cycle, and acts at the end of it. Messages arrive in the middle of the others' cycles, and they
    change the constraints of the tree from then on.

    This runs on a virtual clock of one control step in a single process, instead of on many computers in real time
    as in Sec. 7.1. So the same seed gives the same result, but nothing here measures the real computing time.

    The paper leaves these open, and they are our decisions:
    - The bids of a robot that sent a plan go back to 0. It just took its best path, and this replaces the bid of the
      token holder in Team. The paper does not say when an old bid goes away.
    - A robot that gets the token uses it one cycle later, even in the middle of its own cycle. This passes the token
      once per cycle, as Sec. 7.2.3 waits about 10 s for the token to go around ten robots. plan_for_a_cycle=False
      uses it at the end of the robot's current cycle instead, as in the loop of Algorithm 4, but not in the control
      step it arrived in. This passes the token about 2 times per cycle.
"""

from __future__ import annotations

from typing import List
import numpy as np

from dma_rrt.agent import Agent
from dma_rrt.collision import SIXTEEN_PILLAR_MAP
from dma_rrt.dynamics import PurePursuit
from dma_rrt.team import Team, WITH_BIDS

ASYNC_COORDINATIONS = ("merit", "cooperative_paper", "cooperative", "round_robin")


class AsyncTeam(Team):
    def __init__(
        self,
        start_states: List[np.ndarray],
        goals: List[List[np.ndarray]],
        controller: PurePursuit,
        coordination: str = "merit",
        map_path: str = SIXTEEN_PILLAR_MAP,
        seed: int = 0,
        stop_interval: float = 4.0,
        same_phase: bool = False,
        bid_interval: float = 0.2,
        bids_only_when_better: bool = True,
        plan_for_a_cycle: bool = True,
        paper_tree: bool = False,
        can_turn: bool = True,
    ) -> None:
        """
        :param same_phase: all cycles start at the same time, which is the closest to Team
        :param bid_interval: time between the checks for a new bid of a robot [s]. Algorithm 4 bids once in a cycle
            and Sec. 5.3 sends bids "much more frequently" than plans, without a number
        :param bids_only_when_better: only send a bid that is higher than the last one this robot sent (Sec. 5.3).
            False sends every bid (line 13 of Algorithm 4). The two parts of the paper differ here
        :param plan_for_a_cycle: use the token one cycle after getting it, even in the middle of the robot's own cycle
        """
        if coordination not in ASYNC_COORDINATIONS:
            raise ValueError(f"coordination must be one of {ASYNC_COORDINATIONS}")
        super().__init__(start_states, goals, controller, coordination, map_path, seed, stop_interval, paper_tree, can_turn)
        self._dt = controller.dt
        executor = self.agents[0].executor
        self._steps_per_cycle = executor.steps_per_cycle
        self._samples_per_cycle = executor.samples_per_cycle
        self._steps_per_bid = int(round(bid_interval / self._dt))
        self._bids_only_when_better = bids_only_when_better
        self._plan_for_a_cycle = plan_for_a_cycle
        self._last_bids = [0.0] * len(self.agents)
        self._step_count = 0

        rng = np.random.default_rng(seed)
        self.phases = np.zeros(len(self.agents), dtype=int)
        if not same_phase:
            self.phases = rng.integers(0, self._steps_per_cycle, len(self.agents))
        self._token_use_step = self.phases[self.token_holder] + self._steps_per_cycle
        self.is_plan_sent = False  # a robot sent a plan in the last step
        self.num_plans = 0
        self.num_changed_plans = 0  # plans different from the robot's last plan
        self.num_bids = 0
        for agent, phase in zip(self.agents, self.phases):
            if phase > 0:
                agent.executor.start_at(phase * self._dt)
            else:
                self._begin_cycle(agent)

    def step(self) -> None:
        """
        One control step of all robots
        """
        self._step_count += 1
        self.time = self._step_count * self._dt
        self.is_plan_sent = False
        self.estop_target = None
        holder_id = self.token_holder
        elapsed = self._step_count - self.phases

        for agent in self.agents:
            if elapsed[agent.agent_id] > 0:
                agent.executor.grow(self._samples_in_step(elapsed[agent.agent_id]))
        if self._coordination in WITH_BIDS:
            for agent in self.agents:
                step = elapsed[agent.agent_id]
                if agent.agent_id != holder_id and step > 0 and step % self._steps_per_bid == 0:
                    self._send_bid(agent)
        if self._plan_for_a_cycle and self._step_count == self._token_use_step:
            holder = self.agents[holder_id]
            _, is_path_changed = holder.executor.change_plan()
            self._pass_token(holder, is_path_changed)
        for agent in self.agents:
            step = elapsed[agent.agent_id]
            if step < 0 or step % self._steps_per_cycle != 0:
                continue
            if step > 0:
                has_token = agent.agent_id == holder_id and not self._plan_for_a_cycle
                result = agent.end_cycle(has_token)
                if has_token:
                    self._pass_token(agent, result.is_path_changed)
            self._begin_cycle(agent)

    def _samples_in_step(self, elapsed: int) -> int:
        """
        Spread the samples of a cycle evenly over its control steps
        """
        step = (elapsed - 1) % self._steps_per_cycle
        return (
            self._samples_per_cycle * (step + 1) // self._steps_per_cycle
            - self._samples_per_cycle * step // self._steps_per_cycle
        )

    def _send_bid(self, agent: Agent) -> None:
        bid = agent.compute_bid()
        if self._bids_only_when_better and bid <= self._last_bids[agent.agent_id]:
            return
        self._last_bids[agent.agent_id] = bid
        self.num_bids += 1
        for other in self.agents:
            if other is not agent:
                other.receive_bid(agent.agent_id, bid, agent.planner.root_time)

    def _begin_cycle(self, agent: Agent) -> None:
        agent.executor.begin_cycle()
        for other in self.agents:
            if other is not agent:
                other.receive_root_time(agent.agent_id, agent.planner.root_time)

    def _pass_token(self, agent: Agent, is_path_changed: bool) -> None:
        self.token_counts[agent.agent_id] += 1
        self.num_plans += 1
        self.num_changed_plans += is_path_changed
        if self._coordination in WITH_BIDS:
            winner = agent.select_winner()
        else:
            winner = (agent.agent_id + 1) % len(self.agents)
        if agent.estop_target is not None:
            # line 15 of Algorithm 6: the stopped robot updates its plan next
            winner = agent.estop_target
            self.estop_target = agent.estop_target
            self.num_estops += 1

        message = agent.make_plan_message(winner)
        self._last_bids[agent.agent_id] = 0.0
        for other in self.agents:
            if other is not agent:
                other.receive_plan(message)
                other.bids[agent.agent_id] = 0.0
        self.token_holder = winner
        self._token_use_step = self._step_count + self._steps_per_cycle
        self.is_plan_sent = True
