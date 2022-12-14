import os
from datetime import datetime
from typing import List, Dict

import numpy as np
from prl.environment.Wrappers.augment import AugmentObservationWrapper
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.Wrappers.utils import init_wrapped_env
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.agents.eval.core.evaluator import PokerExperimentEvaluator, DEFAULT_CURRENCY, DEFAULT_VARIANT, \
    DEFAULT_DATE
from prl.baselines.agents.eval.core.experiment import PokerExperiment
from prl.baselines.agents.eval.utils import make_agents, make_participants
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, ActionType, \
    PlayerWithCards, PlayerWinningsCollected, Action, PokerEpisode

POSITIONS_HEADS_UP = ['btn', 'bb']  # button is small blind in Heads Up situations
POSITIONS = ['btn', 'sb', 'bb', 'utg', 'mp', 'co']
STAGES = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]
ACTION_TYPES = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE]


class ExperimentRunner(PokerExperimentEvaluator):
    # run experiments using
    @staticmethod
    def _get_player_stacks(obs, num_players, normalization_sum) -> List[PlayerStack]:
        """ Stacks at the beginning of every episode, not during or after."""
        player_stacks = []
        # Name the players, according to their offset to the button:
        # Button: Player_0
        # SB: Player_1
        # BB: Player_2
        # ...
        # Cut-off: Player_5
        pos_rel_to_observer = np.roll(POSITIONS, obs[COLS.Btn_idx]) if num_players > 2 else np.roll(POSITIONS,
                                                                                                    obs[COLS.Btn_idx])
        stacks = [COLS.Stack_p0,
                  COLS.Stack_p1,
                  COLS.Stack_p2,
                  COLS.Stack_p3,
                  COLS.Stack_p4,
                  COLS.Stack_p5]
        for i in range(num_players):
            player_name = f'Player_{i}'
            seat_display_name = pos_rel_to_observer[i]
            stack = round(obs[stacks[i]] * normalization_sum, 2)
            player_stacks.append(PlayerStack(seat_display_name,
                                             player_name,
                                             stack))
        return player_stacks

    @staticmethod
    def _get_blinds(obs, num_players, normalization_sum) -> List[Blind]:
        """
        observing player sits relative to button. this offset is given by
        # >>> obs[COLS.Btn_idx]
        the button position determines which players post the small and the big blind.
        For games with more than 2 pl. the small blind is posted by the player who acts after the button.
        When only two players remain, the button posts the small blind.
        """
        sb_name = "Player_1"
        bb_name = "Player_2"

        if num_players == 2:
            sb_name = "Player_0"
            bb_name = "Player_1"

        sb_amount = COLS.Small_blind * normalization_sum
        bb_amount = COLS.Big_blind * normalization_sum

        return [Blind(sb_name, 'small blind', sb_amount),
                Blind(bb_name, 'big_blind', bb_amount)]

    @staticmethod
    def _get_winners(showdown_players: List[PlayerWithCards], payouts: dict) -> List[PlayerWithCards]:
        winners = []
        for pid, money_won in payouts.items():
            for p in showdown_players:
                # look for winning player in showdown players and append its stats to winners
                if POSITIONS[pid] == p.name:
                    winners.append(PlayerWithCards(name=p.name,
                                                   cards=p.cards))
        return winners

    @staticmethod
    def _get_money_collected(payouts: Dict[int, str]) -> List[PlayerWinningsCollected]:
        money_collected = []
        for pid, payout in payouts.items():
            money_collected.append(PlayerWinningsCollected(player_name=POSITIONS[pid],
                                                           collected="$" + str(payout),
                                                           rake=None))
        return money_collected

    @staticmethod
    def _parse_cards(cards):
        # In: '3h, 9d, '
        # Out: '[Ah Jd]'
        tokens = cards.split(',')  # ['3h', ' 9d', ' ']
        c0 = tokens[0].replace(' ', '')
        c1 = tokens[1].replace(' ', '')
        return f'[{c0},  {c1}]'

    def _get_remaining_players(self, env) -> List[PlayerWithCards]:
        # make player cards
        remaining_players = []
        for i in range(env.env.N_SEATS):
            # If player did not fold and still has money (is active and does not sit out)
            # append it to the remaining players
            if not env.env.seats[i].folded_this_episode:
                if env.env.seats[i].stack > 0 or env.env.seats[i].is_allin:
                    cards = env.env.cards2str(env.env.get_hole_cards_of_player(i))
                    remaining_players.append(PlayerWithCards(name=POSITIONS[i],
                                                             cards=self._parse_cards(cards)))
        return remaining_players

    def evaluate(self, experiment: PokerExperiment) -> List[PokerEpisode]:
        poker_episodes = []
        n_episodes = experiment.max_episodes
        env = experiment.env
        agents = experiment.agents
        total_actions_dict = {0: 0, 1: 0, 2: 0}
        for ep_id in range(n_episodes):
            # -------- Reset environment ------------
            obs, _, done, _ = env.reset()
            showdown_hands = None
            normalization_sum = float(
                sum([s.starting_stack_this_episode for s in env.env.seats])
            ) / env.env.N_SEATS
            blinds = self._get_blinds(obs, num_players, normalization_sum)
            ante = obs[COLS.Ante] * normalization_sum
            player_stacks = self._get_player_stacks(obs, num_players, normalization_sum)
            actions_total = {'preflop': [],
                             'flop': [],
                             'turn': [],
                             'river': []}
            # make obs
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}
            agent_idx = 0

            while not done:
                # -------- COMPUTE ACTION -----------
                action_vec = agents[agent_idx].act(observation)
                # Record action to episodes actions
                action = int(action_vec[0][0].numpy())
                agent_that_acted = POSITIONS[-int(obs[COLS.Btn_idx])]

                stage = Poker.INT2STRING_ROUND[env.env.current_round]
                action_type = ACTION_TYPES[min(action, 2)]
                raise_amount = env.env.last_action[1]
                episode_action = Action(stage=stage,
                                        player_name=agent_that_acted,
                                        action_type=action_type,
                                        raise_amount=raise_amount)
                actions_total[stage].append(episode_action)
                remaining_players = self._get_remaining_players(env)

                # -------- STEP ENVIRONMENT -----------
                obs, _, done, info = env.step(action)
                # if not done, prepare next turn
                if done:
                    showdown_hands = remaining_players
                    print(ep_id)
                legal_moves = env.env.get_legal_actions()
                observation = {'obs': [obs], 'legal_moves': [legal_moves]}

                # -------- SET NEXT AGENT -----------
                agent_idx += 1
                agent_idx = agent_idx % num_players

                # env.env.cards2str(env.env.get_hole_cards_of_player(0))
            a = "debug"
            winners = self._get_winners(showdown_players=showdown_hands, payouts=info['payouts'])
            money_collected = self._get_money_collected(payouts=info['payouts'])
            poker_episodes.append(PokerEpisode(date=DEFAULT_DATE,
                                               hand_id=ep_id,
                                               variant=DEFAULT_VARIANT,
                                               currency_symbol=DEFAULT_CURRENCY,
                                               num_players=num_players,
                                               blinds=blinds,
                                               ante=ante,
                                               player_stacks=player_stacks,
                                               btn_idx=0,
                                               board_cards=env.env.cards2str(env.env.board),
                                               actions_total=actions_total,
                                               winners=winners,
                                               showdown_hands=showdown_hands,
                                               money_collected=money_collected))
            debug = 'set breakpoint here'
        print(total_actions_dict)
        return poker_episodes


if __name__ == '__main__':
    # move this to example.py or main.py
    # Construct Experiment
    starting_stack_size = 1000
    num_players = 2
    stacks = [starting_stack_size for _ in range(num_players)]
    env_wrapped = init_wrapped_env(env_wrapper_cls=AugmentObservationWrapper,
                                   stack_sizes=stacks,
                                   multiply_by=1)
    max_episodes = 10
    path_to_baseline_torch_model_state_dict = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt.pt"
    agent_list = make_agents(env_wrapped, path_to_baseline_torch_model_state_dict)
    participants = make_participants(agent_list, starting_stack_size)

    evaluator = BaselineEvaluator()
    evaluator.evaluate(
        PokerExperiment(
            env=env_wrapped,
            participants=participants,
            max_episodes=max_episodes,
            agents=agent_list,
            current_episode=0,
            cbs_plots=[],
            cbs_misc=[],
            cbs_metrics=[]
        )
    )
