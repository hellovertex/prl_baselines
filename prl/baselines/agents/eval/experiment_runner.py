from typing import List, Dict
from collections import OrderedDict
import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.agents.eval.core.experiment import PokerExperiment
from prl.baselines.agents.eval.core.runner import ExperimentRunner, DEFAULT_CURRENCY, DEFAULT_VARIANT, \
    DEFAULT_DATE
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, ActionType, \
    PlayerWithCards, PlayerWinningsCollected, Action, PokerEpisode
from prl.baselines.utils.num_parsers import parse_num

POSITIONS_HEADS_UP = ['btn', 'bb']  # button is small blind in Heads Up situations
POSITIONS = ['btn', 'sb', 'bb', 'utg', 'mp', 'co']
STAGES = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]
ACTION_TYPES = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE]


def _make_board(board: str) -> str:
    return f"[{board.replace(',', '').rstrip()}]"


class PokerExperimentRunner(ExperimentRunner):
    # run experiments using
    @staticmethod
    def _get_player_stacks(obs, num_players, normalization_sum, offset) -> List[PlayerStack]:
        """ Stacks at the beginning of every episode, not during or after."""
        player_stacks = []
        stacks = [COLS.Stack_p0,
                  COLS.Stack_p1,
                  COLS.Stack_p2,
                  COLS.Stack_p3,
                  COLS.Stack_p4,
                  COLS.Stack_p5]
        for i in range(num_players):
            player_name = f'Player_{i}'
            seat_display_name = f'Seat {i}'
            stack = "$" + str(parse_num(round(obs[np.roll(stacks[:num_players], -offset)[i]] * normalization_sum, 2)))
            player_stacks.append(PlayerStack(seat_display_name,
                                             player_name,
                                             stack))
        return player_stacks

    @staticmethod
    def _get_blinds(obs, num_players, normalization_sum, offset) -> List[Blind]:
        """
        observing player sits relative to button. this offset is given by
        # >>> obs[COLS.Btn_idx]
        the button position determines which players post the small and the big blind.
        For games with more than 2 pl. the small blind is posted by the player who acts after the button.
        When only two players remain, the button posts the small blind.
        """
        # btn_offset = int(obs[COLS.Btn_idx])
        sb_name = f"Player_{(1 + offset) % num_players}"
        bb_name = f"Player_{(2 + offset) % num_players}"
        if num_players == 2:
            sb_name = f"Player_{offset % num_players}"
            bb_name = f"Player_{(1 + offset) % num_players}"

        sb_amount = "$" + str(parse_num(round(obs[COLS.Small_blind] * normalization_sum, 2)))
        bb_amount = "$" + str(parse_num(round(obs[COLS.Big_blind] * normalization_sum, 2)))

        return [Blind(sb_name, 'small blind', sb_amount),
                Blind(bb_name, 'big blind', bb_amount)]

    @staticmethod
    def _get_winners(showdown_players: List[PlayerWithCards], payouts: dict) -> List[PlayerWithCards]:
        winners = []
        for pid, money_won in payouts.items():
            for p in showdown_players:
                # look for winning player in showdown players and append its stats to winners
                if f'Player_{pid}' == p.name:
                    winners.append(PlayerWithCards(name=f'Player_{pid}',
                                                   cards=p.cards))
        return winners

    @staticmethod
    def _get_money_collected(payouts: Dict[int, str]) -> List[PlayerWinningsCollected]:
        money_collected = []
        for pid, payout in payouts.items():
            money_collected.append(PlayerWinningsCollected(player_name=f'Player_{pid}',
                                                           collected="$" + str(float(payout)),
                                                           rake=None))
        return money_collected

    @staticmethod
    def _parse_cards(cards):
        # In: '3h, 9d, '
        # Out: '[Ah Jd]'
        tokens = cards.split(',')  # ['3h', ' 9d', ' ']
        c0 = tokens[0].replace(' ', '')
        c1 = tokens[1].replace(' ', '')
        return f'[{c0} {c1}]'

    def _get_remaining_players(self, env) -> List[PlayerWithCards]:
        # make player cards
        remaining_players = []
        for i in range(env.env.N_SEATS):
            # If player did not fold and still has money (is active and does not sit out)
            # append it to the remaining players
            if not env.env.seats[i].folded_this_episode:
                if env.env.seats[i].stack > 0 or env.env.seats[i].is_allin:
                    cards = env.env.cards2str(env.env.get_hole_cards_of_player(i))
                    remaining_players.append(PlayerWithCards(name=f'Player_{i}',
                                                             cards=self._parse_cards(cards)))
        return remaining_players

    def _has_folded(self, p: PlayerWithCards, last_action) -> bool:
        return last_action[0] == 0 and f'Player_{last_action[2]}' == p.name

    def _get_showdown_hands(self, remaining_players, last_action) -> List[PlayerWithCards]:
        showdown_hands = []
        for p in remaining_players:
            if not self._has_folded(p, last_action):
                showdown_hands.append(p)
        return showdown_hands

    def _update_cumulative_stacks(self, money_collected: List[PlayerWinningsCollected]):
        for p in money_collected:
            self.money_from_last_round[p.player_name] += float(p.collected[1:])

    def _subtract_blinds_from_stacks(self, blinds: List[Blind]):
        for b in blinds:
            self.money_from_last_round[b.player_name] -= float(b.amount[1:])

    def _get_stack_sizes(self, experiment: PokerExperiment):
        # returns cumulative stack sizes if all players have stacks > 0
        # otherwise returns default stack sizes starting list
        default_stack_size = experiment.env.normalization
        stack_sizes_list = []
        for pname, money in self.money_from_last_round.items():
            if money == -default_stack_size:
                return [default_stack_size for _ in range(experiment.num_players)]
            stack_sizes_list.append(round(default_stack_size) + round(money))
        return stack_sizes_list

    def run(self, experiment: PokerExperiment) -> List[PokerEpisode]:
        poker_episodes = []
        n_episodes = experiment.max_episodes
        num_players = experiment.num_players
        env = experiment.env
        total_actions_dict = {ActionType.FOLD.value: 0,
                              ActionType.CHECK_CALL.value: 0,
                              ActionType.RAISE.value: 0}
        # init action generators
        if experiment.from_action_plan:
            participants = []
            iter_actions = iter(experiment.from_action_plan)
        else:
            participants = experiment.participants
            iter_actions = iter([])

        self.money_from_last_round = OrderedDict()
        for i in range(num_players):
            self.money_from_last_round[f'Player_{i}'] = 0
            # ----- RUN EPISODES -----
        for ep_id in range(n_episodes):
            # -------- Reset environment ------------
            stack_sizes_list = self._get_stack_sizes(experiment)
            env.env.set_stack_size(stack_sizes_list)
            obs, _, done, _ = env.reset(experiment.env_config)
            agent_idx = button_index = ep_id % num_players  # always move button to the right
            showdown_hands = None
            normalization_sum = float(
                sum([s.starting_stack_this_episode for s in env.env.seats])
            ) / env.env.N_SEATS
            blinds = self._get_blinds(obs, num_players, normalization_sum, agent_idx)
            self._subtract_blinds_from_stacks(blinds)
            ante = '$0.00'
            assert obs[COLS.Ante] == 0  # games with ante not supported
            player_stacks = self._get_player_stacks(obs,
                                                    num_players,
                                                    normalization_sum,
                                                    agent_idx)
            actions_total = {'preflop': [],
                             'flop': [],
                             'turn': [],
                             'river': [],
                             'as_sequence': []}

            # make obs
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}
            # ---- RUN GAME ----
            while not done:
                # -------- ACT -----------
                if experiment.from_action_plan:
                    action = next(iter_actions)
                    # if isinstance(action, tuple):
                    #     action =
                else:
                    action_vec = participants[agent_idx].agent.act(observation)
                    action = int(action_vec[0][0].numpy())
                    # todo convert ActionSpace (integer repr) to Action (tuple repr)
                    action = env.int_action_to_tuple_action(action)

                # -------- STEP ENVIRONMENT -----------
                remaining_players = self._get_remaining_players(env)
                stage = Poker.INT2STRING_ROUND[env.env.current_round]
                obs, _, done, info = env.step(action)
                # -------- RECORD LAST ACTION ---------
                a = env.env.last_action
                raise_amount = float(a[1])  # if a[0] == ActionType.RAISE else -1
                episode_action = Action(stage=stage,
                                        player_name=f'Player_{agent_idx}',
                                        action_type=ACTION_TYPES[a[0]],
                                        raise_amount=raise_amount)
                self.money_from_last_round[f'Player_{a[2]}'] -= a[1]
                actions_total[stage].append(episode_action)
                actions_total['as_sequence'].append(episode_action)
                total_actions_dict[a[0]] += 1
                # if not done, prepare next turn
                if done:
                    showdown_hands = self._get_showdown_hands(remaining_players, a)
                    print(ep_id)
                legal_moves = env.env.get_legal_actions()
                observation = {'obs': [obs], 'legal_moves': [legal_moves]}

                # -------- SET NEXT AGENT -----------
                agent_idx = env.current_player.seat_id

                # env.env.cards2str(env.env.get_hole_cards_of_player(0))
            winners = self._get_winners(showdown_players=showdown_hands, payouts=info['payouts'])
            money_collected = self._get_money_collected(payouts=info['payouts'])
            self._update_cumulative_stacks(money_collected)
            board = _make_board(env.env.cards2str(env.env.board))
            poker_episodes.append(PokerEpisode(date=DEFAULT_DATE,
                                               hand_id=ep_id,
                                               variant=DEFAULT_VARIANT,
                                               currency_symbol=DEFAULT_CURRENCY,
                                               num_players=num_players,
                                               blinds=blinds,
                                               ante=ante,
                                               player_stacks=player_stacks,
                                               btn_idx=button_index,
                                               board_cards=board,
                                               actions_total=actions_total,
                                               winners=winners,
                                               showdown_hands=showdown_hands,
                                               money_collected=money_collected))
            debug = 'set breakpoint here'
        print(total_actions_dict)
        return poker_episodes
