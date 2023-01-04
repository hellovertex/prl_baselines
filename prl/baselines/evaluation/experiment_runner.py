import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.evaluation.core.experiment import PokerExperiment, DEFAULT_DATE, DEFAULT_VARIANT, DEFAULT_CURRENCY
from prl.baselines.evaluation.core.runner import ExperimentRunner
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, ActionType, \
    PlayerWithCards, PlayerWinningsCollected, Action, PokerEpisode
from prl.baselines.utils.num_parsers import parse_num

POSITIONS_HEADS_UP = ['btn', 'bb']  # button is small blind in Heads Up situations
POSITIONS = ['btn', 'sb', 'bb', 'utg', 'mp', 'co']
STAGES = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]
ACTION_TYPES = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE]


def _make_board(board: str) -> str:
    return f"[{board.replace(',', '').rstrip()}]"


@dataclass
class PlayerWithCardsAndPosition:
    cards: str  # '[Ah Jd]' <-- encoded like this, due to compatibility with parsers
    name: str
    seat: Optional[str] = None
    position: Optional[str] = None  # c.f. Positions6Max


class PokerExperimentRunner(ExperimentRunner):
    # run experiments using
    def __init__(self):
        self.player_names = None
        self.money_from_last_round = None
        self.iter_action_plan = None
        self.participants = None
        self.run_from_action_plan = None
        self.total_actions_dict = None
        self.num_players = None
        self._times_taken_to_compute_action = []
        self._times_taken_to_step_env = []

    def _get_player_stacks(self, seats, num_players, btn_idx) -> List[PlayerStack]:
        """ Stacks at the beginning of every episode, not during or after."""
        # seats are gotten from the environment and start with the button
        # our agent who has the button can be at a different index than 0 in our agent list
        # We must roll the seats, such that [BTN, ...] -> [...,BTN,...]
        #
        player_stacks = []
        seats = list(np.roll(seats, btn_idx))
        for seat_id, seat in enumerate(seats):
            player_name = f'{self.player_names[seat_id]}'
            seat_display_name = f'Seat {seat_id + 1}'  # index starts at 1
            stack = "$" + str(seat.starting_stack_this_episode)
            player_stacks.append(PlayerStack(seat_display_name,
                                             player_name,
                                             stack))
        return player_stacks

    def _get_money_collected(self,
                             env,
                             initial_stacks: List[PlayerStack],
                             payouts: Dict[int, str],
                             btn_idx: int) -> List[PlayerWinningsCollected]:
        money_collected = []
        # env.env.seats always start with the button
        # initial_stacks are our fixed agents, where anyone could have the button
        # agent_idx relative to button
        for pid, payout in payouts.items():
            # Note: indices are relative to button
            gain = env.env.seats[pid].stack - int(initial_stacks[(pid - btn_idx) % self.num_players].stack[1:])
            # Note: indices are reversed relative to our agents
            money_collected.append(
                PlayerWinningsCollected(player_name=f'{self.player_names[(pid + btn_idx) % self.num_players]}',
                                        collected="$" + str(int(gain)),
                                        rake=None))
        return money_collected

    def _get_blinds(self, num_players, btn_idx, bb, sb) -> List[Blind]:
        """
        observing player sits relative to button. this offset is given by
        # >>> obs[COLS.Btn_idx]
        the button position determines which players post the small and the big blind.
        For games with more than 2 pl. the small blind is posted by the player who acts after the button.
        When only two players remain, the button posts the small blind.
        """
        # btn_offset = int(obs[COLS.Btn_idx])
        sb_name = f"{self.player_names[(1 + btn_idx) % num_players]}"
        bb_name = f"{self.player_names[(2 + btn_idx) % num_players]}"
        if num_players == 2:
            sb_name = f"{self.player_names[btn_idx]}"
            bb_name = f"{self.player_names[(1 + btn_idx) % num_players]}"

        sb_amount = "$" + str(parse_num(sb))
        bb_amount = "$" + str(parse_num(bb))

        return [Blind(sb_name, 'small blind', sb_amount),
                Blind(bb_name, 'big blind', bb_amount)]

    def _get_winners(self, showdown_players: List[PlayerWithCards],
                     payouts: dict,
                     btn_idx: int) -> List[PlayerWithCards]:
        winners = []
        for pid, money_won in payouts.items():
            for p in showdown_players:
                # look for winning player in showdown players and append its stats to winners
                if f'{self.player_names[(pid + btn_idx) % self.num_players]}' == p.name:
                    winners.append(PlayerWithCards(name=p.name,
                                                   cards=p.cards))
        return winners

    @staticmethod
    def _parse_cards(cards):
        # In: '3h, 9d, '
        # Out: '[Ah Jd]'
        tokens = cards.split(',')  # ['3h', ' 9d', ' ']
        c0 = tokens[0].replace(' ', '')
        c1 = tokens[1].replace(' ', '')
        return f'[{c0} {c1}]'

    def _get_remaining_players(self, env, btn_idx) -> List[PlayerWithCards]:
        # make player cards
        remaining_players = []
        for i in range(env.env.N_SEATS):
            # If player did not fold and still has money (is active and does not sit out)
            # append it to the remaining players
            if not env.env.seats[i].folded_this_episode:
                if env.env.seats[i].stack > 0 or env.env.seats[i].is_allin:
                    cards = env.env.cards2str(env.env.get_hole_cards_of_player(i))
                    remaining_players.append(
                        PlayerWithCards(name=f'{self.player_names[(i + btn_idx) % self.num_players]}',
                                        cards=self._parse_cards(cards)))
        return remaining_players

    def _has_folded(self, p: PlayerWithCards, last_action, btn_idx) -> bool:
        return last_action[0] == 0 and f'{self.player_names[(last_action[2] + btn_idx) % self.num_players]}' == p.name

    def _get_showdown_hands(self,
                            remaining_players: List[PlayerWithCards],
                            last_action,
                            btn_idx) -> List[PlayerWithCards]:
        showdown_hands = []
        for p in remaining_players:
            if not self._has_folded(p, last_action, btn_idx):
                showdown_hands.append(p)
        return showdown_hands

    def post_blinds(self, obs, num_players, btn_idx, env):
        blinds = self._get_blinds(num_players, btn_idx, env.env.BIG_BLIND, env.env.SMALL_BLIND)
        ante = '$0.00'
        assert obs[COLS.Ante] == 0  # games with ante not supported
        return ante, blinds

    def _run_game(self, env, initial_observation, btn_idx):
        done = False
        showdown_hands = None
        info = {'player_hands': []}  # monkey patched
        # todo: populate player_hands with default hands
        observation = initial_observation
        # determine who goes first
        agent_idx = btn_idx if self.num_players < 4 else (btn_idx + 3) % self.num_players
        # --- SOURCE OF ACTIONS ---
        actions_total = {'preflop': [],
                         'flop': [],
                         'turn': [],
                         'river': [],
                         'as_sequence': []}
        while not done:
            # -------- ACT -----------
            t0 = time.time()
            if self.run_from_action_plan:
                a = next(self.iter_actions)
                action = a.action_type, a.raise_amount
                # if isinstance(action, tuple):
                #     action =
            else:
                action_vec = self.participants[agent_idx].agent.act(observation)
                action = int(action_vec[0][0].numpy())
                action = env.int_action_to_tuple_action(action)
            self._times_taken_to_compute_action.append(time.time() - t0)
            # -------- STEP ENVIRONMENT -----------
            remaining_players = self._get_remaining_players(env, btn_idx)
            stage = Poker.INT2STRING_ROUND[env.env.current_round]
            current_bet_before_action = env.current_player.current_bet
            t0 = time.time()
            obs, _, done, info = env.step(action)
            self._times_taken_to_step_env.append(time.time() - t0)
            # -------- RECORD LAST ACTION ---------
            a = env.env.last_action
            raise_amount = max(a[1], 0)  # if a[0] == ActionType.RAISE else -1
            # set raise amount to zero if it is preflop, the acting player is the big blind
            # and the action is call with amount equal to big blind
            # then we actually have to make it a check
            if a[2] == env.env.BB_POS and a[1] == env.env.BIG_BLIND and a[0] == 1:
                if stage == 'preflop':
                    raise_amount = 0
            episode_action = Action(stage=stage,
                                    player_name=f'{self.player_names[agent_idx]}',
                                    action_type=ACTION_TYPES[a[0]],
                                    raise_amount=raise_amount,
                                    info={
                                        'total_call_or_bet_amt_minus_current_bet': raise_amount - current_bet_before_action
                                    })

            actions_total[stage].append(episode_action)
            actions_total['as_sequence'].append(episode_action)
            self.total_actions_dict[a[0]] += 1
            # if not done, prepare next turn
            if done:
                showdown_hands = self._get_showdown_hands(remaining_players, a, btn_idx)

                break
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}

            # -------- SET NEXT AGENT -----------
            # btn_idx is the index of the button relative to our agent list
            # seat_id is relative to the button
            # so to translate seat_id to agent_idx, we must roll this by btn_idx
            agent_idx = (btn_idx + env.env.current_player.seat_id) % self.num_players

        return actions_total, showdown_hands, info

    def get_player_hands(self, env, btn_idx):
        player_cards = [env.cards2str(env.get_hole_cards_of_player(i)) for i in range(len(env.seats))]
        hands = ['[' + h.rstrip()[:-1] + ']' for h in player_cards]
        hands = [h.replace(',', '') for h in hands]
        agent_hands = list(np.roll(hands, btn_idx))
        positions = [Positions6Max(i) for i in list(np.roll([j for j in range(len(env.seats))], btn_idx))]
        player_with_cards_and_positions = []
        for seat_id, hand in enumerate(agent_hands):
            cards = hands[seat_id]
            name = f'{self.player_names[seat_id]}'
            position = f'{positions[seat_id].name}'
            player_with_cards_and_positions.append(PlayerWithCardsAndPosition(cards=cards,
                                                                              name=name,
                                                                              position=position))
        return player_with_cards_and_positions

    def _run_single_episode(self,
                            env,
                            env_reset_config,
                            num_players,
                            btn_idx,
                            ep_id) -> PokerEpisode:
        # --- SETUP AND RESET ENVIRONMENT ---
        obs, _, done, _ = env.reset(env_reset_config)
        initial_player_stacks = self._get_player_stacks(env.env.seats,  # before reset
                                                        num_players,
                                                        btn_idx)
        ante, blinds = self.post_blinds(obs, num_players, btn_idx, env)

        if self.run_from_action_plan:
            self.iter_actions = iter(next(self.iter_action_plan))

        # --- RUN GAME LOOP ---
        legal_moves = env.env.get_legal_actions()
        initial_observation = {'obs': [obs], 'legal_moves': [legal_moves]}
        actions_total, showdown_hands, info = self._run_game(env,
                                                             initial_observation,
                                                             btn_idx)
        assert showdown_hands is not None
        assert info is not None

        winners = self._get_winners(showdown_players=showdown_hands,
                                    payouts=info['payouts'],
                                    btn_idx=btn_idx)
        money_collected = self._get_money_collected(env, initial_player_stacks, payouts=info['payouts'],
                                                    btn_idx=btn_idx)

        board = _make_board(env.env.cards2str(env.env.board))

        # player_hands: List[PlayerWithCardsAndPosition] = self.get_player_hands(env.env)
        player_hands = self.get_player_hands(env.env, btn_idx)
        return PokerEpisode(date=DEFAULT_DATE,
                            hand_id=ep_id,
                            variant=DEFAULT_VARIANT,
                            currency_symbol=DEFAULT_CURRENCY,
                            num_players=num_players,
                            blinds=blinds,
                            ante=ante,
                            player_stacks=initial_player_stacks,
                            btn_idx=btn_idx,
                            board_cards=board,
                            actions_total=actions_total,
                            winners=winners,
                            showdown_hands=showdown_hands,
                            money_collected=money_collected,
                            info={'player_hands': player_hands})

    def _run_episodes(self, experiment: PokerExperiment) -> List[PokerEpisode]:

        env = experiment.env
        env_reset_config = experiment.env_reset_config
        num_players = experiment.num_players
        n_episodes = experiment.max_episodes

        poker_episodes = []
        new_starting_stacks = None

        # ----- RUN EPISODES -----
        for ep_id in range(n_episodes):
            print(ep_id)
            # always move button to the next player
            btn_idx = ep_id % num_players
            # along with all the stacks that are shifted 1 to the right
            # -------- Reset environment ------------
            episode = self._run_single_episode(env=env,
                                               env_reset_config=env_reset_config,
                                               num_players=num_players,
                                               btn_idx=btn_idx,
                                               ep_id=ep_id)
            # rotate stacks, such that next player will be at 0
            # [BTN UTG SB BB MP CO] will become [UTG SB BB MP CO BTN]
            new_starting_stacks = np.roll(
                [p.stack for p in env.env.seats],  # current stacks before advancing button
                -1  # button has advanced by 1, roll back its index relative to seats
            ).astype(int).tolist()
            env.env.set_stack_size(new_starting_stacks)
            poker_episodes.append(episode)
        print(self.total_actions_dict)
        print(f'Average time taken computing actions: '
              f'{np.mean(self._times_taken_to_compute_action) * 1000} ms')
        print(f'Average time taken stepping environment: '
              f'{np.mean(self._times_taken_to_step_env) * 1000} ms')
        return poker_episodes

    def run(self, experiment: PokerExperiment) -> List[PokerEpisode]:
        # for testing,
        # it should be possible to
        # 1) provide hands and boards
        # 2) provide action sequence
        #
        self.num_players = experiment.num_players
        self.total_actions_dict = {ActionType.FOLD.value: 0,
                                   ActionType.CHECK_CALL.value: 0,
                                   ActionType.RAISE.value: 0}
        # Actions can be generated via
        # 1) Using agents <--> participant.agent.act
        # 2) Using a predefined action list <--> next() on an iterator over actions
        if experiment.from_action_plan:
            self.run_from_action_plan = True
            self.participants = []
            self.iter_action_plan = iter(experiment.from_action_plan)
        else:
            self.run_from_action_plan = False
            self.participants = experiment.participants
            self.iter_action_plan = iter([])
            self.player_names = [p.name for p in self.participants]
        return self._run_episodes(experiment)
