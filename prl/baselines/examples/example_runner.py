import time
from dataclasses import dataclass
from typing import List, Dict, Optional

import numpy as np
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.evaluation.core.experiment import PokerExperiment, DEFAULT_DATE, DEFAULT_VARIANT, DEFAULT_CURRENCY, \
    PokerExperiment_EarlyStopping
from prl.baselines.evaluation.core.runner import ExperimentRunner
from prl.baselines.supervised_learning.data_acquisition.core.encoder import Positions6Max
from prl.baselines.supervised_learning.data_acquisition.core.parser import Blind, PlayerStack, ActionType, \
    PlayerWithCards, PlayerWinningsCollected, Action, PokerEpisode
from prl.baselines.utils.num_parsers import parse_num

POSITIONS_HEADS_UP = ['btn', 'bb']  # button is small blind in Heads Up situations
POSITIONS = ['btn', 'sb', 'bb', 'utg', 'mp', 'co']
STAGES = [Poker.PREFLOP, Poker.FLOP, Poker.TURN, Poker.RIVER]
ACTION_TYPES = [ActionType.FOLD, ActionType.CHECK_CALL, ActionType.RAISE]




@dataclass
class PlayerWithCardsAndPosition:
    cards: str  # '[Ah Jd]' <-- encoded like this, due to compatibility with parsers
    name: str
    seat: Optional[str] = None
    position: Optional[str] = None  # c.f. Positions6Max


class PokerExperimentRunner(ExperimentRunner):
    # run experiments using
    def __init__(self):
        self.agent_winnings = None
        self.backend = None
        self.agent_map = None
        self.env_reset_config = None
        self.env = None
        self.player_names = None
        self.money_from_last_round = None
        self.iter_action_plan = None
        self.participants = None
        self.run_from_action_plan = None
        self.total_actions_dict = None
        self.num_players = None
        self._times_taken_to_compute_action = []
        self._times_taken_to_step_env = []

    def _make_board(self) -> str:
        board = self.backend.cards2str(self.backend.board)
        return f"[{board.replace(',', '').rstrip()}]"

    def _get_starting_stacks_relative_to_agents(self) -> List[PlayerStack]:
        """ Stacks at the beginning of every episode, not during or after."""
        # seats are gotten from the environment and start with the button
        # our agent who has the button can be at a different index than 0 in our agent list
        # We must roll the seats, such that [BTN, ...] -> [...,BTN,...]
        #
        player_stacks = []
        # relative to agents --> roll env->agent_list
        for seat_id, seat in enumerate(self.env.seats):
            agent_id = self.agent_map[seat_id]
            player_name = f'{self.player_names[agent_id]}'
            seat_display_name = f'Seat {agent_id + 1}'  # index starts at 1
            stack = "$" + str(seat.starting_stack_this_episode)
            player_stacks.append(PlayerStack(seat_display_name,
                                             player_name,
                                             stack))
        return player_stacks

    def _get_money_collected(self,
                             initial_stacks: List[PlayerStack],
                             payouts: Dict[int, str]
                             ) -> List[PlayerWinningsCollected]:
        money_collected = []
        # env.env.seats always start with the button
        # initial_stacks are our fixed agents, where anyone could have the button
        # agent_idx relative to button
        for pid, payout in payouts.items():
            agent_id = self.agent_map[pid]
            # Note: indices are relative to button
            stack_t0 = self.backend.seats[pid].starting_stack_this_episode
            stack_t1 = self.backend.seats[pid].stack
            # e.g. bb will have gain=20050 - 20000=50 if sb folded to bb
            # in this case sb will have gain=19950 - 20000 = - 50
            # but since sb is not in payouts, the gain must be non-negative
            gain = stack_t1 - stack_t0
            assert not gain < 0
            money_collected.append(
                PlayerWinningsCollected(player_name=f'{self.player_names[agent_id]}',
                                        collected="$" + str(int(gain)),
                                        rake=None))
        return money_collected

    def _get_blinds(self) -> List[Blind]:
        """
        observing player sits relative to button. this offset is given by
        # >>> obs[COLS.Btn_idx]
        the button position determines which players post the small and the big blind.
        For games with more than 2 pl. the small blind is posted by the player who acts after the button.
        When only two players remain, the button posts the small blind.
        """
        agent_id_btn = self.agent_map[Positions6Max.BTN]
        agent_id_sb = self.agent_map[Positions6Max.SB]
        agent_id_bb = self.agent_map[Positions6Max.BB]
        # btn_offset = int(obs[COLS.Btn_idx])
        sb_name = f"{self.player_names[agent_id_sb]}"
        bb_name = f"{self.player_names[agent_id_bb]}"
        if self.num_players == 2:
            # the bb assignment is not a mistake
            # in heads up, the bb has index 1 which is the small blind for >2 players
            sb_name = f"{self.player_names[agent_id_btn]}"
            bb_name = f"{self.player_names[agent_id_sb]}"

        sb_amount = "$" + str(parse_num(self.backend.SMALL_BLIND))
        bb_amount = "$" + str(parse_num(self.backend.BIG_BLIND))

        return [Blind(sb_name, 'small blind', sb_amount),
                Blind(bb_name, 'big blind', bb_amount)]

    def _get_winners(self,
                     showdown_hands: List[PlayerWithCards],
                     payouts: dict) -> List[PlayerWithCards]:
        winners = []
        for pid, money_won in payouts.items():
            for p in showdown_hands:
                agent_id = self.agent_map[pid]
                # look for winning player in showdown players and append its stats to winners
                if f'{self.player_names[agent_id]}' == p.name:
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

    def _get_remaining_players(self) -> List[PlayerWithCards]:
        # make player cards
        remaining_players = []
        seats = self.backend.seats
        for i in range(self.backend.N_SEATS):
            agent_id = self.agent_map[i]
            if not seats[i].folded_this_episode:
                if seats[i].stack > 0 or seats[i].is_allin:
                    cards = self.backend.env.cards2str(self.backend.get_hole_cards_of_player(i))
                    remaining_players.append(
                        PlayerWithCards(name=f'{self.player_names[agent_id]}',
                                        cards=self._parse_cards(cards)))
        return remaining_players

    def _has_folded_last_turn(self, p: PlayerWithCards, last_action, btn_idx) -> bool:
        return last_action[0] == 0 and f'{self.player_names[(last_action[2] + btn_idx) % self.num_players]}' == p.name

    def _get_showdown_hands(self,
                            remaining_players: List[PlayerWithCards],
                            last_action,
                            btn_idx) -> List[PlayerWithCards]:
        showdown_hands = []
        for p in remaining_players:
            if not self._has_folded_last_turn(p, last_action, btn_idx):
                showdown_hands.append(p)
        return showdown_hands

    def post_blinds(self, obs):
        blinds = self._get_blinds()
        ante = '$0.00'
        assert obs[COLS.Ante] == 0  # games with ante not supported
        return ante, blinds

    def record_last_action(self, agent_idx, stage, current_bet_before_action):
        a = self.backend.last_action
        what, how_much, who = a
        # -1 is default for fold and check --> clip -1 to 0
        raise_amount = max(how_much, 0)
        bb_acted = who == self.backend.BB_POS
        quantity_equals_bb = how_much == self.backend.BIG_BLIND
        is_check_call = what == ActionType.CHECK_CALL
        # if it is preflop, the acting player is the big blind
        # and the action is call with amount equal to big blind
        # then we actually have to make it a check
        # instead of call by setting raise amount to zero
        if bb_acted and quantity_equals_bb and is_check_call:
            if stage == 'preflop':
                raise_amount = 0
        self.total_actions_dict[what] += 1
        return Action(stage=stage,
                                player_name=f'{self.player_names[agent_idx]}',
                                action_type=ACTION_TYPES[a[0]],
                                raise_amount=raise_amount,
                                info={
                                    'total_call_or_bet_amt_minus_current_bet': raise_amount - current_bet_before_action
                                })

    def _run_game(self, env, initial_observation, btn_idx):
        done = False
        showdown_hands = None
        info = {'player_hands': []}  # monkey patched
        observation = initial_observation
        # determine who goes first
        agent_idx = self.agent_map[Positions6Max.BTN] if self.num_players < 4 else Positions6Max.UTG
        # --- SOURCE OF ACTIONS ---
        actions_total = {'preflop': [],
                         'flop': [],
                         'turn': [],
                         'river': [],
                         'as_sequence': []}
        while not done:
            # -------------------------------------
            # --------------- ACT -----------------
            # -------------------------------------
            t0 = time.time()
            if self.run_from_action_plan:
                a = next(self.iter_actions)
                action = a.action_type, a.raise_amount
            else:
                action = self.participants[agent_idx].agent.act(observation, None)
            self._times_taken_to_compute_action.append(time.time() - t0)
            # -------------------------------------
            # -------- STEP ENVIRONMENT -----------
            # -------------------------------------
            remaining_players = self._get_remaining_players()
            stage = Poker.INT2STRING_ROUND[self.backend.current_round]
            current_bet_before_action = self.env.current_player.current_bet
            t0 = time.time()
            obs, _, done, info = env.step(action)
            self._times_taken_to_step_env.append(time.time() - t0)
            # -------------------------------------
            # -------- RECORD LAST ACTION ---------
            # -------------------------------------
            episode_action = self.record_last_action(agent_idx=agent_idx,
                                                     stage=stage,
                                                     current_bet_before_action=current_bet_before_action)
            actions_total[stage].append(episode_action)
            actions_total['as_sequence'].append(episode_action)

            if done:
                showdown_hands = self._get_showdown_hands(remaining_players, a, btn_idx)

            legal_moves = self.backend.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}

            # -------- SET NEXT AGENT -----------
            agent_idx = self.agent_map[self.backend.current_player.seat_id]

        return actions_total, showdown_hands, info

    def get_player_hands(self):
        env = self.backend
        player_cards = [env.cards2str(
            env.get_hole_cards_of_player(i)
        ) for i in range(len(env.seats))]
        hands = ['[' + h.rstrip()[:-1] + ']' for h in player_cards]
        hands = [h.replace(',', '') for h in hands]
        offset = self.agent_map[]
        agent_hands = list(np.roll(hands, btn_idx))
        positions = [Positions6Max(i) for i in list(np.roll([j for j in range(len(env.seats))], btn_idx))]
        player_with_cards_and_positions = []
        for seat_id, hand in enumerate(agent_hands):
            cards = agent_hands[seat_id]
            name = f'{self.player_names[seat_id]}'
            position = f'{positions[seat_id].name}'
            if position == 'SB' and len(agent_hands) == 2:
                position = 'BB'  # for two players only, BTN becomes SB and SB becomes BB
            player_with_cards_and_positions.append(PlayerWithCardsAndPosition(cards=cards,
                                                                              name=name,
                                                                              position=position))
        return player_with_cards_and_positions

    def _run_single_episode(self,
                            ep_id) -> PokerEpisode:
        # --- SETUP AND RESET ENVIRONMENT ---
        obs, _, done, _ = self.env.reset(self.env_reset_config)
        initial_player_stacks = self._get_starting_stacks_relative_to_agents()
        ante, blinds = self.post_blinds(obs)

        if self.run_from_action_plan:
            self.iter_actions = iter(next(self.iter_action_plan))

        # --- RUN GAME LOOP ---
        legal_moves = self.backend.get_legal_actions()
        initial_observation = {'obs': [obs], 'legal_moves': [legal_moves]}
        actions_total, showdown_hands, info = self._run_game(initial_observation)
        assert showdown_hands is not None
        assert info is not None

        winners = self._get_winners(showdown_hands=showdown_hands,
                                    payouts=info['payouts'])
        money_collected = self._get_money_collected(initial_player_stacks,
                                                    payouts=info['payouts'])

        board = self._make_board()

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
        exp = experiment
        poker_episodes = []
        # ----- RUN EPISODES -----
        for ep_id in range(exp.max_episodes):
            print(ep_id)

            # along with all the stacks that are shifted 1 to the right
            # -------- Reset environment ------------
            episode = self._run_single_episode(ep_id=ep_id)
            poker_episodes.append(episode)

            # always move button to the next player
            # [BTN UTG SB BB MP CO] will become [UTG SB BB MP CO BTN]
            shifted_indices = {}
            for rel_btn, agent_idx in self.agent_map.items():
                shifted_indices[rel_btn] = (agent_idx + 1) % exp.num_players
            self.agent_map = shifted_indices

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
        self.env = experiment.wrapped_env
        self.env_reset_config = experiment.env_reset_config
        self.backend = experiment.wrapped_env.env

        # maps backend indices to agents/players 
        # need this because we move the button but backend has 
        # button always at position 0
        self.agent_map = {}
        self.agent_winnings = {}
        for i in range(experiment.num_players):
            self.agent_map[i] = i
            self.agent_winnings[i] = 0
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
