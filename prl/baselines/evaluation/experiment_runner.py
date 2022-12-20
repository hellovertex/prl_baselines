from collections import OrderedDict
from typing import List, Dict

from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as COLS
from prl.environment.steinberger.PokerRL import Poker

from prl.baselines.evaluation.core.experiment import PokerExperiment, DEFAULT_DATE, DEFAULT_VARIANT, DEFAULT_CURRENCY
from prl.baselines.evaluation.core.runner import ExperimentRunner
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
    def _get_player_stacks(seats, num_players, offset) -> List[PlayerStack]:
        """ Stacks at the beginning of every episode, not during or after."""
        player_stacks = []
        for seat_id, seat in enumerate(seats):
            player_name = f'Player_{(seat_id + offset) % num_players}'
            seat_display_name = f'Seat {seat_id}'
            stack = "$" + str(seat.stack)
            player_stacks.append(PlayerStack(seat_display_name,
                                             player_name,
                                             stack))
        return player_stacks

    @staticmethod
    def _get_blinds(num_players, offset, bb, sb) -> List[Blind]:
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

        sb_amount = "$" + str(parse_num(sb))
        bb_amount = "$" + str(parse_num(bb))

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
                                                           collected="$" + str(int(payout)),
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
            self.money_from_last_round[p.player_name] += int(p.collected[1:])

    def _subtract_blinds_from_stacks(self, blinds: List[Blind]):
        for b in blinds:
            self.money_from_last_round[b.player_name] -= int(b.amount[1:])

    def _get_stack_sizes(self, experiment: PokerExperiment):
        # returns cumulative stack sizes if all players have stacks > 0
        # otherwise returns default stack sizes starting list
        default_stack_size = int(experiment.env.normalization)
        stack_sizes_list = []
        for pname, money in self.money_from_last_round.items():
            stack_sizes_list.append(int(default_stack_size) + int(money))
        for stack in stack_sizes_list:
            if stack < experiment.env.env.SMALL_BLIND:
                return [default_stack_size for _ in range(experiment.num_players)]
        return stack_sizes_list

    def post_blinds(self, obs, num_players, agent_idx, env):
        blinds = self._get_blinds(num_players, agent_idx, env.env.BIG_BLIND, env.env.SMALL_BLIND)
        self._subtract_blinds_from_stacks(blinds)
        ante = '$0.00'
        assert obs[COLS.Ante] == 0  # games with ante not supported
        return ante, blinds

    def _run_game(self):
        while not done:
            # -------- ACT -----------
            if experiment.from_action_plan:
                a = next(self.iter_actions)
                action = a.action_type, a.raise_amount
                # if isinstance(action, tuple):
                #     action =
            else:
                action_vec = self.participants[agent_idx].agent.act(observation)
                action = int(action_vec[0][0].numpy())
                # todo convert ActionSpace (integer repr) to Action (tuple repr)
                action = env.int_action_to_tuple_action(action)

            # -------- STEP ENVIRONMENT -----------
            remaining_players = self._get_remaining_players(env)
            stage = Poker.INT2STRING_ROUND[env.env.current_round]
            obs, _, done, info = env.step(action)
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
                                    player_name=f'Player_{agent_idx}',
                                    action_type=ACTION_TYPES[a[0]],
                                    raise_amount=raise_amount)
            self.money_from_last_round[f'Player_{agent_idx}'] -= raise_amount
            actions_total[stage].append(episode_action)
            actions_total['as_sequence'].append(episode_action)
            self.total_actions_dict[a[0]] += 1
            # if not done, prepare next turn
            if done:
                showdown_hands = self._get_showdown_hands(remaining_players, a)
                print(ep_id)
                break
            legal_moves = env.env.get_legal_actions()
            observation = {'obs': [obs], 'legal_moves': [legal_moves]}

            # -------- SET NEXT AGENT -----------
            agent_idx = (agent_idx + 1) % num_players


    def _run_single_episode(self, experiment, env, num_players, ep_id) -> PokerEpisode:
        # --- SETUP AND RESET ENVIRONMENT ---
        obs, _, done, _ = env.reset(experiment.env_config)
        agent_idx = button_index = ep_id % num_players  # always move button to the right
        showdown_hands = None
        ante, blinds = self.post_blinds(obs, num_players, agent_idx, env)
        initial_player_stacks = self._get_player_stacks(env.env.seats,
                                                num_players,
                                                agent_idx)
        # --- SOURCE OF ACTIONS ---
        actions_total = {'preflop': [],
                         'flop': [],
                         'turn': [],
                         'river': [],
                         'as_sequence': []}

        self.iter_actions = iter(next(self.iter_action_plan))

        # --- RUN GAME ---
        legal_moves = env.env.get_legal_actions()
        observation = {'obs': [obs], 'legal_moves': [legal_moves]}

            # env.env.cards2str(env.env.get_hole_cards_of_player(0))


        winners = self._get_winners(showdown_players=showdown_hands, payouts=info['payouts'])
        money_collected = self._get_money_collected(payouts=info['payouts'])
        self._update_cumulative_stacks(money_collected)
        board = _make_board(env.env.cards2str(env.env.board))
        return PokerEpisode(date=DEFAULT_DATE,
                            hand_id=ep_id,
                            variant=DEFAULT_VARIANT,
                            currency_symbol=DEFAULT_CURRENCY,
                            num_players=num_players,
                            blinds=blinds,
                            ante=ante,
                            player_stacks=initial_player_stacks,
                            btn_idx=button_index,
                            board_cards=board,
                            actions_total=actions_total,
                            winners=winners,
                            showdown_hands=showdown_hands,
                            money_collected=money_collected)

    def update_stack_sizes_from_last_round(self, experiment: PokerExperiment):
        updated_stacks = self._get_stack_sizes(experiment)
        return experiment.env.env.set_stack_size(updated_stacks)

    def _run_episodes(self, experiment: PokerExperiment) -> List[PokerEpisode]:

        env = experiment.env
        num_players = experiment.num_players
        n_episodes = experiment.max_episodes

        poker_episodes = []
        # ----- RUN EPISODES -----
        for ep_id in range(n_episodes):
            # -------- Reset environment ------------
            episode = self._run_single_episode(experiment, env, num_players, ep_id)
            self.update_stack_sizes_from_last_round(experiment)
            poker_episodes.append(episode)
            return poker_episodes

    def run(self, experiment: PokerExperiment) -> List[PokerEpisode]:
        # for testing,
        # it should be possible to
        # 1) provide hands and boards
        # 2) provide action sequence
        #
        self.total_actions_dict = {ActionType.FOLD.value: 0,
                                   ActionType.CHECK_CALL.value: 0,
                                   ActionType.RAISE.value: 0}
        # Actions can be generated via
        # 1) Using agents <--> participant.agent.act
        # 2) Using a predefined action list <--> next() on an iterator over actions
        if experiment.from_action_plan:
            self.participants = []
            self.iter_action_plan = iter(experiment.from_action_plan)
        else:
            self.participants = experiment.participants
            self.iter_action_plan = iter([])
        # the environment is reset after each episode
        # we keep track of the players earnings and losses
        # so we can reset the environment with the updated statck sizes
        self.money_from_last_round = OrderedDict()
        for i in range(experiment.num_players):
            self.money_from_last_round[f'Player_{i}'] = 0

        print(self.total_actions_dict)
        #self.env =

        return self._run_episodes(experiment)
