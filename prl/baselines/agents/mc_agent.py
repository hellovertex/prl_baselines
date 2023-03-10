from random import random
from typing import Tuple, List, Union

import numpy as np
import torch
from prl.environment.Wrappers.augment import AugmentedObservationFeatureColumns as cols
from prl.environment.Wrappers.base import ActionSpace
from tianshou.utils.net.common import MLP
from torch import softmax

from prl.baselines.cpp_hand_evaluator.monte_carlo import HandEvaluator_MonteCarlo
from prl.baselines.cpp_hand_evaluator.rank import dict_str_to_sk

IDX_C0_0 = 167  # feature_names.index('0th_player_card_0_rank_0')
IDX_C0_1 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_0 = 184  # feature_names.index('0th_player_card_1_rank_0')
IDX_C1_1 = 201  # feature_names.index('1th_player_card_0_rank_0')
IDX_BOARD_START = 82  #
IDX_BOARD_END = 167  #
CARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c'])
BOARD_BITS_TO_STR = np.array(['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
                              'h', 'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T',
                              'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2', '3', '4', '5', '6',
                              '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h', 'd', 's', 'c', '2',
                              '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', 'h',
                              'd', 's', 'c', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J',
                              'Q', 'K', 'A', 'h', 'd', 's', 'c'])
RANK = 0
SUITE = 1


class MCAgent:

    def __init__(self, ckpt_path, num_players, device="cuda"):
        """ckpt path may be something like ./ckpt/ckpt.pt"""
        self._mc_iters = 5000
        self.device = device
        self.num_players = num_players
        self.tightness = .1  # percentage of hands played, "played" meaning not folded immediately
        self.acceptance_threshold = 0.5  # minimum certainty of probability of network to perform action
        self._card_evaluator = HandEvaluator_MonteCarlo()
        self.load_model(ckpt_path)

    def load_model(self, ckpt_path):
        input_dim = 564
        classes = [ActionSpace.FOLD,
                   ActionSpace.CHECK_CALL,  # CHECK IS INCLUDED in CHECK_CALL
                   ActionSpace.RAISE_MIN_OR_3BB,
                   ActionSpace.RAISE_6_BB,
                   ActionSpace.RAISE_10_BB,
                   ActionSpace.RAISE_20_BB,
                   ActionSpace.RAISE_50_BB,
                   ActionSpace.RAISE_ALL_IN]
        hidden_dim = [256]
        output_dim = len(classes)
        self._model = MLP(input_dim, output_dim, hidden_dim).to(self.device)
        ckpt = torch.load(ckpt_path,
                          map_location=self.device)
        self._model.load_state_dict(ckpt['net'])
        self._model.eval()


    @staticmethod
    def card_bit_mask_to_int(c0: np.array, c1: np.array, board_mask: np.array) -> Tuple[List[int], List[int]]:
        c0 = c0.cpu()
        c1 = c1.cpu()
        board_mask = board_mask.cpu()
        c0_1d = dict_str_to_sk[CARD_BITS_TO_STR[c0][RANK] + CARD_BITS_TO_STR[c0][SUITE]]
        c1_1d = dict_str_to_sk[CARD_BITS_TO_STR[c1][RANK] + CARD_BITS_TO_STR[c1][SUITE]]
        board = BOARD_BITS_TO_STR[board_mask.bool()]
        # board = array(['A', 'c', '2', 'h', '8', 'd'], dtype='<U1')
        board_cards = []
        for i in range(0, int(torch.sum(board_mask)) - 1, 2):  # sum is 6,8,10 for flop turn river resp.
            board_cards.append(dict_str_to_sk[board[i] + board[i + 1]])

        return [c0_1d, c1_1d], board_cards

    def look_at_cards(self, obs: np.array) -> Tuple[List[int], List[int]]:
        c0_bits = obs[0][IDX_C0_0:IDX_C0_1].bool()
        c1_bits = obs[0][IDX_C1_0:IDX_C1_1].bool()
        board_bits = obs[0][IDX_BOARD_START:IDX_BOARD_END] # bit representation
        return self.card_bit_mask_to_int(c0_bits, c1_bits, board_bits)

    def _compute_action(self,
                        obs: Union[List, np.ndarray]):
        hero_cards_1d, board_cards_1d = self.look_at_cards(obs)
        # from cards get winning probabilityx
        mc_dict = self._card_evaluator.run_mc(hero_cards_1d,
                                              board_cards_1d,
                                              self.num_players,
                                              n_iter=self._mc_iters)
        # {won: 0, lost: 0, tied: 0}[
        win_prob = float(mc_dict['won'] / self._mc_iters)
        # todo: replace win_prob < .5
        total_to_call = obs[0][cols.Total_to_call]
        # if we have negative EV on calling/raising, we fold with high probability
        potsize = sum([
            obs[0][cols.Curr_bet_p1],
            obs[0][cols.Curr_bet_p2],
            obs[0][cols.Curr_bet_p3],
            obs[0][cols.Curr_bet_p4],
            obs[0][cols.Curr_bet_p5],
        ]) + obs[0][cols.Pot_amt]
        pot_odds = total_to_call / (potsize + total_to_call)
        # ignore pot odds preflop, we marginalize fold probability via acceptance threshold
        if not obs[0][cols.Round_preflop]:
            # fold when bad pot odds are not good enough according to MC simulation
            if win_prob < pot_odds:
                if random() > self.tightness:  # tightness is equal to % of hands played, e.g. 0.15
                    return ActionSpace.FOLD.value
        certainty = torch.max(softmax(self._logits, dim=1)).detach().item()
        # if the probability is high enough, we take the action suggested by the neural network
        if certainty > self.acceptance_threshold:
            assert len(self._predictions == 1)
            prediction = self._predictions[0]
            # if we are allowed, we raise otherwise we check
            # todo: should we add pseudo harmonic mapping here with train/eval modes?
            if self.next_legal_moves[min(int(prediction), 2)]:
                return int(prediction)
            elif ActionSpace.CHECK_CALL in self.next_legal_moves:
                return ActionSpace.CHECK_CALL.value
            else:
                return ActionSpace.FOLD.value
        else:
            return ActionSpace.FOLD.value
        # 3. pick argmax only if p(argmax) > 1-acceptance
        # 4. start with acceptance 1 and decrease until
        # 5. tighntess(mc_agent) == tightness(players) = n_hands_played/n_total_hands
        # 6. n_hands_played should be hands that are not immediately folded preflop
        # 7. gotta compute them from the players datasets

    def compute_action(self, obs: np.ndarray, legal_moves) -> int:
        self.next_legal_moves = legal_moves
        if not type(obs) == torch.Tensor:
            obs = torch.Tensor(np.array([obs]))
        self._logits = self._model(obs)
        self._predictions = torch.argmax(self._logits, dim=1)
        action = self._compute_action(obs)
        return action

    def act(self, obs, legal_moves, report_probas=False):
        """Wrapper for compute action only when evaluating in poke--> single env"""
        if type(obs) == dict:
            legal_moves = np.array([0, 0, 0, 0, 0, 0])
            legal_moves[obs['legal_moves'][0]] += 1
            if legal_moves[2] == 1:
                legal_moves[[3, 4, 5]] = 1
            obs = obs['obs']
            if type(obs) == list:
                obs = np.array(obs)[0]
        if report_probas:
            action = self.compute_action(obs, legal_moves)
            return action, torch.softmax(self._logits, dim=1).detach()
        return self.compute_action(obs, legal_moves)

