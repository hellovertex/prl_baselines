# from typing import List, Type
#
# from prl.environment.Wrappers.base import ActionSpace
#
# from prl.baselines.evaluation.v2.eval_agent import EvalAgentBase
#
#
# def get_rainbow_agent(ckpt_path):
#     params = {'device': "cuda",
#               'lr': 1e-6,
#               'load_from_ckpt': ckpt_path,
#               'num_atoms': 51,
#               'noisy_std': 0.1,
#               'v_min': -6,
#               'v_max': 6,
#               'estimation_step': 3,
#               'target_update_freq': 500  # training steps
#               }
#
#     rainbow_config = get_rainbow_config(params)
#     rainbow = RainbowPolicy(**rainbow_config)
#
#
# class _RandomAgent:
#     def act(obs):
#         return random.randint(0, 2)
#
#
# class ComputeHandRanges:
#     def __init__(self):
#         pass
#
#     def is_first_action(self):
#         pass
#
#     def compute(self, agents: List[Type[EvalAgentBase]]):
#         # 1. create environment
#         while True:
#             # current_position = ...
#             # current_cards = ...
#             action = None
#             curr_agent = None
#             if action == ActionSpace.FOLD:
#                 if not self.is_first_action(curr_agent):
#                     # add card counter for position and cards
#                     pass
#             # step each agent
#             # get cards and count
#             # if current round > preflop: break
#
#         pass
#
#
# if __name__ == '__main__':
#     # todo: load from .ckpt file
#     ckpt_path = '/home/sascha/Documents/github.com/prl_baselines/data/checkpoints/vs_calling_station/ckpt.pt'
#     num_players = 2
#     agents = [get_rainbow_agent(ckpt_path)] + [
#         _RandomAgent() for _ in range(num_players - 1)
#     ]
#     ranges = ComputeHandRanges()
#     result = ranges.compute(agents)  # json formatted
#
