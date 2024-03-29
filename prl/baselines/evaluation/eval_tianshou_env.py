from prl.environment.Wrappers.base import ActionSpace

from prl.baselines.agents.dummy_agents import DummyAgentFold, DummyAgentCall, DummyAgentAllIn
from prl.baselines.evaluation.utils import get_reset_config, pretty_print
from prl.baselines.examples.examples_tianshou_env import make_default_tianshou_env

mc_model_ckpt_path = "/home/sascha/Documents/github.com/prl_baselines/data/ckpt/ckpt.pt"
agent_names = ["Bob_0", "Tina_1", "Alice_2", "Hans_3"]
agent_names2 = ["Hans_3", "Bob_0", "Tina_1", "Alice_2"]
agent_names3 = ["Alice_2", "Hans_3", "Bob_0", "Tina_1"]
agent_names3 = ["Tina_1", "Alice_2", "Hans_3", "Bob_0"]

player_hands = ['[6s 6d]', '[9s 9d]', '[Jd Js]', '[Ks Kd]']
board = '[6h Ts Td 9c Jc]'
env = make_default_tianshou_env(num_players=len(agent_names),
                                agents=agent_names)
agents = [
    DummyAgentAllIn,  # Bob
    DummyAgentCall,  # Tina
    DummyAgentFold,  # Alice
    DummyAgentCall  # Hans
]
state_dict = get_reset_config(player_hands, board)
options = {'reset_config': state_dict}
i = 0
for epoch in range(4):
    obs_dict = env.reset(options=options)
    agent_id = obs_dict['agent_id']
    legal_moves = obs_dict['mask']
    obs = obs_dict['obs']
    while True:
        i = agent_names.index(agent_id)
        action = agents[i].act(obs, legal_moves)
        if obs_dict['mask'][8] == 1:
            action = ActionSpace.NoOp
        print(f'AGNET_ID = {agent_id}')
        try:
            pretty_print(i, obs, action)
        except Exception:
            pass
        print(f'legal_moves = {legal_moves}')
        obs_dict, cum_reward, terminated, truncated, info = env.step(action)
        rews = cum_reward
        agent_id = obs_dict['agent_id']
        print(f'AGENT_ID', agent_id)
        obs = obs_dict['obs']
        print(f'GOT REWARD {cum_reward}')
        if terminated:
            print('------------------------------------')
            print('ROUND OVER -- RESETTING ENVIRONMENT')
            print('------------------------------------')
            # if epoch == 0:
            #     assert rews[1] > 0  # Tina wins with 9s 9d
            # if epoch == 1:
            #     assert rews[1] > 0  # Tina wins with Jd Js
            # if epoch == 2:
            #     assert rews[0] > 0  # Bob wins with Jd Js
            if epoch == 0:
                assert rews[1] > 0  # Tina gewinnt
            if epoch == 1:
                assert rews[1] > 0  # Tina wins with Jd Js
            if epoch == 2:
                assert rews[0] > 0  # Bob wins with Jd Js, Hans was last to act, so it is offset 1 to bob
            if epoch == 3:
                assert rews[3] > 0  # Hans wins with J J
            break


