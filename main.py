import argparse
import dqn
from env_maker import make_env
import torch
import torch.nn as nn
import random
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from tqdm import tqdm
import pickle 
from super_mario_bros.gym_super_mario_bros.actions import RIGHT_ONLY
import gym
import numpy as np
import collections 
import cv2
import matplotlib.pyplot as plt
from super_mario_bros.gym_super_mario_bros._app import cli

def run(training_mode, pretrained, env, eps):
    """
    """
    env = gym.make(env)
    env = make_env(env)  # pre-process env for deep learning
    observation_space = env.observation_space.shape
    action_space = env.action_space.n
    agent = dqn.DQNAgent(state_space=observation_space,
                        action_space=action_space,
                        max_memory_size=30000,
                        batch_size=32,
                        gamma=0.90,
                        lr=0.00025,
                        dropout=0.,
                        exploration_max=1.0,
                        exploration_min=0.02,
                        exploration_decay=0.99,
                        double_dq=True,
                        pretrained=pretrained)
    
    num_episodes = eps
    env.reset()
    total_rewards = []
    
    for ep_num in tqdm(range(num_episodes)):
        state = env.reset()
        state = torch.Tensor([state])
        total_reward = 0
        steps = 0
        while True:
            if not training_mode:
                show_state(env, ep_num)
            action = agent.act(state)
            steps += 1
            
            state_next, reward, terminal, info = env.step(int(action[0]))
            total_reward += reward
            state_next = torch.Tensor([state_next])
            reward = torch.tensor([reward]).unsqueeze(0)
            
            terminal = torch.tensor([int(terminal)]).unsqueeze(0)
            
            if training_mode:
                agent.remember(state, action, reward, state_next, terminal)
                agent.experience_replay()
            
            state = state_next
            if terminal:
                break
        
        total_rewards.append(total_reward)

        print("Total reward after episode {} is {}".format(ep_num + 1, total_rewards[-1]))
        num_episodes += 1      
    
    if training_mode:
        with open("ending_position.pkl", "wb") as f:
            pickle.dump(agent.ending_position, f)
        with open("num_in_queue.pkl", "wb") as f:
            pickle.dump(agent.num_in_queue, f)
        with open("total_rewards.pkl", "wb") as f:
            pickle.dump(total_rewards, f)
        if agent.double_dq:
            torch.save(agent.local_net.state_dict(), "dq1.pt")
            torch.save(agent.target_net.state_dict(), "dq2.pt")
        else:
            torch.save(agent.dqn.state_dict(), "dq.pt")  
        torch.save(agent.STATE_MEM,  "STATE_MEM.pt")
        torch.save(agent.ACTION_MEM, "ACTION_MEM.pt")
        torch.save(agent.REWARD_MEM, "REWARD_MEM.pt")
        torch.save(agent.STATE2_MEM, "STATE2_MEM.pt")
        torch.save(agent.DONE_MEM,   "DONE_MEM.pt")
    
    env.close()
    
    if num_episodes > 500:
        plt.title("Episodes trained vs. Average Rewards (per 500 eps)")
        plt.plot([0 for _ in range(500)] + 
                 np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
        plt.show()

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def main(args):
    """
    [INSERT DOCUMENTATION]
    """
    if args.mode == 'agent':
        print("hurray")
        args.env = 'SuperMarioBros-1-1-v0'
        run(training_mode=args.training_mode, pretrained=args.pretrained, env=args.env, eps=args.episodes)
    else:
        cli.main()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs SuperMarioBrosQL")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training_mode', '-tm',
        type=bool,
        default=True,
        choices=[True, False],
        help='True if we want to train the DQN agent'
    )
    parser.add_argument('--pretrained', '-pre',
        type=bool,
        default=False,
        choices=[True, False],
        help='True if DQN agent is run on pretrained weights'
    )
    parser.add_argument('--episodes', '-eps',
        type=int,
        help='The number of agent-environment interactions from initial to final states'
    )
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBrosRandomStages-v0',
        help='The name of the environment to play'
    )
    parser.add_argument('--mode', '-m',
        type=str,
        default='human',
        choices=['human', 'random', 'agent'],
        help='The execution mode for the emulation'
    )
    parser.add_argument('--actionspace', '-a',
        type=str,
        default='nes',
        choices=['nes', 'right', 'simple', 'complex'],
        help='the action space wrapper to use'
    )
    parser.add_argument('--steps', '-s',
        type=int,
        default=500,
        help='The number of random steps to take.',
    )
    parser.add_argument('--stages', '-S',
        type=str,
        default='1-1',
        nargs='+',
        help='The random stages to sample from for a random stage env'
    )
    args = parser.parse_args()
    main(args)
