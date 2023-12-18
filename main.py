import argparse
import dqn
import os
import imageio
from PIL import Image
import PIL.ImageDraw as ImageDraw 
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

def run(args):
    """
    Runs the DDQN or DQN agent on Super Mario Bros. Can specify if training and/or pretrained.
    """
    env = args.env
    env = gym.make(env)
    eps = args.episodes
    num_episodes = eps
    if args.algorithm == "random":
        env = JoypadSpace(env, RIGHT_ONLY)
        frames = []
        if args.pretrained:
            total_rewards_pkl = append_file_name("total_rewards", args.algorithm, ".pkl")
            with open(total_rewards_pkl, 'rb') as f:
                total_rewards = pickle.load(f)
        else:
            total_rewards = []
        for ep_num in tqdm(range(num_episodes)):
            _ = env.reset()
            terminal = False
            progress = tqdm(range(args.steps))
            total_reward = 0
            for _ in progress:
                if terminal:
                    break
                action = env.action_space.sample()

                # Encode frames for output .mp4
                frame = env.render(mode='rgb_array')
                im = Image.fromarray(frame)

                drawer = ImageDraw.Draw(im)

                if np.mean(im) < 128:
                    text_color = (255,255,255)
                else:
                    text_color = (0,0,0)
                drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {ep_num+1}', fill=text_color)

                frames.append(im)

                _, reward, done, info = env.step(action)
                total_reward += reward
                terminal = done
                progress.set_postfix(reward=reward, info=info)
                # env.render()
            total_rewards.append(total_reward)
            total_rewards_pkl = append_file_name("total_rewards", args.algorithm, ".pkl")
            with open(total_rewards_pkl, "wb") as f:
                pickle.dump(total_rewards, f)
        
        # close the environment
        env.close()

        # Rendering runs of the agent to a .mp4 for visualization
        agent_mp4 = append_file_name("agent", args.algorithm, ".mp4")
        # imageio.mimwrite(agent_mp4, frames)

        if num_episodes > 500:
            plot(total_rewards, args.algorithm)

    else:
        training_mode=args.training_mode
        pretrained=args.pretrained
        if args.algorithm == 'ddqn':
            double_dq = True
        elif args.algorithm == 'dqn':
            double_dq = False
        else:
            return

        env = make_env(env)  # pre-process env for deep learning
        observation_space = env.observation_space.shape # shape of game space (i.e. 4x84x84)
        action_space = env.action_space.n # num discrete actions in game (i.e. 5)
        # instantiate DDQN or DQN agent
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
                            double_dq=double_dq,
                            pretrained=pretrained)
        
        env.reset()
        if args.pretrained:
            if args.algorithm == 'ddqn':
                total_rewards_pkl = "total_rewards_ddqn.pkl"
            elif args.algorithm == 'dqn':
                total_rewards_pkl = "total_rewards_dqn.pkl"
            with open(total_rewards_pkl, 'rb') as f:
                total_rewards = pickle.load(f)
        else:
            total_rewards = []

        frames = []
        for ep_num in tqdm(range(num_episodes)):
            state = env.reset()
            state = np.array([state])
            state = torch.Tensor(state)
            total_reward = 0
            steps = 0
            while True:
                if not training_mode:
                    show_state(env, ep_num)
                action = agent.act(state)
                steps += 1
                
                # Encode frames for output .mp4
                frame = env.render(mode='rgb_array')
                im = Image.fromarray(frame)

                drawer = ImageDraw.Draw(im)

                if np.mean(im) < 128:
                    text_color = (255,255,255)
                else:
                    text_color = (0,0,0)
                drawer.text((im.size[0]/20,im.size[1]/18), f'Episode: {ep_num+1}', fill=text_color)

                frames.append(im)
                
                # Update state, reward and determine if end condition is met
                state_next, reward, terminal, info = env.step(int(action[0]))
                # Use deprecated version of step due to incompatibility with nes-py
                total_reward += reward
                state_next = np.array([state_next])
                state_next = torch.Tensor(state_next)
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

        # Write outputs if we train the agent
        if training_mode:
            ending_position_pkl = append_file_name("ending_position", args.algorithm, ".pkl")
            num_in_queue_pkl = append_file_name("num_in_queue", args.algorithm, ".pkl")
            total_rewards_pkl = append_file_name("total_rewards", args.algorithm, ".pkl")
            with open(ending_position_pkl, "wb") as f:
                pickle.dump(agent.ending_position, f)
            with open(num_in_queue_pkl, "wb") as f:
                pickle.dump(agent.num_in_queue, f)
            with open(total_rewards_pkl, "wb") as f:
                pickle.dump(total_rewards, f)

            if agent.double_dq:
                torch.save(agent.local_net.state_dict(), "dq1.pt")
                torch.save(agent.target_net.state_dict(), "dq2.pt")
            else:
                torch.save(agent.dqn.state_dict(), "dq.pt")

            STATE_MEM_pt = append_file_name("STATE_MEM", args.algorithm, ".pt")
            ACTION_MEM_pt = append_file_name("ACTION_MEM", args.algorithm, ".pt")
            REWARD_MEM_pt = append_file_name("REWARD_MEM", args.algorithm, ".pt")
            STATE2_MEM_pt = append_file_name("STATE2_MEM", args.algorithm, ".pt")
            DONE_MEM_pt = append_file_name("DONE_MEM", args.algorithm, ".pt")
            torch.save(agent.STATE_MEM,  STATE_MEM_pt)
            torch.save(agent.ACTION_MEM, ACTION_MEM_pt)
            torch.save(agent.REWARD_MEM, REWARD_MEM_pt)
            torch.save(agent.STATE2_MEM, STATE2_MEM_pt)
            torch.save(agent.DONE_MEM,   DONE_MEM_pt)
        
        env.close()

        # Rendering runs of the agent to a .mp4 for visualization
        agent_mp4 = append_file_name("agent", args.algorithm, ".mp4")
        imageio.mimwrite(agent_mp4, frames)
        
        if num_episodes > 500:
            plot(total_rewards, args.algorithm)

def show_state(env, ep=0, info=""):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("Episode: %d %s" % (ep, info))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())

def append_file_name(basename, alg, file_ext):
    """
    Transforms given basename into a file name to specify if file relates to ddqn or dqn result.

    basename (string): basename for file output
    alg (string): denotes type of agent; one of ["ddqn", "dqn", "random"]
    file_ext (string): file extension for output (MUST INCLUDE ".", e.g. ".pkl" or ".pt")
    """
    string = "_" + alg
    return basename + string + file_ext

def plot(total_rewards, alg):
    """
    Plots Average Rewards (per 500 eps) vs. Episodes Trained using rolling window 
    arithmetic average.

    total_rewards (list): list of rewards from agent
    alg (string): denotes type of agent; one of ["ddqn", "dqn", "random"]
    """
    # check if preload with .pkl file
    if len(total_rewards) == 0:  
        total_rewards_pkl = append_file_name("total_rewards", alg, ".pkl")
        with open(total_rewards_pkl, 'rb') as f:
            total_rewards = pickle.load(f)
        plt.title("Average Rewards (per 500 eps) vs. Episodes Trained")
    # rolling average window of 500 episodes where arithmetic avg reward 
    # from ep n to n+500 is the value for episode n+500 (0 for eps 0 to 499)
    plt.plot([0 for _ in range(500)] + 
                np.convolve(total_rewards, np.ones((500,))/500, mode="valid").tolist())
    plt.show()
    reward_plot = append_file_name("plot", alg, ".png")
    plt.savefig()
    

def main(args):
    """
    Runs DDQN or DQN agent depending on algorithm input. 
    Refers to gym_super_mario_bros if mode is set to human (i.e. user interaction) or random (i.e. random movements).
    """
    if args.mode == 'agent':
        args.env = 'SuperMarioBros-1-1-v0'
        plot([], "random")
        # run(args)
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
    parser.add_argument('--algorithm', '-alg',
        type=str,
        default='ddqn',
        choices=['ddqn', 'dqn', 'random'],
        help='ddqn for Double Deep Q-Network; dqn for Deep Q-Network; random for random'
    )
    parser.add_argument('--env', '-e',
        type=str,
        default='SuperMarioBrosRandomStages-v0',
        help='The name of the environment to play'
    )
    parser.add_argument('--mode', '-m',
        type=str,
        default='agent',
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
