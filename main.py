import argparse

from super_mario_bros.gym_super_mario_bros._app import cli

def main(args):
    """
    [INSERT DOCUMENTATION]
    """
    if args.mode == 'agent':
        print("hurray")
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
