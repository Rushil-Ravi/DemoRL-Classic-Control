#!/usr/bin/env python3
"""
DRL Project: From Imitation to Optimization
Authors: Rushil Ravi & Isabel Moore
"""

import argparse
import sys
import os

# Add scripts directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))


def main():
    parser = argparse.ArgumentParser(description='DRL Project Pipeline')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        choices=['CartPole-v1', 'LunarLander-v2'],
                        help='Environment name')
    parser.add_argument('--mode', type=str, default='expert',
                        choices=['expert', 'demos', 'bc', 'rl', 'eval', 'all'],
                        help='Which step to run')

    args = parser.parse_args()

    print(f"🚀 DRL Project: From Imitation to Optimization")
    print(f"📊 Environment: {args.env}")
    print(f"⚙️ Mode: {args.mode}")
    print("-" * 50)

    try:
        if args.mode in ['expert', 'all']:
            print("\n" + "=" * 50)
            print("STEP 1: Training Expert Agent (DQN)")
            print("=" * 50)
            import train_expert
            # Call the function that exists in your file
            train_expert.train_expert(env_name=args.env)  # Or whatever your function is called

        if args.mode in ['demos', 'all']:
            print("\n" + "=" * 50)
            print("STEP 2: Collecting Expert Demonstrations")
            print("=" * 50)
            import collect_demos
            # Call the function that exists in your file
            collect_demos.collect_demos(env_name=args.env)  # Or whatever your function is called

        if args.mode in ['bc', 'all']:
            print("\n" + "=" * 50)
            print("STEP 3: Training Behavior Cloning")
            print("=" * 50)
            import train_bc
            # Call the function that exists in your file
            train_bc.train_bc(env_name=args.env)  # Or whatever your function is called

        if args.mode in ['rl', 'all']:
            print("\n" + "=" * 50)
            print("STEP 4: Training RL with BC Initialization")
            print("=" * 50)
            import train_rl
            # Call the function that exists in your file
            train_rl.compare_methods(env_name=args.env)  # Or whatever your function is called

        if args.mode in ['eval', 'all']:
            print("\n" + "=" * 50)
            print("STEP 5: Final Evaluation")
            print("=" * 50)
            import evaluate
            # Call the function that exists in your file
            evaluate.evaluate_all(env_name=args.env)  # Or whatever your function is called

        print("\n" + "=" * 50)
        print("✅ PROJECT COMPLETED SUCCESSFULLY!")
        print("=" * 50)

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Please check that your script files exist in the scripts/ folder")
    except AttributeError as e:
        print(f"\n❌ Function not found: {e}")
        print("Please check function names in your script files")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()