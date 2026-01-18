#!/usr/bin/env python3
"""
MARL Training Script for SOITCS.

Trains Q-tables for Multi-Agent Reinforcement Learning traffic signal control.

Usage:
    python scripts/train_marl.py --episodes 100 --output data/pretrained_models/marl_q_tables.npz
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Train MARL agents for traffic signal control"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of training episodes (default: 50)",
    )
    parser.add_argument(
        "--ticks-per-episode",
        type=int,
        default=3600,
        help="Simulation ticks per episode (default: 3600 = ~2 min)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/pretrained_models/marl_q_tables.npz",
        help="Output path for trained Q-tables",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Path to network configuration (default: use built-in grid)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Initial exploration rate (default: 0.2)",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Epsilon decay per episode (default: 0.995)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot learning curves after training",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from src.core.simulation import Simulation
    from src.core.state import SimulationState
    from src.core.clock import SimulationClock
    from src.core.event_bus import get_event_bus
    from config.settings import get_settings

    # Training metrics
    episode_rewards = []
    episode_avg_delays = []
    episode_epsilons = []

    logger.info(f"Starting MARL training: {args.episodes} episodes")
    logger.info(f"Ticks per episode: {args.ticks_per_episode}")
    logger.info(f"Learning rate: {args.learning_rate}, Initial epsilon: {args.epsilon}")

    # Store Q-tables across episodes (persistent learning)
    persistent_q_tables = None

    for episode in range(args.episodes):
        # Create fresh simulation for each episode
        settings = get_settings()

        # Enable MARL in training mode with CA for vehicle movement
        settings.algorithms.cellular_automata["enabled"] = True
        settings.algorithms.sotl["enabled"] = False  # Let MARL control signals
        settings.algorithms.aco["enabled"] = False
        settings.algorithms.pso["enabled"] = False
        settings.algorithms.som["enabled"] = False
        settings.algorithms.marl["enabled"] = True
        settings.algorithms.marl["inference_only"] = False
        settings.algorithms.marl["learning_rate"] = args.learning_rate
        settings.algorithms.marl["epsilon"] = args.epsilon * (args.epsilon_decay ** episode)
        settings.algorithms.marl["epsilon_min"] = 0.01

        sim = Simulation(
            settings=settings,
            state=SimulationState(),
            clock=SimulationClock(),
            event_bus=get_event_bus(),
            headless=True,
        )

        # Set seed for reproducibility (different per episode for variety)
        sim.set_seed(args.seed + episode)

        # Initialize simulation
        sim.initialize(args.network)

        # Load persistent Q-tables if available
        marl_algo = sim.state.get_algorithm("marl")
        if persistent_q_tables is not None and marl_algo:
            for int_id, q_table in persistent_q_tables.items():
                if int_id in marl_algo._q_tables:
                    marl_algo._q_tables[int_id] = q_table.copy()

        # Enable SOTL control on intersections so MARL can request phase changes
        if sim.state.network:
            for intersection in sim.state.network.intersections.values():
                intersection.enable_sotl(True)

        # Run episode
        snapshots = sim.run_ticks(args.ticks_per_episode)

        # Collect metrics
        if marl_algo:
            curves = marl_algo.get_learning_curves()
            episode_reward = float(np.sum(curves["episode_rewards"]))
            episode_rewards.append(episode_reward)
            episode_epsilons.append(marl_algo.epsilon)

            # Save Q-tables for next episode
            persistent_q_tables = {
                int_id: q_table.copy()
                for int_id, q_table in marl_algo._q_tables.items()
            }

        # Average delay for this episode
        delays = [s.average_delay for s in snapshots]
        avg_delay = np.mean(delays) if delays else 0.0
        episode_avg_delays.append(avg_delay)

        logger.info(
            f"Episode {episode + 1}/{args.episodes}: "
            f"reward={episode_reward:.1f}, "
            f"avg_delay={avg_delay:.2f}, "
            f"epsilon={marl_algo.epsilon:.3f}"
        )

    # Save final Q-tables
    if persistent_q_tables:
        np.savez(
            args.output,
            **{str(k): v for k, v in persistent_q_tables.items()}
        )
        logger.info(f"Q-tables saved to {args.output}")

    # Save learning curves
    curves_path = output_path.parent / "marl_learning_curves.npz"
    np.savez(
        curves_path,
        episode_rewards=np.array(episode_rewards),
        episode_avg_delays=np.array(episode_avg_delays),
        episode_epsilons=np.array(episode_epsilons),
    )
    logger.info(f"Learning curves saved to {curves_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Final epsilon: {episode_epsilons[-1]:.4f}")
    print(f"Avg reward (last 10): {np.mean(episode_rewards[-10:]):.2f}")
    print(f"Avg delay (last 10): {np.mean(episode_avg_delays[-10:]):.2f}")

    # Check for learning (rewards should increase)
    early_rewards = np.mean(episode_rewards[:10]) if len(episode_rewards) >= 10 else 0
    late_rewards = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else 0
    improvement = late_rewards - early_rewards
    print(f"Reward improvement: {improvement:+.2f}")
    print("=" * 60 + "\n")

    # Plot learning curves if requested
    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Episode rewards
        axes[0, 0].plot(episode_rewards)
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].set_title("Episode Rewards")

        # Smoothed rewards
        window = min(10, len(episode_rewards))
        smoothed = np.convolve(
            episode_rewards, np.ones(window) / window, mode="valid"
        )
        axes[0, 1].plot(smoothed)
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Smoothed Reward")
        axes[0, 1].set_title(f"Smoothed Rewards (window={window})")

        # Average delay
        axes[1, 0].plot(episode_avg_delays)
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Average Delay")
        axes[1, 0].set_title("Average Delay per Episode")

        # Epsilon decay
        axes[1, 1].plot(episode_epsilons)
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Epsilon")
        axes[1, 1].set_title("Exploration Rate Decay")

        plt.tight_layout()

        plot_path = output_path.parent / "marl_learning_curves.png"
        plt.savefig(plot_path, dpi=150)
        logger.info(f"Learning curves plot saved to {plot_path}")
        plt.show()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
