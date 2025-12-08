# From Imitation to Optimization: Evaluating Demonstrators in Classic Control Environments

**Course:** Deep Reinforcement Learning  
**Team:** Rushil Ravi (UIN: 836000314) & Isabel Moore (UIN: 229001058)  
**GitHub:** [Repository Link]  
**Video Demo:** [5-minute YouTube presentation]

---

## Project Overview

This project investigates whether starting from expert demonstrations accelerates reinforcement learning compared to learning from scratch. 

We implement a complete pipeline: **Expert DQN → Behavior Cloning → PPO fine-tuning**, comparing Pure RL vs BC→RL approaches on classic control tasks.

**Research Question:** *Can behavior cloning from expert demonstrations provide a better initialization for reinforcement learning?*

---

## Key Results

### CartPole-v1

Experimental results in CartPole demonstrate significant sample efficiency gains but reveal unexpected performance dynamics:

| Method | Final Test Performance | Episodes to Threshold | Sample Efficiency |
|--------|----------------------|----------------------|-------------------|
| Pure RL | 500.00 ± 0.00 | 369 episodes | Baseline |
| BC-Only | **500.00 ± 0.00** | N/A | N/A |
| BC→RL | 200.30 ± 1.93 | **1 episode** | **99.7% faster** |

**Key Findings:**
- BC→RL achieves **99.7% faster learning** (1 vs 369 episodes to threshold)
- **Surprising result:** BC-Only and Pure RL both achieve perfect performance (500.0)
- BC→RL degrades from BC initialization (500.0 → 200.3), suggesting RL fine-tuning can harm strong BC policies
- Demonstrates **exploration-induced forgetting** where PPO updates drive agent away from optimal BC policy

### LunarLander-v3

LunarLander presents a more challenging environment with dramatically different results:

| Method | Final Test Performance | Episodes to Threshold | Sample Efficiency |
|--------|----------------------|----------------------|-------------------|
| Pure RL | -577.19 ± 139.44 | Never (1000+) | Failed |
| BC-Only | **145.82 ± 113.36** | N/A | N/A |
| BC→RL | 114.65 ± 114.04 | 286 episodes | **71.4% faster** |

**Key Findings:**
- BC→RL achieves **71.4% sample efficiency improvement** (286 vs 1000 episodes)
- **Pure RL catastrophically fails** (-577.19 average reward) in this complex environment
- BC-Only outperforms BC→RL (145.82 vs 114.65), again showing RL fine-tuning degradation
- Environment complexity matters: demonstration benefits are critical when random exploration fails

### Cross-Environment Insights

**BC Exceeds Experts:** Remarkably, BC policies outperformed their DQN expert demonstrators in both environments:
- LunarLander: BC 158.51 vs Expert 61.60 (257.3%)
- CartPole: BC 500.0 vs Expert 380.20 (131.5%)

This suggests supervised learning filters exploration noise better than ε-greedy policies.

**The BC→RL Paradox:** While BC→RL shows superior sample efficiency, it consistently underperforms BC-Only in final evaluation, challenging the assumption that RL fine-tuning always improves imitation learning.

---

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/DemoRL-Classic-Control.git
cd DemoRL-Classic-Control
python -m venv rl_env_project
source rl_env_project/bin/activate
pip install -r requirements.txt
```

### Run Pipeline
```bash
# CartPole-v1 (simple environment, ~5 minutes)
python main.py --env CartPole-v1 --mode all

# LunarLander-v3 (complex environment, ~10 minutes)
python main.py --env LunarLander-v3 --mode all

# Individual steps (example for CartPole)
python main.py --env {env_name} --mode expert    # Train DQN expert
python main.py --env {env_name} --mode demos     # Collect demonstrations
python main.py --env {env_name} --mode bc        # Train behavior cloning
python main.py --env {env_name} --mode rl        # Compare RL methods
python main.py --env {env_name} --mode eval      # Final evaluation
```

---

## Implementation

### Algorithms

1. **Deep Q-Network (DQN)** - Expert agent training with epsilon-greedy exploration
2. **Behavior Cloning (BC)** - Supervised learning from expert demonstrations  
3. **Proximal Policy Optimization (PPO)** - Policy gradient RL with clipped objective

### Architecture

All networks use consistent 128→128 hidden layer architecture for fair comparison.

### Training Configuration

**CartPole-v1:**
- **Expert (DQN):** 500 episodes, ε-greedy exploration (1.0→0.01)
- **Demonstrations:** 50 episodes, 25,000 transitions (avg 500.0 steps/episode)
- **BC:** 50 epochs, final loss 0.0851
- **Pure RL (PPO):** 800 episodes from random initialization
- **BC→RL (PPO):** 800 episodes starting from BC weights
- **Success threshold:** 195 average reward

**LunarLander-v3:**
- **Expert (DQN):** 500 episodes, ε-greedy exploration (1.0→0.01)
- **Demonstrations:** 50 episodes, 18,479 transitions (avg 369.6 steps/episode)
- **BC:** 50 epochs, final loss 0.3036
- **Pure RL (PPO):** 1000 episodes from random initialization
- **BC→RL (PPO):** 1000 episodes starting from BC weights
- **Success threshold:** 200 average reward

All experiments use seed=42 for reproducibility. Network architecture: 128→128 hidden layers for all methods.

---

## Project Structure

```
DemoRL-Classic-Control/
├── main.py                 # Pipeline runner
├── scripts/                # Training scripts
│   ├── train_expert.py
│   ├── collect_demos.py
│   ├── train_bc.py
│   ├── train_rl.py
│   └── evaluate.py
├── src/                    # Core modules
│   ├── networks.py
│   ├── environments.py
│   └── seed_utils.py
└── images/                 # Generated plots
```

---

## Evaluation

Results reveal a key paradox: BC→RL learns 71-99% faster than Pure RL but degrades from BC initialization in final performance. BC-Only consistently achieves best results. Pure RL fails catastrophically in complex LunarLander (-577.19) but succeeds in simple CartPole (500.0), showing environment complexity matters critically.

---

## Discussion

### Strengths

Strong sample efficiency gains (71-99% faster learning) confirm demonstration benefits. BC exceeds expert performance by filtering exploration noise. Fair evaluation across two complexity levels provides clear insights into when each approach works.

### Limitations

**Key finding:** RL fine-tuning harms strong BC policies through exploration-induced forgetting, challenging assumptions that RL always improves imitation. Environment complexity critically affects viability—Pure RL fails in LunarLander but succeeds in CartPole. Single seed, fixed hyperparameters, and no adaptive switching limit robustness.

### Future Work

Priorities: multiple seeds, adaptive switching, hyperparameter tuning. Extensions: complex environments (Atari, MuJoCo), iterative methods (DAgger), hybrid approaches (DQfD), continuous control (DDPG/SAC), transfer learning.

---

## Team Contributions

**Rushil Ravi:**  
Expert DQN implementation, demonstration collection pipeline, project infrastructure, debugging and testing

**Isabel Moore:**  
Experimental design, statistical analysis, visualization, literature review, report writing

Both members contributed equally to algorithm implementation, hyperparameter tuning, and result analysis.

---

## References

1. Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*
3. Pomerleau (1991). "Efficient training of artificial neural networks for autonomous navigation." *Neural Computation*
4. Sutton & Barto (2018). "Reinforcement Learning: An Introduction." *MIT Press*
5. Silver et al. (2016). "Mastering the game of Go with deep neural networks." *Nature*
6. Brockman et al. (2016). "OpenAI Gym." *arXiv*
7. Paszke et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*
8. Ross et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." *AISTATS*
9. Hester et al. (2018). "Deep Q-learning from Demonstrations." *AAAI*
10. Ho & Ermon (2016). "Generative Adversarial Imitation Learning." *NeurIPS*
11. Lillicrap et al. (2015). "Continuous Control with Deep Reinforcement Learning." *arXiv*
12. Mnih et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." *ICML*
13. Rajeswaran et al. (2017). "Learning complex dexterous manipulation with demonstrations." *arXiv*
14. Parisotto et al. (2016). "Actor-Mimic: Deep multitask and transfer RL." *arXiv*
15. Rusu et al. (2016). "Policy distillation." *arXiv*

---

## Acknowledgments

We thank the course instructors for guidance on experimental methodology and the RL research community for open-source implementations. This project uses OpenAI Gym/Gymnasium environments and PyTorch framework.

---

**Note:** All code, experimental results, and visualizations are original work by the team. External references and prior work are properly cited throughout.
