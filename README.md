# From Imitation to Optimization: Evaluating Demonstrators in Classic Control Environments

**Course:** Deep Reinforcement Learning  
**Team:** Rushil Ravi (UIN: 836000314) & Isabel Moore (UIN: 229001058)  
**GitHub:** [Repository Link]  
**Video Demo:** [5-minute YouTube presentation]

---

## Project Overview

This project investigates whether starting from expert demonstrations accelerates reinforcement learning compared to learning from scratch. We implement a complete pipeline: **Expert DQN → Behavior Cloning → PPO fine-tuning**, comparing Pure RL vs BC→RL approaches on classic control tasks.

**Research Question:** *Can behavior cloning from expert demonstrations provide a better initialization for reinforcement learning?*

---

## Key Results

### CartPole-v1

Experimental results demonstrate significant advantages of BC initialization:

| Method | Final Performance | Episodes to Threshold | Success Rate |
|--------|------------------|----------------------|--------------|
| Pure RL | 126.50 ± 23.26 | 326 episodes | 15.0% |
| BC→RL | **500.00 ± 0.00** | **1 episode** | **93.0%** |

**Findings:**
- BC→RL achieves **99.7% faster learning** (325× sample efficiency improvement)
- BC→RL reaches **perfect performance** with zero variance
- BC initialization provides strong prior knowledge that dramatically reduces exploration needs

### LunarLander-v3

This environment is also supported for experimentation. To run:
```bash
python main.py --env LunarLander-v3 --mode all
```

LunarLander presents a more challenging continuous control task with:
- 8-dimensional continuous state space
- 4 discrete actions
- Success threshold: 200 average reward
- More complex dynamics requiring precise control

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
- Expert: 500 episodes, ε-greedy exploration
- BC: 50 epochs on ~10,000 expert transitions
- Pure RL: 800 episodes from random initialization
- BC→RL: 800 episodes starting from BC weights
- Success threshold: 195 average reward

**LunarLander-v3:**
- Expert: 500 episodes, ε-greedy exploration
- BC: 50 epochs on collected demonstrations
- Pure RL: 1000 episodes from random initialization
- BC→RL: 1000 episodes starting from BC weights
- Success threshold: 200 average reward

All experiments use seed=42 for reproducibility.

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

### Methodology
- Fixed training budget: 800 episodes for both Pure RL and BC→RL
- Statistical evaluation: 20 test episodes per method with mean ± std
- Metrics: Final performance, sample efficiency, success rate

### Results Analysis
BC→RL demonstrates clear advantages:
- **Sample Efficiency:** Reaches threshold in 1 episode vs 326 for Pure RL
- **Final Performance:** Perfect score (500/500) vs 126.50 for Pure RL  
- **Stability:** Zero variance vs high variance (±72.14) for Pure RL
- **Success Rate:** 93% vs 15% for Pure RL

---

## Discussion

### Strengths
- Strong empirical evidence for BC initialization benefits on CartPole-v1
- Fair experimental comparison with equal training budgets
- Reproducible results with fixed random seeds
- Clear visualization of learning dynamics
- Extensible to multiple environments (CartPole, LunarLander)

### Limitations
- CartPole results limited to simple environment; LunarLander provides more challenging test
- Expert quality directly impacts BC performance
- Requires access to expert demonstrations
- May not generalize to all domains without adaptation

### Future Work
- Complete LunarLander-v3 experiments for comparison with CartPole results
- Test on more complex environments (Atari, MuJoCo)
- Investigate minimum demonstration requirements
- Study impact of imperfect expert demonstrations
- Explore transfer learning across tasks

---

## Team Contributions

**Rushil Ravi:** Expert DQN implementation, demonstration collection pipeline, project infrastructure, debugging and testing

**Isabel Moore:** Experimental design, statistical analysis, visualization, literature review, report writing

Both members contributed equally to algorithm implementation, hyperparameter tuning, and result analysis.

---

## References

1. Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature*
2. Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*
3. Pomerleau (1988). "ALVINN: An Autonomous Land Vehicle in a Neural Network." *NeurIPS*
4. Hussein et al. (2017). "Imitation Learning: A Survey of Learning Methods." *JMLR*
5. Sutton & Barto (2018). "Reinforcement Learning: An Introduction." *MIT Press*
6. Silver et al. (2016). "Mastering the game of Go with deep neural networks." *Nature*
7. Brockman et al. (2016). "OpenAI Gym." *arXiv*
8. Paszke et al. (2019). "PyTorch: An Imperative Style, High-Performance Deep Learning Library." *NeurIPS*
9. Ross et al. (2011). "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." *AISTATS*
10. Hester et al. (2018). "Deep Q-learning from Demonstrations." *AAAI*

---

## Acknowledgments

We thank the course instructors for guidance on experimental methodology and the RL research community for open-source implementations. This project uses OpenAI Gym/Gymnasium environments and PyTorch framework.

---

**Note:** All code, experimental results, and visualizations are original work by the team. External references and prior work are properly cited throughout.
