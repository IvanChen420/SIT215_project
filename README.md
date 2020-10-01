# Reinforcement Learning 

## Pre-requirement

- numpy 
- matplotlib
- gym
- tqdm

## Quick Start 

`` python SIT215-code.py ``

## Design Pattern

### Part I:  Reinforcement Learning Agent
1. Agent interface: `Agent()`
2. Random policy control: `RandomPolicyAgent()`
3. Q-learning control (off-policy): `QLearningAgent()`
4. S-A-R-S-A control (on-policy): `SARSAgent()`

### Part II: Tool Functions

1. `bins()`: Convert continuous variable to discrete variables.
2. `discrete_action_helper(_action, _env_id)`: Convert continuous action value to discrete 
    (for Pendulum case)
3. `discrete_state_helper(_state, _env_id)`: Convert continuous state value to discrete 
    (for Cart-Pole and Pendulum case)
4. `summary()`: Print the average rewards / time-step of different algorithms and cases
5. `visualize()`: Plot 

### Part III:Runner

1. `case_script()`
