import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulation.traffic import generate_traffic
from simulation.environment import Environment
from baseline.rule_based import rule_based_decision
from r1_agent.q_learning  import QLearningAgent
from results.plots import plot_results

traffic = generate_traffic()

# Initialize
env_rl = Environment()
env_rule = Environment()
agent = QLearningAgent()

latency_rl = []
latency_rule = []

# TRAIN RL
for episode in range(500):
    env_rl.reset()
    for t in range(len(traffic)):
        state = agent.get_state(traffic[t], env_rl.servers)
        action = agent.choose_action(state)

        cpu, latency, reward, servers = env_rl.step(traffic[t], action)

        next_state = agent.get_state(traffic[t], servers)
        agent.update(state, action, reward, next_state)

# TEST RL vs RULE
env_rl.reset()
env_rule.reset()

for t in range(len(traffic)):

    # RL
    state = agent.get_state(traffic[t], env_rl.servers)
    action = agent.choose_action(state)
    cpu, latency, reward, _ = env_rl.step(traffic[t], action)
    latency_rl.append(latency)

    # Rule-based
    cpu_rule = (traffic[t] / (env_rule.servers * env_rule.capacity)) * 100
    action_rule = rule_based_decision(cpu_rule, env_rule.servers)
    cpu, latency, reward, _ = env_rule.step(traffic[t], action_rule)
    latency_rule.append(latency)

# Plot
plot_results(traffic, latency_rl, latency_rule)