import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.rule_based import rule_based_decision
from r1_agent.q_learning import QLearningAgent

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = RESULTS_DIR / "dqn_model.npz"
LATEST_MODEL_PATH = RESULTS_DIR / "latest_dqn_model.npz"
METRICS_PATH = RESULTS_DIR / "best_metrics.json"
METRICS_VERSION = 3

try:
    df = pd.read_csv(BASE_DIR / "data" / "cleaned_dataset.csv")
    hours = df["hour"].values
except FileNotFoundError:
    hours = np.tile(np.arange(24), 50)

CAPACITY_PER_SERVER = 60
INITIAL_SERVERS = 3
MIN_SERVERS = 2
MAX_SERVERS = 12
SERVER_COST = 4.0

TRAIN_EPISODES = 1200
STEPS_PER_EPISODE = 720
TEST_STEPS = 1000
DQN_BATCH_SIZE = 32
DQN_TRAIN_EVERY = 4
DQN_TARGET_UPDATE_EVERY = 25

SCALE_COOLDOWN = 3
SCALE_UP_DELAY = 2
SCALE_DOWN_DELAY = 1


def generate_traffic(t, hour):
    daily_wave = 95 * np.sin((2 * np.pi * (hour - 8)) / 24)
    lunch_peak = 90 * np.exp(-((hour - 13) ** 2) / 8)
    evening_peak = 170 * np.exp(-((hour - 20) ** 2) / 7)
    weekly_noise = 35 * np.sin((2 * np.pi * t) / 168)

    base = 230 + daily_wave + lunch_peak + evening_peak + weekly_noise
    requests = np.random.poisson(
    max(40, base * np.random.uniform(0.7, 1.4))
    )

    if np.random.rand() < 0.06:
        requests += np.random.randint(220, 520)

    if np.random.rand() < 0.025:
        requests += np.random.randint(500, 850)

    return int(max(20, requests))


def predict_traffic(history):

    if len(history) < 5:
        return history[-1], 0

    short_window = history[-5:]
    long_window = history[-20:]

    short_avg = np.mean(short_window)
    long_avg = np.mean(long_window)

    recent_max = np.max(short_window)

    trend = short_avg - long_avg

    prediction = (
        short_avg * 0.55
        + recent_max * 0.35
        + trend * 0.25
    )

    return max(20, prediction), trend


def apply_pending_scaling(servers, pending_actions):
    next_pending = []

    for delay, change in pending_actions:
        delay -= 1
        if delay <= 0:
            servers = min(MAX_SERVERS, max(MIN_SERVERS, servers + change))
        else:
            next_pending.append((delay, change))

    return servers, next_pending


def action_to_server_change(action):
    if action == 1:
        return 1
    if action == 2:
        return -1
    if action == 3:
        return 2
    if action == 4:
        return -2
    return 0


def schedule_scaling(action, servers, cooldown, pending_actions):
    if cooldown > 0:
        return servers, cooldown, pending_actions

    server_change = action_to_server_change(action)

    if server_change > 0 and servers < MAX_SERVERS:
        server_change = min(server_change, MAX_SERVERS - servers)
        pending_actions.append((SCALE_UP_DELAY, server_change))
        cooldown = SCALE_COOLDOWN
    elif server_change < 0 and servers > MIN_SERVERS:
        server_change = max(server_change, MIN_SERVERS - servers)
        pending_actions.append((SCALE_DOWN_DELAY, server_change))
        cooldown = SCALE_COOLDOWN

    return servers, cooldown, pending_actions


def get_allowed_rl_actions(predicted_traffic, trend, queue, servers):
    current_capacity = servers * CAPACITY_PER_SERVER
    allowed_actions = [0]

    if (
        servers < MAX_SERVERS
        and (queue > 0 or trend > 25 or predicted_traffic > current_capacity * 0.62)
    ):
        allowed_actions.append(1)

    if (
        servers <= MAX_SERVERS - 2
        and (queue > 160 or (trend > 90 and predicted_traffic > current_capacity * 0.78))
    ):
        allowed_actions.append(3)

    if servers > MIN_SERVERS:
        next_lower_capacity = (servers - 1) * CAPACITY_PER_SERVER
        scale_down_is_safe = (
            queue == 0
            and trend <= 10
            and predicted_traffic < next_lower_capacity * 0.55
        )

        if scale_down_is_safe:
            allowed_actions.append(2)

    if servers > MIN_SERVERS + 1:
        two_lower_capacity = (servers - 2) * CAPACITY_PER_SERVER
        big_scale_down_is_safe = (
            queue == 0
            and trend < -60
            and predicted_traffic < two_lower_capacity * 0.45
        )

        if big_scale_down_is_safe:
            allowed_actions.append(4)

    return allowed_actions


def run_environment_step(requests, servers, queue):
    total_capacity = servers * CAPACITY_PER_SERVER
    effective_capacity = (
    total_capacity
    * np.random.uniform(0.72, 0.96)
)

    incoming_work = queue + requests
    processed = min(incoming_work, effective_capacity)
    queue = max(0, incoming_work - processed)

    utilization = min(1.25, requests / max(1, total_capacity))
    latency = (
    40
    + 75 * (utilization ** 2.4)
    + 0.55 * queue
    + np.random.normal(0, 4)
)
    latency = max(25, latency)

    return queue, latency, utilization, processed


def calculate_reward(latency, queue, utilization, servers, action, predicted_traffic, trend):

    reward = 0

    reward -= latency * 1.8
    reward -= queue * 0.7
    reward -= servers * SERVER_COST * 4.5

    if latency > 120:
        reward -= 300

    if latency > 220:
        reward -= 500

    if queue > 200:
        reward -= 180

    if queue > 450:
        reward -= 400

    if 0.65 <= utilization <= 0.90:
        reward += 50

    elif utilization < 0.35:
        reward -= 100

    elif utilization > 0.97:
        reward -= 60

    if servers >= 10 and predicted_traffic < 450:
        reward -= 180

    expected_capacity = servers * CAPACITY_PER_SERVER

    if action in (1, 3) and predicted_traffic > expected_capacity * 0.82:
        reward += 40

    if action in (1, 3) and trend > 60:
        reward += 25

    if action == 0 and trend > 90:
        reward -= 120

    if action in (2, 4) and predicted_traffic < expected_capacity * 0.40 and queue == 0:
        reward += 45

    if action in (2, 4) and trend > 30:
        reward -= 140

    return reward








def load_agent(model_name="balanced_model.npz"):

    agent = QLearningAgent()

    loaded_compatible_model = False

    model_path = RESULTS_DIR / model_name

    if model_path.exists():

        try:

            agent.load(model_path)

            agent.epsilon = 0

            loaded_compatible_model = True

            print(
                f"Loaded model: {model_name}"
            )

        except ValueError:

            print(
                "Model incompatible."
            )

    else:

        print(
            f"Model not found: {model_name}"
        )

    return agent, loaded_compatible_model


def train_agent():
    """Manual training entry point. This is never called by api.py."""

    agent, _ = load_agent("dqn_model.npz")

    for episode in range(TRAIN_EPISODES):
        servers = INITIAL_SERVERS
        queue = 0
        cooldown = 0
        pending_actions = []
        traffic_history = []

        for t in range(STEPS_PER_EPISODE):
            hour = hours[t % len(hours)]
            requests = generate_traffic(t, hour)
            traffic_history.append(requests)
            predicted_traffic, trend = predict_traffic(traffic_history)

            state = agent.get_state(predicted_traffic, servers, hour, queue, trend)

            can_act = cooldown == 0
            allowed_actions = get_allowed_rl_actions(
                predicted_traffic,
                trend,
                queue,
                servers,
            )
            action = agent.choose_action(state, allowed_actions) if can_act else 0

            servers, cooldown, pending_actions = schedule_scaling(
                action,
                servers,
                cooldown,
                pending_actions,
            )
            servers, pending_actions = apply_pending_scaling(servers, pending_actions)

            queue, latency, utilization, _ = run_environment_step(
                requests,
                servers,
                queue,
            )
            reward = calculate_reward(
                latency,
                queue,
                utilization,
                servers,
                action,
                predicted_traffic,
                trend,
            )

            next_hour = hours[(t + 1) % len(hours)]
            next_predicted, next_trend = predict_traffic(traffic_history)
            next_state = agent.get_state(
                next_predicted,
                servers,
                next_hour,
                queue,
                next_trend,
            )
            next_allowed_actions = get_allowed_rl_actions(
                next_predicted,
                next_trend,
                queue,
                servers,
            )

            if can_act:
                done = t == STEPS_PER_EPISODE - 1
                agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_allowed_actions,
                )

            if (episode * STEPS_PER_EPISODE + t) % DQN_TRAIN_EVERY == 0:
                agent.replay(DQN_BATCH_SIZE)

            cooldown = max(0, cooldown - 1)

        agent.decay_epsilon()

        if episode % DQN_TARGET_UPDATE_EVERY == 0:
            agent.update_target_network()

    agent.epsilon = 0
    agent.save(LATEST_MODEL_PATH)
    return agent
# =====================================
# REAL TEST SIMULATION
# =====================================

def run_simulation(agent, steps=300):

    servers_rl = INITIAL_SERVERS
    servers_rule = INITIAL_SERVERS

    queue_rl = 0
    queue_rule = 0

    rl_cooldown = 0
    rule_cooldown = 0

    rl_pending_actions = []
    rule_pending_actions = []

    latency_rl = []
    latency_rule = []

    utilization_rl = []
    utilization_rule = []

    cost_rl = []
    cost_rule = []

    servers_rl_history = []
    servers_rule_history = []

    queue_rl_history = []
    queue_rule_history = []

    traffic_history = []

    action_history = []

    for t in range(steps):

        hour = hours[t % len(hours)]

        requests = generate_traffic(
            t,
            hour
        )

        traffic_history.append(
            requests
        )

        predicted_traffic, trend = predict_traffic(
            traffic_history
        )

        # =================================
        # RL SYSTEM
        # =================================

        state = agent.get_state(

            predicted_traffic,

            servers_rl,

            hour,

            queue_rl,

            trend
        )

        allowed_rl_actions = get_allowed_rl_actions(

            predicted_traffic,

            trend,

            queue_rl,

            servers_rl
        )

        rl_action = (

            agent.choose_action(
                state,
                allowed_rl_actions
            )

            if rl_cooldown == 0

            else 0
        )

        servers_rl, rl_cooldown, rl_pending_actions = schedule_scaling(

            rl_action,

            servers_rl,

            rl_cooldown,

            rl_pending_actions
        )

        servers_rl, rl_pending_actions = apply_pending_scaling(

            servers_rl,

            rl_pending_actions
        )

        queue_rl, latency_rl_value, util_rl, _ = run_environment_step(

            requests,

            servers_rl,

            queue_rl
        )

        # =================================
        # RULE SYSTEM
        # =================================

        total_capacity_rule = (

            servers_rule
            * CAPACITY_PER_SERVER
        )

        cpu_rule = min(

            100,

            (
                requests
                / total_capacity_rule
            ) * 100
        )

        rule_action = rule_based_decision(

            cpu_rule,

            servers_rule,

            queue_rule
        )

        servers_rule, rule_cooldown, rule_pending_actions = schedule_scaling(

            rule_action,

            servers_rule,

            rule_cooldown,

            rule_pending_actions
        )

        servers_rule, rule_pending_actions = apply_pending_scaling(

            servers_rule,

            rule_pending_actions
        )

        queue_rule, latency_rule_value, util_rule, _ = run_environment_step(

            requests,

            servers_rule,

            queue_rule
        )

        # =================================
        # STORE
        # =================================

        latency_rl.append(
            float(latency_rl_value)
        )

        latency_rule.append(
            float(latency_rule_value)
        )

        utilization_rl.append(
            float(util_rl * 100)
        )

        utilization_rule.append(
            float(util_rule * 100)
        )

        cost_rl.append(
            float(servers_rl * SERVER_COST)
        )

        cost_rule.append(
            float(servers_rule * SERVER_COST)
        )

        servers_rl_history.append(
            int(servers_rl)
        )

        servers_rule_history.append(
            int(servers_rule)
        )

        queue_rl_history.append(
            float(queue_rl)
        )

        queue_rule_history.append(
            float(queue_rule)
        )

        action_history.append(
            int(rl_action)
        )

        rl_cooldown = max(
            0,
            rl_cooldown - 1
        )

        rule_cooldown = max(
            0,
            rule_cooldown - 1
        )

    # =====================================
    # FINAL METRICS
    # =====================================

    avg_rl_latency = np.mean(latency_rl)
    avg_rule_latency = np.mean(latency_rule)

    avg_rl_cost = np.mean(cost_rl)
    avg_rule_cost = np.mean(cost_rule)

    avg_rl_utilization = np.mean(utilization_rl)
    avg_rule_utilization = np.mean(utilization_rule)

    avg_rl_queue = np.mean(queue_rl_history)
    avg_rule_queue = np.mean(queue_rule_history)

    avg_rl_servers = np.mean(servers_rl_history)
    avg_rule_servers = np.mean(servers_rule_history)

    p95_rl_latency = np.percentile(
        latency_rl,
        95
    )

    p95_rule_latency = np.percentile(
        latency_rule,
        95
    )

    rl_sla_violations = np.sum(
        np.array(latency_rl) > 120
    )

    rule_sla_violations = np.sum(
        np.array(latency_rule) > 120
    )

    current_score = (

        avg_rl_latency

        + (0.10 * p95_rl_latency)

        + (0.35 * avg_rl_cost)

        + (0.75 * rl_sla_violations)
    )

    rule_score = (

        avg_rule_latency

        + (0.10 * p95_rule_latency)

        + (0.35 * avg_rule_cost)

        + (0.75 * rule_sla_violations)
    )

    # =====================================
    # CONCLUSION
    # =====================================

    if (
        avg_rule_latency > avg_rl_latency
        and
        avg_rule_cost > avg_rl_cost
    ):

        conclusion = (
            "RL performed better with "
            "lower latency and lower cost."
        )

    elif avg_rule_latency > avg_rl_latency:

        conclusion = (
            "RL achieved lower latency "
            "but cost was higher."
        )

    elif avg_rule_cost > avg_rl_cost:

        conclusion = (
            "RL reduced operational cost "
            "but latency increased."
        )

    else:

        conclusion = (
            "Rule-based system performed better overall."
        )

    # =====================================
    # RETURN
    # =====================================

    return {

        "traffic_history":
            traffic_history,

        "latency_rl":
            latency_rl,

        "latency_rule":
            latency_rule,

        "queue_rl":
            queue_rl_history,

        "queue_rule":
            queue_rule_history,

        "servers_rl":
            servers_rl_history,

        "servers_rule":
            servers_rule_history,

        "cost_rl":
            cost_rl,

        "cost_rule":
            cost_rule,

        "utilization_rl":
            utilization_rl,

        "utilization_rule":
            utilization_rule,

        "avg_rl_latency":
            float(avg_rl_latency),

        "avg_rule_latency":
            float(avg_rule_latency),

        "avg_rl_cost":
            float(avg_rl_cost),

        "avg_rule_cost":
            float(avg_rule_cost),

        "avg_rl_queue":
            float(avg_rl_queue),

        "avg_rule_queue":
            float(avg_rule_queue),

        "avg_rl_servers":
            float(avg_rl_servers),

        "avg_rule_servers":
            float(avg_rule_servers),

        "avg_rl_utilization":
            float(avg_rl_utilization),

        "avg_rule_utilization":
            float(avg_rule_utilization),

        "p95_rl_latency":
            float(p95_rl_latency),

        "p95_rule_latency":
            float(p95_rule_latency),

        "rl_sla_violations":
            int(rl_sla_violations),

        "rule_sla_violations":
            int(rule_sla_violations),

        "current_score":
            float(current_score),

        "rule_score":
            float(rule_score),

        "conclusion":
            conclusion,

        "action_history":
            action_history
    }

if __name__ == "__main__":
    trained_agent = train_agent()
    results = run_simulation(trained_agent, TEST_STEPS)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(results, metrics_file, indent=2)

    print("Manual training completed.")
    print(f"Latest model saved to: {LATEST_MODEL_PATH}")
    print(f"Evaluation metrics saved to: {METRICS_PATH}")
