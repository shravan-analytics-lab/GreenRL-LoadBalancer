import numpy as np
from pathlib import Path

from r1_agent.q_learning import QLearningAgent

BASE_DIR = Path(__file__).resolve().parent

# =====================================
# CONFIG
# =====================================

CAPACITY_PER_SERVER = 60

SERVER_COST = 4

MIN_SERVERS = 2
MAX_SERVERS = 12

# =====================================
# MODELS
# =====================================

MODEL_MAP = {

    "cheap":
        "cheap_model.npz",

    "balanced":
        "balanced_model.npz",

    "reliable":
        "rich_model.npz",
}

# =====================================
# TRAFFIC
# =====================================

def generate_traffic(hour):

    daily_wave = 95 * np.sin(
        (2 * np.pi * (hour - 8)) / 24
    )

    evening_peak = 170 * np.exp(
        -((hour - 20) ** 2) / 7
    )

    base = (
        230
        + daily_wave
        + evening_peak
    )

    requests = np.random.poisson(

        max(
            40,
            base * np.random.uniform(0.7, 1.4)
        )
    )

    # random spikes
    if np.random.rand() < 0.08:

        requests += np.random.randint(
            220,
            600
        )

    return int(max(20, requests))

# =====================================
# APPLY ACTION
# =====================================

def apply_action(action, servers):

    if action == 1:
        servers += 1

    elif action == 2:
        servers -= 1

    elif action == 3:
        servers += 2

    elif action == 4:
        servers -= 2

    return max(
        MIN_SERVERS,
        min(MAX_SERVERS, servers)
    )

# =====================================
# MAIN INFERENCE ENGINE
# =====================================

def run_model(policy, steps):

    model_name = MODEL_MAP[policy]

    model_path = (
        BASE_DIR
        / "results"
        / model_name
    )

    # =================================
    # LOAD TRAINED MODEL
    # =================================

    agent = QLearningAgent()

    agent.load(model_path)

    agent.epsilon = 0

    # =================================
    # INITIAL STATE
    # =================================

    servers = 4

    queue = 0

    traffic_past = []

    # =================================
    # HISTORIES
    # =================================

    traffic_history = []

    latency_history = []

    queue_history = []

    server_history = []

    utilization_history = []

    cost_history = []

    action_history = []

    # =================================
    # MAIN LOOP
    # =================================

    for t in range(steps):

        hour = t % 24

        requests_count = generate_traffic(
            hour
        )

        traffic_past.append(
            requests_count
        )

        # =============================
        # TREND
        # =============================

        trend = 0

        if len(traffic_past) > 10:

            trend = (

                np.mean(traffic_past[-5:])
                -
                np.mean(traffic_past[-10:])
            )

        # =============================
        # CREATE STATE
        # =============================

        state = agent.get_state(

            requests_count,

            servers,

            hour,

            queue,

            trend
        )

        # =============================
        # DQN ACTION
        # =============================

        action = agent.choose_action(

            state,

            [0,1,2,3,4]
        )

        # =============================
        # APPLY SCALING
        # =============================

        servers = apply_action(

            action,

            servers
        )

        # =============================
        # ENVIRONMENT
        # =============================

        capacity = (
            servers
            * CAPACITY_PER_SERVER
        )

        effective_capacity = (

            capacity
            * np.random.uniform(0.72, 0.96)
        )

        incoming = (
            requests_count
            + queue
        )

        processed = min(
            incoming,
            effective_capacity
        )

        queue = max(
            0,
            incoming - processed
        )

        utilization = min(

            100,

            (
                requests_count
                / max(1, capacity)
            ) * 100
        )

        latency = (

            40
            + 75 * ((utilization / 100) ** 2.4)
            + 0.55 * queue
        )

        cost = (
            servers
            * SERVER_COST
        )

        # =============================
        # STORE HISTORIES
        # =============================

        traffic_history.append(
            requests_count
        )

        latency_history.append(
            round(latency, 2)
        )

        queue_history.append(
            round(queue, 2)
        )

        server_history.append(
            servers
        )

        utilization_history.append(
            round(utilization, 2)
        )

        cost_history.append(
            round(cost, 2)
        )

        action_history.append(
            action
        )

    # =================================
    # RETURN RESULTS
    # =================================

    return {

        "policy": policy,

        "avg_latency":

            round(
                np.mean(latency_history),
                2
            ),

        "avg_queue":

            round(
                np.mean(queue_history),
                2
            ),

        "avg_servers":

            round(
                np.mean(server_history),
                2
            ),

        "avg_utilization":

            round(
                np.mean(utilization_history),
                2
            ),

        "avg_cost":

            round(
                np.mean(cost_history),
                2
            ),

        "sla_violations":

            int(
                np.sum(
                    np.array(latency_history)
                    > 120
                )
            ),

        "traffic_history":
            traffic_history,

        "latency_history":
            latency_history,

        "queue_history":
            queue_history,

        "server_history":
            server_history,

        "utilization_history":
            utilization_history,

        "cost_history":
            cost_history,

        "action_history":
            action_history
    }