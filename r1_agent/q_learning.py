from collections import deque
import random
import numpy as np


class QLearningAgent:

    def __init__(self):

        self.input_dim = 7
        self.hidden_1 = 64
        self.hidden_2 = 32
        self.action_count = 5

        self.gamma = 0.92
        self.epsilon = 0.20
        self.learning_rate = 0.0007

        self.memory = deque(maxlen=15000)

        self.weights = self._init_network()
        self.target_weights = self._copy_weights(
            self.weights
        )

    def _init_network(self):

        return {

            "w1": np.random.randn(
                self.input_dim,
                self.hidden_1
            ) * np.sqrt(2 / self.input_dim),

            "b1": np.zeros(self.hidden_1),

            "w2": np.random.randn(
                self.hidden_1,
                self.hidden_2
            ) * np.sqrt(2 / self.hidden_1),

            "b2": np.zeros(self.hidden_2),

            "w3": np.random.randn(
                self.hidden_2,
                self.action_count
            ) * np.sqrt(2 / self.hidden_2),

            "b3": np.zeros(self.action_count),
        }

    def _copy_weights(self, weights):

        return {
            key: value.copy()
            for key, value in weights.items()
        }

    def get_state(
        self,
        requests,
        servers,
        hour,
        queue,
        trend=0
    ):

        traffic_feature = np.clip(
            requests / 1200,
            0,
            1.5
        )

        server_feature = np.clip(
            servers / 12,
            0,
            1
        )

        hour_sin = np.sin(
            2 * np.pi * hour / 24
        )

        hour_cos = np.cos(
            2 * np.pi * hour / 24
        )

        queue_feature = np.clip(
            queue / 1000,
            0,
            2
        )

        trend_feature = np.clip(
            trend / 500,
            -1.5,
            1.5
        )

        utilization_feature = np.clip(
            requests / max(1, servers * 60),
            0,
            1.5
        )

        return np.array(
            [
                traffic_feature,
                server_feature,
                hour_sin,
                hour_cos,
                queue_feature,
                trend_feature,
                utilization_feature,
            ],
            dtype=np.float32,
        )

    def _forward(
        self,
        states,
        weights=None
    ):

        if weights is None:
            weights = self.weights

        z1 = states @ weights["w1"] + weights["b1"]
        a1 = np.maximum(0, z1)

        z2 = a1 @ weights["w2"] + weights["b2"]
        a2 = np.maximum(0, z2)

        q_values = a2 @ weights["w3"] + weights["b3"]

        return q_values, (
            states,
            z1,
            a1,
            z2,
            a2
        )

    def predict(self, state):

        states = np.atleast_2d(state)

        q_values, _ = self._forward(states)

        return q_values[0]

    def choose_action(
        self,
        state,
        allowed_actions=None
    ):

        if allowed_actions is None:

            allowed_actions = list(
                range(self.action_count)
            )

        if random.random() < self.epsilon:

            return random.choice(
                allowed_actions
            )

        q_values = self.predict(state)

        return int(
            max(
                allowed_actions,
                key=lambda action: q_values[action]
            )
        )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_allowed_actions
    ):

        self.memory.append(
            (
                state.copy(),
                action,
                reward,
                next_state.copy(),
                done,
                list(next_allowed_actions),
            )
        )

    def replay(self, batch_size=32):

        if len(self.memory) < batch_size:
            return

        batch = random.sample(
            self.memory,
            batch_size
        )

        states = np.array(
            [item[0] for item in batch],
            dtype=np.float32,
        )

        actions = np.array(
            [item[1] for item in batch],
            dtype=np.int64,
        )

        rewards = np.array(
            [item[2] for item in batch],
            dtype=np.float32,
        )

        next_states = np.array(
            [item[3] for item in batch],
            dtype=np.float32,
        )

        dones = np.array(
            [item[4] for item in batch],
            dtype=bool,
        )

        next_allowed_actions = [
            item[5]
            for item in batch
        ]

        current_q, cache = self._forward(states)

        target_q = current_q.copy()

        next_q, _ = self._forward(
            next_states,
            self.target_weights
        )

        for i in range(batch_size):

            if dones[i]:

                target = rewards[i]

            else:

                allowed_next_q = [

                    next_q[i, action]

                    for action
                    in next_allowed_actions[i]
                ]

                target = (
                    rewards[i]
                    + self.gamma * max(allowed_next_q)
                )

            target_q[i, actions[i]] = target

        self._backpropagate(
            cache,
            current_q,
            target_q
        )

    def _backpropagate(
        self,
        cache,
        predictions,
        targets
    ):

        states, z1, a1, z2, a2 = cache

        batch_size = states.shape[0]

        dq = (
            predictions - targets
        ) / batch_size

        dw3 = a2.T @ dq
        db3 = np.sum(dq, axis=0)

        da2 = dq @ self.weights["w3"].T
        dz2 = da2 * (z2 > 0)

        dw2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.weights["w2"].T
        dz1 = da1 * (z1 > 0)

        dw1 = states.T @ dz1
        db1 = np.sum(dz1, axis=0)

        for name, grad in [

            ("w1", dw1),
            ("b1", db1),

            ("w2", dw2),
            ("b2", db2),

            ("w3", dw3),
            ("b3", db3),
        ]:

            grad = np.clip(
                grad,
                -5,
                5
            )

            self.weights[name] -= (
                self.learning_rate * grad
                + 0.00001 * self.weights[name]
            )

    def update_target_network(self):

        self.target_weights = self._copy_weights(
            self.weights
        )

    def decay_epsilon(self):

        self.epsilon = max(
            0.02,
            self.epsilon * 0.995
        )

    def save(self, path):

        np.savez(path, **self.weights)

    def load(self, path):

        data = np.load(path)

        loaded_weights = {

            key: data[key]

            for key in self.weights
        }

        for key in self.weights:

            if (
                loaded_weights[key].shape
                != self.weights[key].shape
            ):

                raise ValueError(
                    "Saved DQN model incompatible."
                )

        self.weights = loaded_weights

        self.update_target_network()