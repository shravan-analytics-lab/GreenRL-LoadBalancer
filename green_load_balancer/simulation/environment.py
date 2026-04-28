class Environment:
    def __init__(self):
        self.servers = 2
        self.capacity = 100

    def reset(self):
        self.servers = 2

    def step(self, requests, action):
        # Actions: 0 = nothing, 1 = add, 2 = remove
        if action == 1:
            self.servers += 1
        elif action == 2:
            self.servers = max(1, self.servers - 1)

        cpu = (requests / (self.servers * self.capacity)) * 100
        latency = cpu * 2

        reward = -latency - (self.servers * 5)

        return cpu, latency, reward, self.servers