import matplotlib.pyplot as plt

def plot_results(traffic, latency_rl, latency_rule):
    plt.plot(traffic, label="Traffic")
    plt.plot(latency_rl, label="RL Latency")
    plt.plot(latency_rule, label="Rule Latency")
    plt.legend()
    plt.title("RL vs Rule-Based")
    plt.show()