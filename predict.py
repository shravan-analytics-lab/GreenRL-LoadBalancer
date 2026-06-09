from pathlib import Path

from r1_agent.q_learning import QLearningAgent

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "results" / "balanced_model.npz"

agent = QLearningAgent()

agent.load(MODEL_PATH)

print("Model loaded successfully!")

state_example = agent.get_state(
    requests=300,
    servers=5,
    hour=14,
    queue=20,
    trend=30
)

action = agent.choose_action(
    state_example,
    [0,1,2,3,4]
)

print("Predicted Action:", action)