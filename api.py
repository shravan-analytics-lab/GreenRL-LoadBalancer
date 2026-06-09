from flask import Flask, jsonify, request
from flask_cors import CORS

from main import RESULTS_DIR, load_agent, run_simulation


app = Flask(__name__)
CORS(app)


MODEL_MAP = {
    "cheap": "cheap_model.npz",
    "balanced": "balanced_model.npz",
    "reliable": "rich_model.npz",
}


@app.get("/")
def home():
    return jsonify(
        {
            "message": "Green AI Load Balancer API Running",
            "policies": list(MODEL_MAP.keys()),
        }
    )


@app.post("/simulate")
def simulate():
    payload = request.get_json(silent=True) or {}

    policy = str(payload.get("policy", "balanced")).lower()
    if policy not in MODEL_MAP:
        return jsonify({"error": f"Unknown policy: {policy}"}), 400

    try:
        steps = int(payload.get("steps", 300))
    except (TypeError, ValueError):
        return jsonify({"error": "steps must be a number"}), 400

    steps = max(50, min(2000, steps))
    model_name = MODEL_MAP[policy]
    model_path = RESULTS_DIR / model_name

    if not model_path.exists():
        return jsonify({"error": f"Saved model not found: {model_name}"}), 404

    agent, loaded = load_agent(model_name)
    if not loaded:
        return jsonify({"error": f"Could not load model: {model_name}"}), 500

    results = run_simulation(agent, steps)
    results["policy"] = policy
    results["model_name"] = model_name
    results["steps"] = steps

    return jsonify(results)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
