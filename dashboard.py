import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots


API_URL = "http://127.0.0.1:5000/simulate"

POLICY_OPTIONS = {
    "Cheap": "cheap",
    "Balanced": "balanced",
    "Reliable": "reliable",
}


st.set_page_config(
    page_title="Green AI Load Balancer",
    page_icon="AI",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(0, 229, 160, 0.12), transparent 28%),
            linear-gradient(135deg, #071018 0%, #0c1724 45%, #10131f 100%);
        color: #eaf4ff;
    }
    [data-testid="stSidebar"] {
        background: #07111c;
        border-right: 1px solid rgba(118, 255, 214, 0.16);
    }
    .hero {
        padding: 1.4rem 0 0.6rem 0;
    }
    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.2rem;
        color: #f7fbff;
    }
    .hero p {
        color: #a8bacd;
        font-size: 1.02rem;
        margin: 0;
    }
    .metric-card {
        background: rgba(11, 24, 38, 0.92);
        border: 1px solid rgba(118, 255, 214, 0.16);
        border-radius: 8px;
        padding: 1rem;
        min-height: 116px;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.22);
    }
    .metric-label {
        color: #95a8ba;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0;
    }
    .metric-value {
        color: #f5fbff;
        font-size: 1.55rem;
        font-weight: 700;
        margin-top: 0.35rem;
    }
    .metric-sub {
        color: #58dfbd;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }
    .section-title {
        color: #f7fbff;
        font-size: 1.25rem;
        font-weight: 700;
        margin: 1.2rem 0 0.6rem 0;
    }
    .conclusion {
        background: linear-gradient(135deg, rgba(32, 216, 168, 0.16), rgba(76, 143, 255, 0.12));
        border: 1px solid rgba(118, 255, 214, 0.22);
        border-radius: 8px;
        padding: 1rem 1.2rem;
        color: #edfaff;
        font-size: 1.02rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(118, 255, 214, 0.14);
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def call_backend(policy, steps):
    response = requests.post(
        API_URL,
        json={"policy": policy, "steps": steps},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def metric_card(label, value, subtext):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_layout(fig, title, y_title=None):
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=55, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,18,30,0.74)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Simulation step", gridcolor="rgba(255,255,255,0.06)")
    if y_title:
        fig.update_yaxes(title_text=y_title, gridcolor="rgba(255,255,255,0.06)")
    return fig


def line_chart(title, series, y_title):
    fig = go.Figure()
    colors = {
        "RL": "#44e0b6",
        "Rule": "#ffb86b",
        "Traffic": "#62a8ff",
    }
    for name, values in series.items():
        fig.add_trace(
            go.Scatter(
                y=values,
                mode="lines",
                name=name,
                line=dict(width=2.4, color=colors.get(name)),
            )
        )
    return chart_layout(fig, title, y_title)


def traffic_latency_chart(data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            y=data["traffic_history"],
            mode="lines",
            name="Traffic",
            line=dict(color="#62a8ff", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            y=data["latency_rl"],
            mode="lines",
            name="RL Latency",
            line=dict(color="#44e0b6", width=2.4),
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            y=data["latency_rule"],
            mode="lines",
            name="Rule Latency",
            line=dict(color="#ffb86b", width=2.4),
        ),
        secondary_y=True,
    )
    chart_layout(fig, "Traffic vs Latency")
    fig.update_yaxes(title_text="Traffic requests", secondary_y=False)
    fig.update_yaxes(title_text="Latency ms", secondary_y=True)
    return fig


def evaluation_table(data):
    return pd.DataFrame(
        {
            "Metric": [
                "Average latency (ms)",
                "Average queue",
                "Average servers",
                "Average utilization (%)",
                "P95 latency (ms)",
                "SLA violations",
                "Composite score",
            ],
            "RL system": [
                data["avg_rl_latency"],
                data["avg_rl_queue"],
                data["avg_rl_servers"],
                data["avg_rl_utilization"],
                data["p95_rl_latency"],
                data["rl_sla_violations"],
                data["current_score"],
            ],
            "Rule-based system": [
                data["avg_rule_latency"],
                data["avg_rule_queue"],
                data["avg_rule_servers"],
                data["avg_rule_utilization"],
                data["p95_rule_latency"],
                data["rule_sla_violations"],
                data["rule_score"],
            ],
        }
    )


with st.sidebar:
    st.title("Control Panel")
    selected_label = st.selectbox("RL Policy Selector", list(POLICY_OPTIONS.keys()), index=1)
    steps = st.slider("Simulation Steps", min_value=100, max_value=1000, value=300, step=50)
    run_clicked = st.button("Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.caption("Backend: Flask API")
    st.caption("Charts: Plotly dark theme")
    st.caption("Models: saved .npz policies")


st.markdown(
    """
    <div class="hero">
        <h1>Green AI Load Balancer</h1>
        <p>Deep Reinforcement Learning cloud autoscaling simulator with live RL vs rule-based comparison.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "simulation_data" not in st.session_state:
    st.info("Select a policy and run the simulation to load real backend evaluation data.")

if run_clicked:
    with st.spinner("Running real backend simulation..."):
        try:
            st.session_state.simulation_data = call_backend(
                POLICY_OPTIONS[selected_label],
                steps,
            )
        except requests.exceptions.ConnectionError:
            st.error("Flask API is not running. Start it with: python api.py")
        except requests.exceptions.HTTPError as error:
            st.error(f"API error: {error.response.text}")
        except requests.exceptions.RequestException as error:
            st.error(f"Could not run simulation: {error}")


data = st.session_state.get("simulation_data")
if data:
    policy_name = data["policy"].title()
    st.markdown(f'<div class="section-title">Policy: {policy_name}</div>', unsafe_allow_html=True)

    row1 = st.columns(4)
    with row1[0]:
        metric_card("RL latency", f'{data["avg_rl_latency"]:.2f} ms', "Lower is better")
    with row1[1]:
        metric_card("Rule latency", f'{data["avg_rule_latency"]:.2f} ms', "Baseline comparison")
    with row1[2]:
        metric_card("RL cost", f'${data["avg_rl_cost"]:.2f}', "Average operational cost")
    with row1[3]:
        metric_card("Rule cost", f'${data["avg_rule_cost"]:.2f}', "Average operational cost")

    row2 = st.columns(4)
    with row2[0]:
        metric_card("RL SLA violations", f'{data["rl_sla_violations"]}', "Latency > 120 ms")
    with row2[1]:
        metric_card("Rule SLA violations", f'{data["rule_sla_violations"]}', "Latency > 120 ms")
    with row2[2]:
        metric_card("RL composite score", f'{data["current_score"]:.2f}', "Lower is better")
    with row2[3]:
        metric_card("Rule composite score", f'{data["rule_score"]:.2f}', "Lower is better")

    st.markdown('<div class="section-title">Live Histories</div>', unsafe_allow_html=True)

    chart_cols_1 = st.columns(2)
    with chart_cols_1[0]:
        st.plotly_chart(traffic_latency_chart(data), use_container_width=True)
    with chart_cols_1[1]:
        st.plotly_chart(
            line_chart(
                "Queue Comparison",
                {"RL": data["queue_rl"], "Rule": data["queue_rule"]},
                "Queued requests",
            ),
            use_container_width=True,
        )

    chart_cols_2 = st.columns(2)
    with chart_cols_2[0]:
        st.plotly_chart(
            line_chart(
                "Server Scaling Comparison",
                {"RL": data["servers_rl"], "Rule": data["servers_rule"]},
                "Active servers",
            ),
            use_container_width=True,
        )
    with chart_cols_2[1]:
        st.plotly_chart(
            line_chart(
                "Operational Cost Comparison",
                {"RL": data["cost_rl"], "Rule": data["cost_rule"]},
                "Cost units",
            ),
            use_container_width=True,
        )

    st.plotly_chart(
        line_chart(
            "Utilization Comparison",
            {"RL": data["utilization_rl"], "Rule": data["utilization_rule"]},
            "Utilization %",
        ),
        use_container_width=True,
    )

    st.markdown('<div class="section-title">Final Evaluation Table</div>', unsafe_allow_html=True)
    table = evaluation_table(data)
    st.dataframe(
        table.style.format(
            {
                "RL system": "{:.2f}",
                "Rule-based system": "{:.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown('<div class="section-title">Conclusion</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="conclusion">{data["conclusion"]}</div>', unsafe_allow_html=True)
