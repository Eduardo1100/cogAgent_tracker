import marimo

__generated_with = "0.20.4"
app = marimo.App(width="full", app_title="cogAgent Explorer")


@app.cell
def _():
    import marimo as mo

    mo.md("# cogAgent Experiment Explorer")
    return (mo,)


@app.cell
def _(mo):
    import os

    import pandas as pd
    from sqlalchemy import create_engine, text

    DATABASE_URL = os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5432/cogagent",
    )
    # Convert async URL to sync if needed
    sync_url = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql+psycopg2://"
    )
    engine = create_engine(sync_url)

    def query(sql, **kwargs):
        with engine.connect() as conn:
            return pd.read_sql(text(sql), conn, params=kwargs)

    mo.md("**DB connected** ✓")
    return os, query


@app.cell
def _(mo, query):
    experiments_df = query(
        "SELECT id, agent_name, split, start_time FROM experiment_runs ORDER BY id DESC"
    )
    experiment_options = {
        f"#{row.id} — {row.agent_name} ({row.split}) @ {str(row.start_time)[:16]}": row.id
        for _, row in experiments_df.iterrows()
    }
    experiment_selector = mo.ui.dropdown(
        options=experiment_options,
        label="Experiment",
    )
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(experiment_selector, mo, query):
    exp_id = experiment_selector.value
    if exp_id is None:
        mo.stop(True, mo.md("*Select an experiment above.*"))

    run_df = query("SELECT * FROM experiment_runs WHERE id = :id", id=exp_id)
    mo.md("## Run Summary")
    return exp_id, run_df


@app.cell
def _(mo, run_df):
    mo.ui.table(run_df)
    return


@app.cell
def _(exp_id, mo, query):
    episodes_df = query(
        "SELECT * FROM episode_runs WHERE experiment_id = :id ORDER BY game_number",
        id=exp_id,
    )
    mo.md(f"## Episode Runs ({len(episodes_df)} episodes)")
    return (episodes_df,)


@app.cell
def _(episodes_df, mo):
    mo.ui.table(episodes_df)
    return


@app.cell
def _(episodes_df, mo):
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Episode Metrics", fontsize=14)

    # 1. Success per game
    ax = axes[0, 0]
    colors = ["green" if s else "red" for s in episodes_df["success"]]
    ax.bar(episodes_df["game_number"], episodes_df["success"].astype(int), color=colors)
    ax.set_title("Success per Game")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Success")
    ax.set_yticks([0, 1])

    # 2. Inadmissible actions
    ax = axes[0, 1]
    ax.bar(
        episodes_df["game_number"],
        episodes_df["inadmissible_action_count"],
        color="orange",
    )
    ax.set_title("Inadmissible Actions per Game")
    ax.set_xlabel("Game #")
    ax.set_ylabel("Count")

    # 3. Task type success rate
    ax = axes[1, 0]
    task_map = {
        1: "pick_place",
        2: "look_light",
        3: "clean_place",
        4: "heat_place",
        5: "cool_place",
        6: "two_object",
    }
    if "task_type" in episodes_df.columns:
        task_stats = (
            episodes_df.groupby("task_type")["success"]
            .mean()
            .mul(100)
            .rename(index=task_map)
        )
        task_stats.plot(kind="barh", ax=ax, color="steelblue")
        ax.set_title("Success % by Task Type")
        ax.set_xlabel("Success %")
    else:
        ax.set_title("Task type data unavailable")

    # 4. Tokens per game
    ax = axes[1, 1]
    if (
        "prompt_tokens" in episodes_df.columns
        and "completion_tokens" in episodes_df.columns
    ):
        ax.bar(
            episodes_df["game_number"],
            episodes_df["prompt_tokens"],
            label="Prompt",
            color="steelblue",
        )
        ax.bar(
            episodes_df["game_number"],
            episodes_df["completion_tokens"],
            bottom=episodes_df["prompt_tokens"],
            label="Completion",
            color="coral",
        )
        ax.set_title("Tokens per Game")
        ax.set_xlabel("Game #")
        ax.set_ylabel("Tokens")
        ax.legend()
    else:
        ax.set_title("Token data unavailable")

    plt.tight_layout()
    mo.as_html(fig)
    return


@app.cell
def _(exp_id, mo, os):
    import redis

    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        status_key = f"eval:status:{exp_id}"
        status_data = r.hgetall(status_key)
        if status_data:
            mo.md(f"## Live Redis Status\n\n```\n{status_data}\n```")
        else:
            mo.md(f"## Live Redis Status\n\n*No live data at `{status_key}`*")
    except Exception as e:
        mo.md(f"## Live Redis Status\n\n*Redis unavailable: {e}*")
    return


if __name__ == "__main__":
    app.run()
