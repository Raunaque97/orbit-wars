import argparse
import os
from pathlib import Path

from kaggle_environments import make


def main():
    parser = argparse.ArgumentParser(
        description="Run an Orbit Wars match and write Kaggle's HTML replay."
    )
    parser.add_argument("--agent-a", default="main.py")
    parser.add_argument("--agent-b", default="main.py")
    parser.add_argument("--output", default="replay.html")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--budget-ms", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.budget_ms is not None:
        os.environ["ORBIT_WARS_TIME_BUDGET_MS"] = str(args.budget_ms)

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=args.debug)
    env.run([args.agent_a, args.agent_b])

    html = env.render(mode="html", width=args.width, height=args.height)
    output = Path(args.output)
    output.write_text(html, encoding="utf-8")

    final = env.steps[-1]
    print(f"wrote {output.resolve()}")
    for i, state in enumerate(final):
        print(f"player {i}: reward={state.reward}, status={state.status}")


if __name__ == "__main__":
    main()
