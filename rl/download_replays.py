from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def kaggle_bin() -> str:
    return str(Path(sys.executable).with_name("kaggle"))


def list_episode_ids(submission_id: int, limit: int) -> list[int]:
    proc = subprocess.run(
        [kaggle_bin(), "competitions", "episodes", str(submission_id), "-v"],
        check=True,
        text=True,
        capture_output=True,
    )
    ids: list[int] = []
    for row in csv.DictReader(proc.stdout.splitlines()):
        if row.get("state") != "EpisodeState.COMPLETED":
            continue
        if row.get("type") != "EpisodeType.EPISODE_TYPE_PUBLIC":
            continue
        ids.append(int(row["id"]))
        if len(ids) >= limit:
            break
    return ids


def download_replay(episode_id: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            kaggle_bin(),
            "competitions",
            "replay",
            str(episode_id),
            "-p",
            str(out_dir),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_id", type=int)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or Path(f"rl/data/replays/{args.submission_id}")
    for episode_id in list_episode_ids(args.submission_id, args.limit):
        download_replay(episode_id, out_dir)


if __name__ == "__main__":
    main()
