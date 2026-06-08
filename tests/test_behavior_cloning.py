import json

from rl.behavior_cloning import iter_replay_samples, load_or_extract_samples
from rl.model import FeatureSpec


def _obs(step):
    planets = [
        [0, 0, 10.0, 10.0, 2.0, 30, 2],
        [1, -1, 20.0, 10.0, 2.0, 5, 1],
        [2, 0, 80.0, 80.0, 2.0, 20, 2],
    ]
    return {
        "player": 0,
        "step": step,
        "angular_velocity": 0.0,
        "planets": planets,
        "initial_planets": planets,
        "fleets": [],
        "comet_planet_ids": [],
        "comets": [],
    }


def test_bc_samples_owned_sources_and_filters_opening_steps(tmp_path):
    row0_obs = _obs(0)
    row1_obs = _obs(1)
    row1_obs["planets"][0][5] = 24
    replay = {
        "rewards": [1],
        "steps": [
            [{"observation": row0_obs, "action": []}],
            [{"observation": row1_obs, "action": [[0, 0.0, 6]]}],
        ],
    }
    path = tmp_path / "episode-1-replay.json"
    path.write_text(json.dumps(replay))

    samples = list(
        iter_replay_samples(
            [path], spec=FeatureSpec(), max_samples=None, min_step=1, max_step=2
        )
    )

    assert len(samples) == 2
    by_source = {sample["source_index"]: sample for sample in samples}
    assert by_source[0]["has_launch"] is True
    assert by_source[0]["stop_label"] == 0.0
    assert by_source[0]["target_index"] == 1
    assert by_source[0]["graph"]["planet_features"][0, 4].item() == 50.0
    assert by_source[2]["has_launch"] is False
    assert by_source[2]["stop_label"] == 1.0


def test_bc_skips_initial_no_action_replay_row_by_default(tmp_path):
    replay = {
        "rewards": [1],
        "steps": [
            [{"observation": _obs(0), "action": []}],
            [{"observation": _obs(1), "action": [[0, 0.0, 6]]}],
        ],
    }
    path = tmp_path / "episode-1-replay.json"
    path.write_text(json.dumps(replay))

    samples = list(
        iter_replay_samples([path], spec=FeatureSpec(), max_samples=None, max_step=2)
    )

    assert len(samples) == 2
    by_source = {sample["source_index"]: sample for sample in samples}
    assert by_source[0]["has_launch"] is True
    assert by_source[2]["has_launch"] is False


def test_bc_caches_vectorized_samples(tmp_path):
    replay = {
        "rewards": [1],
        "steps": [
            [{"observation": _obs(0), "action": []}],
            [{"observation": _obs(1), "action": [[0, 0.0, 6]]}],
        ],
    }
    path = tmp_path / "episode-1-replay.json"
    path.write_text(json.dumps(replay))
    cache_dir = tmp_path / "sample-cache"

    samples, loaded = load_or_extract_samples(
        [path], spec=FeatureSpec(), max_step=2, cache_dir=cache_dir
    )
    cached_samples, cached_loaded = load_or_extract_samples(
        [path], spec=FeatureSpec(), max_step=2, cache_dir=cache_dir
    )

    assert loaded is False
    assert cached_loaded is True
    assert len(samples) == 2
    assert len(cached_samples) == 2
    assert len(list(cache_dir.glob("bc_samples_*.pt"))) == 1
