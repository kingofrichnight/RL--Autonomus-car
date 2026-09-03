from safeintent_rl.config import load_config, split_env_config


def test_reward_v2_changes_only_reward_coefficients() -> None:
    _, baseline = split_env_config(load_config("configs/intersection.yaml"))
    env_id, reward_v2 = split_env_config(load_config("configs/intersection_reward_v2.yaml"))

    assert env_id == "intersection-v2"
    changed = {key for key in baseline if baseline[key] != reward_v2[key]}
    assert changed == {"collision_reward", "arrived_reward", "high_speed_reward"}


def test_collision_cost_exceeds_maximum_episode_speed_reward() -> None:
    _, config = split_env_config(load_config("configs/intersection_reward_v2.yaml"))
    maximum_speed_return = (
        config["duration"] * config["policy_frequency"] * config["high_speed_reward"]
    )

    assert maximum_speed_return == 7.5
    assert abs(config["collision_reward"]) > maximum_speed_return
