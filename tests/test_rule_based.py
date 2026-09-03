from types import SimpleNamespace

from safeintent_rl.agents import RuleBasedAgent


def make_agent() -> RuleBasedAgent:
    action_type = SimpleNamespace(actions={0: "SLOWER", 1: "IDLE", 2: "FASTER"})
    env = SimpleNamespace(unwrapped=SimpleNamespace(action_type=action_type))
    return RuleBasedAgent(env, brake_ttc=2.0, accelerate_ttc=4.0)


def test_rule_based_agent_brakes_at_critical_ttc() -> None:
    agent = make_agent()
    assert agent.predict(None, min_ttc=2.0) == 0


def test_rule_based_agent_idles_between_thresholds() -> None:
    agent = make_agent()
    assert agent.predict(None, min_ttc=3.0) == 1


def test_rule_based_agent_accelerates_at_safe_ttc() -> None:
    agent = make_agent()
    assert agent.predict(None, min_ttc=4.0) == 2
