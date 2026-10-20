"""PPO/QR-DQN command-class arbitration. Pure arithmetic, no simulator.

This is a new project-specific ensemble candidate, not a novelty claim.
Predicted return tails and forecast margins are imperfect proxies, not safety
certificates. The QR baseline always remains available as a fallback.
"""
import math

WEIGHTS = (0.25, 0.5, 0.75)
QUANTILES = 200
TAIL_COUNT = 40


def require(test, message):
    if not test:
        raise ValueError(message)


def finite_vector(values, n, label):
    require(isinstance(values, (tuple, list)) and len(values) == n, label + ' shape')
    require(all(type(x) in (int, float) and math.isfinite(x) for x in values), label + ' finite')


def select(probabilities, action_quantiles, partition, forecast, weight):
    """Return a complete reproducible decision from Python numeric values.

    action_quantiles is action-major [3][200]. Sort each action's predicted
    quantile values for a lower-20%-tail summary; this is not calibrated CVaR.
    For a command class take its highest expected-return label (lowest label
    on an exact tie). Sum PPO probabilities over all class members. Normalize
    representative Q means across classes to [0,1], using all zeros if tied.
    Alternative classes require positive clearance, forecast margin/time no
    lower than the QR representative, and a return tail no lower than QR's.
    Maximize (1-weight)*normalized_Q + weight*PPO_class_mass among eligible
    classes. Exact score ties retain QR, then the lowest representative label.
    """
    require(type(weight) is float and weight in WEIGHTS, 'unregistered weight')
    finite_vector(probabilities, 3, 'probabilities')
    require(all(0 <= p <= 1 for p in probabilities)
            and abs(math.fsum(probabilities) - 1) <= 1e-6, 'probability simplex')
    require(isinstance(partition, (tuple, list)) and len(partition) == 3
            and all(type(c) is int and 0 <= c <= 2 for c in partition), 'partition')
    require(isinstance(action_quantiles, (tuple, list)) and len(action_quantiles) == 3, 'quantile actions')
    require(isinstance(forecast, (tuple, list)) and len(forecast) == 3, 'forecast actions')
    for q, f in zip(action_quantiles, forecast):
        finite_vector(q, QUANTILES, 'quantiles')
        finite_vector(f, 3, 'forecast')
        require(-1 <= f[0] <= 1 and 0 <= f[1] <= 1 and 0 <= f[2] <= 1, 'forecast range')
    means = [math.fsum(q) / QUANTILES for q in action_quantiles]
    tails = [math.fsum(sorted(q)[:TAIL_COUNT]) / TAIL_COUNT for q in action_quantiles]
    baseline = max(range(3), key=lambda a: (means[a], -a))
    classes = sorted(set(partition))
    members = {c: [a for a in range(3) if partition[a] == c] for c in classes}
    reps = {c: max(members[c], key=lambda a: (means[a], -a)) for c in classes}
    low, high = min(means[a] for a in reps.values()), max(means[a] for a in reps.values())
    details = []
    for c in classes:
        a = reps[c]
        mass = math.fsum(probabilities[b] for b in members[c])
        normalized = 0.0 if high == low else (means[a] - low) / (high - low)
        tests = dict(positive_clearance=forecast[a][0] > 0,
                     clearance_not_lower=forecast[a][0] >= forecast[baseline][0],
                     forecast_time_not_lower=forecast[a][1] >= forecast[baseline][1],
                     return_tail_not_lower=tails[a] >= tails[baseline])
        eligible = c == partition[baseline] or all(tests.values())
        details.append(dict(command_class=c, members=members[c], representative=a,
                            ppo_mass=mass, q_mean=means[a], q_normalized=normalized,
                            lower_tail=tails[a], gate=tests, eligible=eligible,
                            score=(1 - weight) * normalized + weight * mass))
    chosen = max((d for d in details if d['eligible']),
                 key=lambda d: (d['score'], d['command_class'] == partition[baseline],
                                -d['representative']))
    selected = baseline if chosen['command_class'] == partition[baseline] else chosen['representative']
    return dict(baseline=baseline, ppo_action=max(range(3), key=lambda a: (probabilities[a], -a)),
                selected=selected, changed=selected != baseline, weight=weight,
                action_means=means, action_lower_tails=tails, classes=details)
