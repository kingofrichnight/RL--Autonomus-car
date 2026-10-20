"""Frozen candidate v2: soft return-tail penalty and forecast-gated ensemble.

v1 is preserved as the first engineering prototype. No simulator episode was
used to choose this revision. QR predictions describe the learned QR return
distribution, not the new hybrid's actual return or collision probability.
"""
import math
from hybrid_rule import finite_vector, require

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
TAIL_FRACTION = 0.2
TAIL_WEIGHT = 0.25
QUANTILES = 200


def select(probabilities, action_quantiles, partition, forecast, weight, qr_baseline):
    require(type(weight) is float and weight in WEIGHTS, 'unregistered weight')
    finite_vector(probabilities, 3, 'probabilities')
    require(all(0 <= p <= 1 for p in probabilities)
            and abs(math.fsum(probabilities) - 1) <= 1e-6, 'probability simplex')
    require(type(qr_baseline) is int and qr_baseline in (0, 1, 2), 'native QR baseline')
    require(isinstance(partition, (list, tuple)) and len(partition) == 3
            and all(type(c) is int and 0 <= c <= 2 for c in partition), 'partition')
    require(isinstance(action_quantiles, (list, tuple)) and len(action_quantiles) == 3, 'quantile actions')
    require(isinstance(forecast, (list, tuple)) and len(forecast) == 3, 'forecast actions')
    for q, f in zip(action_quantiles, forecast):
        finite_vector(q, QUANTILES, 'quantiles')
        finite_vector(f, 3, 'forecast')
        require(-1 <= f[0] <= 1 and 0 <= f[1] <= 1 and 0 <= f[2] <= 1, 'forecast range')
    means = [math.fsum(q) / QUANTILES for q in action_quantiles]
    tails = [math.fsum(sorted(q)[:40]) / 40 for q in action_quantiles]
    risk_values = [0.75 * mean + 0.25 * tail for mean, tail in zip(means, tails)]
    baseline_class = partition[qr_baseline]
    classes = sorted(set(partition))
    members = {c: [a for a in range(3) if partition[a] == c] for c in classes}
    # Preserve native float32 QR argmax for its class; elsewhere use the
    # highest mean-return label, with a deterministic lowest-label tie.
    reps = {c: (qr_baseline if c == baseline_class else
                max(members[c], key=lambda a: (means[a], -a))) for c in classes}
    low, high = min(risk_values[a] for a in reps.values()), max(risk_values[a] for a in reps.values())
    # A fixed reward-unit floor prevents amplification of near-equal estimates.
    # It is numerical regularization, not a tuned reward/risk threshold.
    denominator = max(high - low, 1e-6)
    details = []
    for c in classes:
        a = reps[c]
        mass = math.fsum(probabilities[b] for b in members[c])
        normalized = (risk_values[a] - low) / denominator
        gate = dict(positive_clearance=forecast[a][0] > 0,
                    clearance_not_lower=forecast[a][0] >= forecast[qr_baseline][0],
                    forecast_time_not_lower=forecast[a][1] >= forecast[qr_baseline][1])
        eligible = c == baseline_class or all(gate.values())
        details.append(dict(command_class=c, members=members[c], representative=a,
                            ppo_mass=mass, q_mean=means[a], lower_tail=tails[a],
                            risk_value=risk_values[a], normalized=normalized,
                            gate=gate, eligible=eligible,
                            score=(1 - weight) * normalized + weight * mass))
    winner = max((d for d in details if d['eligible']),
                 key=lambda d: (d['score'], d['command_class'] == baseline_class, -d['representative']))
    selected = winner['representative']
    return dict(baseline=qr_baseline, selected=selected, changed=selected != qr_baseline,
                ppo_action=max(range(3), key=lambda a: (probabilities[a], -a)),
                weight=weight, classes=details, normalization_denominator=denominator)
