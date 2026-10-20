"""Descriptive post-exit review of all 30 fixed recordings; no new episodes."""
from collections import Counter
import json, struct
from run_check import ROOT, sha, verify, write


def main():
    verify()
    complete=json.loads((ROOT/'run_v1/complete.json').read_text())
    audit=json.loads((ROOT/'run_v1/audit.json').read_text())
    control=json.loads((ROOT/'CONTROL_REVIEW.json').read_text())
    assert audit['status']==control['status']=='pass'
    controls={r['stage']:r for r in control['runs']}
    results=[]
    for summary in complete['episodes']:
        stage=summary['stage'];path=ROOT/'run_v1'/f'{stage}.decisions.jsonl'
        pre=json.loads((ROOT/'run_v1'/f'{stage}.initial.json').read_text())
        accelerations=[];warnings=0;accelerating_warnings=0;impact_exclusions=0;example=None
        for text in path.read_text().splitlines():
            row=json.loads(text);obs=struct.unpack('<163f',bytes.fromhex(row['observation_hex']))
            # FASTER command is not necessarily positive acceleration at maximum speed.
            warning=row['action']==2 and obs[112]<=0
            first=row['physics'][0]
            if warning:
                warnings+=1
                positive=not first['crashed'] and (first['ego'][3]-pre['ego'][3])*15>1e-9
                accelerating_warnings+=positive
                if example is None:
                    example={'input_time_s':pre['simulated_seconds'],'faster_clipped_margin_m':10*obs[112],
                             'positive_first_tick_acceleration':bool(positive)}
            speed=pre['ego'][3]
            for tick in row['physics']:
                if tick['crashed']:impact_exclusions+=1
                else:accelerations.append((tick['ego'][3]-speed)*15)
                speed=tick['ego'][3]
            pre=row['scene']
        assert len(accelerations)==controls[stage]['noncollision_ego_ticks_verified']
        assert impact_exclusions==controls[stage]['impact_ticks_excluded_from_acceleration']
        results.append({'stage':stage,'outcome':controls[stage]['outcome'],
            'max_acceleration_mps2':max(accelerations),'min_acceleration_mps2':min(accelerations),
            'faster_with_nonpositive_forecast_decisions':warnings,
            'of_those_positive_initial_acceleration':accelerating_warnings,
            'collision_ticks_excluded':impact_exclusions,'example':example,'source_sha256':sha(path)})
    counts=dict(Counter(r['outcome'] for r in results))
    write(ROOT/'CASE_REVIEW.json',{'status':'pass','episodes':30,'outcome_counts':counts,'runs':results,
        'method':'All fixed saved traces. Acceleration = 15*(v_next-v), collision ticks excluded. Positive means >1e-9 m/s2 to reject numerical noise. Nonpositive predicted clearance is a warning feature, not proof of an actual conflict.',
        'complete_sha256':sha(ROOT/'run_v1/complete.json'),'control_sha256':sha(ROOT/'CONTROL_REVIEW.json')})
    lines=['# Wider roads: all-policy engineering check v8','',
        'All 30 configured combinations were rerun once: three maps, five saved policies and medium/heavy traffic. One model seed (42), one previously observed scenario per map, zero training. These are exploratory demonstrations, not comparative success-rate estimates.','',
        '## Changes and checks','',
        '- Physical route lanes: 5 to 6 m. NPC lanes: 4 to 5 m, with their parallel centres separated accordingly. Turn centre-lines and the 35 m approach are retained.',
        '- Finish contact and independent finish reconstruction use the full 6 m width. The existing off-route centre-distance tolerance now follows lane width plus half the vehicle width; it is not full-body containment.',
        '- The replay renders those actual widths. It highlights FASTER commands with a nonpositive forecast and clears stale event selections when stepping away from an event.',
        '- Retained v7 corrections: nearest eligible actor ordering, admission before the next input, forward-only NPCs, jammed-route junction reservations and the NPC-only collision guard.',
        '- All native/model decisions, observations, route metrics, rewards and termination conditions were independently reconstructed. Every NPC swept-body pair was checked, including ego-collision ticks. Every noncollision ego speed tick matched its saved policy target and native speed controller.',
        '- 68 unique engineering tests passed. The first 59 passed in the initial suite; all nine new width checks passed after correcting a NumPy compatibility error in their test helper. Initial test errors and corrected results are both retained. No learned-policy episode was consumed by those tests.','',
        '## Recorded outcomes','',str(counts),'',
        '| Case | Outcome | Time (s) | Turns | Stops | Slow with positive FASTER forecast (s) | Acceleration min / max (m/s²) | FASTER with nonpositive forecast / actually accelerating |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    summaries={s['stage']:s for s in complete['episodes']}
    for r in results:
        s=summaries[r['stage']];c=controls[r['stage']]
        lines.append(f"| {r['stage']} | {r['outcome']} | {s['simulated_seconds']:.1f} | {s['turns_completed']}/{s['required_turns']} | {c['stop_entries_below_0_5_mps']} | {c['low_speed_positive_faster_forecast_seconds']:.1f} | {r['min_acceleration_mps2']:.2f} / {r['max_acceleration_mps2']:.2f} | {r['faster_with_nonpositive_forecast_decisions']} / {r['of_those_positive_initial_acceleration']} |")
    lines+=['','## Interpretation and remaining issues','',
        'Widening is an environment change, including traffic positions and interactions. A changed outcome cannot be attributed to a better learned policy. Original PPO, QR-DQN, Hybrid 25%, CAPE and CAGE weights/rules and all earlier evidence remain unchanged.',
        '', 'Aggressive or hesitant learned actions remain visible. FASTER can merely maintain the 9 m/s maximum; the table separates command warnings from actual initial acceleration. Collision ticks are excluded from acceleration calculations. The native proportional speed controller uses targets 0, 4.5 and 9 m/s and a 0.6 s time constant, without a comfort limiter.',
        '', 'Slow-with-positive-forecast time uses an input speed below 0.5 m/s and a positive FASTER margin. This is not proof of an empty or safe road. Forecasts use at most 14 eligible actors, a three-second horizon and constant NPC velocities. They do not supply intentions, reservation state, route-progress memory or a safety guarantee. CAPE/CAGE address specific alias conditions; they are not universal progress controllers.',
        '', 'The guard only changes NPC motion. Abrupt NPC holds and queues remain simulator approximations. The audit found no NPC–NPC body overlap in these recorded runs; ego collisions and legitimate queue timeouts remain outcomes. No forced ego progress, comfort smoothing, safety shield, policy tuning or new training was introduced.',
        '', 'There is no overall duration limit. Native body contact at the finish after all turns produces COMPLETE; collision/off-route take precedence. Twenty simulated seconds within 0.5 m causes a timeout, including in a legitimate queue. At any terminal event, the recorded replay pauses all cars.',
        '', 'This is not CARLA, a realistic motorway-speed study, a generalization test or evidence of real-world safety. Only the 30 configured demonstration cases were checked.',
        '', '## Files','',
        '[Replay](CITY_REPLAY.html) · [Native outcome tables](RESULTS.md) · [Case diagnostics](CASE_REVIEW.json) · [All-decision audit](run_v1/audit.json) · [Controller/NPC audit](CONTROL_REVIEW.json) · [Protocol](protocol.json)',
        '',f"Release SHA256: `{sha(ROOT/'release.json')}`"]
    with (ROOT/'README.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print('ALL_CASE_REVIEW_PASS',counts)

if __name__=='__main__':main()
