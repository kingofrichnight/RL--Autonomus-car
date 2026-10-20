# Driving Lab: city-grid intersections and Progress Assist v1

Thirty recorded evaluations (4 new40second heavy-jam runs and26 retained completions): three maps, five saved policies with Progress Assist, medium/heavy traffic. No training. One model seed (42), one previously observed scenario per map; exploratory demonstrations, not success-rate estimates.

## Rules

- v9 highway route lanes are 7 m and traffic lanes 6 m; city 8 m / 6 m and jam 6 m / 5 m. All ten city turns now coincide with the populated street grid. Highway/jam retain v9 branch-road geometry and recordings. These branches have no additional NPC spawning. The city has a dedicated central ego corridor; traffic runs in separated outer corridors. Forecast and assistance thresholds are unchanged from v11. Finish contact, off-route checks and rendering use those widths. It retains v7, which corrects actor ranking on every map to physical proximity and moves admission before the next observation. The saved weights are unchanged but input ordering changes: this is a labelled transfer demonstration. The 14-slot cap and legacy rear filter remain; no complete-perception claim.
- New telemetry shows last noncollision-tick acceleration, the selected-action forecast, and outlines actors included in the preceding decision input. Playback defaults to 1x. Progress Assist qualifies after five clear forecasts, retains recovery with hysteresis, caps turn targets, and brakes on predicted conflict. This is a heuristic, not a safety guarantee.
- City and jammed-route NPCs reserve crossings in arrival order, yield outside conflict boxes, check downstream space and remain forward-only; this is a traffic-dynamics correction, not policy training. All maps now use forward-only NPCs, consistent observation ranking and corrected admission timing.
- No overall simulation-time or decision-count limit. Completion, collision and off-route remain terminal.
- Replay display: a checkered stripe marks the finite finish line. Playback stops and displays COMPLETE at the first saved replay frame where the car body touches it without collision/off-route and after all required turns. The replay shows 100% completion then. The new native environment checks the same finish contact at each physics tick. Display classifications agree with all 30 displayed native outcomes; centre-progress metrics and native end times remain in the table below and source files.
- Standstill timeout: 40 simulated seconds within 0.5m of the most recent movement anchor. Moving farther resets the timer. Checked at 15Hz, including legitimate queues.
- Each route has a 40m added approach. Spawn is 35m before the original entrance; queue spawns stay beyond it.
- NPC-only coordination rejects conflicting proposed physics movements and holds those NPCs at their preceding clear poses with zero speed. This can stop traffic abruptly or cause queues; it is not a realistic braking model. Native ego collisions are unchanged.
- All original studies and recordings are preserved. These changed-environment results must not replace or be pooled with any earlier demonstration or study results.

## City district · 10 turns

Ten turns at street-grid intersections, with separated corridors and NPC turn-taking at crossings. Traffic cap: 44 medium / 72 heavy. Start 35 m before map entry. No fixed duration; 40 s within 0.5 m ends the run.

Car-shaped NPC coordination permits adjacent-lane passing and holds conflicting movements; abrupt holds are a simulator aid, not realistic braking. Progress Assist may override saved speed actions; it is not a safety guarantee. City NPCs now take turns entering junction conflict boxes; spawning inside those boxes is rejected. This can change initial traffic placement and counts. Ego proposals and assist are unchanged. City v12: 8 m ego corridor, 6 m NPC corridors, traffic centres 8 m from street axis; highway 7 m / 6 m; jam 6 m / 5 m; city turns are on populated grid junctions; highway/jam have native branch roads through turns; lane centres, finish and off-road thresholds use the wider geometry. Retained from v7: all maps use physical-distance input ranking and corrected admission timing. Saved weights are unchanged; assisted actions and reasons are recorded separately. 

| Policy | Traffic | Outcome | Turns | Progress | Time | NPC hold-ticks |
|---|---|---|---:|---:|---:|---:|
| ppo | medium | Finished | 10/10 | 99.8% | 167.67s | 0 |
| qr | medium | Finished | 10/10 | 99.8% | 151.07s | 0 |
| hybrid025 | medium | Finished | 10/10 | 99.8% | 151.07s | 0 |
| cape | medium | Finished | 10/10 | 99.8% | 151.07s | 0 |
| cage | medium | Finished | 10/10 | 99.8% | 151.07s | 0 |
| ppo | heavy | Finished | 10/10 | 99.8% | 160.13s | 0 |
| qr | heavy | Finished | 10/10 | 99.8% | 159.07s | 0 |
| hybrid025 | heavy | Finished | 10/10 | 99.8% | 159.07s | 0 |
| cape | heavy | Finished | 10/10 | 99.8% | 159.07s | 0 |
| cage | heavy | Finished | 10/10 | 99.8% | 159.07s | 0 |

## Highway connectors

Four turns through crossroads with straight-through road alternatives. Traffic cap: 40 medium / 60 heavy. Start 35 m before map entry. No fixed duration; 40 s within 0.5 m ends the run.

Car-shaped NPC coordination permits adjacent-lane passing and holds conflicting movements; abrupt holds are a simulator aid, not realistic braking. Progress Assist may override saved speed actions; it is not a safety guarantee. Low model-compatible speeds (0, 16.2, 32.4 km/h), not motorway-speed validation.City v12: 8 m ego corridor, 6 m NPC corridors, traffic centres 8 m from street axis; highway 7 m / 6 m; jam 6 m / 5 m; city turns are on populated grid junctions; highway/jam have native branch roads through turns; lane centres, finish and off-road thresholds use the wider geometry. Retained from v7: all maps use physical-distance input ranking and corrected admission timing. Saved weights are unchanged; assisted actions and reasons are recorded separately. 

| Policy | Traffic | Outcome | Turns | Progress | Time | NPC hold-ticks |
|---|---|---|---:|---:|---:|---:|
| ppo | medium | Finished | 4/4 | 99.7% | 104.47s | 0 |
| qr | medium | Finished | 4/4 | 99.7% | 104.47s | 0 |
| hybrid025 | medium | Finished | 4/4 | 99.7% | 104.47s | 0 |
| cape | medium | Finished | 4/4 | 99.7% | 104.47s | 0 |
| cage | medium | Finished | 4/4 | 99.7% | 104.47s | 0 |
| ppo | heavy | Finished | 4/4 | 99.7% | 109.67s | 0 |
| qr | heavy | Finished | 4/4 | 99.7% | 110.27s | 0 |
| hybrid025 | heavy | Finished | 4/4 | 99.7% | 110.27s | 0 |
| cape | heavy | Finished | 4/4 | 99.8% | 110.73s | 0 |
| cage | heavy | Finished | 4/4 | 99.8% | 110.73s | 0 |

## Jammed three-turn route

Three turns, slow queues and crossing traffic. Traffic cap: 32 medium / 52 heavy. Start 35 m before map entry. No fixed duration; 40 s within 0.5 m ends the run.

Car-shaped NPC coordination permits adjacent-lane passing and holds conflicting movements; abrupt holds are a simulator aid, not realistic braking. Progress Assist may override saved speed actions; it is not a safety guarantee. City v12: 8 m ego corridor, 6 m NPC corridors, traffic centres 8 m from street axis; highway 7 m / 6 m; jam 6 m / 5 m; city turns are on populated grid junctions; highway/jam have native branch roads through turns; lane centres, finish and off-road thresholds use the wider geometry. Retained from v7: all maps use physical-distance input ranking and corrected admission timing. Saved weights are unchanged; assisted actions and reasons are recorded separately. Retained: nearest eligible NPCs ranked by physical distance; admissions occur before the next observation. Changed input ordering with unchanged model weights. Junction reservations and forward-only braking retained. Queue speed caps: 12.6 / 6.5 km/h. A legitimate queue can trigger the standstill timeout.

| Policy | Traffic | Outcome | Turns | Progress | Time | NPC hold-ticks |
|---|---|---|---:|---:|---:|---:|
| ppo | medium | Finished | 3/3 | 99.3% | 86.73s | 0 |
| qr | medium | Finished | 3/3 | 99.3% | 86.60s | 0 |
| hybrid025 | medium | Finished | 3/3 | 99.4% | 86.93s | 0 |
| cape | medium | Finished | 3/3 | 99.3% | 86.60s | 0 |
| cage | medium | Finished | 3/3 | 99.3% | 86.60s | 0 |
| ppo | heavy | Finished | 3/3 | 99.4% | 191.60s | 0 |
| qr | heavy | Standstill timeout (40s) | 2/3 | 60.7% | 160.20s | 0 |
| hybrid025 | heavy | Standstill timeout (40s) | 2/3 | 60.7% | 162.00s | 0 |
| cape | heavy | Standstill timeout (40s) | 2/3 | 60.7% | 160.20s | 0 |
| cage | heavy | Standstill timeout (40s) | 2/3 | 60.7% | 160.20s | 0 |

## Recording provenance

The26 retained successful recordings were originally made with a20second standstill deadline. They never reached that deadline and finish before the new40second condition can matter. The4 earlier timeouts were rerun from the initial seed; no frames were resumed, padded or relabelled. All30 movement clocks are independently reconstructed against40seconds.

## Verification and limits

The native trace audit passed. A separate posthoc compatibility checker corrects the original checker assumption that traffic lists stay identical at the former cutoff: continuing the hybrid run removes one NPC beyond its lane endpoint. Exact physics, proposals, actions and retained traffic states are verified. The failed optional checker and its error remain preserved. See TIMEOUT_COMPATIBILITY.json.

All 19563 displayed decisions and every physics-step standstill timer, route metric and reward were reconstructed. Saved-model/rule actions were checked. NPC continuous rectangle separation was checked on every physics tick without an ego collision. Ego-collision terminal frames remain visible; the NPC guard does not shield the ego. Every replay frame matches its native record.

Steering comes from the native route controller. Policies choose speed only using the existing observation interface. This is not CARLA, a new trained policy, realistic motorway validation or evidence of real-world safety. One scenario per map cannot establish general superiority. A run can remain active indefinitely if it keeps moving without reaching a terminal condition.

Release SHA256: `c32d358cfa9aba60ca610a66502a26b15ce1a39530594f9c27f6133074477d57`. No earlier failed audits or scientific outputs were modified. This version logs every physics tick, so its progress reconstruction does not require the preceding demonstration posthoc collision-peak allowance.

[Independent audit](run_v1/audit.json) · [Protocol](protocol.json) · [Replay](CITY_REPLAY.html) · [Delivery validation](DELIVERY_VALIDATION.json)
