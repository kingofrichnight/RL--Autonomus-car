# Driving Lab: 40-second standstill timeout

All maps and five assisted policies now allow40 simulated seconds within0.5m of the last movement anchor. Moving farther resets the timer. No overall duration cap. Completion, collision and off-route still end immediately.

Four earlier heavy-jam timeouts were rerun from their initial seeds. The26 successful prior recordings remain unchanged: every physics tick was checked to establish that they finished before either deadline. The original recording deadline is explicit in catalog.json. No resumed or padded recordings.

City junction coordination, geometry, traffic, model weights, forecasts and Progress Assist are unchanged. Longer waiting does not guarantee that a blockage clears.

| Case | Outcome | Seconds | Original recording deadline |
|---|---|---:|---:|
| highway_medium_ppo | complete | 104.5 | 20s |
| highway_medium_qr | complete | 104.5 | 20s |
| highway_medium_hybrid025 | complete | 104.5 | 20s |
| highway_medium_cape | complete | 104.5 | 20s |
| highway_medium_cage | complete | 104.5 | 20s |
| highway_heavy_ppo | complete | 109.7 | 20s |
| highway_heavy_qr | complete | 110.3 | 20s |
| highway_heavy_hybrid025 | complete | 110.3 | 20s |
| highway_heavy_cape | complete | 110.7 | 20s |
| highway_heavy_cage | complete | 110.7 | 20s |
| jam_medium_ppo | complete | 86.7 | 20s |
| jam_medium_qr | complete | 86.6 | 20s |
| jam_medium_hybrid025 | complete | 86.9 | 20s |
| jam_medium_cape | complete | 86.6 | 20s |
| jam_medium_cage | complete | 86.6 | 20s |
| jam_heavy_ppo | complete | 191.6 | 20s |
| district_medium_ppo | complete | 167.7 | 20s |
| district_medium_qr | complete | 151.1 | 20s |
| district_medium_hybrid025 | complete | 151.1 | 20s |
| district_medium_cape | complete | 151.1 | 20s |
| district_medium_cage | complete | 151.1 | 20s |
| district_heavy_ppo | complete | 160.1 | 20s |
| district_heavy_qr | complete | 159.1 | 20s |
| district_heavy_hybrid025 | complete | 159.1 | 20s |
| district_heavy_cape | complete | 159.1 | 20s |
| district_heavy_cage | complete | 159.1 | 20s |
| jam_heavy_qr | timeout | 160.2 | 40s |
| jam_heavy_hybrid025 | timeout | 162.0 | 40s |
| jam_heavy_cape | timeout | 160.2 | 40s |
| jam_heavy_cage | timeout | 160.2 | 40s |

The original optional compatibility checker failed because continued simulation removes an NPC beyond the map at the former cutoff. A separate posthoc geometry check verified this expected maintenance difference; both checker versions and the original error are preserved.
[New-run audit](run_v1/audit.json) · [All30 timeout compatibility checks](TIMEOUT_COMPATIBILITY.json) · [Results](RESULTS.md)
