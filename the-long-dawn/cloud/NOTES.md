# FARM notes (the render farm: cloud/farm.py + cloud/farm_node.py)

## STATE (27 Sep, 20:56Z; the director is away until ~23:29Z)

**Running now.** One farm daemon on this Mac (`python3 the-long-dawn/cloud/farm.py status` shows it) drives three pools:

- **Pool p1:** workspace `long-dawn-farm` on the first Autoresearch account.
  - Up to 15 nodes (one bearer endpoint each).
  - Spend ~$27; the farm refuses new nodes at $200; the platform hard cap is $230.
- **Pool p2:** the second account (org byeron98), workspace `default`.
  - Up to 15 nodes.
  - Spend ~$3; the farm refuses new nodes at $450; the default cap leaves ~$640 of headroom below its $798 credit.
- **The M4 laptop over ssh** (`ldf-m4`). Free, and guarded: it works only on AC power while its owner's own load is under 40% of its cores, runs at most 8 threads, and everything it holds lives in `~/ldfarm`.

**Capacity.** cpu-8 is full in both accounts, so CPU units spill onto h100-1:

- Up to 15 per pool while cpu-8 queues.
- Each pool keeps 1 cpu-8 plus up to 3 h100 waiting in the platform queue: its place in line, which is also what starts new machines.
- Pool 1's parked h100 disks (ldf-g01..g12) wake in ~15-30 s. Pool 2's h100 creates land as capacity frees.

**Queue.** The finals in `_local_logs/handoff/LAUNCHED_BY_MAIN.md`, plus lane tests:

- Tests go first.
- An approved final's first unit starts within 5 min.
- Running finals keep about 1 node in 3.
- About 77 node-hours of finals were queued at 20:53Z. On a 14-28 core h100, each unit is re-cut into more processes at dispatch.

**Delivered today.** embers_A3_a9a10 (480/480), about 60 look-dev tests, and the RUN-C reveal/scroll/illum finals (in progress).

## How to operate

```
python3 the-long-dawn/cloud/farm.py <job.json ...> [--nodes N] [--test [K]] [--frames A-B] [--missing] [--gpu] [--dry-run]
python3 the-long-dawn/cloud/farm.py status | cancel <request> | stop-idle | shutdown
```

- **Requests.** Every call is a request to the daemon, which is started on demand and exits after 5 idle minutes. The caller follows the request log and exits when it ends (0 done, 1 incomplete/failed, 2 cancelled). Ctrl-C cancels the request. `--detach` submits and returns.
- **Signals.** Killing or signalling the daemon never stops the fleet. The next daemon (any farm.py call starts one) re-attaches to every running agent and its in-flight unit, and resumes each request from the frames it already landed. `farm.py shutdown` is the one full stop.
- **Idle nodes.** A node stops the moment the queue has nothing for it. An agent that hears nothing for 10 minutes stops its own node.
- **Files.**
  - `~/.cache/ldfarm/`: `daemon.log`, `req/<id>.log|.json`, `inbox/`, `leases/`, `runs.jsonl`.
  - `~/.config/longdawn-farm/`: the p1 token, `token.meta.json`, `ssh_nodes.json`.
  - `~/.config/longdawn-farm2/token.json`: the p2 token.
  - Never print a token.
- **Mission pages.** `long-dawn-render-farm` in each account. The daemon posts an hourly per-pool line and an alert at 80% of a pool's spend stop.

## Open items

- **Both tokens expire 2026-10-04.** Re-vend p1 through the givemeanode connector (vend_workspace_token, workspace `long-dawn-farm`, ttl 168) on the davidgringras account. That connector dropped in this session.
- **Tickets.** tkt-jqtbh asks to raise the endpoint cap from 16 to 48, which would lift the 15-per-pool limit. tkt-36kp5 reports cpu-8 creates failing on "carve slot occupied" instead of queuing.
- **MONTAGE driver failure reporting (27 Sep handoff fix):** the earlier meltc failure exposed a common-driver bug:
  `render.py` could return 0 after Blender or post-processing failed. The driver now returns nonzero for either
  failure and for incomplete posted frames; `melt_local.py` retains its additional output-existence check.
  This fixes reporting, not shot code or visual quality. Farm image decoding/dimension checks remain independent.
- **Measured.**

  | Node | Work | Speed |
  |---|---|---|
  | cpu-8 | embers | ~0.45 s/frame per lane, 4 lanes |
  | cpu-8 | RUN-C ink | ~35-44 s/frame per process |
  | h100 | Cycles ring still | 8.9 s/frame, plus a one-time 6-min CUDA JIT per new node |
  | M4 | Metal Cycles | ~44 s/frame warm, plus a one-time 165 s denoiser compile |
  | M4 | numba ink | on par with cpu-8 |

- **Pricing** (weekend): cpu-8 $0.00804/min, h100-1 $0.0528/min.
