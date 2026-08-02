# Plan

Living working doc for this repo: the benchmark protocol, per-phase digests of completed work
(what was built, and what still binds), and the full spec of the active phase. `STATUS.md` is
the current-state board (read it first); `README.md` is the public summary; `CLAUDE.md` is the
working rules. Update this file as design decisions land.

Locked specs for completed Phases 0–4 moved verbatim to `PLAN_ARCHIVE.md` (2026-08-01). Any
reference to a Phase 0–4 spec — here, in code/test docstrings, or in README.md — resolves
there. Grep it; never read it whole. Before re-opening any decision named in a "Still binds"
line below, read that phase's archived spec first: those decisions were paid for, and a
summary is not grounds to re-open them.

## Benchmark protocol (Phases 1–4)

Every headline result comes from **≥3 independent training seeds** (5+ for the capstone), reported as mean ± std across seeds — deep RL is brittle with respect to random seed (per Spinning Up), so single-run numbers don't count. The per-run eval protocol (fixed eval seeds, deterministic policy, N episodes) is unchanged.

Execution: any run longer than ~5 minutes — i.e. every real training run (a 5M-step MinAtar run is ~55 min) — is launched by the maintainer in their own terminal; Claude hands over the exact command (env vars included) and picks up from the logged output and checkpoints. Learned on a prior project: training launched through Claude Code tooling runs ~10x slower. Short smokes and tests stay in-session.

## Phase 0 — shared harness + tabular warmup (COMPLETE)

**Built:** the training harness every later algorithm reuses unchanged — `python -m rl.train`
end to end (env factory → rollout → logging → periodic eval → checkpoint), seeding utility,
fixed-seed eval protocol, checkpoint save/load, `scripts/watch.py`; proven by a random-policy
pipeline check and tabular Q-learning on FrozenLake (final eval 0.67 ± 0.47 vs 0.02 random).
Full spec: `PLAN_ARCHIVE.md` §Phase 0.

**Still binds:** `tests/test_harness.py` stays green for the life of the project — the
known-good path when a reward curve goes flat. The locked metric names, eval protocol, and
agent interface live in CLAUDE.md's architecture invariants.

## Phase 1 — DQN, discrete track (COMPLETE)

**Built:** DQN with replay buffer, target network, ε-greedy, and Double/Dueling/n-step config
toggles (`rl/agents/dqn.py`); benchmarked on all 5 MinAtar games at 5M steps × 3 seeds, 5/5
replicate the paper (see README results). Full spec: `PLAN_ARCHIVE.md` §Phase 1.

**Still binds:**
- No gradient clipping in DQN — audited, deliberate, and changing it breaks comparability
  with the 63-run Phase 1 campaign (§Phase 1 for the audit against the PyTorch DQN tutorial).
- Single-threaded torch for tiny nets — the threading pathology is measured three independent
  times in this repo (MinAtar DQN, Connect 4 PPO, the SAC probe); default to `torch_threads: 1`.

## Phase 2 — PPO, both tracks (COMPLETE)

**Built:** PPO discrete and continuous — vectorized collection seam, `RolloutBuffer` + GAE,
discrete PPO on MinAtar (beats DQN on 3 of 5 games), continuous PPO on MuJoCo with running
obs/reward normalization landed as checkpointed harness surface; REINFORCE on-ramp. Chunk
3/4/5 specs locked by three-lens reviews and implemented as written, no deviations. Full
spec: `PLAN_ARCHIVE.md` §Phase 2.

**Still binds:**
- The four canonical continuous forks — unbounded `Normal` + `ClipAction`; state-independent
  free `log_std`; running obs stats, wrapper-side and checkpointed; SB3 reward-norm operation
  order — chosen for the PPO-vs-SAC hold-everything-fixed discipline (§Phase 2).
- Normalizer state is harness surface: checkpointed, restored through one helper, and
  `train()` raises on a normalize flag for scalar-path algos (§Phase 2).
- Advantage recomputed once per rollout (never per epoch), MSE critic loss, per-minibatch
  advantage norm — audited against the TorchRL tutorial and kept canonical (§Phase 2).

## Phase 3 — SAC, continuous track (COMPLETE)

**Built:** SAC (twin Q, squashed Gaussian with the stable log-prob form, auto-tuned α, Polyak
via `rl/common/polyak.py`), validated against published SAC anchors before the PPO comparison
(M3), then benchmarked against Phase 2 PPO on 3 MuJoCo envs; 64×64-Tanh architecture ablation
and PPO-raw control campaign. Full spec: `PLAN_ARCHIVE.md` §Phase 3.

**Still binds:**
- SAC runs on RAW envs while PPO's MuJoCo configs normalize — the confound is closed by the
  PPO-raw control, and any cross-algorithm claim must respect the per-anchor protocol table
  (§Phase 3).
- The `(B,)`-shape guards and their tests are load-bearing — the `(B,B)` broadcast traps
  (log-prob flatten, critic target) train plausibly and only detonate off the gate env
  (§Phase 3).
- Snapshots/copies must never alias live tensors — the `log_alpha` rebind failure class;
  round-trip persistence tests must go through `torch.save`, not in-memory (§Phase 3).

## Phase 4 — Connect 4 self-play on-ramp (COMPLETE)

**Built:** the self-play machinery, validated in isolation — `Connect4Env` with open_spiel
differential oracle, opponent pool with strided retention (`rl/selfplay/`), negamax solver
validated on Pons' benchmark (4,400/4,400), Bradley-Terry Elo tournament harness, the
forgetting demonstration (naive vs pool, one-key diff), the probe-lever campaign, and the
supervised-on-solver-labels diagnostic. Full spec: `PLAN_ARCHIVE.md` §Phase 4.

**Still binds (live Phase-5 priors):**
- ~50% is the EQUILIBRIUM of self-play, not a failure; learning is judged against fixed
  external anchors, never the pool, and `eval_opponent` never resolves to a pool member
  (§Phase 4 pre-registered expectations — carried verbatim into Phase 5 below).
- PPO hyperparameters do not port across episode-length/reward regimes; γ=1.0 with a
  terminal-only ±1 reward; lr anneal off in self-play — it would suppress the phenomenon
  under study (§Phase 4 config).
- Pool snapshots are deepcopies — `agent.state_dict()` aliases live tensors; frozen opponents
  must be probed as actually frozen (§Phase 4 self-play machinery).
- Encoder/architecture ceilings are diagnosed with a supervised instrument, never inferred
  from RL curves — the Phase 4 supervised diagnostic is the pattern (§Phase 4; SESSION_LOGS
  2026-07-29).
- Best rung ≠ final rung — report both; the checkpoint ladder exists for this (§Phase 4).

## Phase 5 — capstone (decided 2026-07-25: Pokémon Showdown Gen 1)

**Env:** Pokémon Showdown Gen 1 singles, battle phase only (no teambuilding) — poke-env driving a local Node.js Showdown server, starting format `gen1randombattle`. **Hero algorithm:** the Phase 2 PPO with self-play. Legal actions change every turn (fainted Pokémon can't be switched to, moves run out of PP, forced switches), which is why the action-masking contract lives in the harness now (landed mid-Phase 2, env-agnostic, all-True default) while everything Pokémon-specific stays deferred until Phase 4 completes: no poke-env dependency, no battle logic, no Pokémon observation encoders during Phases 2–4.

**API corrections from the Phase 4 review (poke-env 0.15.0 source, 2026-07-26) — the plan named a dead API.** `Gen*EnvSinglePlayer` was **removed in 0.8.4** (2025-04-20) along with `EnvPlayer` and `OpenAIGymEnv`. The live surface is **`SinglesEnv`**: subclass, implement `calc_reward(battle)` and `embed_battle(battle)`, assign `observation_spaces`. Three further shape facts, each a bounded known cost rather than a surprise: (1) **the opponent does not live inside poke-env's env** — `PokeEnv` is a PettingZoo `ParallelEnv` with two server-connected `_EnvPlayer`s and no `opponent` parameter, and the opponent enters one level up via `SingleAgentWrapper(env, opponent)`; what transfers from Phase 4 is the learner-facing contract (`(obs, float, terminated, truncated, info)`, an opponent held as a plain policy object, a `calc_reward(state)` hook), not the opponent's *location*. (2) **poke-env puts the action mask in the OBSERVATION, not `info`** — it rewrites the obs space to `Dict({"observation", "action_mask"})`, as PettingZoo classic does; our harness contract matches Shimmy/OpenSpiel instead, so Phase 5 needs one adapter wrapper lifting `obs["action_mask"]` into `info`. (3) **`truncated` is load-bearing here where it is dead on Connect 4** — `calc_term_trunc` sets `terminated=True` only for a decisive wipe, while **forfeits, ties and timer losses all return `truncated=True`**, and `reset`/`close` inject `ForfeitBattleOrder()`. Reward shape transfers cleanly (`calc_reward(self, battle) -> float`, no sign-flip machinery — egocentricity is a property of the state handed to each seat). Vectorization does not: poke-env is one battle per env instance and scales via `SubprocVecEnv` over `SingleAgentWrapper` factories.

**Direct precedent (found 2026-07-25):** Huang & Lee, *"A Self-Play Policy Optimization Approach to Battling Pokémon"* (IEEE CoG 2019) is this exact architecture — PPO + self-play on Pokémon Showdown, actor-critic with GAE and an entropy bonus, and masked softmax over illegal actions. It reached 1677 Glicko-1 on the ladder and beat `pmariglia` (the tree-search SOTA bot) 612–388, over 3.84M self-play matches (~6 days, ~$91 on GCP). **Three corrections from the Phase 4 review, all of which weaken the original reading:** (a) they **renormalize after the softmax** (`π_i = s_i π'_i / (sᵀπ')`) where we mask *logits* with a finite sentinel — same contract, different gradient path, so "our masking contract, independently arrived at" overstates it; (b) "they needed dense reward shaping" is an **inference** — no ablation is reported, they say only "to speed up learning", and Generals.io (arXiv:2606.23348) argues shaping is a throughput artifact ("at low throughput few games finish, so the terminal signal alone is too sparse") and finds sparse reward converges *more* cleanly; (c) **their own §V-C reports catastrophic forgetting**: RL-meta, fine-tuned from RL-rb for only ~10% additional training on a narrowed opponent distribution, wins **77/500 (15.4%)** against the model it came from, in the format it was originally trained on. That is published, quantitative forgetting in this domain, and it substantially weakens "naive self-play was sufficient" — naive self-play was sufficient *to reach 1677 against ladder humans*, which is the defensible claim. Their feedforward-with-LSTM-as-future-work finding stands, so recurrence remains an option rather than a precondition.

**A stronger, more recent anchor: Metamon** (Grigsby et al., *Human-Level Competitive Pokémon via Scalable Offline RL with Transformers*, RLJ 2025, arXiv:2504.04395). SynRL-V2 reaches **Gen1OU GXE 79.9%, Glicko-1 1761 ± 35 over 613 human ladder battles**, peak global rank #31 in Gen1OU. Protocols differ from Huang & Lee's (different tier, different era, GXE vs raw Glicko) and **must not be mixed**. Two findings directly load-bearing for us: they **do not mask actions** ("if the agent selects an invalid action, it is replaced by a random valid action") and name invalid-action selection from PP stalls as "their most noticeable flaw" — direct evidence our masking contract is right; and their self-play arm overfit to its own checkpoints (§5.3, quoted in Phase 4's pre-registered expectations), fixed by deliberate opponent and team diversification.

**Milestone ladder (each independently shippable):** beat `MaxBasePowerPlayer` → beat `SimpleHeuristicsPlayer` → self-play with a historical-checkpoint opponent pool → optional: live Showdown ladder Elo.

**Headline metric:** win rate vs `SimpleHeuristicsPlayer` over ≥1000 battles, multiple seeds. Ladder Elo is an optional flourish, not the metric. Budget the eval variance off Huang & Lee's **1000 matches per matchup**, not off Phase 4's 400 — theirs is driven by team randomization, which `gen1randombattle` has and Connect 4 does not.

**Fallback if self-play stalls:** Procgen generalization study (train/test level gap) — the previous lean, kept ready.

**Hardware (revised 2026-07-28 — the "rented cloud GPU" line was inherited from the Procgen-era capstone and did not survive contact with the repo's own measurements; see the session log).** Online self-play runs **CPU-first; no GPU is provisioned for it.** Reasoning of record: (1) the tiny-net threading pathology is measured three independent times in this repo — MinAtar DQN 278 → ~1,550 steps/s single-threaded, Connect 4 PPO 2,196 → 8,473 (3.9×) at `torch_threads: 1`, and the Phase-3 SAC probe where 4 threads *dropped* throughput 425 → 327 even at 256×256 nets — per-op work this small loses to parallelization overhead, and a GPU is the same mistake at kernel-launch scale; (2) the capstone encoder (structured vectors, MLP-scale, small Gen 1 embedding tables at most) is the same regime as nets this repo already runs at 5.7k–12.5k steps/s on one core; (3) the env step is a websocket round-trip to the Node server plus poke-env protocol parsing — milliseconds against microsecond forwards, so the GPU-accelerable fraction of wall-clock is small; (4) on-policy PPO holds no standing dataset to keep a device utilized. **A GPU re-enters for exactly three things**: an offline supervised arm (the BC diagnostic below), the Procgen fallback (image observations — the setting the old line was written for), or an encoder that outgrows MLP scale (a Metamon-shaped transformer pivot). **Pre-registered Phase-5 throughput measurements** (each an evening once the collection loop exists, before any provisioning decision, under the existing throughput-gate discipline): (a) per-turn latency breakdown of one battle — server compute vs websocket RTT vs poke-env parsing vs `embed_battle` encoding (if the encoder dominates, vectorizing it beats any hardware change); (b) asyncio concurrency curve, aggregate turns/s vs battles-in-flight in one process — the GIL ceiling; (c) multi-process scaling, workers × battles-per-worker, one Showdown server per worker vs shared; (d) forward-pass share at the real encoder, batch-1 vs batched. **Collection-loop structural contract (decided now, while it is cheap — before the loop is written):** battle coroutines submit observations to a single inference seam rather than calling the policy directly, so batch-1, micro-batched, and lockstep-vector inference stay config choices rather than rewrites. Note the batch-1 small-tensor pathology belongs to poke-env's native asyncio `Player` model; the `SubprocVecEnv`-over-`SingleAgentWrapper` route already batches at the vector boundary, at the cost of head-of-line blocking on the slowest battle each turn. Whether micro-batching pays is measurement (d)'s question, not an assumption. W&B merges local + cloud runs into one dashboard if a cloud instance is ever used.

**Optional named diagnostic (the one workload where a GPU rents): behavioral cloning on Showdown replays** — the same encoder + policy head trained supervised on a replay corpus, establishing an architecture ceiling against which self-play results are interpreted (self-play ≈ BC ceiling ⇒ the encoder is the bottleneck; self-play ≪ BC ceiling ⇒ the training is). Phase 4's supervised-on-solver-labels diagnostic is this same instrument at smaller scale, and the Pons-metrics finding there (no training variant moved tactical quality) is exactly the class of question it answers. Scoping deferred to Phase 5 start: `gen1randombattle` replay availability and parsing, and encoding from the acting player's observable state only. **Scoping RESOLVED (2026-07-30, parallel-session advisory, folded in and deleted per the advisory precedent): GO-WITH-CAVEATS.** Corpus clears the bar (~109k archived `gen1randombattle` replays ≈ 2.7M decisions on the HolidayOugi HF archive — count not primary-verified, license unstated — with the official `search.json` API confirmed live and accumulating; prefer a self-scrape through the documented JSON endpoints, ~1 req/s etiquette, over the unlicensed dump). The work is the PARSER, not the data: no random-battle format has an off-the-shelf spectator→first-person parser (Metamon's released datasets are Gen 1–4 OU/NU/UU/Ubers + Gen 9 OU only; its MIT parser is read-for-reference prior art, self-described "no way to be perfect"), and one omniscience leak is verified live — replay logs store EXACT HP for both sides (e.g. 241/481) despite the HP-percentage rule, while the live client gives each seat the opponent's HP at /100 resolution, so the parser must round opponent-side HP to /100 (own side stays exact) or the BC arm trains on precision the deployed encoder never sees. Corpus accumulation is bursty and partly tournament-sourced — stratify by rating/source. Open questions (schema drift across years, possible end-of-battle full-team reveal per the approved 2019 full-info-replays thread, exact primary-source count) live in the 2026-07-30 session-log entry.

### Self-play priors carried from Phase 4 (verbatim)

- **~50% is the EQUILIBRIUM of self-play, not a failure.** With a randomized first player, a policy playing a frozen copy of itself scores ~0 net. Every "is it learning?" question is answered against the **fixed external anchors**, never against the pool — which is why `eval_opponent` can never resolve to a pool member. Published support: Metamon (RLJ 2025, arXiv:2504.04395 §5.3) let SynRL-V1 battle recent checkpoints of itself, got a model "significantly better against itself" that gave "inconsistent improvement against real players" — "battle replays make it clear that the model believes it is playing SynRL-V1" — in Gen 1 OU, the capstone's exact setting.

**What this deliberately does NOT de-risk** (budget separately): the async multi-battle collection layer — our `SyncVectorEnv` of N in-process copies does not map onto poke-env's asyncio-over-websockets to a single Node server, and that is the largest remaining capstone piece; long horizons (≤42 plies vs Gen 1's >100 turns, so γ/λ/rollout length all need re-tuning); partial observability; reward shaping; and eval-variance budgeting. Connect 4 is also, per Czarnecki et al., cyclic mainly in its mid-strength band, so the cycling detector gets built and only partially exercised.

## Session log

Moved to `SESSION_LOGS.md` (2026-07-29) — every dated entry, verbatim.
Any "see the session log" reference in this file resolves there. Append
new entries THERE as work lands, not here.
