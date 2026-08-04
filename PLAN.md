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

Every headline result comes from **≥3 independent training seeds** (5+ for the capstone), reported
as mean ± std across seeds — deep RL is brittle with respect to random seed (per Spinning Up), so
single-run numbers don't count. The per-run eval protocol (fixed eval seeds, deterministic policy, N
episodes) is unchanged.

Execution: any run longer than ~5 minutes — i.e. every real training run (a 5M-step MinAtar run is
~55 min) — is launched by the maintainer in their own terminal; Claude hands over the exact command
(env vars included) and picks up from the logged output and checkpoints. Learned on a prior project:
training launched through Claude Code tooling runs ~10x slower. Short smokes and tests stay
in-session.

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

**Env:** Pokémon Showdown Gen 1 singles, battle phase only (no teambuilding) — poke-env driving a
local Node.js Showdown server, starting format `gen1randombattle`. **Hero algorithm:** the Phase 2
PPO with self-play. Legal actions change every turn (fainted Pokémon can't be switched to, moves run
out of PP, forced switches), which is why the action-masking contract lives in the harness now
(landed mid-Phase 2, env-agnostic, all-True default) while everything Pokémon-specific stays
deferred until Phase 4 completes: no poke-env dependency, no battle logic, no Pokémon observation
encoders during Phases 2–4.

**API corrections from the Phase 4 review (poke-env 0.15.0 source, 2026-07-26) — the plan named a
dead API.** `Gen*EnvSinglePlayer` was **removed in 0.8.4** (2025-04-20) along with `EnvPlayer` and
`OpenAIGymEnv`. The live surface is **`SinglesEnv`**: subclass, implement `calc_reward(battle)` and
`embed_battle(battle)`, assign `observation_spaces`. Three further shape facts, each a bounded known
cost rather than a surprise: (1) **the opponent does not live inside poke-env's env** — `PokeEnv` is
a PettingZoo `ParallelEnv` with two server-connected `_EnvPlayer`s and no `opponent` parameter, and
the opponent enters one level up via `SingleAgentWrapper(env, opponent)`; what transfers from Phase
4 is the learner-facing contract (`(obs, float, terminated, truncated, info)`, an opponent held as a
plain policy object, a `calc_reward(state)` hook), not the opponent's *location*. (2) **poke-env
puts the action mask in the OBSERVATION, not `info`** — it rewrites the obs space to
`Dict({"observation", "action_mask"})`, as PettingZoo classic does; our harness contract matches
Shimmy/OpenSpiel instead, so Phase 5 needs one adapter wrapper lifting `obs["action_mask"]` into
`info`. (3) **`truncated` is load-bearing here where it is dead on Connect 4** — `calc_term_trunc`
sets `terminated=True` only for a decisive wipe, while **forfeits, ties and timer losses all return
`truncated=True`**, and `reset`/`close` inject `ForfeitBattleOrder()`. Reward shape transfers
cleanly (`calc_reward(self, battle) -> float`, no sign-flip machinery — egocentricity is a property
of the state handed to each seat). Vectorization does not: poke-env is one battle per env instance
and scales via `SubprocVecEnv` over `SingleAgentWrapper` factories.

**Direct precedent (found 2026-07-25):** Huang & Lee, *"A Self-Play Policy Optimization Approach to
Battling Pokémon"* (IEEE CoG 2019) is this exact architecture — PPO + self-play on Pokémon Showdown,
actor-critic with GAE and an entropy bonus, and masked softmax over illegal actions. It reached 1677
Glicko-1 on the ladder and beat `pmariglia` (the tree-search SOTA bot) 612–388, over 3.84M self-play
matches (~6 days, ~$91 on GCP). **Three corrections from the Phase 4 review, all of which weaken the
original reading:** (a) they **renormalize after the softmax** (`π_i = s_i π'_i / (sᵀπ')`) where we
mask *logits* with a finite sentinel — same contract, different gradient path, so "our masking
contract, independently arrived at" overstates it; (b) "they needed dense reward shaping" is an
**inference** — no ablation is reported, they say only "to speed up learning", and Generals.io
(arXiv:2606.23348) argues shaping is a throughput artifact ("at low throughput few games finish, so
the terminal signal alone is too sparse") and finds sparse reward converges *more* cleanly; (c)
**their own §V-C reports catastrophic forgetting**: RL-meta, fine-tuned from RL-rb for only ~10%
additional training on a narrowed opponent distribution, wins **77/500 (15.4%)** against the model
it came from, in the format it was originally trained on. That is published, quantitative forgetting
in this domain, and it substantially weakens "naive self-play was sufficient" — naive self-play was
sufficient *to reach 1677 against ladder humans*, which is the defensible claim. Their
feedforward-with-LSTM-as-future-work finding stands, so recurrence remains an option rather than a
precondition.

**A stronger, more recent anchor: Metamon** (Grigsby et al., *Human-Level Competitive Pokémon via
Scalable Offline RL with Transformers*, RLJ 2025, arXiv:2504.04395). SynRL-V2 reaches **Gen1OU GXE
79.9%, Glicko-1 1761 ± 35 over 613 human ladder battles**, peak global rank #31 in Gen1OU. Protocols
differ from Huang & Lee's (different tier, different era, GXE vs raw Glicko) and **must not be
mixed**. Two findings directly load-bearing for us: they **do not mask actions** ("if the agent
selects an invalid action, it is replaced by a random valid action") and name invalid-action
selection from PP stalls as "their most noticeable flaw" — direct evidence our masking contract is
right; and their self-play arm overfit to its own checkpoints (§5.3, quoted in Phase 4's
pre-registered expectations), fixed by deliberate opponent and team diversification.

**Milestone ladder (each independently shippable):** beat `MaxBasePowerPlayer` → beat
`SimpleHeuristicsPlayer` → self-play with a historical-checkpoint opponent pool → optional: live
Showdown ladder Elo.

**Headline metric:** win rate vs `SimpleHeuristicsPlayer` over ≥1000 battles, multiple seeds. Ladder
Elo is an optional flourish, not the metric. Budget the eval variance off Huang & Lee's **1000
matches per matchup**, not off Phase 4's 400 — theirs is driven by team randomization, which
`gen1randombattle` has and Connect 4 does not.

**Fallback if self-play stalls:** Procgen generalization study (train/test level gap) — the previous
lean, kept ready.

**Scope decisions from the prior-work dig (2026-08-03, maintainer-ratified; evidence in the
2026-08-03 session-log entries and `prior_work/`):**

- **MCTS is an OPEN follow-up phase — deferred, not ruled out.** Inference-time-only policy
  improvement in Wang's pattern: training is untouched, the trained policy stays the artifact, and
  search bolts on at evaluation — so it composes with every training lever and no current work
  forecloses it. The Phase-4-era premise "tree search needs a forward model the capstone will not
  have" is REVISED: the forward model exists upstream in our own vendored server —
  `State.serializeBattle`/`deserializeBattle` (`showdown/sim/state.ts:61,84`) and
  `Battle.toJSON`/`fromJSON`/`resetRNG(null)`/`restart()`/`undoChoice`
  (`showdown/sim/battle.ts:318,322,360,1968,3029`) — and Wang's fork adds only two stream commands
  plus constrained team regeneration (diffs: `prior_work/wang_fork_diffs.md`). Gen 1 shrinks the
  determinization further (no items/abilities/Hidden-Power typing; volatile constraints reduce to
  Disable, lock states, Transform; port target `showdown/data/random-battles/gen1/teams.ts`).
  Deferred because the real cost is the search stack (Wang: 20 workers, 1000–2000 rollouts/move,
  ~10 s/move — evaluation ~100× slower) and search's measured edge is smallest in Gen 1 (PokéAgent
  2025: MCTS #8 in Gen1OU where pure policies took #1/#2). Standing consequence now: keep the value
  head healthy — search truncates rollouts at leaves with V.
- **"Pure self-play" is retired as an identity constraint.** The capstone agent may use teachers,
  shaping, and offline data. Concretely in scope for the BC-warm-start design session: BC init
  from `SimpleHeuristicsPlayer` (VGC-Bench: +25–30 pts vs SH at a matched 5M budget; ps-ppo used
  BC-fit-to-the-heuristic as an architecture screen), faint-based reward shaping (ps-ppo: ±0.1
  against the ±1 terminal; potential-based if policy invariance is wanted; their
  post-hoc-alignment off-by-one is the known trap), and the P5b LR-anneal verdict — CREDITED
  2026-08-04 (0.392 → 0.443 pooled at 6M; the anneal joins the recipe; annealed ckpts cannot be
  warm-extended, so any 12M arm is from-scratch). Design the
  recipe as a pre-registered stack, not one lever at a time.
- **Speed before the next science chapter (directed 2026-08-03).** After the P5b read, a
  throughput session precedes the BC-warm-start design session. The rationale is meta-level
  compounding: cheaper experiments raise hypothesis turnover — more levers tried per week, each
  verdict steering the next — which compounds in a way raw steps/s cannot (that only tops out at
  the machine ceiling). Budget was itself a credited lever, so cheaper steps also discount every
  later pre-registration. Measured basis
  (2026-07-29 measurement (c)): one shared Showdown server peaks at TWO workers and declines
  (the Node process saturates; the live 3-wide campaign uses ~3 of 14 cores, per-run throughput
  685 → 587), while one-server-per-worker scaled to ~7.5k decisions/s at W = 4–8. Work items:
  (1) a server-port knob in the env seam + one Showdown server per lane — goal: restore
  ≥685 steps/s per run at 3-wide and measure lane scaling W = 3–6 through the FULL training loop
  (the (c) curve was collection-only); (2) go/no-go on the deferred decision-lockstep facade,
  gated on the already-named measurement (e) — the facade is the lever for long SINGLE runs
  (12M+/Wang-scale), lane count is the lever for probe science; (3) long-run hygiene:
  **RESOLVED 2026-08-04 — see the session-log entry; both premises above are superseded.**
  (1) needed NO code: the shared-server ceiling was `simulator: 1` in the gitignored
  `showdown/config/config.js`, not the node process. At `simulator: 4` the shared server
  scales to W = 6–8 and BEATS one-server-per-lane by 26–50%, so **server sharding is retired
  as the unlock** and the port knob was never written. Lane scaling measured through the full
  loop: W=3 → 659 steps/s per lane, W=6 → 556 (3→6 costs 15.6% per lane, returns +41%
  aggregate); the ≥685 goal was NOT met. (2) is CLOSED as a **self-play-scoped** item —
  batching opponent forwards is 2.04× on the component but only ~2.5% of the loop under
  self-play and exactly 0% under `opponent: heuristics`; measurement (e) was never run and is
  not the gate. Revisit only when a self-play chapter is designed, priced as a code-cost
  tradeoff. **The finding that supersedes the framing above: collection-only benchmarks
  overstate full-loop gain ~7× (29% collection → 3.7% end-to-end). The loop is
  update-and-encode bound, not collection bound**, which contradicts the hardware note below,
  the collection-loop architecture work, and the surrogate-tuning interest. Next item is
  instrumenting the loop split (collect / encode / update / eval).
  `caffeinate`, and Wang's room-cleanup server hack if the poke-env-#332 slowdown signature ever
  appears. Engineering session under log-entry discipline — stated goals, no science claims.
  Calibration from the dig: Wang's 150M/4d is ~434 steps/s aggregate on 80 cloud workers (~5.4
  per core, both-perspective counting); the laptop already does ~1,760 full-loop. His scale is
  reachable by wall-clock, not hardware. Both-players collection is a self-play-only 2× (a
  scripted opponent's seat is off-policy data) — note for self-play chapters, not this one.

**Hardware (revised 2026-07-28 — the "rented cloud GPU" line was inherited from the Procgen-era
capstone and did not survive contact with the repo's own measurements; see the session log).**
Online self-play runs **CPU-first; no GPU is provisioned for it.** Reasoning of record: (1) the
tiny-net threading pathology is measured three independent times in this repo — MinAtar DQN 278 →
~1,550 steps/s single-threaded, Connect 4 PPO 2,196 → 8,473 (3.9×) at `torch_threads: 1`, and the
Phase-3 SAC probe where 4 threads *dropped* throughput 425 → 327 even at 256×256 nets — per-op work
this small loses to parallelization overhead, and a GPU is the same mistake at kernel-launch scale;
(2) the capstone encoder (structured vectors, MLP-scale, small Gen 1 embedding tables at most) is
the same regime as nets this repo already runs at 5.7k–12.5k steps/s on one core; (3) the env step
is a websocket round-trip to the Node server plus poke-env protocol parsing — milliseconds against
microsecond forwards, so the GPU-accelerable fraction of wall-clock is small; (4) on-policy PPO
holds no standing dataset to keep a device utilized. **A GPU re-enters for exactly three things**:
an offline supervised arm (the BC diagnostic below), the Procgen fallback (image observations — the
setting the old line was written for), or an encoder that outgrows MLP scale (a Metamon-shaped
transformer pivot). **Pre-registered Phase-5 throughput measurements** (each an evening once the
collection loop exists, before any provisioning decision, under the existing throughput-gate
discipline): (a) per-turn latency breakdown of one battle — server compute vs websocket RTT vs
poke-env parsing vs `embed_battle` encoding (if the encoder dominates, vectorizing it beats any
hardware change); (b) asyncio concurrency curve, aggregate turns/s vs battles-in-flight in one
process — the GIL ceiling; (c) multi-process scaling, workers × battles-per-worker, one Showdown
server per worker vs shared; (d) forward-pass share at the real encoder, batch-1 vs batched.
**Collection-loop structural contract (decided now, while it is cheap — before the loop is
written):** battle coroutines submit observations to a single inference seam rather than calling the
policy directly, so batch-1, micro-batched, and lockstep-vector inference stay config choices rather
than rewrites. Note the batch-1 small-tensor pathology belongs to poke-env's native asyncio `Player`
model; the `SubprocVecEnv`-over-`SingleAgentWrapper` route already batches at the vector boundary,
at the cost of head-of-line blocking on the slowest battle each turn. Whether micro-batching pays is
measurement (d)'s question, not an assumption. W&B merges local + cloud runs into one dashboard if a
cloud instance is ever used.

**Optional named diagnostic (the one workload where a GPU rents): behavioral cloning on Showdown
replays** — the same encoder + policy head trained supervised on a replay corpus, establishing an
architecture ceiling against which self-play results are interpreted (self-play ≈ BC ceiling ⇒ the
encoder is the bottleneck; self-play ≪ BC ceiling ⇒ the training is). Phase 4's
supervised-on-solver-labels diagnostic is this same instrument at smaller scale, and the
Pons-metrics finding there (no training variant moved tactical quality) is exactly the class of
question it answers. Scoping deferred to Phase 5 start: `gen1randombattle` replay availability and
parsing, and encoding from the acting player's observable state only. **Scoping RESOLVED
(2026-07-30, parallel-session advisory, folded in and deleted per the advisory precedent):
GO-WITH-CAVEATS.** Corpus clears the bar (~109k archived `gen1randombattle` replays ≈ 2.7M decisions
on the HolidayOugi HF archive — count not primary-verified, license unstated — with the official
`search.json` API confirmed live and accumulating; prefer a self-scrape through the documented JSON
endpoints, ~1 req/s etiquette, over the unlicensed dump). The work is the PARSER, not the data: no
random-battle format has an off-the-shelf spectator→first-person parser (Metamon's released datasets
are Gen 1–4 OU/NU/UU/Ubers + Gen 9 OU only; its MIT parser is read-for-reference prior art,
self-described "no way to be perfect"), and one omniscience leak is verified live — replay logs
store EXACT HP for both sides (e.g. 241/481) despite the HP-percentage rule, while the live client
gives each seat the opponent's HP at /100 resolution, so the parser must round opponent-side HP to
/100 (own side stays exact) or the BC arm trains on precision the deployed encoder never sees.
Corpus accumulation is bursty and partly tournament-sourced — stratify by rating/source. Open
questions (schema drift across years, possible end-of-battle full-team reveal per the approved 2019
full-info-replays thread, exact primary-source count) live in the 2026-07-30 session-log entry.

### P4 — encoder-ceiling BC diagnostic (pre-registered 2026-08-02; instrument landed 2026-08-01)

**Question.** Is the ~0.4 plateau vs SimpleHeuristics below what the 611-dim encoder + [512,512]
trunk provably supports? A feature audit of SimpleHeuristicsPlayer's source (2026-08-02 session-log
entry) answers the information half analytically; the run verifies it end to end and adds the
learnability half. Diagnostic outside the milestone ladder (Phase-4 contamination framing): the
clone never touches a pool, a tournament, or a milestone number.

**The audit, in one paragraph (evidence gathered before the bands were set — that is its job).**
SH's realized gen1 policy is a near-closed-form function of encoded features. Forced switches
(measured 20.5% of decision rows) are argmax `_estimate_matchup`, whose four terms — both
directional type multipliers, spe base-stat comparison, both hp fractions — are literal per-mon
encoder features, with ties broken by team order, which is slot order, which is encoded. Move choice
is argmax of bp × STAB × stat-ratio × accuracy × expected_hits × type multiplier — every factor
encoded except `expected_hits` (multi-hit moves; exposed on 1.8% of move rows, chosen on 0.29%).
**SH's setup-move branch is dead code upstream**: poke-env 0.15.0 compares `move.target` (int enum
`Target.SELF`) to the string `"self"`, always False — confirmed analytically (the verbatim predicate
matches zero gen1 moves) and empirically (status-clicked-while-damage-available 4/7,140, all
explained by immunity zeroing every damage score and the tie resolving to slot 0). Hazard, dynamax
and tera branches are dead in gen1. The stochastic fallback (`active is None`) never fires: 0
label disagreements in 8,943 triple-called live decisions, both actives present on every row —
**label noise ≈ 0**. Two consequences shape the bands: (1) a low-agreement FAIL cannot indict the
encoder's information content — the audit forecloses that reading; it would indict the trunk or the
optimization (itself a finding: if supervised SGD cannot fit a near-closed-form target on this
trunk, PPO never had a chance) or the BC method (compounding drift). (2) The existence proof
sharpens: a faithful clone scores the mirror baseline b ≈ 0.49 vs SH (measured 0.485 at n=2,000,
0.492 at n=400), and 0.49 > 0.42, the re-analysis's plateau asymptote. Side fact for the record: the
dead branch means SimpleHeuristicsPlayer is weaker than nominal in every poke-env 0.15.0 stack —
purely internal comparability for us (every milestone number used this same SH), possibly worth an
upstream report.

**Arms.** Primary: 20k SH-vs-SH battles (~450k decisions, ~3 min at the measured 2,825 decisions/s)
via `scripts/make_bc_dataset.py`, then `scripts/train_bc.py` at [512,512], 40 epochs, seeds 0/1/2
(one dataset, three fit seeds — init + battle-split + shuffle; collection noise is not the binding
term). Data check: one 10k-battle-subset fit (seed 0; the subset excludes the primary seed-0 val
battles, so both checkpoints score on the common val set). Conditional, built/run only if
triggered: a [1024,1024] capacity probe (R2 partial/fail) and one DAgger round — clone-visited
states relabeled by SH, refit, re-eval (R3 drift branch). Baseline b = the recorder's win rate over
the 20k collection battles themselves (n=20k, se 0.0035, ties count as non-wins — free from the
collection run).

**Reads, in order (locked):**

- **R0 — collection sanity:** b ∈ [0.45, 0.55]; ~22–23 decisions/battle; forced-switch share
  0.20 ± 0.05 (probe cross-check). Outside → HOLD interpretation.
- **R1 — fit health:** 3-seed val free-agreement spread ≤ 0.02 (multi-choice rows — decisions with
  >1 legal action; the uniform-over-legal floor is ~0.19); 20k-vs-10k common-val Δ < 0.01 = data
  non-binding (if ≥ 0.01: one pre-authorized doubling to 40k battles, nothing else changes).
- **R2 — agreement (fit gate, NOT the verdict):** val free-agreement ≥ 0.93 → audit verified
  (prediction ~0.97 given the enumerated residues). 0.90–0.93 → partial: capacity probe + R4 before
  any claim. < 0.90 → fit failure; explicitly not an encoder-information indictment (see audit);
  capacity probe, then investigate.
- **R3 — win rate (headline):** best-val checkpoint, deterministic, 1,000 battles/seed through
  `scripts/eval_checkpoint.py` (same seed rung as the campaign finals), pooled 3,000 (se 0.009)
  against b. **≥ b − 0.04** (≈3σ, and it absorbs the battle_against-vs-SingleAgentWrapper instrument
  mismatch we did not calibrate) → verdict: the architecture supports ≥ ~0.49 vs SH, so the
  0.408/0.42 plateau sits ≥ 7 points below a representable, supervised-learnable policy ⇒ **the
  plateau is training-side** (signal / distribution / optimization), not representational. The
  one-directional caveat attaches: nothing here shows PPO can REACH that policy under terminal-only
  reward — the claim is about where the ceiling is not. **< b − 0.04 with R2 passed** → compounding
  drift or strategically concentrated errors: run the DAgger round; closes → BC-method artifact,
  verdict as above; does not close → R4 names the sites — concentrated on expected_hits/status rows
  ⇒ a real-but-priced encoder gap (move identity; confirms the embedding follow-up), diffuse ⇒
  unaudited gap, back to a design session. **< b − 0.04 with R2 < 0.90** → no plateau verdict until
  the fit failure is understood.
- **R4 — disagreement concentration (always run; in-session analysis, no script changes):** val
  disagreements tabulated over {multi-hit-exposed, all-status-moves, forced-switch,
  switch-out-trigger, rest}. The audit predicts the first bucket dominates any shortfall.

**Contamination disclosed:** the machinery smoke saw 0.756 free-agreement at 3 epochs / 40k rows
(still climbing); the probe stats (b, forced share, branch deadness) were gathered in the design
pass — they are audit evidence the bands were deliberately set on. **Non-goals:** no value labels
(the collector records no outcomes; re-collection is 3 min if that ever changes); no cross-play vs
RL checkpoints. A passing clone is also a warm-start candidate above the RL best (0.49 > 0.408) —
flagged as a separate milestone-ladder decision, explicitly not taken here.

### Self-play priors carried from Phase 4 (verbatim)

- **~50% is the EQUILIBRIUM of self-play, not a failure.** With a randomized first player, a policy
  playing a frozen copy of itself scores ~0 net. Every "is it learning?" question is answered
  against the **fixed external anchors**, never against the pool — which is why `eval_opponent` can
  never resolve to a pool member. Published support: Metamon (RLJ 2025, arXiv:2504.04395 §5.3) let
  SynRL-V1 battle recent checkpoints of itself, got a model "significantly better against itself"
  that gave "inconsistent improvement against real players" — "battle replays make it clear that the
  model believes it is playing SynRL-V1" — in Gen 1 OU, the capstone's exact setting.

**What this deliberately does NOT de-risk** (budget separately): the async multi-battle collection
layer — our `SyncVectorEnv` of N in-process copies does not map onto poke-env's
asyncio-over-websockets to a single Node server, and that is the largest remaining capstone piece;
long horizons (≤42 plies vs Gen 1's >100 turns, so γ/λ/rollout length all need re-tuning); partial
observability; reward shaping; and eval-variance budgeting. Connect 4 is also, per Czarnecki et al.,
cyclic mainly in its mid-strength band, so the cycling detector gets built and only partially
exercised.

## Session log

Moved to `SESSION_LOGS.md` (2026-07-29) — every dated entry, verbatim.
Any "see the session log" reference in this file resolves there. Append
new entries THERE as work lands, not here.
