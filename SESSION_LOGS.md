# Session log

Dated working-session entries for the whole project — findings, decisions,
and run records, verbatim. Split out of PLAN.md on 2026-07-29 so sessions
can load the living spec without paying for the history: read this file
SELECTIVELY (grep a date, phase, or finding), never whole — it is large by
design. Ordering: the current-phase block sits at the top in ascending date
order; earlier phases follow below.

- 2026-07-27 (chunk 1 implementation) — **Phase 4 chunk 1 landed per the locked spec; 6 commits, 187
  tests green (52 new), 52 mutations run (49 caught, 3 deliberate equivalence controls survived),
  tree clean.** *Board* (`rl/envs/connect4.py`): canonical NumPy, row 0 = bottom, `drop()` negates;
  heights derived rather than cached, so nothing can desynchronize from the position. Five win
  fixtures cross-checked against open_spiel; the 42-move draw and the win-on-the-42nd-disc found by
  **random search** (809 and 2603 games) rather than hand-built. **Two real test gaps found by
  mutation, both the same class**: numpy's negative indices WRAP, so a ray running off row 0 reads
  row 5 and one off column 0 reads column 6, reporting a phantom line of four on a board with none —
  discriminating positions found by searching against a deliberately mutated win check. *Env*:
  learner-centric, opponent inside `step()`, all four perspective sites probed including the
  recorded-not-fixed terminal `next_obs` flip (verified inert at `rollout.py:125`). *Oracle*: every
  open_spiel convention confirmed by probe, not assumed; the carve-out pin uses the **AST**, not a
  grep, because every file that legitimately documents the ban (CLAUDE.md, pyproject.toml, the
  test's own docstring) contains the forbidden name as prose and a text scan reports the rule's own
  statement as a violation. **Measured complementarity** on 16 board mutations: fixtures alone catch
  15, the oracle alone 12, and the three the oracle misses are structural (it never plays an illegal
  move, never compares observation planes). *Harness*: `env_kwargs` as CALLER kwargs —
  probe-confirmed and pinned, gymnasium deep-copies a registered spec's kwargs but not the caller's,
  which is what lets chunk 2 share one pool across sub-envs; `make_eval_env` routes all four eval
  sites; `eval/win_rate` from `info["outcome"]` with a test asserting it holds at 0.25 under a
  reward-sign inversion that flips `return_mean`; checkpoint ladder by threshold crossing, accepted
  on a **round trip into a frozen opponent** rather than on a file appearing. **Masking finally has
  power**: Connect 4 is the repo's first training env with varying masks, so `lr=0 => approx_kl ==
  0` is a real test — measured 10.2% of rollout rows carry an illegal column, dropping the mask
  moves an importance ratio by up to 1.402x, and approx_kl comes out at exactly 0.0; a standing
  control records that the same defect is undetectable on CartPole, where 100% of masks are
  all-True. **Three methodology bugs caught and fixed mid-session, each of which would have produced
  false greens**: (1) the mutation harness had no baseline check, and a red baseline makes every
  mutation report "caught" including the controls; (2) mutated and original source lines of the SAME
  BYTE LENGTH restored within the same mtime second, so python served bytecode compiled from the
  mutated source long after the file was restored — the working tree behaved like mutated code and
  two unrelated tests failed; fixed with `PYTHONDONTWRITEBYTECODE` plus a `__pycache__` purge and a
  restore assertion; (3) the first tactic probe walked whole games and scored every decision point,
  which over-counts misses (an ignored winning move is still there next turn and is scored again,
  while a perfect policy is scored once) — re-run with one independent position per sample. **Two of
  my own test bugs found by mutation, not review**: a terminal-mask probe that won in a board with
  no full column, so all-True was trivially satisfied; and twice, an "opponent wins" case where the
  learner played columns 2,3,4,5 — itself a horizontal four, so the learner won the test written for
  its opponent.
- 2026-07-27 (chunk 1 gate) — **Gate passed on the criterion that was locked; the one added
  criterion that failed turned out to measure opponent specialization, not a defect.** Fixture
  probes are the gate proper and are green. The training smoke ran 3 seeds (0/100/200) x 200k steps
  in **28 s wall**, concurrent, from a clean tree (`git_dirty: false`). Baselines were measured
  BEFORE training (a property of the initialisation, not a result): a freshly-initialised policy
  beats `heuristic` 2.50 / 0.75 / 0.00%. Finals: **44.3 / 48.3 / 33.5%** vs heuristic and 88.5 /
  80.5 / 81.0% vs random, curves monotone and **still climbing at 200k**, entropy 1.92 -> 0.29-0.42
  (max ln 7 = 1.946), `eval/return_std` 0.94-1.00 so no collapsed eval set. Strict improvement
  passed on every seed by ~0.34-0.48. **A 0.60 absolute bar was pre-registered, then recalibrated to
  0.25 BEFORE running** once the baselines showed a random policy beats `heuristic` only 1.3% of the
  time — 0.60 from a 1.3% start in 195 PPO updates was an unfair ask, and the recalibration was
  decision-relevant rather than cosmetic since the run landed at 0.42. **The added `>=0.95 vs
  random` floor failed (0.81-0.89) and the investigation is the finding**: an unbiased tactic probe
  (validated by controls — `HeuristicOpponent` scores exactly 1.000, random 0.187) shows the agent
  takes an available immediate win only 24-27% of the time, and **that does not improve from 200k to
  1M steps** even as its win rate vs heuristic climbs 0.44 -> 0.65. The explanation was stated as a
  prediction and then tested: training against `heuristic` means every threat the agent builds is
  blocked on the very next ply, so live convertible wins are rare in its training distribution. An
  otherwise identical agent trained against `random` reaches **97.8% vs random** (clearing the
  floor) while dropping to **20.3% vs heuristic** — specialization in both directions. So the floor
  was mis-specified: it assumed transfer from a heuristic-specialised policy, and PLAN's "beats
  random >>90%" is a property of the final pool-trained agent, not of a heuristic-trained one.
  **Pre-registered for chunk 2**: the pool contains weak early snapshots, so pool training should
  recover the vs-random number without a heuristic arm; if it does not, the specialization is
  stronger than this reading and the opponent-diversity lever (AlphaStar PFSP, already named) comes
  forward. **Also measured, and it contradicts a locked number**: over 42,725 learner decision
  points under random play the mask is all-True at **83.8%** and single-legal at **0.05%**, where
  the spec's locked figures are 63.8% and 0.53%. Most likely the trained self-play distribution
  (longer games, ~16% draws, more full columns) versus random play; recorded rather than reconciled,
  since it changes no decision and makes the hand-written single-legal fixture MORE necessary, not
  less. Independent confirmations of spec numbers: mean **10.56** learner steps/episode against the
  spec's 10.6, and draws 0/300 under random play against the spec's 0.27%.
- 2026-07-27 (chunk 2 implementation) — **Pool machinery, train wiring, kernel knob and both arm
  configs landed; 4 commits, 208 tests green (21 new), 15/15 seeded mutations caught with all 3
  equivalence controls surviving. The pathfinder has not run yet — the chunk stays open.** *Pool*
  (`rl/selfplay/pool.py`): `AgentOpponent` deep-copies the whole agent **at construction** — the one
  place the copy cannot be forgotten by a call site, since `state_dict()` aliases live tensors — and
  samples via `torch.multinomial` with its own `torch.Generator`, because `Categorical.sample()`
  only draws from the global stream, and sharing that stream would let the number of learner steps
  between opponent moves change every seeded opponent decision. `SnapshotPool` is itself an
  `Opponent`: `push()` freezes (the install-point contract), `select()` runs the 80/20 draw from the
  env's own per-episode stream (the pinned draw order is unchanged — the select slot just consumes
  draws now), overflow evicts the second-oldest so the step-0 snapshot anchors the pool's span.
  **One spec collision found and resolved: at `pool_size: 1` the locked evict-second-oldest rule
  would keep the random init forever and evict every later snapshot — the naive arm would train
  against its own initialisation for the entire run — so a size-1 pool replaces its sole member
  instead.** The defect is pinned by a seeded mutation (unconditional `del members[1]`: caught).
  *Wiring* (`rl/train.py`): `selfplay.opponent: "self"` is swapped for the live pool object before
  env construction (the caller-kwargs seam shares it across all 8 sub-envs); the step-0 push lands
  after `make_logger` and before the loop's first `reset`; cadence pushes fire only when `update()`
  reports a drained rollout, every `push_every_updates`, and `selfplay/pool_size` is logged from the
  train loop per the locked namespace rule. The `"self"` spelling buys a free guard: an unreplaced
  string — including `eval_opponent: self` — dies in `make_opponent` as *unknown opponent*, so
  nothing can silently evaluate against the pool; pool mode on a scalar-loop algorithm raises too
  (it has no rollout boundary to push at, and would otherwise train against the frozen step-0
  snapshot while looking like self-play). *Kernel fork*: `ConvQNet` gains `kernel_size` (default 3 —
  every existing config, checkpoint and DQN call site byte-identical), threaded as a PPO hparam so
  the 4×4 probe arm is a config key at pathfinder time; param counts pinned at the spec's measured
  83,816 (k3, both nets) and 26,135 (k4 actor). *Configs*: `connect4_{pool,naive}.yaml` at the
  locked 2M numbers, differing only in `pool_size` (20 vs 1; `latest_prob` kept in the naive file,
  inert, to preserve the one-key diff); both parse and train end-to-end at a truncated budget.
  **Next session: the pathfinder** — pool arm seed 0 plus the two probe arms in the maintainer's
  terminal, the verify criteria (win rate vs the fixed anchor climbs, pool grows, entropy holds),
  and the pre-registered vs-random recovery check via a both-anchors scoring pass over the
  checkpoint ladder.
- 2026-07-27 (chunk 2 pathfinder) — **Chunk 2 closes: the machinery is validated and the strength
  findings are the two pre-registered escape hatches firing, not bugs.** Three 2M runs (pool / λ=1.0
  / kernel=4, seed 0 each, maintainer's terminal, all stamped `git_dirty: false`, ~12.5–13.4k
  steps/s median). *Machinery*: `selfplay/pool_size` hits 20 at step 389,120 — exactly push 20 × 20
  updates × 1024 — with 98 lifetime pushes matching 1 + ⌊1953/20⌋; `eval/return_std` ≥ 0.39
  everywhere (no collapsed eval set); in-training `eval/win_rate` vs heuristic agrees with the
  post-hoc ladder scoring (~0.20 at 2M) — the metric cross-checks. New committed tooling:
  `scripts/score_ladder.py` (every rung × both anchors) and `scripts/mutations/chunk2_pool.py`.
  *Verify criteria*: pool grows ✓; win rate climbs ✓ but weakly (~0.08→0.20–0.23, plateaued by
  ~200–400k); **entropy collapses ✗** — 1.91 → ~0.2–0.37 at 1M → 0.035–0.119 at 2M, min 0.007–0.049
  against ln 7 = 1.946. *The pre-registered recovery check FAILED*: final vs-random 0.882 / 0.895 /
  0.865 (pool / λ1 / k4) against the ≥0.95 floor — the same band as the chunk-1 heuristic-trained
  agents (0.81–0.89), nowhere near the random-trained 0.978 — and the trajectory is
  flat-to-DECLINING (pool arm 0.925 at 200k → 0.777 at 1M → 0.882 at 2M), so 1.8M further steps of
  self-play bought nothing against either external anchor. *Mechanism, from three independent
  readouts*: (1) the rebuilt win-in-1 tactic probe (1000 independent positions; controls: heuristic
  1.000, random 0.218) puts conversion at **0.251–0.314** — pool self-play did not teach taking an
  available win, refuting chunk 1's distribution explanation (vs the pool, convertible wins ARE
  common: the step-0 anchor never blocks); (2) the entropy collapse means near-deterministic play
  against near-deterministic copies (AgentOpponent samples, but sampling a ~0.05-entropy policy is
  still deterministic-ish), shrinking game-tree coverage to whatever the 20% historical draws and
  the first-player flip provide — Tesauro's substitute noise sources, of which the entropy bonus is
  one, are exactly what vanished; (3) `loss/value` falls to 0.14–0.26, far BELOW the pre-registered
  0.84 learned-nothing floor — the critic predicts self-play outcomes very well, i.e. games among
  snapshots became mutually predictable: the Metamon §5.3 signature ("the model believes it is
  playing SynRL-V1") on our own env. **Consequence**: the specialization is stronger than the
  chunk-1 reading anticipated; per the pre-registration the **PFSP weighting (`f_hard(x)=(1−x)^p`
  over the historical draw) comes forward**, joined by an entropy-floor lever (larger `entropy_coef`
  or a schedule) — both as chunk-4 probe arms, with probe-before-campaign vs probe-as-arms an open
  decision. Note the phase's framing holds: a mediocre agent is pre-registered success, the
  deliverable is the loop and the harness — and chunk 3's tournament (alpha-beta anchors, Elo, Pons
  metrics) is what will measure these agents properly. *Fork verdicts*: λ — no separation anywhere
  (final 0.882 vs 0.895 vs-random, 0.200 vs 0.230 vs-heuristic; λ=1.0 noisier mid-run, dipping to
  0.652 at 600k) → **0.95 stays**; kernel — k4 consistently ahead vs heuristic across all 10 rungs
  (mean 0.244 vs 0.179, paired eval seeds) at 26k vs 84k params → suggestive, **2 confirming seeds
  before the chunk-4 campaign locks**.
- 2026-07-27 (mechanism diagnostics, closing the loop before chunk 3) — **Decision: proceed to chunk
  3 (maintainer). The reason of record is not "it's the plan": chunk 3's deliverables are
  instruments, and an instrument's correctness doesn't depend on the quality of what it measures — a
  tournament calibrated across a weak-to-mediocre spread is arguably better-calibrated than one on a
  narrow strong band, alpha-beta is agent-independent, and the solver yields ground-truth
  optimal-move accuracy, a far sharper diagnostic than vs-random win rate. Probing entropy/PFSP
  first would measure the fix with the blunt instrument.** Before moving, the mechanism claim was
  tested against its benign twin, both diagnostics run on the three final checkpoints. **(1) Value
  loss by state distribution** (200 one-per-game positions per generator; targets = mean of K=8
  mirror-self-play continuations from each position, so the continuation policy is held fixed and
  only the state distribution varies): self-play/random/heuristic MSE = 0.783/1.112/1.228 (pool),
  0.065/0.733/0.824 (λ1), **0.031/0.966/0.994 (k4)** — the malign twin confirmed at 10–30× for the
  two probe arms, directionally (1.4–1.6×) for the pool arm. "The value function is simply good" is
  refuted: it is good only where the policy lives. **(2) Coverage measured directly, not inferred**
  (200 sampled games vs the latest snapshot — the 80% case; control random-vs-random: 200/200
  distinct, mean common prefix 0.2): pool **133/200 distinct** (long 30.3-ply games, prefix 14.0),
  λ1 **11/200**, k4 **8/200** with mean common prefix 12.1 of mean length 13.0 — the probe arms
  replay near-identical short games to within ~1–2 plies of the end. The two diagnostics cohere:
  self-play value MSE tracks distinct-game count exactly, and the pool arm's anomalously high
  self-play MSE is explained by its genuinely higher-variance long games, not by a better-calibrated
  critic. **Kernel-fork caveat recorded: k4's vs-heuristic edge coexists with the WORST coverage
  collapse — one more reason the 2 confirming seeds are mandatory, and they should log both
  diagnostics.** **Third lever added (maintainer)**: mix the fixed opponents into the pool draw at a
  small rate — with 80% of games against the current policy the pool is structurally
  self-referential regardless of `entropy_coef`, and PFSP reweights *within* the self-lineage
  without leaving it; a few percent of fixed-bot games keeps external coverage alive by construction
  and is the only lever aimed directly at the vs-random decline. Kept small or the agent exploits
  those two bots specifically. **Chunk-4 method note**: track vs-random as a CURVE, not an endpoint
  (probe arms set `checkpoint_every: 100000` → 20-point `score_ladder` curves) and align where the
  decline begins against entropy's descent — that separates "entropy_coef too low" from "pool
  insufficiently diverse". **Named for chunk 3+**: a supervised-on-solver-labels run of the same
  architecture separates "the encoder is the ceiling" from "the training is the ceiling".
- 2026-07-27 (chunk 3 step 1) — **Solver core landed: bitboard + negamax/alpha-beta + flagged
  bounded TT + brute-force differential oracle. 3 commits, 229 tests green (21 new), mutation
  battery 10/10 real mutations caught with 3/3 equivalence controls surviving
  (`scripts/mutations/chunk3_solver.py`, committed).** *Bitboard* (`rl/selfplay/solver.py`): Pons'
  layout — 7 bits/column, the sentinel bit being the bitboard twin of chunk 1's numpy negative-index
  wrap (a no-sentinel mutation produces exactly the phantom-line class, caught) — canonical like
  `Connect4Board`, `play()` hands over `current ^ mask`. Cross-checked ply-by-ply against the numpy
  board on random playouts plus every hand-pinned fixture, which chains it to the open_spiel oracle
  for free. *Search*: centre-first, size-bounded replace-on-collision TT (~1M entries, one encoded
  int each), `(value, EXACT|LOWER|UPPER)` per the locked spec; scores in Pons' convention so labels
  and score-regret consume solver output directly; win-before-full pinned inside the search by the
  WIN_ON_42 fixture (+1, not draw). **The methodology finding of the step: the TT-flag mutations
  survived the endgame differential TWICE.** At ≤10 empties nearly every node resolves through the
  win scan within a ply or two, so fail-soft bounds nearly always EQUAL exact values and
  bounds-stored-as-EXACT is invisible — the shallow-differential blind spot is the same shape as the
  spec's "Pons validation structurally cannot see it", and it confirms the
  corruption-rises-with-depth claim from the other side. The guard with power is a **consistency
  test at 18–22 stones** (positions filtered to 2k–300k fresh-solve nodes): fresh full-window solve
  must equal a solve over a table poisoned by a null-window sweep — the chapter-8 usage pattern
  pre-exercised, provably equal for correct flags, leaks bounds when broken. *Perf*: **642k nodes/s
  sustained** vs the scratch probe's 894k — the gap is copy-per-node `Bitboard` allocation — scaling
  the chapter-8 Begin-Easy estimate from ~12 to ~17 min; inlining ints in `_negamax` is the named
  lever if step 2's measured wall-clock demands it. *Next*: step 2 — chapter-8 iterative-deepening
  null-window driver, Pons downloader into gitignored `data/`, validation runs handed to the
  maintainer's terminal.
- 2026-07-27 (chunk 3 step 2) — **Chapter-8 driver + Pons validation: 3000/3000 correct on End-Easy
  (0.2 s), Middle-Easy (14.0 s — 281 s pre-chapter-8, a 20× cut) and Middle-Medium (541.5 s),
  ~555–586k nodes/s sustained; Begin sets handed to the maintainer's terminal.** `solve()` is now
  Pons' dichotomic null-window driver (`int(x / 2)` not `x // 2` — the reference C++ truncates
  toward zero, Python floors, and they differ on negative bounds), checked against a plain
  full-window negamax at depths brute force cannot reach. **The battery caught one of its own
  equivalence controls being mis-specified, which is the finding of the step**: the
  strict-`>`-cutoff "control" survived pre-chapter-8 and was CAUGHT once the driver's narrow-window
  traffic existed — relaxing the cutoff lets alpha rise to equal beta and the next child be searched
  with a ZERO-width window, where the TT-hit path (`if alpha >= beta: return value`) returns bounds
  facing the wrong way. Not equivalent at all: a violation of `_negamax`'s implicit alpha < beta
  caller contract, now an explicit assert (making the catch deterministic) and a reclassified real
  mutation. Battery stands at 12/12 real caught, 4/4 controls surviving. `scripts/pons_benchmark.py`
  downloads on first use (plain http primary, raw.githubusercontent fallback) into gitignored
  `data/`; one solver shared per set on purpose (positions transpose, warm entries are valid bounds
  anywhere — the flag tests pin this). *Next*: Begin sets in the maintainer's terminal (~20+ min
  each at our 0.62× of the scratch probe's speed), then step 3 `AlphaBetaOpponent`.
- 2026-07-27 (chunk 3 step 3) — **`AlphaBetaOpponent` landed and registered
  (`alphabeta2`/`alphabeta4`); 236 tests green, battery 15/15 real mutations caught, 5/5 controls
  surviving.** Fixed-depth negamax with **0 at the horizon** — purely tactical, no positional guess,
  so anchor strength is a function of depth alone and reproducible. Two design points worth the
  record: (1) root move scores are computed **full-window per move, no root alpha threading** — a
  rising alpha lets tied moves fail low below their true value and silently drop out of the tie set
  by column order, which would bias the uniform tie-break carrying the locked 89/100-vs-2/100
  eval-diversity property; (2) the **win scan runs before the horizon check** in `_limited`, which
  is what gives depth d vision of plies 1..d — swapping them turns alphabeta2 into alphabeta1 with
  no error anywhere (pinned by mutation, caught via the seeded block test). Full-depth
  `alphabeta_move_scores` equals exact per-move solver values on random endgames — a differential
  between the two search implementations. Tie-break draws from the env's per-episode stream like the
  heuristic's fallback; a control pins that tests assert distributional properties, never exact rng
  stream consumption. *Next*: step 4 `rl/selfplay/elo.py` (BT-MLE by MM, Ford check, floor/ceiling
  drops, stratified bootstrap), then step 5 `scripts/tournament.py`.
- 2026-07-27 (chunk 3 steps 4–5 + pool-ladder campaign) — **Elo harness and tournament landed per
  the locked spec, and the full 500-games/pair campaign already ran on the pool ladder: 52,500 games
  in ~54 s.** The scratch estimate of 311 ms/game for `alphabeta4` (hence "~40-min campaign")
  belonged to the pre-bitboard implementation — measured now at ~1 ms/game, the campaign is a
  **minutes job and no longer needs the maintainer's terminal**. *Harness* (`rl/selfplay/elo.py`, 15
  tests, battery 9/9 real + 2/2 controls): MM fit refuses non-Ford matrices; iterative floor/ceiling
  drops; stability test at 200/2k/20k; stratified bootstrap pinned by the all-draws fixture (CI
  width exactly 0, the spec's own sd-0.021 counterexample); triple fraction only ever reported
  against its acyclic null band. *Tournament* (`scripts/tournament.py` + `play_game` in
  `opponents.py`, committed so the chunk-4 coverage diagnostic can reuse it): per-matchup rng +
  fresh matchup-seeded `AgentOpponent`s (replay isolation), freeze-at-install, exact N/2 colour
  split, `best_checkpoint.pt` excluded. **Campaign results**
  (`runs/connect4_pool_s0/tournament.json`, seed 0, B=1000, 0 resamples failed, every player rated
  in 1000/1000 — full Ford connectivity, no drops): `alphabeta4` +204.7 [187.5, 222.1] ≫
  `alphabeta2` ≡ 0 ≈ `heuristic` −11.4 [−24.3, +1.9] ≫ ladder top (`ckpt_2M` −121.8, `final` −124.5,
  statistically indistinguishable) → monotone-ish decline → `ckpt_200k` −412.7 ≫ `random` −566.2.
  The run spans **~291 Elo of learning**; CI half-widths ~13 Elo. Pre-registered strength
  expectations: beats random ≫90% ✓ (442-Elo gap ⇒ ~93%), loses to alphabeta4 ✓, "roughly
  competitive with alphabeta2" ✗ — 124 Elo below, the chunk-2 coverage-collapse finding restated by
  a calibrated instrument. **The finding: the cycling detector fired.** Intransitive triples 0.0352
  of 455 vs acyclic null band [0.0000, 0.0022] at 500 games/pair — **16× the null's upper edge**,
  real intransitivity in the mid-ladder, consistent with Czarnecki et al.'s spinning-top geometry at
  exactly the intermediate-strength band this agent occupies. Draws 0.5% (the eval-set degeneracy
  note said ~16% for self-play pairs; anchor pairs dominate here). *Still open in chunk 3*:
  Begin-set Pons validation (maintainer's terminal, ~20+ min/set); tournaments for the `lam1`/`k4`
  ladders are now cheap enough to run alongside. *Named earlier, now unblocked*: the
  supervised-on-solver-labels diagnostic.
- 2026-07-28 (Pons validation closes at the tractability boundary the spec predicted) —
  **Begin-Easy: 1000/1000 correct in 1047 s** (maintainer's terminal) — and the wall-clock validates
  the perf model: ~12 min scratch estimate ÷ our measured 0.62× node rate ≈ the 17.5 min observed.
  **Begin-Medium: 400/400 correct at the point it was stopped**, but measured INTRACTABLE at this
  solver level: 9.6 h for 40% of the set, ~140 s/position and worsening (heavy-tailed, per the
  never-extrapolate rule the remainder is unknown but ≥ another day; Begin-Hard is strictly deeper).
  This is the boundary the locked spec itself drew — chapter 8's 660× was only ever claimed for
  Begin-Easy, against the pre-chapter-8 "54 hours to 17 days" for the Begin group — so the run was
  stopped rather than continued. **Solver validation closes at 4,400/4,400 externally corroborated
  positions (End-Easy, Middle-Easy, Middle-Medium, Begin-Easy full; Begin-Medium 400-position
  partial), zero mismatches**, on top of the brute-force differential that is the primary
  correctness argument. Reaching Begin-Medium/Hard would need Pons chapters 9–13 (loss anticipation,
  threat ordering, optimized TT) and serves no Phase-4 goal: agent metrics run over solver-exhausted
  subsets with coverage always reported, and those sets are equally intractable for the child solves
  the metrics need. Named lever only if a future phase wants it.
- 2026-07-28 (kernel fork SETTLED: k3 stays) — **The 4 confirming-seed runs trained (maintainer's
  terminal, all four `git_dirty: false`) and all four ladders tournamented; with 3 seeds per arm the
  kernel fork closes on the pre-registered prior.** Finals, anchor-relative: **k3 pool {−124.5,
  −189.8, −131.7}, k4 {−200.8, −171.7, −231.8}** — k3 ahead by ~53 Elo on means, ahead on 2 of 3
  same-seed pairings (s0 by 76, s2 by 100; s1 the reversal, k4 by 18), ahead in 8 of 9 cross-arm
  pairwise comparisons (rank-sum p≈0.10 — directional, not decisive, and it does not need to be:
  **the burden was on k4 to confirm an advantage, and it shows a deficit**). The chunk-2
  vs-heuristic edge was anchor specialization, full stop. **Campaign config locks: `kernel_size` 3
  (default), `gae_lambda` 0.95.** Secondary findings across all 7 tournaments now on disk: **the
  cycling detector fires in every single one** (fractions 0.035–0.108 vs null-band tops
  0.002–0.013); **best rung ≠ final in 4 of 7 runs** — late-training regression is the norm, not an
  anomaly (k4_s2 fell 47 Elo from its 1.2M peak; pool_s1 fell 31 from 1.2M) — directly motivating
  the chunk-4 forgetting measurement; within-arm seed spread is ~65 Elo (pool s1 is a genuinely weak
  seed: proxy 0.458, cycles 6× its band), justifying 3 seeds/arm in the campaign. The mechanism
  diagnostics on these ladders (the "re-log both" note) remain open pending the committed tooling;
  the fork verdict did not need them — the tournament was sufficient — but they still feed the
  probe-lever design.
- 2026-07-28 (cross-arm tournaments: lam1 + k4) — **Both probe-arm ladders tournamented (500/pair,
  B=1000, 0 failed resamples, full connectivity both), and the calibrated scale revises both fork
  readings.** **λ fork, strengthened**: lam1 final **−183.0 [−194.9, −171.7]** vs pool final −124.5
  [−137.4, −111.8] — **~60 Elo behind, CIs disjoint**, where the score_ladder pass had read "no
  separation" (the tournament, with agent-vs-agent games and Elo resolution, is simply the sharper
  instrument). 0.95 stays, now with a margin attached. **Kernel fork, FLIPPED**: k4 final **−200.8
  [−212.9, −189.5]**, ~76 Elo below pool — despite the score_ladder vs-heuristic edge (0.244 vs
  0.179) that made it "suggestive". Both are real: k4 improved against the eval anchor specifically
  while getting worse against the field — the diagnostics entry's caveat (worst coverage collapse,
  8/200 distinct) cashing out exactly as feared. The 2 confirming seeds (k3 vs k4, seeds 1–2) now
  adjudicate a likely-negative rather than confirm a positive; **prior: k3 stays unless the seeds
  surprise**. **Probe-arm ladders are genuinely non-monotone**: lam1 troughs at 600k–800k
  (−427.6/−388.5, both BELOW its 200k rung at −274.7 — a ~150-Elo regression then recovery); k4's
  1.2M–1.4M (−326.3/−309.8) sit below its 200k (−264.6). **Cycling replicates in all three arms**:
  triple fraction 0.0352 / 0.0725 / 0.1077 (pool/lam1/k4) vs acyclic null bands topping at 0.0022 /
  0.0066 / 0.0088 — 16×/11×/12× above. **AlphaStar min-winrate proxy, computed from the same
  matrices** (scratch pass over tournament.json counts; single-seed baselines for chunk 4): pool
  **0.610**, lam1 **0.487**, k4 **0.481** — ordered exactly as coverage collapse predicts. k4's tail
  is the sharpest single fact of the night: its final checkpoint's min-winrate is **0.09** (2M rung
  0.15) against an earlier self the BT fit places ~8 Elo away — an extreme pairwise intransitivity
  BT compresses silently, which is precisely the failure mode the triple-fraction-vs-null-band
  detector exists to expose (and k4's 0.108 is the largest measured). Draws 0.5%/0.3%/0.9%.
  *Decision queue for chunk-4 lock*: (1) commit the two mechanism diagnostics as tooling (coverage
  probe can now reuse `play_game`); (2) 4 confirming-seed runs (pool + k4 × seeds 1–2,
  `--seed`/`--run-name` overrides exist) from a CLEAN tree, then tournament + diagnostics on their
  ladders; (3) then lock the campaign config.
- 2026-07-28 (chunk 4 step 1: mechanism diagnostics committed and run on all 7 finals) —
  **`scripts/coverage_probe.py` + `scripts/value_mse_probe.py` landed per the 2026-07-27 scratch
  methods (3 commits, 256 tests green), and the "re-log both" note closes: both probes run on every
  final on disk.** `play_game` gained optional `start`/`moves` params (played on a copy; the probes
  need mid-game continuations and move sequences); pinned by tests plus a new battery
  (`chunk4_probes.py`, 3/3 real caught, 1/1 control survives), with the chunk-3 battery re-run
  intact (17/17 + 5/5) since its `old` strings border the edit. Full results (distinct/200,
  prefix/length, MSE self-play/random/heuristic): pool_s0 **88**, 15.4/30.5, 0.713/1.136/1.099;
  pool_s1 **8**, 17.5/18.9, 0.024/0.708/0.756; pool_s2 **27**, 13.2/18.1, 0.023/0.690/0.806; k4_s0
  **7**, 12.3/13.1, 0.027/0.710/0.829; k4_s1 **36**, 23.2/29.9, 0.149/0.799/0.743; k4_s2 **2**,
  7.0/7.0, 0.002/0.793/0.853; lam1_s0 **11**, 9.4/11.2, 0.050/0.662/0.752 (control 200/200 distinct,
  prefix 0.2, every run). Three findings. **(1) Coverage collapse is the norm across seeds, not a
  probe-arm artifact**: 6 of 7 finals sit at 2–36 distinct games of 200 — pool_s0, the one seed the
  scratch diagnostics happened to run on, is the OUTLIER at 88, so chunk 4's probe levers are
  addressing the general case, not a quirk. **(2) The extreme: k4_s2's final plays essentially ONE
  deterministic 7-ply game against itself** (2/200 distinct, common prefix 7.0 of mean length 7.0,
  self-play MSE 0.002) — the same checkpoint with the 0.09 min-winrate tail and the −47-Elo late
  regression, tying the cross-arm entry's sharpest anomaly to its mechanism. **(3) The malign twin
  confirmed on all 7**: self-play MSE 0.002–0.713 vs off-distribution 0.66–1.14 (gap 1.6×–400×), and
  self-play MSE tracks distinct-game count monotonically across all seven runs — the critic is good
  exactly where the policy lives and nowhere else. Coverage rank also tracks tournament Elo rank
  (one inversion, pool_s2/k4_s1): less-collapsed finals rate higher against the field. Replication
  vs scratch: probe arms nearly exact (lam1 11/200 → 11/200, k4_s0 8 → 7), pool_s0 133 → 88 (its
  long, high-variance games; the committed tooling's seeded numbers are the record from here).
  Committed-probe JSONs live at `runs/*/{coverage,value_mse}.json`. *Next*: naive arm (3 training
  runs, maintainer's terminal), then its tournaments + diagnostics, then the forgetting tooling
  (AlphaStar proxy promoted from the scratch snippet).
- 2026-07-28 (naive arm trained, tournamented, diagnosed — the pre-registered forgetting
  demonstration lands on its primary measure) — **The campaign's training is COMPLETE:
  `connect4_naive_{s0,s1,s2}` ran in the maintainer's terminal (all stamped `git_dirty: false` at
  `e674128`), each ladder tournamented (500/pair, B=1000, 0 failed resamples, full connectivity) and
  diagnosed. Headline: the AlphaStar min-winrate proxy separates the arms with NO overlap — naive
  {0.309, 0.392, 0.250} vs pool {0.610, 0.458, 0.609}; every naive seed sits below every pool seed,
  and below every probe-arm value too (k4 0.481–0.567, lam1 0.487). Naive self-play forgets how to
  beat its own past selves; the pool largely prevents it — which is the demonstration Phase 4 exists
  to produce.** (Proxy convention, calibrated to reproduce the recorded pool baselines exactly:
  step-ordered rungs plus `final`, `min_{j<i} wr(i vs j)` averaged over i ≥ 1, draws 0.5; still a
  scratch pass — committed tooling is the next step.) Supporting facts, each from the calibrated
  instruments: **(a) Naive finals rate {−223.9, −175.1, −217.0} vs pool {−124.5, −189.8, −131.7}** —
  ~57 Elo behind on means, and naive's best final (s1, −175.1) is worse than pool's median. **(b)
  The naive ladders are violently non-monotone**: best rung ≠ final in ALL 3 — s2's best rung is its
  200k (−129.3, ~88 Elo ABOVE its own final: 1.8M further steps made it worse); s0 craters ~200 Elo
  at 1.4M–1.8M (−327.4/−398.4) then partially recovers; s1 troughs −337.2 at 1.4M. **(c) The proxy's
  tails are catastrophic in the AlphaStar sense**: naive_s0's 1.4M rung wins 4.6% against an earlier
  self, naive_s2's 1.8M wins 1.6% — Huang & Lee's §V-C 15.4% forgetting, reproduced and exceeded on
  our own env. **(d) Cycling fires in all 3** (0.0527/0.0352/0.0835 vs bands topping
  0.0132/0.0088/0.0066 — 4×/4×/12.7×), now 10 of 10 tournaments repo-wide. **(e) Diagnostics**:
  coverage 35/13/2 distinct of 200 (naive_s2 is the k4_s2 signature — one deterministic ~9-ply game
  against itself, self-play MSE 0.000); value MSE self/random/heuristic 0.331/0.733/0.760,
  0.681/0.873/0.822, 0.000/0.762/0.940. **One deviation recorded rather than smoothed: naive_s1
  breaks the "self-play MSE tracks distinct-game count" monotonicity** (13 distinct yet MSE 0.681,
  the second-highest measured) — its games are short (mean 9.3 plies) and its positions early (mean
  ply 4.5), where outcomes among even a few distinct games are genuinely mixed; the claim from the
  7-run entry weakens to "tracks within long-game regimes". Draws 0.8/0.4/0.4%. *Next*: promote the
  proxy snippet to committed tooling plus the regression-rate secondary against its simulated null
  band, then the Pons agent-metrics script, then figure + README.
- 2026-07-28 (forgetting tooling committed; both measures on all 10 ladders) — **`alphastar_proxy` /
  `regression_rate` / `regression_null_band` landed in `rl/selfplay/elo.py` with
  `scripts/forgetting.py` on top (2 commits, 261 tests green; battery `chunk4_forgetting.py` 4/4
  real caught, 2/2 controls survive), and the demonstration now stands on committed,
  mutation-guarded instruments.** One design decision was open because the scratch snippet never
  pinned it, settled with the maintainer: **the regression-rate null is the monotone REARRANGEMENT**
  — the run's own fitted rung ratings reassigned in sorted order over steps (same strength multiset,
  zero forgetting by construction), then binomial resimulation over actual decisive counts,
  mirroring `cycle_null_band` otherwise. This is the reading that makes the 2026-07-26 review triple
  coherent: a never-learns run's ~48% bare rate lands INSIDE its (flat) band and stops ranking as
  the worst forgetter; the mutation battery pins the rearrangement itself (skip the sort → caught by
  a permutation-invariance test). Committed numbers, all 10 ladders (proxy; regression vs band top):
  **naive 0.309/0.382 vs 0.127, 0.392/0.345 vs 0.109, 0.250/0.473 vs 0.091 — ALL THREE 3–5× ABOVE
  their zero-forgetting bands**; pool 0.610/0.073 vs 0.055 (marginally above), 0.458/0.236 vs 0.110
  (above — the known weak seed), 0.609/0.055 vs 0.073 (**inside**); k4 0.481/0.255 (above),
  0.527/0.109 (inside, at the edge), 0.567/0.145 (above); lam1 0.487/0.182 (above). **Primary and
  secondary now agree from independent angles**: proxy separation naive-vs-pool has no overlap, and
  the secondary places every naive seed far outside its null while the pool arm is marginal-to-clean
  (its one clear violation is pool_s1, already flagged by proxy 0.458 and 6×-band cycling). The
  probe arms sit between the arms on both measures, ordered as their coverage collapse predicts.
  `forgetting.json` written per run dir. *Next*: Pons agent-metrics script (the last instrument),
  then campaign script + figure + README; open decision — probe levers before or as campaign arms.
- 2026-07-28 (Pons agent metrics on the 6 campaign finals — the last instrument, and it refuses to
  flatter) — **`solver_move_scores` (exact per-move values; win-in-1 checked before the child solve
  because `solve()` refuses a won child) + `scripts/pons_agent_metrics.py` landed (2 commits, 262
  tests green, battery extended to 5/5 real caught); sweep over the 6 campaign finals in the
  maintainer's terminal (~2.2 min/run, middle_easy child solves dominate).** Two results. **(1) The
  arms do NOT separate on absolute tactical quality.** Optimal-move agreement, pool vs naive:
  end_easy {0.585, 0.602, 0.592} vs {0.582, 0.570, 0.593}; middle_easy {0.337, 0.389, 0.335} vs
  {0.353, 0.339, 0.358}; blunder rate 0.20–0.26 everywhere; regret ~1.5 (end) / 4.1–4.8 (middle)
  Pons units. So the pool's advantages — ~57 Elo, proxy separation, null-band cleanliness — are
  ROBUSTNESS-TO-OPPONENTS effects, not move-quality effects: pool training prevents forgetting
  without buying tactics. Consistent with chunk-2's arm-independent win-in-1 conversion (~0.25–0.31,
  and blunder ~0.25 rhymes with it), and it sharpens the still-named supervised-on-solver-labels
  question — whether the ceiling is the encoder or the training signal, since no training variant
  tried moves it past this band. **(2) The value metrics restate the off-distribution finding on
  EXTERNAL ground truth, more harshly than the MSE probe did: on the End/Middle sets, every critic's
  Brier (0.27–0.36) is WORSE than a constant-0.5 predictor's 0.25**, sign accuracy 0.50–0.59.
  Arithmetic cross-check that the two instruments agree: off-distribution value MSE ~1.1 in V units
  ⇒ Brier ≈ MSE/4 ≈ 0.28, as measured. The Begin sets read best (sign accuracy up to 0.73, Brier to
  0.21 — every game passes through the opening, so early positions are the least off-distribution)
  but still only straddle the uninformative baseline. Coverage: policy sets end_easy + middle_easy
  in full (middle_medium opt-in by runtime; Begin intractable per the closed validation); value
  metrics all 6 sets, decisive fractions 568–1000/1000. `pons_metrics.json` per run dir. **Every
  chunk-4 instrument has now run on the full campaign**; remaining: `runs/connect4_campaign.sh`,
  figure, README section, and the maintainer's probe-lever decision.
- 2026-07-29 (probe-lever campaign: 9 runs, all instruments — one lever hits its target, each fails
  differently elsewhere) — **All three levers ran as full arms (3 seeds each, maintainer's terminal;
  instruments via gitignored `runs/lever_instruments.sh`: tournament + Pons metrics + vs-random
  `score_ladder` curve per ladder, ~60 min, plus in-session coverage/value-MSE/forgetting).
  Provenance note: all 9 stamped sha `f421d23`, but 8 stamped `git_dirty: true` — attributed at
  verification time to exactly one UNTRACKED advisory .md appearing between launches (`git status`
  showed nothing else); source byte-identical to `f421d23`, runs accepted.** Numbers (final Elo;
  proxy; vs-random final/max; distinct/200; agreement end/middle): **entropy** −66.5/−273.6/−112.2;
  0.594/0.362/0.527; 0.892/.943, 0.890/.963, 0.877/.958; 33/94/176; 0.52–0.58/0.29–0.38. **pfsp**
  −166.2/−41.3/−171.3; 0.455/0.491/0.427; 0.820/.912, 0.823/.900, 0.875/.900; 15/34/39;
  0.58–0.61/0.34–0.37. **mix** −208.5/−102.3/−123.2; 0.424/0.597/0.503; 0.900/.935, **0.915/.915,
  0.948/.948**; 5/108/6; 0.55–0.58/0.31–0.34. Verdicts, against each lever's pre-registered aim:
  **(1) Fixed-opponent mixing is the only lever that moved its target** — the vs-random decline is
  ELIMINATED on 2 of 3 seeds (final = curve max; third seed 0.900 vs pool's ~0.88 endpoint) at no
  Elo cost vs pool — while its self-play coverage stayed collapsed (5–6 distinct on those same
  seeds): external coverage was bought by construction, exactly as designed, not via exploration.
  **(2) The entropy floor buys what it claims — coverage (up to 176/200, best measured) — and
  contains the first monotone ladder in the repo** (entropy_s0: final = best rung, −66.5,
  second-best final of all 15; regression rate INSIDE its null band, also a first for a strong run)
  — but seed variance is extreme (s1: −273.6 with proxy 0.362 despite healthy 94-distinct coverage)
  and the vs-random endpoint didn't move. **(3) PFSP holds the best single final of all 15 runs (s1,
  −41.3) and the best arm mean (−126), but the worst pool-family forgetting proxies (0.427–0.491)
  with catastrophic tails (mins 0.04–0.05)** — concentrating games on hard opponents trades
  robustness-to-past-selves for current strength, a coherent and now-measured trade. **(4) No lever
  moves tactics**: agreement stays in the same band as all six campaign finals (fourth independent
  confirmation) — the ceiling question now rests entirely on the scheduled supervised diagnostic.
  **(5) Cycling fires 9/9** (5.5–18.6× band tops): 19 of 19 tournaments repo-wide. **(6) Coverage
  and strength decouple in both directions** (entropy_s1: 94 distinct, −273.6; mix_s2: 6 distinct,
  vs-random 0.948) — coverage is a mechanism readout, never a success metric, as the spec
  pre-registered. 20-rung ladders tighten CIs to ~9–10 Elo; regression rate above its band in 8 of
  9.
- 2026-07-29 (supervised-on-solver-labels: SCHEDULED — the five-times-named diagnostic gets its
  decision) — **An advisory from a parallel session (no repo access) asked for the decision to be
  made explicitly: accepted, folded into this session's queue, advisory file deleted after folding
  in (the hardware-advisory precedent).** Two of its premises were stale — the campaign config had
  already locked and the kernel fork already settled — so the surviving rationale is INTERPRETIVE,
  and it grew sharper with the lever results: four independent training variants now leave
  optimal-move agreement in the same 0.29–0.61 band, and the README's read of the whole phase turns
  on whether that ceiling is the encoder or the training signal. Scope (one correction to the
  advisory's "broad distribution with exact labels": exact labels for the early game are the
  measured-intractable Begin regime, so both training and evaluation live in the tractable band —
  the same wall the Pons validation drew): positions from random playouts at ≥12 stones, one per
  game, solved under a node-budget cap (frontier measured 2026-07-28: 8–33% yield below 12 stones,
  67% at 12, ~100% at 18+, median sub-ms); policy targets from `solver_move_scores` (children of a
  12-stone position are 13+, all cheap), value target the outcome sign (γ=1 critic semantics); Pons
  sets held out; **both kernels (k3 + k4)** — de-confounding k4's capacity from its self-play
  dynamics; evaluated through `scripts/pons_agent_metrics.py` unchanged (behind an explicit
  allow-non-selfplay flag, not a weakened guard). **Pre-registered read, recorded before any result:
  supervised ≈ the RL band ⇒ the encoder is the ceiling; supervised ≫ the band ⇒ the training signal
  is, and the gap prices the levers' headroom.** Runs before the README so the write-up carries the
  ceiling number.
- 2026-07-29 (supervised diagnostic: the answer is BOTH, split by distribution — and it unifies the
  phase) — **Dataset (100,000 positions, stones 12–41, 99.1% decisive, Pons sets held out by
  bitboard key) + both kernel runs trained and evaluated (maintainer's terminal, ~90 min mostly
  label generation).** Numbers. **In-distribution** (5% held-out validation, recomputed in-session
  from the seed-deterministic split): optimal-move agreement **0.855 (k3) / 0.850 (k4)**, value sign
  accuracy 0.903/0.893. **On the Pons sets** (policy: agree/blunder/regret): end_easy
  0.608/0.241/1.45 (k3), 0.615/0.233/1.45 (k4); middle_easy **0.465**/0.205/3.80,
  **0.438**/0.220/4.01 — against the RL band's 0.52–0.61 (end) and 0.29–0.39 (middle). Value on
  middle_easy: sign 0.783/0.761, **Brier 0.159/0.165 — decisively better than the 0.25 uninformative
  baseline that every RL critic sat WORSE than**; begin_hard 0.437/0.465 (distant balanced outcomes
  defeat it, below coin-flip). Three conclusions, each answering a named question: **(1) The encoder
  is NOT the representational ceiling — but distribution is.** Given exact labels on states it
  trains over, the same 84k-param net picks the optimal move 85.5% of the time; on the
  differently-distributed Pons sets that falls to 0.44–0.62. Two separate gaps: RL→supervised on the
  eval sets is real but modest (middle_easy +0.05–0.13 over the band's top), while
  in-distribution→Pons is enormous (~0.4). The binding constraint on eval-set tactics is
  GENERALIZATION OFF THE TRAINING DISTRIBUTION at this net scale — which is the chunk-2
  coverage-collapse mechanism restated: the RL agents' effective training distributions were even
  narrower than random playouts, and the whole phase's tactical band is that narrowness made
  visible. The architecture was never the problem; where the policy lives is. **(2) The kernel
  fork's settled verdict is now unconfounded**: supervised k3 and k4 are indistinguishable
  everywhere (in-dist 0.855 vs 0.850; Pons deltas ≤0.03, mixed sign) — k4 has no extra
  representational capacity, so its self-play deficit was self-play dynamics, and k3-by-default
  stands on capacity grounds too. **(3) The value story confirms from the other side**: a critic
  trained on a broad distribution carries real signal onto the eval sets (Brier 0.16), so the RL
  critics' worse-than-uninformative 0.27–0.36 was never a critic problem — it was their collapsed
  state distribution. One number that did NOT move: blunder rate (supervised 0.21–0.24 vs RL
  0.20–0.26) — sign-class mistakes on off-distribution positions persist even at 0.855
  in-distribution capability, consistent with the distribution reading. **The levers' headroom,
  priced as pre-registered**: on the eval sets the training signal is worth at most ~0.1 agreement;
  everything beyond that requires broadening the state distribution the policy actually visits —
  which is exactly what fixed-opponent mixing (the one lever that hit its target) does, and what the
  capstone's team-randomized `gen1randombattle` does by construction. *Next*: wrap-up —
  `runs/connect4_campaign.sh`, figure, README.
- 2026-07-29 (PHASE 4 COMPLETE) — **Wrap-up landed: gitignored `runs/connect4_campaign.sh` (the full
  reproduction record, training through instruments through the supervised diagnostic),
  `assets/connect4_forgetting.png` (small multiples per arm + the proxy dot panel; palette
  re-validated), and the README Phase 4 section — written, as scheduled, with the ceiling number in
  hand. 271 tests green, all batteries passing (chunk1–4: every real mutation caught, every control
  surviving), tree clean.** Phase totals for the record: 15 self-play training runs + 2 supervised,
  19 tournaments (~1M tournament games), 4,400 externally validated solver positions, 5 committed
  instruments (`tournament`, `forgetting`, `coverage_probe`, `value_mse_probe`,
  `pons_agent_metrics`) + 2 generators (`make_solver_dataset`, `train_supervised`), 6 mutation
  batteries. The phase's three durable findings: naive self-play forgets and the pool prevents it
  (no proxy overlap); intransitivity is structural at this band (19/19 detectors); and the tactical
  ceiling is the visited state distribution, not the architecture — the finding that decides what
  the capstone must get right (opponent/state diversity) and what it need not fear (encoder capacity
  at this scale). **Next: Phase 5.**
- 2026-07-28 (capstone hardware line revised — a parallel-session advisory evaluated and accepted) —
  **The "rented cloud GPU" line was confirmed inherited from the Procgen-era capstone plan (it
  survived the 2026-07-26 Phase-5 rewrite in PLAN.md, CLAUDE.md and README.md) and is replaced:
  online self-play is CPU-first; a GPU enters only for offline supervised arms, the Procgen
  fallback, or an encoder grown past MLP scale.** The advisory (a separate chat session, no repo
  access) cited the MinAtar threading finding; the repo actually holds three independent
  measurements of the same tiny-net pathology (MinAtar 278→~1,550; Connect 4 2,196→8,473; SAC
  425→327 *with more* threads), plus ms-scale websocket env steps against µs-scale forwards and
  on-policy's inability to keep a device utilized — the advisory and the repo's measurements agree
  everywhere they overlap. Throughput measurement is **deferred to Phase 5 start** (it needs the
  poke-env + Node stack, forbidden until Phase 4 closes) with a four-item pre-registered list now in
  the Phase 5 section. Two additions landed with the revision: the **BC-on-replays diagnostic**
  recorded as named-optional (the one genuinely GPU-shaped workload; Phase 4's
  supervised-on-solver-labels is the same instrument at small scale), and a **collection-loop
  structural contract** — a single inference seam between battle coroutines and the policy — adopted
  now because it is cheap before the loop exists and keeps batching a config choice rather than a
  rewrite. One advisory detail corrected in place: the batch-1 forward pathology belongs to
  poke-env's native asyncio `Player` model, while the `SubprocVecEnv` route already batches at the
  vector boundary (at head-of-line-blocking cost). Procgen-era sweep (the advisory's standing note):
  grep found no other stale assumptions — the three hardware lines were the full residue. Advisory
  file (untracked) deleted after folding in.
- 2026-07-29 (Phase 5 opens: env plumbing landed) — **Origin synced (23 Phase-4 commits pushed,
  explicit go-ahead), then the capstone's step 1 landed as approved: poke-env pin, local server, env
  + adapter, 278 tests green (7 new), 2 commits.** *Dependency*: `poke-env==0.15.0` (still
  PyPI-latest — the exact version the 2026-07-26 API review audited, so every correction in the
  Phase 5 section applies as written; drags in pettingzoo 1.26.1 + websockets 16.0). *Server*:
  `smogon/pokemon-showdown` pinned at `59da482` by `scripts/setup_showdown.sh` into gitignored
  `showdown/` (pin-by-sha shallow fetch verified against GitHub); config = the example plus
  `exports.repl = false` — the REPL unix sockets crash `EINVAL` at boot on macOS + Node 25 (two
  CRASH stacks in a clean boot log, zero with repl off; the battle worker is unaffected either way).
  *Env* (`rl/envs/showdown.py`, registered `Showdown-v0` through `make_env` like Connect4):
  `ShowdownSingles(SinglesEnv)` with terminal-only reward = outcome (the Phase 4 shape) and a
  deliberately minimal 10-dim placeholder encoder (move base powers, type multipliers vs the
  opposing active, fainted fractions — the encoder-design step replaces it), plus the `ShowdownEnv`
  adapter doing exactly the two contract translations the review pre-registered: mask lifted from
  the obs `Dict` into `info["action_mask"]`, and `info["outcome"]` read from `battle.won`/`lost`,
  never from term/trunc (forfeit/timer arrive `truncated` but decided — the server sends `|win|`;
  only a tie leaves `won` None). Gen 1 action space = 10 (6 switches + 4 moves, no gimmicks), pinned
  by test. **One API fact beyond the review: plain `int` actions crash poke-env's `action_to_order`
  (it calls `action.item()` on the gimmick check) — the adapter casts to `np.int64`.** *Tests*:
  outcome/reward/encoder/opponent-factory units run fully offline (`start_listening=False`); the
  integration test plays a live episode acting only through the mask under poke-env's `strict=True`,
  which is the masking proof — any illegal converted order raises — and skips when nothing listens
  on :8000. *Smoke numbers*: scripted RandomPlayer-vs-MaxBasePower, 5 battles in 0.9 s; through the
  full `make_env` stack, 10 episodes per fixed opponent — mask-random loses 0/10 to `max_power` and
  `heuristics`, wins 8/10 vs `random` (milestone-1 headroom confirmed in both directions), at
  **~1,100 agent-steps/s, one env, one process, no policy net** — the env-only prior the four
  pre-registered throughput measurements start from. *Next*: the throughput measurements need the
  collection-loop seam (the single-inference-seam contract) — decide with the maintainer whether
  that seam or the milestone-1 train wiring comes first; encoder design after the measurements say
  what `embed_battle` may cost.

- 2026-07-29 (collection seam + throughput measurements (a) and (b)) — **The single-inference-seam
  contract is now code (`rl/collect.py`: `InferenceSeam` + `SeamPlayer`, batch-1 servicing, numpy
  boundary, counters as the timing hooks; 281 tests green, 3 new) and the first two pre-registered
  measurements ran (`scripts/showdown_throughput.py`, policy = real mlp[64,64] actor+critic on the
  placeholder encoder, torch 1 thread).** Architecture fact the numbers ride on: poke-env schedules
  every battle coroutine onto its singleton `POKE_LOOP` (daemon thread), so all in-flight battles in
  a process share one loop and one seam services them all — batch-1 inference blocks the same loop
  that services the websockets, the pathology represented honestly. **(a) Per-turn latency, 20
  battles / 1,257 decisions, one in flight: ~0.72 ms/decision — encode 0.018 ms, batch-1 inference
  0.098 ms, env gap 0.541 ms (p50 0.496, p90 0.677)** — the gap is ~75% of a decision and decomposes
  into protocol parsing 0.079 ms (both seats), websocket ping RTT 0.086 ms, residual ~0.38 ms
  (server compute + event-loop scheduling). The placeholder encoder is 2.5% of a decision — encoder
  headroom is large, as the hardware note predicted. Instrumentation lesson recorded: timing the
  async `_handle_message` spans awaited suspensions (it contains the whole decision path) and
  produced 5.1 s of "parsing" inside a 0.94 s run — parse is timed at the sync
  `Battle.parse_message`/`parse_request` instead. **(b) Concurrency curve, one process: ~1,700
  decisions/s at 1 battle in flight → saturation ~3,400 at 16, flat through 32, slight decline at 64
  (256 battles/point); batch-1 inference share of wall rises 0.15 → ~0.27 at the plateau.** Two
  readings: the asyncio ceiling is ~2× the serial rate and arrives early (16 battles); and by Amdahl
  even FREE inference buys ≤1.37× at the plateau — micro-batching's best case (measurement (d)'s
  question) is bounded before the real encoder exists. Scale check: 3.4k decisions/s is the same
  order as Connect 4 PPO's 8.5k steps/s — a 2M-decision collection ≈ 10 min/process at the plateau,
  before learner overhead. *Next*: measurement (c) multi-process scaling (workers ×
  battles-per-worker, one server per worker vs shared); (d) waits on the real encoder; then
  milestone-1 train wiring picks its collection route from (b)+(c).

- 2026-07-29 (throughput measurement (c): multi-process scaling — the collection stack's numbers are
  now complete except (d)) — **W spawned workers, each the (b)-plateau loop (16 in flight, 128
  battles/worker, barrier-synced battling spans, pid-based account names — poke-env's default
  per-process class-name counter would collide on a shared server), aggregate = total decisions /
  slowest worker's wall; machine = 14 logical / 10 performance cores.** **Shared single server
  (:8000): 3,646 / 5,910 / 5,100 / 5,141 / 4,533 decisions/s at W = 1/2/4/8/12 — peaks at TWO
  workers and declines**, with mean per-worker inference share falling 0.26 → 0.05: the lone node
  process saturates and the python workers idle against it. **One server per worker (ports 8100+i,
  booted and torn down by the script): 2,468 / 4,377 / 7,341 / 7,545 / 7,276 at W = 1/2/4/8/12 —
  scales to ~7.5k decisions/s (~115 battles/s) at W = 4–8, flat after**: the machine ceiling with
  server and worker sharing the same cores. One anomaly recorded: per-worker W=1 (2,468) reads BELOW
  shared W=1 (3,646) — the per-worker server is freshly booted (cold JIT) where :8000 has been
  serving all evening; the crossover is at W≥3 regardless. **Provisioning arithmetic that falls
  out**: milestone-1-scale collection (2M decisions) ≈ 4.5 min at the multi-process ceiling or ~10
  min in ONE process — single-process native-asyncio collection through the seam (3.4–3.6k/s)
  already outruns the whole Connect 4 pipeline (8.5k steps/s but WITH learning), and the Gym-adapter
  serial path (1.1k/s) is the floor, not the plan. CPU-first re-confirmed from the collection side:
  inference share never exceeds ~0.27 anywhere on the curve, so no accelerator changes any of these
  numbers. *Remaining*: measurement (d) (forward-pass share, batch-1 vs batched) waits on the real
  encoder; milestone-1 wiring now picks its collection route with (a)–(c) in hand — the open
  question is seam-loop-native vs `SubprocVecEnv`-over-Gym for TRAINING (the rollout buffer
  boundary), not for raw collection, where the seam route has won on measurement.

- 2026-07-29 (collection-architecture fork settled by a three-review pass; two env bugs found and
  fixed; milestone-1 config landed and smoked) — **The maintainer directed a three-reviewer
  adversarial pass (independent sessions, different lenses: correctness / harness integration /
  shipping pragmatism) over the proposed seam-native collector, and the proposal was REJECTED
  unanimously — the decision of record is now: milestone 1 trains on the zero-new-code path,
  `SyncVectorEnv` over `Showdown-v0`.** The three independent kill shots, each verified in source or
  measured live: (1) *correctness* — `PPOAgent.update` recomputes `old_logp`/values at drain on the
  premise "the policy hasn't changed since it acted" (`ppo.py:388-399`); battles spanning an
  `update()` violate it and the recompute forces ratio=1 on stale rows, silently, while also
  breaking the repo's `lr=0 ⇒ approx_kl==0` masking probe; (2) *concurrency* — "stop servicing the
  seam" is not a barrier (a request already inside the forward reads tensors `optimizer.step()` is
  mutating; `Categorical.sample` and `randperm` share the global torch generator across threads);
  (3) *economics* — measured process CPU/wall is **0.97 at just 8 battles in flight**, so the async
  collector recovers idle that does not exist: ≤15% over a lockstep facade at 2–4× the code, on a
  milestone whose slowest path (~1.2k dec/s, measured through `make_vec_env` working TODAY with
  `reset_mask` intact) finishes 2M decisions in ~30 min. **Two live bugs found in the landed
  adapter, both fixed with a stub-tested pump/remap in `ShowdownEnv.step` (battery `phase5_env.py`:
  4/4 real caught, 1/1 control survives)**: (i) *phantom transitions, measured 6.4% of raw steps vs
  max_power* — at `battle._wait` poke-env never asks our seat to move, `PokeEnv.step` silently
  discards the caller's action yet returns a full transition with a placeholder one-legal mask, rows
  with zero policy gradient that still skew advantage normalization, the entropy readout (Phase 4's
  health metric), and episode lengths; now absorbed inside `step()` (with a `waits_absorbed` counter
  and a live regression test) and guarded by an assert so any future discard path fails loudly; (ii)
  *truncation double-count* — poke-env marks forfeits/ties/timer losses `truncated=True`, and
  `compute_gae` keeps the bootstrap on truncated rows, stacking γ·V(final) on top of the terminal ±1
  at γ=1.0; every learner-visible finish is a completed game (reset/close-injected forfeits never
  surface), so finishes now map to `terminated=True` unconditionally. Also verified for later: the
  pause premise holds server-side (an order held 150 s mid-battle resumed cleanly; the 20 s
  `ping_timeout` is the real budget if POKE_LOOP itself ever blocks). **Named and deferred**: the
  decision-lockstep facade behind `make_vec_env` (owns both seats → batches opponent forwards, keeps
  `rl/train.py`/`RolloutBuffer`/PPO/pool wiring untouched) is the follow-up if the wall binds, gated
  on pre-registered **measurement (e)** — challenge-to-first-request slot-reset latency plus actual
  facade decisions/s vs the (b) curve, because 1,100-vs-3,400 was concurrency-vs-serial, not
  architecture-vs-architecture; the seam collector stays scoped to self-play scale with its three
  preconditions written down (collection-time logp/value storage, drain-to-quiescence barrier,
  forfeit-leak tagging). **`configs/showdown_maxbp_ppo.yaml` landed** (Phase 4 recipe where reasons
  transfer: γ=1.0 terminal-only, λ=0.95, clip 0.2, entropy 0.01; mlp[64,64] on the placeholder
  encoder; 8 envs × 128 rollout = the 1024 cadence) **and an 8,192-step smoke through the full
  harness ran clean**: all locked metric names emitted, eval/win_rate 0.2–0.4 vs max_power for an
  untrained policy with win-rate/return arithmetic mutually consistent, episode length 25–44,
  **~1,050 steps/s through the complete loop** — collection plus learning, on the predicted number.
  287 tests green. *Next*: the 2M milestone-1 run in the maintainer's terminal from a clean tree,
  then read the curve.

- 2026-07-29 (MILESTONE 1 PASSED: PPO beats MaxBasePowerPlayer — and the encoder is confirmed as
  milestone 2's lever) — **The 2M-step run (`showdown_maxbp_s0`, maintainer's terminal, clean tree
  at `31662e9`, `git_dirty: false`, ~34 min at 983 steps/s median) trained clean, and the headline
  stands on the locked protocol: FINAL checkpoint (never best — selection bias), 1,000 fresh battles
  at seeds past the training-eval ladder — 663/1000, win rate 0.663 ± 0.029 (95% CI), ties 2.5%.**
  The CI floor (0.634) clears the beat-the-bot bar without argument. Curve reads, all healthy:
  `eval/win_rate` 0.29 at 100k → plateau ~0.65 ± 0.05 from 500k on, final rung 0.72 (in-training
  rungs are 100-episode estimates, SE ≈ 0.05 — the 0.52 dip at 1.8M is noise-band); entropy 1.74 →
  0.58 with no collapse (against masked-max ≈ ln 10 = 2.30) — the phantom-row fix means this number
  is now trustworthy; `approx_kl` ≤ 0.004 throughout; `loss/value` RISES 0.16 → 0.34, the benign
  twin this time — early play loses near-deterministically (predictable −1), a ~0.65-win policy has
  genuinely uncertain outcomes (outcome variance ≈ 0.9, so 0.34 still carries real signal); episode
  length 33 → 24.5 (it wins faster, not just more). **The milestone-2 price, measured the same night
  (500-battle scratch probe, same final checkpoint, `eval_opponent` overridden to heuristics): 0.262
  ± 0.039** — the three-review pass's prediction confirmed in BOTH directions: the 10-dim
  placeholder encoder, whose only tactical content is move base power and type multipliers, is
  exactly sufficient against a bot that ignores type effectiveness and roughly a random policy's
  showing against `SimpleHeuristicsPlayer` (which reads HP, status, boosts, switch value — none of
  which the encoder represents). Phase 4's closing lesson lands intact on the new domain: the
  ceiling is what the policy can SEE and VISIT, not the training signal — `eval/win_rate` vs
  max_power plateaued by 500k with 1.5M steps buying ~nothing, the same shape as every Connect 4
  arm. *Next, in order*: encoder design (the real Phase 5 design work — observed-state features per
  PLAN's capstone spec; run measurement (d) forward-pass share at the real encoder while at it),
  then the milestone-2 run vs `SimpleHeuristicsPlayer`; the lockstep facade stays parked unless the
  wall binds. `final_eval_maxbp_1000.json` in the run dir is the headline artifact.

- 2026-07-30 (encoder design landed: the Gen 1 observable-state encoder replaces the placeholder;
  measurement (d) complete; milestone-2 config ready) — **Maintainer-approved design session settled
  the four pre-stated decision points: (1) NO species embedding — species identity enters only
  through base stats + types, both derivable from the observed species, so the obs stays a flat Box
  and the harness is untouched (an embedding table is the priced follow-up if milestone 2 stalls);
  (2) engineered type-chart multipliers AND raw type one-hots — the multiplier is the directly
  decision-relevant scalar under terminal-only reward, the one-hots let the net learn what the
  scalar can't express; (3) boosts as raw stage/6; (4) full 5-slot opponent bench blocks
  (revealed-or-zeros behind a revealed flag). OBS_DIM 10 → 611.** Layout (rl/envs/showdown.py, all
  offsets documented in fill helpers): global(5: turn, fainted fractions, force_switch, trapped) |
  our 6 team blocks (32 each: hp, fainted, is-active, status one-hot, level, 5 base stats, 15-type
  one-hot, best-multiplier matchup both directions vs the opposing active — the switch-value signal
  SimpleHeuristicsPlayer reads) | our active extras (16: 7 boosts, 7 volatiles, status_counter,
  preparing) | our 4 move blocks (23 each: known, bp, acc, PP fraction, multiplier vs foe,
  physical/status flags, priority, 15-type one-hot) | opponent mirror (revealed mons and revealed
  moves only — the information-set line). **Slot alignment pinned from poke-env source
  (singles_env.py): switch action i = list(battle.team.values())[i], move action 6+j =
  list(active.moves.values())[:4][j] — team and move blocks use exactly those orderings so the
  policy can associate slot-i features with action i.** One representational gap, documented not
  fudged: Gen 1 Light Screen is a per-mon volatile the sim emits as "|-start|...|Light Screen",
  which poke-env 0.15.0 maps to Effect.UNKNOWN (no LIGHT_SCREEN member) — dropped rather than
  parser-forked; Reflect parses fine. **Measurement (d) (new `showdown_throughput.py d`, offline
  forward microbench at the real encoder): batch-1 forward 58µs at 611 dims, per-sample 1.6µs at
  batch 128 (~37× batching headroom that nothing currently needs). Measurement (a) rerun live at the
  real encoder: encode 0.079ms/decision (4.4× the placeholder's 0.018 but still ~11% of the 0.71ms
  decision), inference 0.089ms, env gap unchanged at 0.545ms (~75%) — the encoder is NOT the
  bottleneck and no collection-architecture change is warranted, exactly as the hardware note
  predicted.** 8,192-step smoke through the full harness vs heuristics (config pattern, tensorboard
  logger): all locked metrics emitted, eval/win_rate 0–0.05 for a near-untrained policy (consistent
  with mask-random's 0/10), entropy 1.69 vs masked-max 2.30, value loss falling, win-rate/return
  arithmetic mutually consistent (−0.85 at 20 eps = 1W/1T/18L), **~870–1,000 steps/s through the
  complete loop — the 611-dim encoder costs ~8% against milestone 1's ~1,050, so the 2M milestone-2
  run stays ~35–40 min**. 290 tests green (5 new encoder tests — layout, slot order, per-block
  features, bounds — replacing the 2 placeholder tests; both live-server integration tests pass at
  the new encoder under strict=True). `configs/showdown_heuristics_ppo.yaml` landed: the milestone-1
  recipe verbatim except `opponent: heuristics` — hidden_sizes stays [64,64] on Phase 4's capacity
  finding. *Next*: the 2M milestone-2 run in the maintainer's terminal from a clean tree, then the
  locked-protocol 1000-battle final-checkpoint eval.

- 2026-07-30 (milestone-2 first run at the real encoder: NOT passed at 2M — but the curve never
  plateaued, which is the finding) — **`showdown_heur_s0` (2M steps vs SimpleHeuristicsPlayer,
  maintainer's terminal, 37.7 min at ~850 steps/s median). Provenance: stamped sha `7a4155f` with
  `git_dirty: true` — unavoidable and correct this once: the encoder the run trains on IS the
  uncommitted 2026-07-30 session work (six files, verified as the whole diff); the tree goes into
  the commits byte-identical, so the stamp resolves to those commits.** Headline on the locked
  protocol (FINAL checkpoint, 1000 fresh battles, seeds past the training ladder,
  `final_eval_heur_1000.json`): **292/1000, win rate 0.292 ± 0.028 (95% CI [0.264, 0.320]), ties
  1.7% — the beat-the-bot bar is not met.** The curve is the real result: `eval/win_rate` 0.06 at
  100k → 0.31–0.40 band at 1.8–2M and **still climbing at the wall — no plateau, where milestone 1
  plateaued by 500k with 1.5M wasted steps.** The encoder unambiguously moved the needle within-run
  (untrained smoke 0.00–0.05 → 0.40 peak rung; the milestone-1 placeholder policy sat at 0.26 and
  could not represent HP/status/boosts at all — cross-protocol, so indicative not paired). Secondary
  reads: entropy 1.72 → 0.355, lower than milestone-1's 0.58 endpoint and still falling — a watch
  item for a longer run (the Phase 4 entropy-floor lever is the tested instrument if it collapses);
  `loss/value` rises 0.15 → 0.26, the benign twin again (outcome variance at p≈0.3 is ~0.82, so 0.26
  MSE carries real signal); `approx_kl` ≈ 0.002 throughout; episode length 25 → 20. *Next, discussed
  with the maintainer*: budget is the obvious single-variable lever — a longer run (~19 min/M steps)
  before touching entropy floors or opponent mixing; both stay named fallbacks from Phase 4 if the
  curve stalls short of 0.5.

- 2026-07-31 (distribution lever at fixed-bot scale: NOT credited — the pre-registered band read
  fires 'at/below', and milestone-3 self-play moves up; first 3-seed read of the campaign) —
  **`showdown_mix512_s{0,1,2}` (70/20/10 heuristics/max_power/random MixturePlayer on the 512 trunk,
  6M × 3 seeds CONCURRENT — the Stage-0 pattern's first campaign use: ~653 steps/s each, all three
  in one ~2.6 h window; wandb behaved, exactly one offline-run dir per run). Locked-protocol
  headlines: 0.348 / 0.358 / 0.363, pooled 1069/3000 = 0.356 ± 0.017 — and the matched-budget
  control, run for exactly this comparison: the PURE-heuristics 512 run's 6M checkpoint under the
  same locked protocol scores 0.324 ± 0.029.** The pre-registered read (config header) was the
  RUNG-BAND comparison, and it fires 'at/below': mixture 4–6M bands 0.357/0.348/0.337 (mean 0.347)
  vs the capacity run's 0.346 — indistinguishable. The locked-eval comparison leans positive (+0.032
  pooled-vs-control, z ≈ 1.9, p ≈ 0.06 — marginal; plausibly inflated by the control checkpoint
  sitting in a rung dip, 0.324 vs its own 0.346 band) — so the honest bracket on the mixture effect
  at this scale is 0 to +0.03: REAL at most as a nudge, nowhere near the 0.09–0.14 gap, and the
  per-battle mechanism worked as designed (live-smoked 8/3/1 pick split; mixture seeds hold entropy
  HIGHER at 6M — 0.32–0.43 vs pure's ~0.30 — the random/max_power battles keep the policy stochastic
  without translating into eval strength). **Per pre-registration: fixed-bot state coverage is NOT
  the remaining constraint — milestone-3 self-play moves up the queue.** Methodological bonus, the
  campaign's first n=3: seed-std of the locked headline is 0.008 — far tighter than Phase 4's Elo
  spreads led us to fear at this budget — so the earlier n=1 lever reads (budget, capacity)
  retroactively gain credibility, and 3-seed reads are cheap enough (~2.6 h) to be the default going
  forward. Lever ledger for milestone 2 to date, all locked-protocol: encoder 0.262→(enabling),
  budget 0.292→0.358 (credited, exhausted), capacity 0.324@6M→0.408@12M (credited, biggest single
  lever, curve still creeping), distribution-via-fixed-bots ≤+0.03 (not credited). *Next*:
  milestone-3 self-play machinery — the Phase-4 pool + the seat-2 Player adapter driving our policy
  (not yet written), with the parked lockstep facade as the collection architecture of record there
  (async review 2026-07-30) and eval anchored on heuristics throughout (the milestone-2 bar stays
  the scoreboard; H&L and Metamon both got their strength from self-play + diversity, not fixed-bot
  grinding).

- 2026-07-31 (Stage-0 probe: concurrent seeded runs PASS — async shelved, campaign moves to 3-seed
  lever reads) — **The pre-registered free-lever probe (`configs/showdown_probe_100k.yaml`,
  timing-only 100k at the 512 recipe, maintainer's terminal per the throughput-measurement
  discipline): solo 859 steps/s (matches the 12M run's 840 median — probe representative); three
  concurrent runs 689/694/691 — a uniform −20% per run, under the ≤25% bar, aggregate 2,073 steps/s
  = 2.4× solo.** Decision of record, per the async-review pre-registration: **the async vec-env
  branch stays SHELVED** — concurrent seeds beat its honest estimate (~1,900 steps/s on one run) on
  aggregate throughput while also buying n=3 per lever read, at zero code and zero risk to the
  pool-identity seam. Server headroom note: 3 runs ≈ 24 battles in flight against a ~5.9k dec/s
  measured ceiling — do not extrapolate to 4+ without re-probing. A 3-seed 6M lever read now costs
  ~2.4 h wall. *Next*: MixturePlayer (lever 2, on the 512 trunk), then the first 3-seed run of the
  campaign.

- 2026-07-31 (CAPACITY WAS BINDING: [512,512] breaks the [64,64] ceiling — 0.408 at 12M and the
  curve still hasn't flattened) — **`showdown_heur_512_s0` (12M steps, [512,512] ≈ 1.16M params,
  maintainer's terminal, clean tree at `35314be`, 4.2 h at 840 steps/s median — the capacity lever
  cost ~2% throughput, not the predicted ~30%; even reviewer 3's live-measured 765 was pessimistic
  because early-run segments include eval rungs). Locked-protocol headline (FINAL checkpoint, 1000
  fresh battles, `final_eval_heur_1000.json`): 408/1000, win rate 0.408 ± 0.030 (95% CI [0.378,
  0.438]), ties 2.1%.** The pre-registered read (config header) fires cleanly: **the plateau MOVED —
  capacity was binding.** Two comparisons, ordered by cleanliness: (1) matched-budget
  single-variable read — at 4–6M the 512 rung band averages 0.346 where the [64,64] 6M run's
  same-band average was 0.312 and flat; (2) the shape difference is the louder fact — [64,64]
  flattened hard at ~0.31 from 2.5M on, while [512,512] climbs monotonically in 2M bands (0.219 /
  0.287 / 0.346 / 0.392 / 0.410) and is STILL creeping at 12M (9–12M vs 6–9M: +0.018, decelerating
  but not flat; last-10 rungs 0.30–0.50). The 0.408 headline vs the budget run's 0.358: CIs graze
  ([0.378,0.438] vs [0.328,0.388]) — but that pairing confounds capacity with budget; the
  attribution rests on (1)+(2). Mechanism reads, notably different at 1.16M params: entropy sits in
  a 0.29–0.36 band ALL run (no monotone decay to 0.27 like [64,64] — the bigger net holds
  exploration while improving, which may itself be part of why it keeps climbing); `approx_kl` ends
  ~0.013 vs [64,64]'s ~0.003 (larger net moves more per update at the same lr — worth watching
  against clip 0.2, not yet alarming); stalls stay rare (9 of 425k episodes ≥500 turns);
  win-rate/return arithmetic consistent (0.408/−0.163/2.1% ties). **Milestone-2 bar (0.5) NOT yet
  met — gap 0.09 — but for the first time in Phase 5 the curve is unflattened at the wall, so the
  bar is plausibly reachable rather than ceiling-blocked.** Phase 4's "capacity was never the
  problem" is now formally scoped: true at Connect4/10-dim scale, FALSE at 611-dim — the
  supervised-diagnostic instrument (BC arm, scoping GO'd 2026-07-30) can now price how much more
  capacity is worth from the offline side. *Next (unchanged morning agenda, sharpened)*: (1) Stage-0
  free-lever probe (3 concurrent 100k runs vs solo) — decides async shelving AND enables 3-seed
  reads; (2) MixturePlayer on the 512 trunk — the distribution lever now runs on an un-throttled
  net, exactly the stacking order the async review argued for; (3) candidate runs after the probe,
  in single-variable discipline: 512 + mixed opponents (lever 2 proper) vs 512 continued/bigger
  (capacity step 2 — weaker: the curve's deceleration says distribution likely binds next); budgets
  ~3–6M per the review's read-resolution finding — though note the 512 curve resolved its OWN read
  only past 6M, so "reads resolve by 3M" is per-lever, not a law.

- 2026-07-30 (async-collection proposal: three-review pass returns APPROVE-WITH-CONDITIONS ×3 — and
  demotes it behind free levers; protocol pre-registered) — **Maintainer-directed three-reviewer
  adversarial pass (independent Opus sessions: correctness / systems / experiment-design) over the
  proposed `vec_mode: async` (AsyncVectorEnv option in `make_vec_env`, Showdown-only, motivated by
  ~1 busy core of 14 during the 12M capacity run). Decision of record: the flag's design is sound
  (~25-line diff) but it is SHELVED behind zero-code levers; build it only if the wall re-binds.**
  The reframe (reviewer 3, verified against our own runs): the wall is partly self-inflicted — every
  lever read so far resolves by ~3M while we run 12M, and the machine + server fit ~3 CONCURRENT
  seeded runs (one process uses ~1k steps/s of a ~5-6k-ceiling server), which is 3×
  experiments/evening with zero code AND fixes the campaign's n=1-per-lever weakness; measured live
  during review, the 512 run does ~765 steps/s (~4.4 h wall), not the predicted ~600. The async
  prior re-priced honestly: realistic 1.8–2.6×, hard Amdahl ceiling 3.9× at the 512 config — "3–5×"
  retracted; ratio ≥4.0 in any future A/B is a falsifier (broken measurement), not a win. Two
  kill-shot-class findings if ever built, unanimous: (1) cloudpickled factories SEVER the
  shared-opponent-identity seam (`make.py:24-29`'s documented contract) — each worker would get a
  private SnapshotPool copy, `pool.push` never reaches envs, self-play silently trains vs a frozen
  step-0 opponent while `selfplay/*` metrics lie; existing identity tests reach through `.envs`
  (sync-only) and cannot catch it; mandatory guard: async hard-rejects non-scalar env_kwargs. (2)
  `context="spawn"` must be pinned: the parent is ALWAYS poke-env-dirty (gymnasium builds a dummy
  env parent-side — which also leaks 2 websockets/accounts for the run's life since
  `PokeEnv.close()` never closes websockets); fork deadlocks on POKE_LOOP, and forkserver is
  deterministic-catastrophic because `set_seed` runs before env construction → all 8 workers draw
  byte-identical account names → server `|nametaken|` swallowed inside a fire-and-forget task →
  presents as 8× 60 s "Agent is not challenging" timeouts naming the wrong cause. Further
  conditions: `vec_mode` a VALIDATED Config field (typo must raise, never silently run sync —
  load_config only type-checks today); teardown try/finally + `close(timeout=)` (untimed pipe
  recv/join + poke-env's untimed `_challenge_task.result()` = one wedged websocket hangs training
  exit forever); pid-based account names regardless of platform; re-baseline sync with `simulator:
  4` (config.js currently 1 — the single node simulator child IS measurement (c)'s shared-server
  saturation mechanism) before crediting async anything; measurement (e) (reset latency) is a
  PREREQUISITE, not a follow-up — under lockstep the whole vector stalls on one env's challenge
  round trip, and R alone spans the estimate 1.6–4.1×. The naive two-1M-run speedup test was
  unanimously rejected as noise-blind (eval/win_rate never traverses the changed code; rung SE ≈0.05
  hides phantom-class bugs; 1M is mid-climb; battles server-rolled so pairing is fiction) and
  replaced by a pre-registered ladder: (A) offline sync-vs-async bit-equality differential on a
  deterministic env INCLUDING the partial reset_mask path + mutation battery `phase5_vec.py`
  (dispatch→always-sync, dropped validation, dropped guard, dropped spawn; +1 control); (B)
  throughput A/B = six 100k runs S-A-S-A-S-A alternated on a fresh exclusive server, adopt iff
  median ratio ≥2.0 AND min(async)>max(sync); (C) equivalence on the same runs via mechanism
  invariants (entropy/KL/clip bands, episode-length KS, waits_absorbed rate ±20% — exposed via
  `envs.call`, and the lr=0 ⇒ approx_kl==0 probe under async, noting reviewer 1's caution that the
  probe canNOT detect cross-env permutation, which is what (A) pins); (D) 1M soak before any
  overnight async run (no resume path exists — a crash at hour 3 costs the evening); (E) first async
  campaign run = 6M replication, CI must overlap [0.328, 0.388]. If the gate fails the branch is
  DROPPED, not merged default-off (CLAUDE.md dead-configurability rule). Also recorded: reviewer
  consensus that gymnasium 1.3.0's async reset_mask/DISABLED/info-aggregation semantics are
  bit-compatible with sync (verified in source, both call the shared `_add_info`), async's
  `step_wait` is a full barrier so the seam-collector kill shots do NOT apply, `num_envs` under sync
  is a non-lever (serial stepping; raising it silently changes update cadence), and per-run
  `server_configuration` passthrough (~3 lines) is strictly more leverage per line than async — it
  composes with concurrent runs toward measurement (c)'s 7.5k/s per-worker-server ceiling and is a
  prerequisite for async ever scaling past one server. Milestone-3 note: the parked lockstep facade
  remains the architecture of record there (in-process, preserves pool identity); async is at most a
  milestone-2 expedient. *Next, in order (morning agenda)*: (1) 512 capacity verdict — locked
  1000-battle eval on `showdown_heur_512_s0` final checkpoint (auto-fires via watcher if the session
  holds; else run `scripts/eval_checkpoint.py runs/showdown_heur_512_s0/ckpt_012000000.pt --episodes
  1000 --out runs/showdown_heur_512_s0/final_eval_heur_1000.json` and read the curve); (2) Stage-0
  free-lever probe — 3 concurrent 100k runs vs 1 solo, steps/s degradation, decides whether async
  stays shelved; (3) MixturePlayer design+build (lever 2, ~20-30 lines, async-compatible by
  construction); (4) drop run budgets to ~3M now that reads resolve by then.

- 2026-07-30 (budget lever: credited, then exhausted — 6M run lands 0.358, flat from 2.5M on;
  capacity is next) — **`showdown_heur_6m_s0` (the 2M recipe verbatim at total_steps 6M,
  maintainer's terminal, clean tree at `dfdfbdc`, 116 min at ~860 steps/s). Locked-protocol headline
  (FINAL checkpoint, 1000 fresh battles, `final_eval_heur_1000.json`): 358/1000, win rate 0.358 ±
  0.030 (95% CI [0.328, 0.388]), ties 1.8%.** The verdict has two halves, both clean: (1) the 2M and
  6M CIs DO NOT overlap ([0.264, 0.320] vs [0.328, 0.388]) — budget bought a real +0.066, and the
  curve says where: the climb the 2M wall truncated completes by ~2.5M; (2) from there the curve is
  DONE — plateau halves 2.5–4.2M and 4.2–6M average 0.306 vs 0.312 (Δ0.006 across 1.75M steps), so
  further budget buys ~nothing and the remaining gap to the 0.5 bar is ~0.14. Secondary reads:
  entropy 1.68 → 0.267, still drifting slowly, never cratered — low-but-stable, not collapse, so the
  entropy floor stays third in the lever order; stall-to-tie episodes exist but stay rare (3 of 214k
  training episodes ≥500 turns; eval ties 1–4%/rung, 1.8% at the headline) — tie-farming is real,
  logged, and not the plateau's cause. **Architecture research (same evening, sources in the entry):
  the published bracket for beating heuristics at random battles is ~0.5M–1.5M params — Huang &
  Lee's gen7randombattle PPO used 1,327,618 params (structured: 128-dim entity embeddings, per-mon
  encoders, max-pool, shared action heads; 929/1000 vs most-damage, 612/1000 vs tree-search
  pmariglia, naive self-play, light shaping ±0.0125/faint), and Metamon's first heuristic-beating
  models were 500k–4M-param BC-RNNs with 15M chosen as the transformer floor for underfitting. Our
  trunk is ~45k params on a 611-dim obs — 30× under the closest precedent. Opponent-mixing canon
  verified while at it: OpenAI Five 80/20 current/past (the maintainer's remembered "20% random
  stuff"; AlphaZero never mixed opponents — its 25% was Dirichlet root noise), AlphaStar ~35%
  self-play + PFSP over the league, and Phase 4's own latest_prob 0.8 + fixed_mix 0.05 sits squarely
  in that canon — mixing's real home is milestone 3's pool, where the machinery already exists.**
  *Next (launching tonight)*: `showdown_heur_512_s0` — [64,64] → [512,512] (~1.16M params,
  Huang-&-Lee scale), 12M-step overnight budget, rungs unchanged so the first 6M is the exact paired
  capacity-vs-budget read; pre-registered in the config header: plateau shifts up ⇒ capacity was
  binding; stays ~0.31 ⇒ capacity exonerated and the distribution levers (MixturePlayer, then
  entropy floor) take over.

- 2026-07-30 (BC diagnostic scoping: GO-WITH-CAVEATS — a parallel research session's advisory,
  verified premises folded in, file deleted per the advisory precedent) — **The five-times-named
  diagnostic is FEASIBLE on data: ~109,147 archived `gen1randombattle` replays (~2.7M decisions at
  ~25/battle) in the HolidayOugi/pokemon-showdown-replays HF archive (README-reported count, NOT
  primary-verified; license unstated), and the official `replay.pokemonshowdown.com/search.json` API
  verified live and accumulating (~100–120 replays/day from a single ~10 h window —
  order-of-magnitude only). Programmatic JSON access is the documented, sanctioned path (WEB-API.md;
  no published rate limit — self-throttle ~1 req/s; full self-scrape ≈ low-tens-of-hours), preferred
  over the unlicensed third-party dump beyond prototyping.** The binding cost is the PARSER: Metamon
  (arXiv:2504.04395, code MIT, datasets CC-BY-NC) released parsed replays for Gen 1–4 OU/NU/UU/Ubers
  + Gen 9 OU ONLY — no random-battle format anywhere in the ecosystem has a spectator→first-person
  reconstruction, and Metamon's own docs call the problem inherently imperfect ("the server sends
  info to the players that it does not save to its replay... there is no way to be perfect"). **One
  leak verified on a live replay fetched during scoping: the raw log stores EXACT HP fractions for
  both sides (`|-damage|p2a: Chansey|241/481`) despite the `HP Percentage Mod` rule tag.** Sharpened
  against our own stack: poke-env's live battles give each seat the OPPONENT'S HP at /100 resolution
  while own-side HP is exact, so the parser rule is round-opponent-to-/100, keep-own-exact —
  otherwise the BC arm trains on precision the deployed encoder never sees (a train/deploy skew, not
  merely leaked omniscience). Quality caveat: accumulation is bursty and partly tournament-sourced
  (a Jan–Jul 2023 slice sampled as entirely `smogtours-*`) — stratify by rating/source, don't treat
  the corpus as IID ladder play. *Open questions carried*: (1) primary-source total count (~2,183
  paginated requests would settle it); (2) real accumulation rate (two timestamped queries N days
  apart); (3) log-schema drift across years; (4) whether a full-team reveal exists at battle end
  (2019 "full information replays" thread was Approved by Zarel — sample complete logs end-to-end
  before freezing parser assumptions); (5) HolidayOugi license clarification if ever used beyond
  private prototyping. Scoping was research-only — no repo files touched by the advisory session;
  PLAN.md's BC paragraph now carries the resolved verdict.

- 2026-07-31 (milestone-3 design session: three-review adversarial pass, then the machinery landed —
  PoolPlayer, init_from warm start, pool-health metrics, cross-play eval; campaign is warm-started
  self-play vs matched control) — **Design drafted (PoolPlayer = the seat-2 Player adapter; four
  forks: facade-vs-batch-1, anchor mixing, from-scratch-vs-warm-start, push cadence), then a
  maintainer-directed three-Opus adversarial pass (correctness / experiment design / systems).
  Verdicts, several against the draft: (A) NO lockstep facade — unanimous, but on corrected
  evidence: the draft's throughput constants were misattributed ([64,64] microbench read as
  [512,512] — `showdown_throughput.py` hardcodes [64,64]); at [512,512] batch-1 is
  compute-saturated, batching headroom ~zero, honest facade ceiling ~5%, and its real prize
  (decoupling battles-in-flight from num_envs, ~1.9×/process) is unbankable against a shared server
  at ~80% of ceiling — `simulator: 4` in `showdown/config/config.js` is the one-line lever if the
  wall ever binds, ahead of any facade. (B) Anchor mixing KILLED for run 1: 10% heuristics battles
  would train on the eval bot (self-fulfilling headline), the Phase-4 fixed_mix evidence re-read
  says it bought coverage "by construction" on 2 of 3 seeds with zero strength gain and WORSE
  forgetting proxies, and the proposed `mix:self=` mechanism had a provable stale-report bug (after
  an anchor battle `pool.report`'s identity match hits whichever member played last — silent,
  directionally biased PFSP corruption). Future anchors, if forgetting fires: max_power or a policy
  anchor, never heuristics. (C) Run 1 FLIPPED to warm-start + matched control (the 2-1 nominal split
  for from-scratch dissolved once init_from measured ~5 lines and the science argument stood:
  from-scratch changes opponent identity + strength trajectory + stationarity at once, and 12M ≈ 6%
  of H&L's from-scratch budget — 3.84M matches × both-sides collection ≈ 192M learner transitions —
  so a null is unattributable; warm-start + control is paired, single-knob, both arms in one ~5h
  evening). From-scratch 12M stays the pre-registered narrative arm if the paired read is flat
  (expected 0.20–0.35). (D) push_every_updates 150, not 20 or 100: the Bansal-validated quantity is
  the ~half-run history span — strided retention keeps anchor + 19 newest, so cadence 150 at 6M
  spans 48.6% where Phase 4's own setting spanned 19.5% (δ≈0.2, an unnoticed artifact) — and cadence
  has zero throughput content (deepcopy measured 1.3 ms, not the draft's ~150 ms). Two protocol
  kills adopted: the STALL-EQUILIBRIUM gate (terminal-only ±1 with tie=0 at γ=1 makes stalling
  dominate losing for BOTH seats under mirror self-play, and eval/win_rate cannot distinguish
  0.35W/0.60T from 0.35W/0.60L — R0: locked-eval ties ≤4.2%, mean episode length ≤1.5× the 12M
  run's, else the run reports as a degeneracy finding; related: the encoder's `turn/50` clock
  saturates at 50, dead vs heuristics, load-bearing under long self-play games — recorded, not
  fixable, warm start freezes the layout), and NO OVERNIGHT ON UNMEASURED THROUGHPUT (the one
  unmeasured term is the SimpleHeuristicsPlayer.choose_move cost the baseline already pays — ±5%
  band — so a 100k solo probe + 500k 3-wide shakeout, ~25 min, reads the facade trigger and the
  health gates before 5+ hours ride on new code). Verified true under attack, notably:
  `embed_battle` is genuinely egocentric on the seat-2 battle object (fields are per-battle; slot
  orderings match `action_to_order` on that same object) and `PPOAgent` deepcopies clean (no
  env/logger refs; snapshot ≈ 19–24 MB → pool ≈ 400–470 MB, noise). Sourcing corrections adopted:
  the 0.008 seed-std belongs to the 6M mixture n=3 (0.408 is n=1); plan 3-wide at ~653 steps/s
  (campaign-measured), not the probe's 690; pre-review test count was 296 with live, not 294.**
  **LANDED (all committed, 308 tests green INCLUDING both live tests twice on a free :8000 — the
  handoff's flake is cleared): `PoolPlayer` in rl/envs/showdown.py (sync choose_move per
  SingleAgentWrapper's non-awaitable assert; one per sub-env wrapping the ONE shared pool;
  per-battle member draw on battle_tag change with own-attribute tracking — reset_battles() is
  called on the opponent every battle; wait-state assert = the seat-2 twin of the discarded-action
  guard; first-reset seed latched per sub-env so member draws decorrelate — a shared stream would
  collapse pool diversity 8-fold), ShowdownEnv outcome→pool.report wiring (isinstance, never
  getattr: nothing cross-checks PFSP stats), `init_from` config field + train.py load BEFORE the
  step-0 push (ordering test: push-first would anchor the pool at random init and winrate_anchor
  would read ~1.0 forever; refuses lr_anneal_steps — the restored update count clamps lr to ~0
  silently), Showdown fixed_mix>0 hard-reject (measured: HeuristicOpponent crashes on a 611-dim obs
  but RandomOpponent silently plays legal uniform-random moves, unreported),
  `selfplay/winrate_anchor` + `winrate_latest` + `anchor_games` logged from train.py at rollout
  boundaries pre-push, read positionally (stats[0]/stats[-1] — the two indices eviction cannot
  misalign; winrate_anchor = the in-run H&L §V-C detector, under warm start the anchor IS the 0.408
  parent, ~se 0.01 by run end), `--opponent-checkpoint` cross-play on eval_checkpoint.py (seat 1
  deterministic per locked protocol, seat 2 samples per pool contract — run both orientations), and
  four configs (sp6m + cont6m differ in the selfplay block only; sp_probe 100k timing-only;
  sp_shakeout 500k 3-wide with pass gates). 8,192-step live smoke through the full harness: all
  locked metrics + the three new series; mean self-play return +0.014 over 292 episodes (the mirror
  equilibrium, and end-to-end proof of outcome wiring); ties 2.1%; entropy 0.296→0.321 (the 512
  run's band — warm start confirmed from inside; eval rungs 0.3–0.4 confirm it from outside); pool
  1→5 on the smoke cadence; winrate_anchor 0.42–0.47 on 33→104 cumulative games.** *Next, in order
  (probe gates each step)*: (1) solo probe `showdown_sp_probe.yaml` (~2 min; ≥700 steps/s ⇒ facade
  stays parked); (2) 3-wide shakeout `showdown_sp_shakeout.yaml` (~13 min; gates in header — steps/s
  ≥575 each, entropy ≥0.20, ties ≤4%, ep-length ≤~37, pool_size 4, winrate_anchor ~0.35–0.65); (3)
  overnight campaign — sp6m ×3 seeds then cont6m ×3, ~5.1 h, locked 1000-battle finals per seed; (4)
  evening-2 reads R0–R5 incl. the cross-play round robin (SP/CT/parent, both orientations), then
  exactly one of: latest_prob 0.8→0.5 (credited), contamination-free anchor arm (R4 fires), or the
  pre-registered from-scratch 12M narrative arm (flat).

- 2026-08-01 (milestone-3 run 1: self-play NOT credited at matched continuation budget — windowed
  anchor flat at 0.5 all run, cross-play dead even; control arm sets a new 3-seed best 0.432; all
  gates passed, all reads pre-registered) — **Campaign ran exactly as designed (launch 01:03 UTC,
  clean tree `35d4399`, 3-wide waves: sp6m ×3 at ~553 steps/s / 3.0 h, cont6m ×3 at ~600 / 2.8 h;
  solo probe beforehand: 717 steps/s ≥ the 700 architecture gate ⇒ facade stays parked; 3-wide
  shakeout 552–554 vs the derived 575 — inside the pre-stated judgment band, launched; the only
  launch incident was wandb prompting for login in a fresh shell — no credentials exist on this
  machine, all prior campaigns were ambient-offline — fixed with `WANDB_MODE=offline`, noted as a
  to-do to make offline the code default in logging.py). Locked finals (FINAL ckpt_006000000, 1000
  fresh battles each): SP 436/375/414 → pooled 1225/3000 = 0.408 ± 0.018; CT 421/446/428 → pooled
  1295/3000 = 0.432 ± 0.018. The reads, in pre-registered order: R0 GATE PASSED (training ties
  1.2–1.3%, ep len ~26.7 stable, eval ties 1.1–2.4% vs the 4.2% gate — the stall-equilibrium risk
  did not materialize); R1 Δ = −0.023, inside the ±0.025 indistinguishability floor (z ≈ −1.8),
  leaning CT; R2 4–6M bands SP 0.399 vs CT 0.410, same story; R3 CROSS-PLAY (matched seeds, both
  orientations, 6×1000 battles) 3008/6000 = 0.501 — NO strength difference; R4 — the maintainer's
  named primary signal and the campaign's cleanest fact — windowed winrate_anchor (cumulative
  counters differenced per 1M) sits at 0.465–0.551 ≈ 0.5 in EVERY window on EVERY seed (late-window
  n≈400, se 0.025; cumulative n≈10k/seed): the learner NEVER pulled away from its frozen 0.408
  parent and never sank — no improvement, no forgetting, H&L §V-C did not fire; R5 entropy
  0.384–0.391 late (top of the 512 band), no collapse. Verdict per pre-registration: SELF-PLAY NOT
  CREDITED as a training distribution at matched init + 6M continuation budget — and R4 sharpens it
  beyond "not credited": the null is not anchor-blindness (the intransitivity worry), because the
  learner didn't beat its own parent either; warm-started pool self-play at Phase-4 recipe
  (latest_prob 0.8, cadence 150, pool 20) produced zero measurable strength gradient anywhere.
  Meanwhile the control read is a real result twice over: continued fixed-bot training bought +0.024
  over the parent (18M cumulative, curve still creeping, expected-value band 0.42–0.44 hit dead
  center) and 0.432 pooled is the new best-ever headline — the first 3-SEED number in the lineage,
  retiring the n=1 caveat, seed-std 0.010. One asymmetry worth keeping: CT gained +0.024 on its own
  training distribution yet ties SP head-to-head — the anchor gain reads as
  heuristics-specialization, not generalizable strength, which is the R2-bias argument confirmed in
  the other direction. Seed spread note: SP finals spread 0.061 (s1 0.375) vs CT's 0.013 — self-play
  continuation variance is real. Milestone-2 bar: not met by either arm (0.432 best). *Open for the
  run-3 design discussion (the pre-registered tree's "R1 flat and R3 flat" branch fires →
  from-scratch 12M×3 narrative arm, expected 0.20–0.35, budget caveat 6% of H&L's 192M; but the tree
  predates knowing CT +0.024 / SP +0.000, so the alternative on the table is the latest_prob 0.8→0.5
  lever — 80% mirror-vs-near-current games is the plainest mechanism candidate for the zero gradient
  — at half the wall of the narrative arm; hypotheses worth pricing: budget, latest_prob curriculum,
  warm-start local equilibrium)*. Artifacts: final_eval_heur_1000.json ×6, xplay_vs_*.json ×6,
  histories in the wandb offline dirs.
- 2026-08-01 (run-3 design session: three-Opus adversarial pass + in-session re-analysis of run-1's
  own histories — the run-1 framing does not survive; every draft candidate killed except the
  narrative arm; from-scratch 12M×3 chosen at cadence 150, launch gated on the dropped R3 parent
  cells) — **Process: the run-3 draft (candidates: A latest_prob 0.8→0.5 warm-started 6M×3; A′ pool
  pre-seeded from parent-lineage rungs; B from-scratch 12M×3; C = A then B in one overnight; D
  bar-chasers) went through a three-Opus pass (experiment design/statistics, RL mechanism, research
  strategy/ops); every load-bearing number below was re-verified in-session from the extracted wandb
  histories before adoption.** *The re-analysis:* **(1) Run-1's framing overstates both halves.** CT
  never measurably improved in-run either — pooled eval late-3M minus early-3M +0.0034 ± 0.0073
  (z=0.47); SP −0.0119 ± 0.0073 (z=−1.63); CT's +0.024-over-parent is a single-endpoint read at
  z≈1.3; and at the seed-paired level (where the recipe claim lives) the CT−SP finals contrast is
  t(2)=0.92, p=0.45, MDE ≈ 0.14 — the ±0.025 "resolution floor" was a battle-level number licensing
  a recipe-level claim ~5× underpowered. **(2) The fixed-bot curve saturates geometrically**: parent
  per-2M return gains +0.153/+0.103/+0.061/+0.027/+0.016 (ratio ≈0.65), extrapolated asymptote ≈0.42
  vs heuristics (recorded as an extrapolation, not a measurement) — the 0.5 bar is unreachable by
  more fixed-bot budget in this configuration, killing bar-chaser D and re-reading CT's +0.024 as
  consuming most of the lineage's remaining headroom. **(3) The run-1 pool was PROVEN
  strength-homogeneous**: winrate_latest last-half means 0.4993/0.4986/0.5013 per seed; pooled
  last-3M windowed anchor 0.5028 ± 0.0084 — reweighting a draw over equal-strength members is a
  no-op, killing A at any dose by measurement. **(4) Run 1 did not exclude self-play improvement at
  the size the lineage was capable of**: achievable rate ~0.005 win-prob/1M and decaying; the
  per-seed anchor rule needed +0.05 to fire (20–30% power vs its own best case); SP's highest-n
  instrument (pooled training return, n≈98k/window) reads +0.0025 ± 0.0013, z=1.9 — edge of
  resolution, not zero. "Zero gradient" is retired for "below instrument resolution on a
  lineage-wide plateau." **(5) A significant seat effect hides in run-1's own cross-play**: the
  deterministic seat beats the sampling seat by +0.018 ± 0.0065 (z=2.8) at equal parameters;
  both-orientation averaging cancels it (R3's 0.501 stands) but any single-orientation read is
  biased ~2 points — protocol fact, recorded. **(6) R3 was executed incompletely**: the
  pre-registration locked a SP-final/CT-final/parent round robin; only the SP↔CT leg ran (6 xplay
  files on disk, no parent cells). Also noted: eval_checkpoint.py's paired-episode docstring promise
  is false on Showdown (per-battle return correlation ≤0.04 over all 21 run-pairs — the server rolls
  the teams); harmless, but no analysis plan may rely on it. *Candidate verdicts (convergent):* **A
  KILLED** (homogeneity above; its one real effect — tighter anchor precision — comes free by
  pooling seeds, already se 0.0084). **A′ KILLED at this rung** — two verified defects: pool.py
  evicts index 1 on overflow, flushing every pre-seed by ~push 19 of 39 (step ~2.9M of 6M), and
  pre-seeding silently redefines stats[0] = winrate_anchor into a spurious-success detector; deeper:
  one-trajectory rungs are ontogenetic not strategic diversity, and per-minibatch advantage
  normalization gives below-frontier games unit-scale gradient — a noise-floor increase, not just
  waste. If ever revived: pre-seed from the independent-seed mix512 runs, protected-slot retention,
  anchor ordering pinned. **C KILLED** (re-priced ~10.5 h against the 5.9 h measured precedent —
  run-1's true wall by ckpt mtimes, not the logged ~5.1 — and it puts the valuable arm in the
  unmeasured tail). **D KILLED** (finding 2; capacity step 2 stays queued). **B SURVIVES
  RELABELLED**: it cannot discriminate H-budget/H-mixture/H-equilibrium (all predict a rise from
  random init, where the achievable rate is ~9× the plateau's) — it is the narrative/ceiling arm the
  pre-registered tree selected when "R1 flat and R3 flat" fired, and both outcomes carry the
  write-up; the informative-only-on-success caveat from 2026-07-31 stands, accepted. *Decisions
  (maintainer):* **run 3 = B**, from-scratch 12M×3, **cadence 150 not the span-preserving 300**
  (Phase 4's working from-scratch precedent is the nearer regime on both axes — span 24.3% vs its
  19.5%, latest-staleness 154k vs 307k; a deliberate, recorded deviation from the half-run-span
  rule, which was adopted for run 1, which failed); lr_anneal_steps 0 pinned; R0 widened for the
  random-init regime (0–3M recorded not gated); winrate_anchor expected to saturate (strength read =
  eval rungs; winrate_latest is the mirror-equilibrium check). **Launch GATED on two things:** (i)
  closing the R3 parent cells — 12×1000 battles both orientations, decision rule pre-registered:
  both-orientation pooled SP-vs-parent (wins/total, ties count as losses, run-1 R3 convention)
  within 0.5 ± 0.013 (2se at n=6000) ⇒ run-1 null confirmed ⇒ launch; outside ⇒ HOLD, the tree
  re-opens; CT-vs-parent ≈0.5 confirms the +0.024 as specialization/endpoint noise, ≥0.55 would
  revise the re-analysis — and (ii) the scratch 500k 3-wide shakeout (random-init throughput and
  episode profile are unmeasured; the 12M wall estimate is recomputed from its mtimes before any
  overnight). Configs landed: showdown_scratch12m.yaml, showdown_scratch_shakeout.yaml. *Queued from
  the reviews, not committed to:* **P4 — BC-clone SimpleHeuristics through the 611-dim encoder +
  [512,512]** (the encoder-ceiling test, now the live suspect given the 0.42 asymptote; decisive if
  it FAILS, one-directional if it passes — to be pre-registered as such; a diagnostic outside the
  milestone ladder per the Phase-4 contamination framing; ~1 machinery session); P3 team-luck
  variance decomposition (~20 min, prices the SNR story); P5 rollout_steps 512 (the config's only
  true SNR knob — per-minibatch advantage normalization makes an lr test meaningless, explicitly
  rejected) and an entropy_coef 0.003 2M probe with a real prediction split; strategy advice
  recorded as recommendations: a stop rule for milestone 3 (ships after a bounded set — open), the
  0.5 bar stays unmoved with cross-play co-reported and the bar's date attached, replay-BC reframed
  as capstone INITIALIZER rather than diagnostic. HANDOFF.md folded in and restored to stub.
- 2026-08-01 (run-3 gates: R3 parent cells closed — run-1 null confirmed at se 0.0065 and CT's
  +0.024 does not generalize; scratch shakeout all-green; 12M×3 overnight launched) — **The dropped
  R3 leg (12×1000 battles, both orientations): SP-final-vs-parent pooled 3030/6000 = 0.5050 — inside
  the pre-registered [0.487, 0.513] launch band (z=+0.77 at se 0.0065). Six million steps of
  warm-started self-play left the policy within ±1.3 points of its parent: the lineage's tightest
  null, closing the read the 2026-08-01 re-analysis found missing. CT-final-vs-parent 3059/6000 =
  0.5098 — the control's +0.024-on-heuristics does NOT appear head-to-head (nowhere near the 0.55
  revision trigger): heuristics-specialization/endpoint-noise CONFIRMED by direct measurement rather
  than transitive inference.** Both pairs show the deterministic-seat edge in the first orientation
  (+0.02–0.04), consistent with the re-analysis's measured +0.018 seat effect. Operational lesson
  recorded: the handed-over 12-eval && chain (~2.7k chars) mangled on paste — the first 4 evals
  survived, the remaining 8 ran from a bash script in the session tmp dir; long command sets go in
  scripts from now on, never single chains. *Shakeout (500k ×3 concurrent, the first measurement of
  the random-init regime, all gates pre-registered in the config header):* steps/s **575/569/568**
  at 3-wide by mtimes (≥500 gate; faster than warm-SP's 553); entropy declining to 0.49/0.60/0.60
  (in [0.3, 1.8]); training ties 1.1–1.2%; ep len 34.4–36.4 mean, p99 80–86 (longer than the warm
  regime's 26.7, as predicted for random play — the widened R0 gate was right to exist); pool
  reaches 4 on cadence; **winrate_anchor vs the random step-0 init 0.925/0.917/0.922 (n≈4.8–5.2k
  each) — learning from scratch is unambiguous, the sharpest possible contrast with the warm-start
  regime's 0.5028**; eval rungs climb 0.04→0.20 / 0.08→0.25 / 0.10→0.28 inside 500k (rungs noisy at
  se 0.05; the anchor is the signal). Watch item for the morning read, not a blocker: entropy is
  falling fast — R4's <0.15-median trigger may engage mid-run (record-and-plan per pre-registration;
  never mid-run changes). **GO issued on all gates; 12M×3 launched 09:33:50–09:34:02, all three
  stamped `056b78f` / `git_dirty: false`, 3 live processes verified; wall estimate ~5.9 h at the
  slowest measured 568 steps/s.** Next session: locked finals (1000 fresh battles/seed), the R1–R4
  reads, R3 cross-play if R2 lands in band.
- 2026-08-01 (run 3 COMPLETE: from-scratch self-play LEARNS on Showdown — finals 0.380 ± 0.009
  pooled, above the pre-registered band; 0.484 head-to-head vs the equal-budget fixed-bot parent;
  all health gates green) — **The narrative arm ran exactly as designed: 12M ×3 seeds,
  09:34→15:39–15:41 (6h05, the 5.9 h estimate held), ~546 steps/s effective with evals, all stamped
  `056b78f` clean; every pre-registered read taken in order.** R0 PASSED (post-3M ties 1.1–1.2% vs
  the 4.2% gate, ep len 26.7–26.8 vs ≤40 — episode length converged to the warm regime's ~26.7 from
  the random-init ~35 as play got competent; no stall equilibrium). R1 PASSED decisively (windowed
  anchor vs the random init 0.972–0.982 already at 4M, final cumulative 0.949–0.955 at
  n≈10.6–10.9k/seed). R2: per-2M eval means climb monotonically to ~8–10M then flatten — s0
  0.247/0.312/0.343/0.390/0.395/0.365, s1 0.253/0.300/0.355/0.354/0.366/0.364, s2
  0.245/0.338/0.354/0.370/0.383/0.386; **locked finals (final ckpt, 1000 fresh battles/seed):
  0.369/0.398/0.373 → pooled 0.3800 ± 0.0089, seed spread 0.029 — ABOVE the pre-registered 0.20–0.35
  band**, short of parent parity (0.408) and the control (0.432); milestone-2 bar not met. R3
  CROSS-PLAY (both orientations, 1000/pair, matched seeds): **scratch-vs-parent 2902/6000 = 0.4837 ±
  0.0065 (z≈−2.5) — a small but resolvable deficit at EQUAL 12M budget against a policy trained on
  the eval bot's own distribution**; scratch-vs-sp6m 2845/6000 = 0.4742 ± 0.0064, per-seed
  0.434/0.498/0.489 — s0's cross-play weakness matches its last-2M eval dip (0.395→0.365): the
  Phase-4 "best rung ≠ final" late-regression pattern recurs on one seed of three (finals stay on
  the final ckpt per the locked protocol; best_checkpoint feeds no reported number). R4 CLEAN:
  winrate_latest 0.505–0.506 all run (mirror equilibrium as designed); late entropy medians
  0.397–0.427 and the <0.15-for-5-rungs trigger never engaged on any seed — **the Tesauro-dice
  prediction held: server-rolled teams supply the exploration that Connect 4's deterministic board
  did not, and the Phase-4 entropy collapse did not reproduce**. *The milestone-3 three-arm arc is
  now complete and coherent:* (1) warm-started self-play at matched budget moved nothing — 0.5050 vs
  parent at se 0.0065; (2) continued fixed-bot training gained only on its own anchor — +0.024 on
  heuristics, 0.5098 head-to-head: specialization; (3) from-scratch self-play, never having seen the
  eval bot, lands within ~3 points of the equal-budget fixed-bot policy on BOTH measures — 0.380 vs
  0.408 on the anchor, 0.484 head-to-head. **Self-play produces a genuine generalist that approaches
  the lineage's ~0.4 plateau from below; the plateau, not the training distribution, is the binding
  constraint — consistent with the re-analysis's geometric-asymptote read (~0.42) and Phase 4's
  visited-state-distribution finding, and it points every follow-up at the same place: P4, the
  encoder-ceiling BC diagnostic.** Next queue (per the design session, none launched): P4
  encoder-ceiling diagnostic; the milestone-3 write-up + stop-rule decision; P3 (team-luck variance)
  and P5 (rollout_steps 512) as mechanism follow-ups. Artifacts: final_eval_heur_1000.json ×3,
  xplay_vs_parent.json ×3, xplay_vs_scratch12m_s*.json ×3, xplay_vs_sp6m.json /
  xplay_vs_scratch12m.json ×3 pairs, full wandb offline histories.
- 2026-08-01 (P4 machinery: the BC-clone instrument built, smoke-tested end to end, and priced — the
  whole diagnostic is an in-session run, not a terminal one) — **Handoff folded in (nothing durable
  was missing from the log) and the stub restored; the encoder-ceiling diagnostic's two halves
  landed with the design/pre-registration deliberately NOT taken.** *Data path* (`RecordingPlayer`,
  rl/collect.py): a scripted bot plays its own battles over the websocket while the expert —
  resolved through the ONE opponent-spec resolver, so the expert surface is exactly the
  training-opponent surface — supplies orders on our battle object (MixturePlayer's delegation
  pattern); every decision is recorded as (obs, mask, action, battle_id) using SeamPlayer's exact
  encode/mask/convert trio, so a row is bitwise what a learner would have seen. Three guards, each
  load-bearing and each tested: the recorded action is **round-tripped back through
  `action_to_order` and string-compared to the expert's own order** (the move index is relative to
  `active_pokemon.moves`, or `available_moves` for Struggle — a mis-index would teach the clone to
  name a DIFFERENT move under the very conversion deployment uses, and the only symptom would be a
  weak clone, i.e. the diagnostic's own failure verdict); `action >= 0` BEFORE that check, because a
  default order round-trips to itself on its way to indexing the mask from the wrong end; and
  `mask[action]`. Rows carry a battle id because an honest holdout splits on battles, never rows.
  *Collector* (`scripts/make_bc_dataset.py`): asyncio `battle_against`, gitignored data/ .npz,
  prints the recorder's win rate / decisions-per-battle / forced-row share / switch-move split.
  *Trainer* (`scripts/train_bc.py`): masked cross-entropy on the EXACT capstone actor (make_agent +
  the showdown_heur_512 hparams, so the checkpoint re-evals through eval_checkpoint.py unchanged),
  battle-level holdout, agreement reported twice — overall and over multi-choice rows, against a
  uniform-over-legal floor — plus per-epoch metrics JSON; actor-only optimizer (no value labels
  exist, and handing the critic to Adam would imply otherwise). *Verification*: 315 tests green (9
  new: 5 offline recorder tests on a hand-built Gen 1 position with real Pokemon/Move objects, 1
  live-server recorder test, 2 wandb-mode tests, plus the existing suite); full collect -> train ->
  eval_checkpoint smoke run live. **The one failing test
  (test_full_episode_contract_against_live_server, 'Can not reset player's battles while they are
  still running') is PRE-EXISTING — reproduced on the stashed, unmodified tree — and fires only when
  the whole suite runs with a server up; it passes when its file runs alone.** *Operational numbers
  measured, and they change the shape of the run*: collection **2,825 decisions/s** (2,000 battles =
  45k decisions in 16 s), training **~0.7 s/epoch at 40k rows x [512,512]**, and a 1,000-battle
  deterministic re-eval extrapolates to **~50 s** (50 episodes in 2.6 s) — so the entire P4
  diagnostic runs in-session in minutes, and dataset size is not a constraint (1M decisions is ~6
  min of collection). Also closed the standing operational nit: **wandb now defaults to
  `mode=offline`** in `WandbLogger` (an explicit `WANDB_MODE` still wins), so no launch depends on
  remembering the export; verified against the real wandb 0.28.1 signature. *Contamination
  disclosure, stated because pre-registration comes next*: the machinery smoke (2,000
  heuristics-vs-heuristics battles, 3 epochs, [512,512]) reached **0.756 val agreement on
  multi-choice rows against a 0.188 uniform-over-legal floor** and was still climbing — a pathfinder
  observation from a throwaway run, NOT the diagnostic; no win-rate number was produced, and the
  headline read (the clone's win rate vs the bot it copied) is untouched. *Deliberately left to the
  design session*: the pre-registered bands and the one-directional caveat, dataset size and
  epochs/early stopping, WHERE the rows are collected (these are the expert's own visited states —
  Phase 4's 2026-07-29 finding was that in-distribution and off-distribution capability differ
  enormously, so this is a decision, not a default), whether the value head gets labels, and whether
  a mutation battery is warranted. *Next*: design/pre-registration pass, then the milestone-3
  write-up + stop-rule decision.

- 2026-07-21 — Repo scaffolded: structure, README, CLAUDE.md, `.gitignore`, pinned `pyproject.toml`.
  Initial commit.
- 2026-07-22 — Pushed to GitHub. Created `deep-rl` conda env; installed pinned deps and smoke-tested
  CartPole/FrozenLake/LunarLander. Added working-style and dev-env sections to CLAUDE.md. Retired
  the handoff doc into this file. Checked the plan against OpenAI Spinning Up: added the multi-seed
  benchmark protocol and the optional VPG on-ramp.
- 2026-07-22 (later) — Random-policy milestone: logger seam (W&B + TensorBoard), env factory,
  `Agent` interface (+ `state_dict` for checkpointing), `RandomAgent`, fixed-seed eval, checkpoint
  stub, `rl.train` entry point, CartPole sanity test. Verified: `pytest` green; full 5000-step
  `cartpole_random.yaml` run through W&B (offline) hit the ~22 random baseline (eval 24.7 ± 11.1).
  Post-review fix: eval episode seeds are now constants (`EVAL_SEED_OFFSET + episode`), decoupled
  from the training seed so multi-seed benchmark runs share one eval distribution. Next: tabular
  Q-learning on FrozenLake.
- 2026-07-22 (later still) — **Phase 0 complete.** Tabular Q-learning on slippery FrozenLake:
  `QLearningAgent` (ε-greedy, per-step Bellman update, ε annealed over first half of training),
  `frozenlake_q.yaml`, and the train loop's first real `update()` call (per-step, on the fresh
  transition; bootstraps through truncation but not termination). 200k steps → eval 0.67 ± 0.47 vs
  0.02 random baseline on identical eval episodes; checkpoint restore reproduces 0.67 exactly. Also
  pulled `scripts/watch.py` forward from Phase 1 (render_mode passthrough in the env factory +
  checkpoint-driven agent rebuild) — verified live on the FrozenLake policy. Next: Phase 1 DQN
  (replay buffer → `rl/buffers/base.py` gets its first consumer).
- 2026-07-22 (Phase 1 start) — **Linear-Q on-ramp landed**: `rl/buffers/base.py` (thin `Buffer` ABC)
  + `rl/buffers/replay.py` (NumPy ring buffer), `rl/networks/mlp.py` (`hidden_sizes=[]` → single
  linear layer), `rl/agents/dqn.py` (online + target net, ε-greedy, Huber TD loss, agent-owned
  replay; `update()` stores the fresh transition then trains on a sampled batch, so the train loop
  stayed untouched apart from registering `algo: dqn`), `configs/cartpole_linear_q.yaml`, DQN smoke
  test in `test_harness.py`. First live contact with the deadly triad, on schedule: at lr 1e-3 the
  linear net **diverges** — Q predictions climb to ~281 vs the ~100 theoretical ceiling, eval 9.45
  (below the ~22 random baseline). Same skeleton with `hidden_sizes: [64, 64]` at that lr solves
  CartPole (eval 500 ± 0, Q-pred mean ~104 ≈ the true ceiling — skeleton validated, not a bug).
  Committed config uses lr 1e-4, where linear-Q learns modestly (eval 46.5 ± 14.9, single seed,
  pipeline proof not a headline number). Next: promote to DQN proper (`hidden_sizes: [64, 64]`
  config), then Double/Dueling/n-step toggles and the multi-seed CartPole/LunarLander benchmark.
- 2026-07-22 (DQN proper) — **Toggles + sanity benchmark landed.** Double DQN / dueling / n-step as
  config toggles, defaults off (`rl/agents/dqn.py`: `NStepAccumulator` assembling n-step transitions
  with per-transition `gamma^m` discounts stored in the buffer; `rl/networks/mlp.py`: `DuelingMLP`).
  Transition tuple grew to include `truncated` (n-step must flush at every episode boundary, not
  just termination — both agents updated). `--seed`/`--run-name` CLI overrides make multi-seed runs
  a shell loop. `tests/test_dqn.py`: hand-computed n-step boundary cases + all-toggles smoke run (9
  tests green). 3-seed vanilla benchmark, final `eval/return_mean` mean ± std across seeds:
  **CartPole 221 ± 243** (per-seed 500 / 105.5 / 58.2 — every seed reaches ~500 mid-run, then
  oscillates; Q-preds all sit at the ~100 ceiling while realized returns vary, i.e. textbook policy
  churn / overestimation that Double DQN exists to damp), **LunarLander 182 ± 55** (241.8 / 171.4 /
  132.7; seed 0 clears the 200 "solved" bar, all still trending up at 300k steps). Numbers are the
  vanilla baseline for MinAtar-phase ablations, not headlines. Next: MinAtar (install `minatar`),
  toggle ablations, headline vs published baseline.
- 2026-07-22 (best-checkpoint + 500k rerun) — **Best-checkpoint saving landed; baselines confirmed
  sound.** Train loop now snapshots `best_checkpoint.pt` whenever an eval sets a new high-water mark
  (final-only checkpointing was sampling an oscillating policy at an arbitrary phase of its churn);
  benchmark protocol now reports final *and* best. LunarLander budget 300k → 500k. Rerun results
  (mean ± std across 3 seeds): **CartPole final 221 ± 243, best 437 ± 109** (bests: 500 / 500 /
  310.7 — seed 1 held a perfect policy at step 95k and lost it in the last 5k steps); **LunarLander
  final 181 ± 27, best 257.5 ± 15.5** (bests: 243.8 / 274.4 / 254.4 — **every seed clears the 200
  "solved" bar at peak**, spread is tight; finals wobble below via churn). Interpretation: vanilla
  DQN reliably *finds* solving policies on both sanity envs; what it can't do is *hold* them — the
  known stability gap the Double/Dueling/n-step ablations will quantify on MinAtar. Best-checkpoint
  re-evals reproduce their trigger values exactly (eval protocol verified through save/restore).
  Next: MinAtar.
- 2026-07-22 (MinAtar wired) — **MinAtar landed: env + conv path + paper-matched config.**
  Decisions: small CNN matching the paper net (Conv 16@3×3 → FC 128 — removes the architecture
  confound from the headline comparison), `-v0` env ids (full shared 6-action set + sticky actions
  p=0.1 + difficulty ramping = the paper's setup), Adam default with `optimizer: rmsprop` as a wired
  config knob (paper-exact centered RMSprop is a planned ablation, not a memory). Pinned
  `minatar==1.0.15`; its env ids only register via a `gymnasium.envs` entry point that gymnasium 1.0
  removed, so `make_env` registers them explicitly. New: `rl/envs/wrappers.py` (`ChannelFirst`:
  (10,10,C) bool planes → (C,10,10), dtype passthrough), `rl/networks/conv.py` (`ConvQNet`, dueling
  flag; picked automatically by obs rank — no config key), `configs/minatar_breakout_dqn.yaml`
  (paper hyperparams: buffer 100k, batch 32, ε 1→0.1 over 100k, target sync 1000, 5M steps),
  `tests/test_minatar.py` (5 tests). Replay buffer now stores obs in the env dtype (bool = ~200MB
  for 100k Seaquest transitions, vs 800MB in float32). 14 tests green; 15k-step Breakout smoke ran
  clean through offline W&B. Throughput surprise: ~278 steps/s with default torch threading vs
  **~1,550 steps/s with `OMP_NUM_THREADS=1`** (tiny net + thread thrash) — a 5M run is ~55 min
  single-threaded, so the full 5-game × 3-seed sweep is ~14 core-hours and can parallelize across
  cores. Next: full Breakout run vs the paper curve, then scope the ablations.
- 2026-07-22 (torch threads) — **`torch_threads` config field (default 1), set first thing in
  `train()`.** Locks in the 5x+ MinAtar speedup found above without depending on shell state —
  reproducibility lives in the config, not in whoever remembers an export. Overridable for
  capstone-scale nets where intra-op threading may genuinely pay. Verified: config-driven call alone
  (no env var) hits ~1,500 steps/s on the Breakout smoke, matching `OMP_NUM_THREADS=1`; README
  documents the env var anyway as belt-and-suspenders (OpenMP sizes its pool at import time). 14
  tests green.
- 2026-07-23 — **Breakout replicates the paper; ablation campaign scoped and capture infra
  hardened.** First full 5M Breakout run (seed 0, Adam): smoothed training return lands on the
  paper's ~10 plateau (10.2–10.6 over the last 1M steps) with a slower early ramp — ~2M frames to
  reach what the paper hits by 0.5M, plausibly Adam-vs-RMSprop (now the Tier-2 ablation question).
  Eval sits *above* training return (~16 vs ~10): training return pays the ε=0.1 tax, which is
  expensive in Breakout (one random action drops the ball); the paper reports the ε-contaminated
  training return, so the apples-to-apples comparison is `rollout/episode_return`. Churn signature
  from the sanity envs is here too: best eval 25.45 at 1.9M vs final 10.65. Campaign: 30 runs = 5
  games × 3 seeds vanilla (Tier 1, headline vs paper) + Breakout × 3 seeds each of rmsprop / double
  / dueling / nstep3 / all3 (Tiers 2–3); ~4.5 h wall-clock at 8-wide. Capture layer landed first
  (reviewed by a multi-agent adversarial pass pre-launch): self-contained run dirs (`config.yaml`
  snapshot that round-trips through `load_config`, `meta.yaml` with git sha/dirty + package
  versions, W&B offline data colocated under `runs/<name>/wandb/`), atomic checkpoint writes,
  `checkpoint.pt` refreshed at every eval so a dead run still leaves best + latest + full history,
  `eval_returns` protocol split exposing per-episode returns, `scripts/extract_history.py` (offline
  W&B binary → CSV; handles truncated files and the legacy `./wandb` layout),
  `scripts/eval_checkpoint.py` (100-episode re-eval for de-biased headline numbers — best_checkpoint
  is a max over ~50 noisy 20-episode evals, so its recorded score is winner's-cursed; the re-eval
  skips the training-time seed window since deterministic policy + fixed seeds replay those episodes
  exactly, and uses the same fresh window for every run so per-episode paired comparisons stay valid
  — run it on final checkpoints too so best-vs-final gaps compare like with like). 17 tests green (3
  new in `tests/test_run_capture.py`). Review's one confirmed blocker: launch from a *committed*
  tree, else all 30 `meta.yaml`s stamp a sha that lacks the capture layer itself. Follow-up review
  passes (diff + eval semantics) came back clean, with one documented subtlety: MinAtar's
  sticky-action carry (`Environment.last_action`) survives `reset()`, so deterministic-eval episodes
  are weakly order-dependent (~1% of episodes flip, means shift ~0.1%) — left as-is to keep env
  dynamics paper-faithful; per-episode pairing across runs is approximate, protocol re-runs remain
  bit-identical. Next: commit, launch campaign, then curves vs paper + ablation table.
- 2026-07-24 — **30-run campaign complete (all runs 5M steps, zero failures, single sha `aa34b9c`,
  ~5h at 8-wide ≈ 1,000 steps/s/worker).** Tier-1 last-500k training return (mean ± std, 3 seeds) vs
  paper Fig-3 endpoint: Breakout **10.27 ± 0.38** vs ~10 (match), Freeway **51.14 ± 0.11** vs ~50.5
  (match), Asterix **13.25 ± 0.77** vs ~16.5 (below, still climbing at 5M), Space Invaders **32.31 ±
  4.51** vs ~45 (below, plateaued ~1.5M), Seaquest **5.36 ± 2.33** vs ~20 (well below, huge seed
  spread 2.2–7.6). Likely confound for the gaps: paper Fig 3 uses the per-game *optimal* step-size
  from their sensitivity sweep; we ran lr 2.5e-4 everywhere. Breakout rmsprop ablation refutes the
  optimizer hypothesis: centered RMSprop is *slower* than Adam at every mark (5M: 9.09 vs 10.36
  training return) — Adam was not the reason our ramp trailed the paper's. Breakout ablations,
  de-biased 100-episode re-evals (mean ± std across seeds): final — n-step 3 **25.11 ± 2.96** > all3
  23.82 ± 1.59 > vanilla 23.28 ± 0.55 > dueling 21.70 > double 21.24 > rmsprop 21.21; best-vs-final
  churn gap — vanilla +2.45 ± 3.02 vs double **−0.09 ± 1.46** (Double holds its best policy;
  direction matches the overestimation story, but 3-seed stds overlap — suggestive, not conclusive);
  Double also *lowers* training return (8.2 vs 10.3), i.e. stabilizes at a cost on this env.
  Winner's curse quantified: training-time 20-episode best overstates the 100-episode re-eval by
  ~3.5–4.5 points (~15%) on every variant — the de-biased protocol mattered. Figure: 6-panel
  curves-vs-paper + ablation panel (scratchpad; move into repo with the README results section).
  Open follow-ups: per-game lr sensitivity probe for the Seaquest/SI gap; whether to promote n-step
  3 to all 5 games for a vanilla-vs-improved headline pair.
- 2026-07-24 (diagnostics) — **Seaquest/SI gap solved: it's the optimizer, per-game.** 21-run
  diagnostic (Adam lr sweep 1e-4/5e-4, centered-RMSprop probe, +3 vanilla Seaquest seeds): **RMSprop
  at the reference lr reproduces the paper on both gap games** — Seaquest 19.6 (paper ~20), Space
  Invaders 44.7 (paper ~45), including the curve shape Adam couldn't produce (SI climbing through
  4–5M where Adam plateaus at 1.5M). Adam is per-game lr-sensitive here (1e-4 improves both games,
  5e-4 wrecks both); 6 vanilla Seaquest seeds confirm the Adam plateau is real, not a seed draw.
  Breakout remains the counterexample (Adam ≥ RMSprop), so the honest headline is: optimizer choice
  interacts per-game, and the paper's curves are RMSprop curves. Bonus discovery: Seaquest Adam-5e-4
  learns *immortal-but-scoreless* policies (hide + refill oxygen; best eval ~1–6) and MinAtar
  registers no time limit, so two runs spent 5+ h inside a single greedy-eval episode at ~99% CPU —
  the eval protocol now caps episodes at `max_steps=10_000` (~50x a normal episode; binds only on
  effectively-immortal policies; every previously completed number unaffected). Crash-resilient
  capture paid off: killed runs left best/latest checkpoints + full histories. Next: rmsprop on
  Asterix + Freeway (completes the 5-game replication under the paper's optimizer) and the lr5e4
  seed reruns under the capped protocol.
- 2026-07-26 (chunk 5 implementation) — **Continuous PPO landed per the locked spec, no deviations;
  every new code path exercised end-to-end on real MuJoCo. 6 commits, 78 tests green (25 new), tree
  clean.** *Deps:* `gymnasium[...,mujoco]==1.3.0` + `mujoco==3.10.0` pinned exactly (the extra
  requires only `>=2.1.5`; the physics version moves benchmark numbers). Spaces confirmed as the
  review predicted — obs float64, actions float32. *Buffer:* action storage takes the space's shape
  and dtype, mask array allocated only for Discrete. *Normalizers* (`rl/envs/normalize.py`, new):
  `RunningMeanStd` (float64 Chan merge, checkpointable), vector `NormalizeObservation`,
  `FrozenNormalizeObservation` sharing the live object for eval, `NormalizeReward` in SB3's
  accumulate → update → zero-on-any-done order. **Writing the partial-reset test immediately caught
  a real bug the review had not predicted: `SyncVectorEnv.reset` does `options.pop("reset_mask")`,
  mutating the caller's dict**, so reading the mask after delegating always answers None and every
  non-reset row's stale observation would have been folded into the statistics at each episode
  boundary — both wrappers now capture the mask first. *Agent:* Box spaces select `GaussianActor`
  (mlp mean + free `log_std`, init 0) by the same no-config-key rule the obs rank uses;
  `_logp_entropy` is the single fork point so recompute, surrogate and entropy bonus cannot drift;
  `act()` forks before the mask-tensor line and returns (act_dim,)/(N, act_dim) float32;
  `flat_actions` uses `flatten(0, 1)`; new `loss/policy_std`. **The discrete path is proven bitwise
  unchanged** against the pre-change agent for both MLP and conv nets — identical init, identical
  parameters after two full update cycles, identical metrics, identical `act()` RNG stream — so
  chunk 3/4 runs stay reproducible. *Wiring:* train() shares the live stats with a frozen eval
  wrapper and refuses normalize flags on a scalar-path algo; `save_checkpoint` serializes stats at
  save time; watch/record/eval_checkpoint restore through one helper that RAISES on missing stats;
  episode returns accumulate `infos.get("raw_reward", rewards)`; `make_env` applies ClipAction to
  Box envs then restores the honest bounds (ClipAction's ±inf declaration would have broken
  `action_space.sample()` and SAC's Phase-3 squashing, while its clip closes over the inner env's
  real bounds, so the restore is free). *Mutation-checked:* reintroducing `reshape(-1)`, logging
  normalized returns, or normalizing only the reset rows each fails its test. **One test was found
  vacuous by mutation and rewritten** — the terminal-return check asserted quantities the test
  itself computed and passed under the buggy order; it now mirrors the correct recursion across a
  real truncation boundary and compares statistics (~5 units of separation against a 5e-4
  tolerance). *Configs:* `pendulum_ppo` (gate) + `halfcheetah/hopper/walker2d_ppo` (annotated
  reference + two pointers), all four verified to build agents matching each env's action dim and to
  run end-to-end including a re-eval through the frozen-normalizer restore. **Throughput ~5,700
  steps/s single-threaded → 1M ≈ 4-6 min**, so the 9-run campaign is well under half an hour (vs
  MinAtar's 55 min/run). `runs/continuous_campaign.sh` written with the wandb-stagger and dirty-tree
  guards. **Pendulum sanity gate PASSED in-session** (27 s at ~12.5k steps/s, so it stayed a smoke
  rather than a handover): eval climbs −1074 → −109 over 150k, de-biased 100-episode re-evals
  **−136.2 ± 91.7 (final) / −135.8 ± 91.1 (best)** — better than the tuned RL-Zoo PPO reference
  (−172.2) despite running the shared γ=0.99 recipe, so the deliberately conservative gate language
  went unused. Health bands all green on the real run: `policy_std` 0.99 → 0.62 with no early
  collapse, approx_kl ≤ 3.5e-3, clip_frac 0.036 → 0 (the anneal, by construction — judged early per
  the chunk-4 rule), value loss settling ~2e-3 in scaled-reward units as documented, and
  `rollout/episode_return` logged in raw units (−1101 → −120) which is the live proof the raw_reward
  path works outside tests. The re-eval also exercised the checkpoint → statistics → frozen-eval
  seam on a trained policy: wrong or missing stats would have read near the −1200 random baseline
  instead of −136. Next: maintainer runs the HalfCheetah pathfinder, then the 9-run campaign.
- 2026-07-26 (chunk 5 campaign) — **PHASE 2 COMPLETE. 9-run continuous campaign done in 5 minutes;
  the implementation reproduces the reference on Hopper to within 0.2%.** All 9 runs clean on one
  sha (`73acb0e`), `git_dirty: false`, zero wandb stubs (the stagger held). **The framing correction
  that mattered**: the first table I built compared our de-biased greedy re-evals against the
  CleanRL anchors — which is the chunk-4 mistake in mirror image, since those anchors are
  *stochastic training returns*. Recomputed like-for-like (training return vs training return, last
  100k steps, 3 seeds): **Hopper 2380 ± 698 vs 2383 → 1.00×**, Walker2d 3122 ± 284 vs 2288 → 1.36×,
  HalfCheetah 2437 ± 1021 vs 1443 → 1.69×. Hopper landing exactly on its anchor is the correctness
  result — an independent implementation reproduced from scratch. The two overshoots each have a
  candidate explanation this campaign cannot settle, and both are recorded as such: **Walker2d is
  confounded by v5 itself** (right-foot friction 0.9 → 1.9 plus the healthy-reward fix — the env I
  graded "directional only" at spec time, which is exactly where it bit), and **HalfCheetah is the
  pure-truncation env where our GAE's bootstrap-through-truncation should pay** against CleanRL's
  treat-truncation-as-terminal (their #457/#198), confirmed 100% truncation by `episode_length` =
  1000 on every episode of every run, direction predicted by Pardo et al. — but its seed spread
  (1746/3881/1686) is far too wide for a 3-seed claim, so it stays suggestive. **De-biased
  100-episode greedy re-evals** (final / best): HalfCheetah 2730 ± 1241 / 2732 ± 1259, Hopper 2912 ±
  748 / **3360 ± 166**, Walker2d **4122 ± 55** / 4122 ± 55. New finding worth carrying into Phase 3:
  **the deterministic-eval premium is large and env-dependent** (+12% HalfCheetah, +22% Hopper, +32%
  Walker2d) — a Gaussian policy's exploration noise costs real return in proportion to how
  unforgiving the env is, which is why anchor comparisons must use training return on both sides.
  **Hopper is the only env with real churn** (seed 0 finishes 1869 having peaked 3209 — falling is
  terminal and unrecoverable, so a late bad update is expensive); Walker2d's best and final are the
  same policy on all 3 seeds, HalfCheetah's differ by ≤26. Health bands held campaign-wide as on the
  pathfinder; `policy_std` never collapsed early on any run. Figure `assets/mujoco_ppo_campaign.png`
  plots training return only, with per-seed traces drawn rather than a smooth band because the
  spread IS the finding, and the anchor as a neutral rule — one measure per axis, deliberately not
  mixing the two protocols. README Phase 2 continuous section written; **every numeric claim in it
  was verified programmatically against the run artifacts before commit, which caught one wrong
  claim** ("best-vs-final gap of exactly zero on all six runs" — true for Walker2d, ≤26 for
  HalfCheetah). Phase 2 status flipped to done, both tracks. Next: Phase 3 SAC, which benchmarks
  against these exact runs on the same three envs with the env stack held fixed — and whose first
  prerequisite is the Polyak soft-update helper in `rl/common/` (still absent).
- 2026-07-26 (chunk 5 pathfinder) — **HalfCheetah pathfinder green on every gate; cleared to
  scale.** Single seed, 1M steps, sha `87450c9`, clean tree, ~9,900 steps/s (≈4 min wall including
  20 eval passes — 14x faster than a MinAtar run). Eval climbs monotonically −19 → **1864.9 ± 17.4**
  and is still rising at 1M; de-biased 100-episode re-eval **1866.8 ± 21.0** (final == best, the
  anneal-off-style low churn seen on the discrete track). **That is ABOVE the CleanRL
  identical-recipe anchor of 1442 ± 46**, which needs saying carefully rather than celebrating: at
  least part of the gap is protocol — our headline is a deterministic mean-action eval while CleanRL
  reports stochastic training return, and our own training return (1747) sits below our eval (1865)
  by about the expected amount. A second, more interesting hypothesis is testable and pre-registered
  here: **HalfCheetah never terminates, so every one of its episode boundaries is a truncation**
  (confirmed — `rollout/episode_length` is 1000 for the entire run), and our GAE bootstraps through
  truncation where CleanRL's treats it as terminal (their open issues #457/#198). This is precisely
  the env where chunk 2's per-row `next_obs` design should pay, and Pardo et al. 2018 predict the
  direction. Not a claim until the 3-seed campaign lands, and it cannot be separated from the
  protocol difference without re-running their code — documented as a finding candidate, not a
  result. *Health bands:* clip_frac 0.155 @50k → 0.205 @250k → 0.264 @500k → 0 @1M — in the 0.1–0.3
  band across the peak-lr window and driven to zero late by the anneal exactly as the chunk-4 rule
  anticipates ✓; `policy_std` declines smoothly 0.89 → 0.45 → 0.24 → 0.16 with **no early collapse**
  (the failure signature this metric exists for) ✓; value loss 0.037 → 0.0014, stable and in the
  ~1e-3 scaled-reward range the spec predicted ✓; `loss/entropy` goes negative (7.8 → −2.7), which
  is correct rather than alarming — a Gaussian's differential entropy is negative below σ ≈ 0.242,
  and −2.68 matches the analytic value for σ=0.16 over 6 dims to within the within-update drift.
  **One band drifted high**: approx_kl peaks at ~0.0196 around 500k against the pre-registered
  ≤1.5e-2. Read as benign and left alone — it is the repo's documented "safety diagnostic, not
  sufficiency" metric, it co-moves with clip_frac (the clip is firing, which is the mechanism doing
  its job), CleanRL runs MuJoCo with `target_kl=None` as we do, and returns improve monotonically
  straight through that window. Noted so the campaign can watch it rather than rediscover it. Next:
  the 9-run campaign via `runs/continuous_campaign.sh`.
- 2026-07-26 (chunk 5 spec + review) — **Chunk 5 (continuous PPO on MuJoCo) spec drafted,
  three-lens-reviewed and LOCKED in one pass.** The four pre-drafted forks all resolved to the
  canonical side and all survived review: unbounded Normal + ClipAction with unclipped log-probs,
  state-independent free log_std init 0, running obs normalization as a checkpointed vector-level
  wrapper, running discounted-return reward scaling on the training env only. Review
  (published-evidence / reference-implementation / adversarial-with-probes, run as three parallel
  agents): **8 keep / 8 amend / 0 reject**. The catches that paid: **(1) act_dim=1 broadcast trap**
  (adversarial, probe-confirmed) — a continuous update() that keeps the discrete `reshape(-1)` hands
  Normal a (B,) actions tensor against a (B,1) mean, producing a (B,B) log-prob matrix whose
  `.sum(-1)` is plausibly-shaped garbage with first-ratio exactly 1; it *trains*, passes a Pendulum
  smoke (act_dim=1), and detonates only at HalfCheetah — hence flatten(0,1) plus hand-checks at
  act_dim ≥ 2 with shape asserts are spec items; **(2) reward-normalizer order-of-operations**
  (reference lens) — the draft's `G ← γ(1−done)G + r` drops each episode's largest-magnitude return
  sample from the variance estimate; SB3's accumulate → update → zero-on-any-done order adopted;
  **(3) partial-reset corruption** (adversarial, probed) — the vector loop replaces obs wholesale
  and non-reset rows return raw, so the obs wrapper must normalize ALL rows while updating stats
  only for reset rows; **(4) gymnasium's own vector normalization wrappers are structurally unusable
  here** (source-confirmed asserts on NEXT_STEP autoreset and all-True reset masks), making the
  custom wrappers forced rather than stylistic; **(5) evidence corrections** (published lens) —
  Andrychowicz C63 actually favors tanh over clip (+30% HalfCheetah), so D1's rationale now rests on
  reference fidelity + saving SAC's tanh machinery for Phase 3, with a stated HalfCheetah handicap;
  and the evidence-favored init std is 0.5, making `log_std_init −0.7` the pre-registered first
  probe-tier lever. Anchors secured apples-to-apples: CleanRL's identical-recipe v4 numbers at 1M
  (HalfCheetah 1442 ± 46, Hopper 2383 ± 272, Walker2d 2288 ± 572) with per-env v4→v5 transfer grades
  (clean / near-clean / directional); "~1500 on HalfCheetah is on-anchor, not a bug" and "SAC wins
  big on HalfCheetah in Phase 3" are both pre-registered against future bug hunts. Checkpoint-stats
  consistency was formally discharged (evaluate → save runs with zero env steps between, so
  best_checkpoint's normalizer stats are bitwise the selection-time stats). Full amended spec in the
  Phase 2 section. mujoco resolves to 3.10.0 against gymnasium 1.3.0's `>=2.1.5` extra; not yet
  installed. Next: maintainer lock + go-ahead → pyproject pin + install → implement per spec with
  per-step commits → Pendulum sanity → HalfCheetah pathfinder → 9-run campaign.
- 2026-07-26 (DQN Breakout lr probe) — **Fairness control closed: the Breakout tie survives a
  matched tuning budget.** PPO's Tier-1 lr came from a five-point sweep at 5M while DQN's Breakout
  number had never been lr-swept (Phase 1 swept lr only on Seaquest/SpaceInvaders/Asterix, the games
  where its curves trailed), so Breakout — the one game the de-biased re-evals called a tie — had
  the least defensible asymmetry. 6 runs, 3 seeds each at lr 5e-4 and 1e-3, seeds 0/1/2 paired
  against the existing vanilla runs, configs differing from vanilla in `lr` alone. **DQN does not
  improve at higher lr; it degrades**: final 100-episode re-eval 23.28 ± 0.55 (lr 2.5e-4) → 21.06 ±
  2.52 (5e-4) → **15.09 ± 2.60** (1e-3); n-step 3 at the default lr remains DQN's best at 25.11 ±
  2.96. So both algorithms are now swept on Breakout and both land at ~25 (PPO 25.91 ± 2.63),
  intervals overlapping heavily — a genuine tie, not an artifact of who got tuned. **The asymmetry
  is the finding**: lr 1e-3 is exactly what took PPO from 5.5 to 25.9, and the same value costs DQN
  a third of its score. Mechanism is gradient work per sample — DQN takes ~1024 gradient steps per
  1024 transitions where PPO takes 16, so at any shared nominal lr DQN sits near its stability
  ceiling while PPO is starved; identical config numbers, ~64× different effective learning rates.
  This retires "we gave both algorithms the same learning rate" as a fairness argument — it would
  have been actively misleading. Remaining caveat, documented rather than spent on: Freeway's DQN
  number is still un-swept (PPO leads 61.3 vs 59.3 there), though on both games where DQN has now
  been swept, higher lr only hurt it. README Phase 2 section updated; its last TODO is closed.
- 2026-07-26 (Phase 3 spec + review) — **Phase 3 (SAC) spec drafted, three-lens-reviewed and LOCKED.
  19 keep / 24 amend / 2 reject** — both rejects were errors in the draft's *justifications*, not
  its design choices, so every decision survived. Phase 2's 11 commits pushed first. **Three forks
  the maintainer settled up front**: SAC runs RAW (no normalizers) with a PPO-raw control campaign
  to close the confound by measurement; Tier-1 nets 256×256 ReLU plus a 64×64-Tanh ablation at **3
  seeds** on HalfCheetah (PPO's architecture whole, not a width×activation 2×2, with the one-env
  limitation stated); and SAC validated against published SAC benchmarks *before* the PPO comparison
  is treated as meaningful. A throughput probe run before drafting made the architecture fork
  affordable either way (256×256 ≈ 425 steps/s ⇒ 39 min/1M; `torch_threads=1` still wins — 4 threads
  *drops* it to 327). **The catch that paid for the review: the draft's update pseudocode dropped a
  `squeeze`.** `Q` emits (B,1) against (B,) rewards, so the TD target broadcasts to (B,B),
  `F.mse_loss` accepts it with only a UserWarning, and every critic output is regressed to the batch
  mean (gradient correlation −0.82 with the correct one) — while the *actor* loss is accidentally
  exact under the same broadcast (mean of an outer difference = difference of means), so the policy
  still moves and nothing looks wrong. Proven end-to-end on Pendulum: **−97.6 correct vs −1720.2
  as-written**, and a B=1 unit test passes it. This is chunk 5's act_dim=1 trap one layer down.
  Runners-up: **the α optimizer's lr was missing entirely** and the natural guess is wrong (CleanRL
  runs it at `q_lr` 1e-3, not `policy_lr` 3e-4 — 3.3× off on the one loop that sets the
  entropy/reward exchange rate); **the proposed log-prob oracle couldn't catch the bug it existed to
  catch** (`TanhTransform()` alone omits the affine scale — off by exactly `A·log 2` — and NaNs in
  float32 at the |u|=12 case the tests specify); **CleanRL's published docs table is stale v2 while
  its live benchmark is v4**, +15% on HalfCheetah, so grading the pathfinder against 9,635 would
  have passed a SAC 15% under recipe; and a degeneracy audit found that `action_bias = 0` and
  `terminated ≡ False` on **both** the Pendulum gate and the HalfCheetah pathfinder, so the
  terminal-bootstrap path is dead code on both gates — mitigated with a synthetic-terminal test plus
  a short Hopper smoke before the campaign. Two comparability claims were **rejected outright** and
  must not reach the README: `loss/policy_std` is not PPO-comparable (state-dependent pre-squash
  latent, bounded to (0.0067, 7.39) by construction, vs PPO's global unbounded raw-action scale) and
  `loss/q_pred_mean` shares DQN's name but no env is run by both agents. Also corrected from the
  draft: SB3 does *not* include the `−Σ log s` scale term (CleanRL is the only reference that does),
  the stable log-prob's real justification is the epsilon killing the correction's **gradient** in
  saturation rather than float32 precision, `Box.sample()` without `make_env`'s bounds-restore
  returns N(0,1) not ±inf, the campaign is ~79 min at 12-wide not 35–50, and the
  return-vs-wall-clock figure is a standard openrlbenchmark output rather than novel (the *paired
  same-machine* PPO-vs-SAC comparison is the novel part). One draft concern was **dropped** as
  refuted: `_scalar_loop`'s per-episode averaging distorts α by ≤0.33% over 1M steps. Full locked
  spec in the Phase 3 section. Next: implement per spec with per-step commits (maintainer authorized
  up front), Pendulum gate in-session, then the pathfinder.
- 2026-07-26 (Phase 3 implementation) — **SAC landed per the locked spec, no deviations; 5 commits,
  117 tests green (39 new), tree clean. Both cheap gates passed in-session.** *Polyak*
  (`rl/common/polyak.py`): params-only soft update, shared rather than agent-internal so a
  soft-update DQN stays a cheap ablation; 5 mutations, all caught. *Buffer*: `ReplayBuffer`
  generalized to Box actions mirroring chunk 5's `RolloutBuffer` change, mask arrays allocated only
  for Discrete, `obs_dtype` now the agent's choice (DQN keeps the env's for MinAtar's bool planes;
  SAC narrows MuJoCo float64 → float32, the same principle inverted). **The discrete path is proven
  bitwise unchanged** — old and new buffers driven through 400 DQN update steps on vanilla,
  double+dueling+n-step-3, and conv/bool obs: losses identical, max |dw| = 0.000e+00.
  **`tests/test_replay.py` is new: this buffer had no direct tests at all**, only DQN's indirect
  ones, which cannot see a Box action truncated to int64; 7 mutations, all caught. *Agent*
  (`rl/agents/sac.py`): CleanRL's recipe with the spec's two divergences (stable tanh correction,
  fresh α in the target); every critic output squeezed. **Mutation testing earned its keep again —
  19 mutations, 4 survived, 3 of which were real test gaps.** The hand-computed TD-target test could
  NOT see a dropped squeeze, because with constant target critics every row of the (B,B) broadcast
  is identical and its mean equals the correct one — fixed by making the target critics vary across
  the batch, with the test now asserting that precondition explicitly. Nothing checked that the
  entropy bonus reaches the target (that test runs at α=1e-12 by construction). And the
  deterministic-act test was **circular** — it compared `act()` against `deterministic_action()`,
  the method under test, so dropping the action scale or bias changed both sides and passed; it now
  rebuilds the expected action from the raw head output. The fourth survivor is not a bug:
  `alpha_optimizer.zero_grad()` clears the actor's contribution before the α backward, so detaching
  α in the actor loss is intent rather than load-bearing. *Configs*: the PPO-raw controls are
  GENERATED from the shipped PPO files so only the two flags and `run_name` can drift
  (diff-verified) — and generating the ablation arm the same way silently failed, because the
  `^`-anchored regex missed the keys indented under `agent:`, leaving `halfcheetah_sac_mlp64.yaml` a
  pure rename; caught by diffing every generated config against its source, and the arm is now
  verifiably 64×64 Tanh at 17,614 params against the reference's 217,870. **Pendulum gate PASSED**
  (100 s, sha 281096b, clean tree): eval −1332 @5k → **−109 @10k** and flat thereafter — our PPO
  needed 150k steps to reach −136 on this env, so SAC scores better in ~1/15th the samples, which is
  the phase's thesis visible on the cheapest env we own. α decays 0.998 → 0.007; `loss/entropy`
  converges to −1.107 against a target entropy of exactly −1. **Hopper smoke added** (47 s) to cover
  what neither gate can: `terminated` is never True on Pendulum or HalfCheetah, and it logged 417
  episodes of length 9–147 against a 1000-step limit, i.e. real terminations — with entropy
  converging to −3.03 against `H_target = −3`, the temperature loop landing on target at a second
  action dimension. **Two readings to carry to the pathfinder:** `time/steps_per_sec` must be judged
  over the POST-WARM-UP window (the first 5000 steps take no gradient step and run ~120× faster — a
  whole-run mean reads 6030 against a true 460, the same shape of rule as chunk 4's peak-lr
  clip_frac window); and `loss/mu_absmax`, added to settle whether a converged policy really
  saturates, **reaches 7.7 on Pendulum and 5.6 on Hopper** — past the |u| ≈ 7.6 where the
  epsilon-floored tanh correction starts distorting, so the stable form was not defensive
  programming. `runs/sac_campaign.sh` written (gitignored, like chunk 5's — a committed launcher
  would dirty the tree every run): 9 PPO-raw controls first at ~5 min to derisk the launcher, then
  12 SAC runs at ~79 min. Next: maintainer runs the HalfCheetah pathfinder, then the campaign.
- 2026-07-26 (Phase 3 campaign) — **PHASE 3 COMPLETE. 21 runs in 82 minutes; SAC validated against
  published benchmarks, then benchmarked against our PPO on two axes.** All 21 runs on one sha
  (`c15b931`), `git_dirty: false`, zero wandb stubs. **M3 first, as required: the implementation is
  checked before the comparison is allowed to mean anything.** Protocol-matched against published
  SAC — greedy re-evals to the deterministic anchors, training return to CleanRL's — Hopper (105% of
  the paper, 108% CleanRL) and Walker2d (101% paper, 102% CleanRL) land at or above published;
  HalfCheetah sits 84–102%, which is inside the *inter-source* disagreement, since the published
  HalfCheetah numbers span 9535–12139 (27%) among themselves. **Headline, both ways:** SAC beats PPO
  on all three envs against BOTH the matched-stack control and PPO's normalized best — HalfCheetah
  6.5×/3.7×, Walker2d 3.3×/1.5×, Hopper 1.4×/1.1× (a tie inside ±522). Because it holds under
  PPO-normalized, the pre-registered qualification was not needed. **The finding that justifies the
  second axis: SAC needs 1.3× (HalfCheetah), 4.6× (Walker2d), 6.5× (Hopper) of PPO's ENTIRE wall
  clock merely to match PPO's final score**, then keeps climbing — ~24× the compute per 1M steps (72
  vs 3 min under campaign concurrency; corroborated by 442 vs ~9,900 steps/s solo). Per sample SAC
  dominates; per minute PPO does, and which matters depends on whether samples or compute is scarce
  — decisive for the capstone, where a sample is a websocket round-trip. **Architecture ablation
  (M2):** SAC with PPO's nets whole (64×64 Tanh) keeps **78%** of the reference score and still
  beats PPO 5.1× on matched envs — architecture is a real but minority contributor, not the
  explanation. Stated limitation: one env. **The PPO-raw control paid for itself twice**: it priced
  the env stack (PPO loses 20–55% raw) and answered a question it was not built for — normalization
  buys PPO *stability*, with Walker2d's seed spread going ±284 → ±563 and one seed reaching 620
  against another's 1907. **Two findings the literature does not publish numbers for:** SAC's
  deterministic-eval premium is much SMALLER than PPO's (+7.3% vs +12.0% HalfCheetah, +5.2% vs
  +32.1% Walker2d) — the inverse of the naive guess, and mechanistically right, since entropy is in
  SAC's objective so it is optimized to perform while noisy; and **Hopper churn is an ENV property,
  not an algorithm one** — SAC churns there too and harder (all 3 seeds finish below peak, against
  PPO's 1 of 3), which Phase 2's evidence alone could not have separated. **Verification caught a
  real error before commit, again**: the new README section used sample std (`statistics.stdev`)
  where every existing table uses population std (`np.std`), so all nine ± values were wrong and
  PPO's would have silently contradicted the Phase 2 section directly above it. Figure
  `assets/mujoco_sac_vs_ppo.png` plots the same curves twice — per step and per wall-clock minute,
  small multiples with shared y per column, never a dual axis — using categorical slot 2 for SAC
  while both PPO arms keep slot 1 (same algorithm, two env stacks; color follows the entity, dash
  carries the variant). Palette validated: worst CVD ΔE 24.7. Next: Phase 4, the Connect 4 self-play
  on-ramp.
- 2026-07-26 (Phase 4 spec + review) — **Phase 4 (Connect 4 self-play on-ramp) spec drafted,
  three-lens-reviewed and LOCKED. 6 reject / 5 implementation blockers / ~30 amend.** The phase is
  structurally different from 1–3: **there is no published anchor** (no PPO-self-play-on-Connect-4
  result exists, and AlphaZero is forbidden by the hard constraint), so the review is the only
  external correctness check and correctness has to come from *exact oracles* instead of curves —
  which promoted `open_spiel` (dev-only, `pyspiel.load_game` only, carve-out added to CLAUDE.md) and
  a negamax solver from nice-to-have to the phase's spine. That **overrides PLAN.md's own "no solver
  build required"**: Pons' labels give the value of a *position*, and a move is optimal only if its
  child's value is −v(p), so blunder rate is not computable from labels alone. Published-evidence
  lens re-scoped by the maintainer (no recipe to find) onto general self-play practice + the
  competitive-Pokémon literature; adversarial lens required to return **executable probes, not
  prose**. Four adversarial claims were re-derived independently before acceptance — two came out
  worse than reported, one better. **The catch that paid for the review: `eval/win_rate` defined as
  `return > 0` reads the very reward a sign bug inverts.** Measured on real training runs: a flipped
  loss reward reports **1.000** and passes the gate at its ceiling, while the "near 0%" reading the
  spec named as the signature appears on a policy that genuinely wins **81.3%** — so the metric now
  comes from an env-supplied `info["outcome"]`. Runners-up, each with a probe: **the chunk-1 gate
  caught 0 of 4 seeded defects** (`RandomOpponent.move` ignores `obs`, so a wrong *opponent*
  perspective is a provable no-op — 50 games bit-identical — and swapped-learner-planes and
  dropped-epoch-mask both scored *higher* than clean); **regression rate as the forgetting headline
  reads 47.8% on a run that never learns** against 13.9% for genuine forgetting and 2.1% for a fast
  learner, so AlphaStar's published min-win-rate-vs-all-past-versions proxy becomes primary; **the
  value-head MAE metric ranks a constant-zero critic (0.218) above a perfect one (0.643)**; **a
  transposition table without bound flags corrupts 0.13%→0.40% of results with the error rate
  *rising* with depth**, which Pons validation structurally cannot see (labels are root values,
  blunder rate needs child values); and five blockers — `selfplay: dict = {}` is a `ValueError` that
  kills every import in the repo, an unseeded pool `IndexError`s on `_vector_loop`'s very first
  `envs.reset()`, `normalize_obs` un-freezes the pool by the back door (vector wrapper above an
  in-env opponent), `rl/train.py:137` builds its eval env with a bare `make_env`, and the terminal
  all-False mask violates `masking.py`'s ≥1-legal invariant. **Published evidence changed two design
  decisions and retired one inherited claim**: Bansal et al.'s δ-uniform ablation measures a plain
  recency deque (δ≈0.8) as second-worst of four, so pool retention becomes **strided** (evict the
  second-oldest); Czarnecki et al. measured Connect Four's game geometry and place it in the
  spinning-top class whose **cyclic dimension is widest at intermediate skill** — exactly where a
  "mediocre agent" sits — retiring the unsourced "Connect 4 is near-transitive"; and Generals.io
  (arXiv:2606.23348) turns out to be a near-exact protocol match (PPO, two-player self-play, sparse
  terminal ±1, **γ=1.0, λ=0.9, clip 0.2**, superhuman), independently confirming γ=1.0. Measured,
  not projected: a solver reproducing Pons' node counts to **0.3%** with 1000/1000 correct answers
  shows the spec'd algorithm covers 3000 of 6000 positions and **chapter 8's iterative deepening is
  mandatory** (Begin-Easy 660× → ~12 min), and that **NumPy is the *slowest* board representation**
  (351k nodes/s vs a Python-int bitboard's 894k). Also measured: 10.6 learner steps/episode, **31%
  of episode log records silently merge** (vs MinAtar's 2.8e-05), draws are 0.27% of random games,
  and the tournament — not training — is the >5-minute job. **Two forks left open on purpose, to be
  settled by a 4-minute probe arm rather than argument**: `gae_lambda` 0.95 vs 1.0 (the lenses
  agreed on the arithmetic and disagreed on the conclusion), and `ConvQNet`'s 3×3 kernel against a
  game whose winning pattern is a line of four. **Phase 5 corrected too**: `Gen*EnvSinglePlayer` was
  removed from poke-env in 0.8.4 so the plan named a dead API, poke-env puts the action mask in the
  *observation* rather than `info` (one adapter wrapper), Huang & Lee renormalize after the softmax
  rather than masking logits, their dense-shaping "requirement" is an unablated inference that
  Generals.io contradicts, their own §V-C reports **77/500 (15.4%)** catastrophic forgetting, and
  Metamon (RLJ 2025) is a stronger, more recent Gen 1 anchor that also independently validates our
  masking contract. Next: chunk 1 — board + learner-centric env + the open_spiel oracle, gated on
  the per-site fixture probes rather than on a training run.
- 2026-07-25 (chunk 4 campaign) — **The locked config was mis-tuned, not miswritten; lr was the
  whole story. 30-run campaign done, PPO beats DQN on 3 of 5 games.** The pathfinder ran clean (8.9
  min, all four health bands green, clip_frac 0.087 inside the recalibrated band) and still landed
  at **5.47** last-500k training return vs DQN's 10.27 — triggering the spec's own sanity rule ("PPO
  well below DQN on Breakout is a bug signal"). **The bug hunt came back clean four ways**: CartPole
  PPO on the post-chunk-4 tree reproduced chunk 3's first eight evals *identically* (158 136 261 241
  500 448 344 278, diverging later only through float noise from the reshaped recompute);
  critic-vs-discounted-return-to-go correlation **1.000**; the actor was state-responsive with the
  correct directional sign (corr with ball offset +0.124 vs DQN's +0.113); dead-unit rates matched
  the demonstrably-working critic. So: tuning, not code. One hypothesis was raised and killed by its
  own evidence — MinAtar Breakout's 6-action set has **four behaviourally identical "stay" actions**
  (`minimal_action_set` is `[0,1,3]`; `u`/`d`/`f` fall through `act()`), which explained the entropy
  plateau at 1.26 and why greedy eval ≈ sampled return, but the policy sat at 86.2% stay mass,
  *above* uniform's 66.7%, so the entropy bonus was pulling toward moving, not away; masking to
  `{n,l,r}` reproduced the identical rollout. **1.5M single-seed probes isolated lr as the only
  lever**: 2.5e-4+anneal 4.97 · 5e-4+anneal 5.51 · 5e-4 **8.42** · 7.5e-4 **12.15** · 1e-3
  **12.65**; annealing cost 35% at equal base lr, **clip 0.2 cost 39%**, γ 0.999 and 64 envs were
  both neutral. Web research corroborated externally: gymnax's own Breakout lr sweep is **flat at ~7
  for a full 10M at lr 1e-4**, and our anneal made the time-averaged lr **1.25e-4** — squarely in
  their documented dead zone. Two framing corrections came out of that search: our clip 0.1 departs
  from a **5-of-5 MinAtar reference consensus on 0.2** (kept anyway, on our own evidence — those
  references run the easier variant), and **the external anchors are not apples-to-apples at all**:
  gymnax and pgx MinAtar are `use_minimal_action_set=True` (Breakout **3 actions**), have **no
  sticky actions**, and cap episodes; the spec's "gymnax reports Breakout 28" is also questionable
  (their shipped lr-5e-4 config reads ~22 on the published figure). Our DQN-vs-PPO comparison is
  untouched by this — both ran the identical v0 env and trunk. Single-seed 5M runs then showed the
  1.5M ordering **did not survive** (5e-4 19.60 · 7.5e-4 16.15 · 1e-3 24.29), i.e. seed noise was
  too large to pick on one run, so **both 5e-4 and 1e-3 went to a full 3-seed campaign**: 30 runs,
  55 min, single sha `8c52c9c`, clean tree. Free determinism check: the two runs that collided with
  earlier probes of identical config+seed reproduced them to the exact logged-row count (42232,
  35952). **De-biased 100-episode greedy re-evals of the final checkpoint (mean ± std over seeds
  0/100/200) — the cross-algorithm headline, since it taxes both algorithms identically**: Breakout
  PPO **25.91 ± 2.63** vs DQN 23.28 (n-step-3 25.11) — *a tie*; Freeway **61.34 ± 0.50** vs 55.78
  (rmsprop 59.25); Asterix **31.51 ± 1.19** vs 23.85 (best 25.64); Seaquest **24.48 ± 19.36** vs
  7.11 (rmsprop **28.70**) — a coin flip, per-seed 15.5/6.6/51.4; Space Invaders **276.93 ± 37.93**
  vs 64.37 (rmsprop 90.65) — **3× DQN's best**. So PPO wins decisively on Space Invaders and
  Asterix, modestly on Freeway, ties Breakout, and is unreliable on Seaquest. **The training-return
  comparison badly overstated this** (it read as PPO 2× on Breakout) exactly as the spec predicted —
  recorded here because it was briefly believed mid-session before the re-evals corrected it. Bonus
  finding: with the anneal **off**, PPO's final ≈ best on re-eval (Breakout final 25.91 > best
  24.89, the winner's curse on a 20-episode selection), so PPO's low churn is now a genuine
  stability signal rather than the anneal artifact the spec warned about. **Open fairness caveat**:
  PPO got a 5-point lr sweep at 5M; DQN never got an lr sweep on Breakout or Freeway at all (only
  optimizer + algorithmic ablations). Matters most on Breakout, which is a tie; the Asterix and
  Space Invaders gaps are too large to be tuning artifacts. **Spec deviations, both
  evidence-backed**: Tier-1 lr 2.5e-4 → **1e-3**, and the lr anneal — an explicit
  Andrychowicz-backed spec item — **turned off**.
- 2026-07-25 (chunk 4 implementation) — **PPO on MinAtar landed per the locked spec, no deviations;
  every health band already reads green on a 200k probe.** Three commits. *Agent*
  (`rl/agents/ppo.py`, the only source file touched — `conv.py` reused untouched as specified):
  rank-3 obs select `ConvQNet` for both heads (DQN's no-config-key rule, DQN's trunk, dueling off);
  `act()` branches on rank then **batches the single path** (crash #1 — chunk 3's "no unconditional
  unsqueeze" was an MLP-ism, restated as "branch on rank, batch the single path"); the update-start
  recompute runs on `(T·N)`-flattened tensors and reshapes `values`/`next_values`/`old_logp` back to
  `(T, N)` for GAE (crash #2 — the buffer's `(T, N, C, H, W)` is rank 5 and conv2d rejects it);
  `_orthogonal_init` iterates `net.modules()` and now covers `Conv2d` at gain √2 per CleanRL
  `layer_init`; `lr_anneal_steps` kwarg, default 0 = off, linear to ~0, keyed off the checkpointed
  update counter. The MLP path's init is **bitwise unchanged** — verified param-by-param against the
  pre-change implementation (same layers, same order, same RNG draws), so chunk 3's CartPole run
  stays reproducible. *Tests*: anneal endpoints hand-checked against the real 128×8 / 5M schedule at
  updates 0 / 2441 / 4881 (2.5e-4 → 1.250208e-4 → 9.28e-8, positive at the last update), default-off
  and past-the-end clamp, `act()` on a single unbatched rank-3 obs in both policy modes
  (mask-respecting in each) and on batched rank-4, plus a MinAtar PPO train-loop smoke sized to
  force **one full rollout fill and one eval pass** — the two crash paths are invisible to anything
  shorter. Each of the three fixes was **mutation-checked**: reverting it makes the corresponding
  test fail with exactly the error the spec review probed. 53 tests green (5 new). *Configs*: five
  `minatar_<game>_ppo.yaml`, `num_envs` top-level, Breakout annotated as the reference and the other
  four carrying the sibling pointer; all five load and build, per-game channel counts adapting
  automatically (4 / 6 / 7 / 10). **In-session 200k Breakout probe** (seed 0, no eval, tensorboard):
  **~9,250 steps/s single-threaded → 5M ≈ 9 min of collection+training vs DQN's ~55 min**,
  confirming the spec's throughput projection with room to spare (PPO takes 16 grad steps per 1024
  transitions where DQN takes ~1024). Training return climbs 1.16 → 3.17 → 3.94 → 4.34 across the
  four 50k windows. All four health bands green this early: **clip_frac 0.132–0.148** — inside the
  recalibrated 0.05–0.2 band for clip 0.1, which closes the chunk-3 CartPole `clip_frac ≈ 0`
  observation as "easy env + big batch" exactly as the spec's decision rule anticipated, so **no
  probe tier is indicated**; approx_kl 2.8e-3 (band ≤ 1e-2); entropy 1.791 (= ln 6, the near-uniform
  init) gliding to 1.310 with no collapse; value loss stable. Caveat: 200k of 5M at one seed — the
  band is formally judged over the pathfinder's first ~1M. Next: the Breakout pathfinder run in the
  maintainer's terminal, then the 15-run campaign, then the README Phase 2 results section.
- 2026-07-25 (action masking) — **Masking contract landed harness-wide, provably a no-op on every
  spine env** (5 commits, per the capstone handoff doc; scope held: no poke-env, no battle logic, no
  benchmark re-runs). Contract: envs emit `info["action_mask"]` (bool [A], True = legal;
  `ActionMask` wrapper injects all-True for Discrete-action envs, innermost so observation wrappers
  keep their attrs; Box envs never see a mask), `Agent.act(obs, action_mask=None,
  deterministic=False)`, update batches grew to `(..., mask, next_mask)`, and every algorithm masks
  through `rl/common/masking` — finite `-1e8` sentinel (never `-inf`: `Categorical.entropy()` at
  `-inf` is `0·-inf = NaN`), `masked_logits` bitwise-identity under all-True, where-guarded
  `masked_entropy`, and `masked_sample` by **rejection** so the all-True RNG stream is identical to
  a bare `space.sample()`. Per algorithm: random/tabular sample legal-only; tabular's **bootstrap
  max is masked too** (the handoff's table said acting-argmax only, but its own §5 argument applies
  identically — extension noted); DQN masks ε-sampling, greedy argmax, and both target paths
  (vanilla max + Double's online argmax), with the replay schema now `(s, a, R, s', term, disc,
  mask, next_mask)` and the n-step accumulator stamping every emission with the **bootstrap
  state's** next_mask (episode-end partials share the terminal state's mask — correctly shaped even
  when terminated makes it unused); PPO/REINFORCE store per-row masks and reapply them at the
  recompute **and on every epoch's forward** (a collection-only mask would silently corrupt every
  importance ratio; the critic is never masked). Vector loop: probed on gymnasium 1.3.0 — step infos
  aggregate to `(N, A)`; partial-reset infos cover only reset rows (**non-reset rows are all-False
  placeholders**), so masks merge via `np.where(done[:, None], …)`; the truncation case is correct
  **by construction** because our autoreset is DISABLED (step info always describes the true final
  state — the returned-obs-belongs-to-the-new-episode bug is a NEXT_STEP-autoreset artifact we don't
  have; scalar envs never autoreset). Verified two ways: the bitwise `masked_logits` identity test,
  plus an **empirical before/after check** — a fixed-seed 20k-step Breakout DQN run at the
  pre-change and post-change trees produced identical metric histories, all 1,645 rows cell-for-cell
  (rejection sampling is what makes the ε stream survive). New `tests/envs/masked_dummy.py` (random
  legal subsets, raises on illegal, variable episode lengths so vectorized sub-envs desynchronize)
  drives end-to-end runs for random (scalar), PPO (vector + partial-reset merge + masked eval), and
  DQN (ε/greedy/n-step target); hand tests pin the DQN target with the illegal action holding the
  highest Q (1-step and n-step paths separately) and PPO's first-epoch ratio ≡ 1 under active masks.
  48 tests green (14 new). Chunk-4 lock-rule deviation entry: the locked spec's act()/update()
  sketches now compose with this contract (act gains `action_mask`; the conv-obs unsqueeze fix and
  the mask branch combine; the rollout buffer's add() carries masks). Next: implement chunk 4 (PPO
  on MinAtar) per the locked spec, on top of the masking contract.
- 2026-07-25 (capstone decision) — **Capstone decided: Pokémon Showdown Gen 1 singles (battle phase
  only) via poke-env + a local Node.js Showdown server, starting format `gen1randombattle`; hero
  algorithm PPO + self-play.** Milestone ladder (each independently shippable): beat
  MaxBasePowerPlayer → beat SimpleHeuristicsPlayer → self-play with a historical-checkpoint opponent
  pool → optional live-ladder Elo. Headline: win rate vs SimpleHeuristicsPlayer over ≥1000 battles,
  multi-seed (Elo is a flourish, not the metric). Documented fallback if self-play stalls: Procgen
  generalization study (train/test level gap). CLAUDE.md rule flipped from "undecided — no capstone
  scaffolding" to "decided — capstone-specific code deferred until Phase 3 completes" (no poke-env,
  no battle logic, no Pokémon encoders during Phases 2–3; env-agnostic harness contracts the
  capstone needs may land earlier). Immediate consequence, landing next: the action-masking harness
  contract — Showdown's legal actions change every turn (fainted Pokémon, PP depletion, forced
  switches) — all-True default, provable no-op on the spine envs.
- 2026-07-25 (chunk 4 review) — **Chunk 4 spec locked after its own three-Opus-lens review** (same
  lenses as chunk 3: published-evidence / reference-implementation / adversarial; 10 keep / 3 amend
  / 0 reject). The review paid for itself in implementation blockers: **two probe-confirmed conv
  crashes caught at spec time** — (1) the draft's act() branch-only rank generalization forwards a
  single (C,10,10) obs into ConvQNet with no batch dim (Flatten eats the channel dim → mat1/mat2
  RuntimeError), killing the first eval at 100k and every watch/record/eval_checkpoint rebuild —
  fixed by unsqueeze-then-index on the single path (DQN's own pattern; "no unconditional unsqueeze"
  was an MLP-ism, restated as "branch on rank, batch the single path"); (2) the update() recompute
  forwards the (T, N, C, 10, 10) rank-5 buffer tensor into conv2d, dying on the first rollout fill —
  recompute now runs flattened and reshapes back to (T, N) for GAE. Also load-bearing:
  `_orthogonal_init`'s `for m in net` raises on non-Sequential ConvQNet → `net.modules()` (and
  dueling must stay off or the head-gain rule mis-inits `value` — probed); `num_envs` must stay a
  top-level config field or it collides with make_agent's injected kwarg; the MinAtar smoke must
  force a full fill + an eval or both crashes ship invisibly. Headline restructured: **greedy
  100-episode re-evals become the only cross-algorithm number** — DQN's training return pays a
  constant ε=0.1 tax, PPO's a shrinking sampling tax, so training-return gaps are
  exploration-mechanism artifacts (curves stay as per-algorithm context; PPO's best-vs-final gap is
  also anneal-compressed — not a stability signal). clip_frac rule recalibrated: band ~0.05–0.2
  under clip 0.1, judged over the peak-lr first ~1M (the anneal drives it ≈0 late by construction);
  probe direction fixed **up** — lr 5e-4 first, then γ 0.999 (every MinAtar-JAX ref runs one or
  both; they also use clip 0.2, 64 envs, flatten-MLPs — a second coherent recipe, divergences now
  stated per-knob instead of elided). New external anchors: gymnax-reported PPO at 10M — Breakout
  28, Freeway 58, Asterix 15, SpaceInvaders 131; Seaquest unpublished (gymnax doesn't register it).
  New review-found facts: Freeway is a fixed 2501-step timer → 8 lockstep sub-envs share one W&B log
  step forever (documented; headline unaffected); Seaquest immortal-sub-env risk gets campaign
  monitoring (training cap rejected — reward-stream parity). Framing corrections: separate nets
  *departs from* the shared-trunk conv lineage (support: PureJaxRL's separate stacks + chunk-3
  continuity), value-clip omission rests on Engstrom/Andrychowicz, not the separate-nets logic. Good
  news probed too: SyncVectorEnv + reset_mask works on MinAtar sub-envs, seed+i fan-out confirmed
  against 1.3.0 source, all divisibility arithmetic exact, per-game channel counts auto-adapt. Next
  session: implement chunk 4 per the locked spec.
- 2026-07-25 (chunk 4 spec draft) — **Chunk 3 committed (6c0bc6b); chunk 4 (PPO on MinAtar) spec
  drafted — DRAFT pending the review gate.** Full draft in the Phase 2 section. Positions taken:
  separate ReLU `ConvQNet` actor/critic reusing the DQN trunk arch, shared trunk explicitly rejected
  (would refactor the just-reviewed agent and reopen the value_coef/value-clip caveats); lr anneal
  as agent hparam `lr_anneal_steps` (0 = off); clip_eps 0.1 per the pixel-PPO lineage, CartPole
  keeps 0.2; benchmark seeds 0/100/200 so 8-wide sub-env seed windows don't overlap under
  gymnasium's seed+i fan-out; harness numbers identical to the DQN campaign; the chunk-3 clip_frac
  observation gets its diagnostic test at MinAtar difficulty. Numbers crib CleanRL/ppo2 Atari
  pending the published-evidence lens (PureJaxRL etc.). Next: three-lens review → lock → implement →
  Breakout pathfinder run → 15-run campaign.
- 2026-07-25 (PPO chunk 3) — **PPOAgent landed per the locked spec, no deviations; PPO solves
  CartPole through the vectorized path.** `rl/agents/ppo.py`: separate Tanh actor/critic via a new
  `activation` kwarg on `mlp()` (default ReLU — DQN/REINFORCE untouched), orthogonal init (√2 hidden
  / 0.01 policy head / 1.0 value head, zero biases), one Adam eps=1e-5 over the param union,
  recompute-at-update-start (values, next_values, old_logp in one no-grad pass), GAE → 4 epochs × 4
  transition-level minibatches with per-minibatch advantage norm, single grad clip over the union;
  the ratio/clip math is a module-level `clipped_surrogate_loss` returning the approx_kl/clip_frac
  diagnostics. `make_agent` grew the vectorized-constructor contract (single spaces + `num_envs`,
  getattr fallbacks so scalar rebuilds in watch/record/eval_checkpoint just work — exercised live by
  the re-eval below); `_VecRandomAgent` moved to it. `tests/test_ppo.py`: hand-computed surrogate
  cases (upper/lower clip, pessimistic-min asymmetry, mean reduction, kl estimator),
  fill-train-clear cadence with the near-uniform-init entropy check, vector train-loop smoke. 34
  tests green. Single-seed 150k `cartpole_ppo.yaml` run (~10 s wall, ~15k steps/s): eval first hits
  500 ± 0 at 25k (update 24), churns mid-run (dip to 189 at 45k), holds 473–500 from 95k on, final
  eval 492.9 ± 21.6; de-biased 100-episode re-eval of the final checkpoint **475.4 ± 48.5**
  (REINFORCE final: 499.98 — on CartPole the machinery buys nothing over REINFORCE; the case for PPO
  starts at MinAtar and the continuous track). Health bands: approx_kl max 1.8e-3 over all 146
  updates (band ≤ 1e-2 ✓), entropy glides 0.693 → 0.554 with no collapse ✓, **clip_frac 0.000–0.002
  — below the 0.1–0.3 healthy band**. The spec's own discriminator reads ≈0 as "lr too timid", not a
  bug signature, and the evidence agrees it's conservatism, not breakage: the clip does fire
  occasionally (metric isn't dead), ratios stay so close to 1 that reuse is trivially safe, and the
  policy solves the env in 24 updates. Left as-is — config is locked, CartPole is solved, and the
  lr/anneal question already belongs to the MinAtar configs. Overlay figure
  `assets/cartpole_ppo_vs_reinforce.png` (eval return vs step, both 150k runs; dataviz-skill
  palette; README embed deferred). Next: decide chunk 4 — MinAtar PPO (conv actor-critic, lr anneal
  per spec) or start the continuous track (`gymnasium[mujoco]`).
- 2026-07-25 (visuals) — **`scripts/record.py` landed: greedy rollouts → annotated GIFs.**
  Checkpoint-driven like watch.py (agent + env rebuilt from the checkpoint, deterministic policy,
  `render_mode="rgb_array"`); MinAtar's 10×10 float frames nearest-upscale to ~320px with an
  episode/step/return HUD bar underneath the playfield; per-frame GIF durations hold each episode's
  last frame ~1.2s so the final return is readable; `--max-steps 2000` recorder cap (the
  immortal-policy lesson applied to recording); `--seed` makes a clip reproducible. Pinned
  `pillow==12.3.0` (already transitive via matplotlib, now a declared direct import). Showcase clips
  recorded at seed 0: Seaquest rmsprop_s0 best 32/73/36 (the 961-step middle episode is the full
  oxygen loop — shoot fish, rescue divers, surface to trade a diver for oxygen), Breakout nstep3_s0
  best 32/53/35, LunarLander s0 best 254/273/266 (all three land; 30 fps). Seaquest clip promoted to
  `assets/minatar_seaquest_dqn_rollout.gif` and embedded in README Results; record.py documented in
  README Run. 18 tests green. Next: Phase 2 PPO — decide the REINFORCE/VPG on-ramp at phase start.
- 2026-07-25 (PPO design review) — **Chunk 3 spec locked after a three-Opus-subagent review**
  (lenses: published ablation evidence / reference implementations + their issue trackers /
  adversarial break-it pass). 7/8 non-numeric decisions were unanimous keeps. Strongest validations:
  the truncation-aware GAE + autoreset-DISABLED design matches SB3's time-limit bootstrapping and
  dodges CleanRL's two still-open truncation bugs (cleanrl#457, #198) plus the Farama
  vector-autoreset footgun; the recompute-at-update-start design (no stored values/log-probs) was
  verified exactly correct — no param-mutation path exists mid-rollout, and float noise between the
  collection and recompute passes is ~1e-6 against a 0.2 clip. Three amendments adopted: **Tanh**
  hidden activations for PPO's MLPs via an `activation` kwarg on `mlp()` (every feedforward PPO
  reference uses Tanh; our numeric defaults are cribbed from CleanRL's Tanh-validated configs),
  **Adam eps=1e-5** (canonical PPO detail, flagged independently by two reviewers), **lr 2.5e-4 not
  1e-3** for CartPole (1e-3 is 4× the reference default and with 4 reuse epochs feeds the
  solved-then-collapsed pattern; annealing stays deferred to MinAtar). Traps folded into the spec:
  advantage-norm eps, float32 cast on both obs and next_obs, single grad-clip over the param union,
  act() branching on obs rank, approx_kl/clip_frac logging. Full spec in the Phase 2 section. Next
  session: implement chunk 3 per spec.
- 2026-07-25 (PPO chunk 2) — **Rollout buffer + GAE landed, unit-tested against hand-computed
  cases.** `rl/buffers/rollout.py`: `RolloutBuffer` — the second buffer pattern from base.py finally
  has its consumer path (fixed (T, N) horizon, filled/drained/cleared; obs keep the env dtype like
  replay). Design choice: every row stores its own next_obs (a second obs array, trivial at rollout
  scale) so rows are self-contained — GAE never reaches across rows for successors, needs no special
  bootstrap at the buffer end, and a truncated row's next_obs is the true final observation, so
  time-limit truncations bootstrap correctly (the REINFORCE wart, fixed where it matters; note most
  reference PPO implementations get truncation wrong). `compute_gae`: delta_t = r +
  gamma·V(s')·(1−terminated) − V(s); A_t = delta + gamma·lam·(1−done)·A_{t+1} — lam is the explicit
  bias-variance dial between the phases' two poles (lam=0 pure TD/DQN regime, lam=1
  Monte-Carlo-minus-baseline/REINFORCE regime); termination masks the bootstrap, any done cuts the
  chain, matching the repo-wide truncation convention. `tests/test_rollout.py`: 6 tests, every GAE
  expectation worked by hand (lam=0 ≡ TD errors, gamma=lam=1 ≡ MC−baseline via telescoping,
  termination masks a poison next-value, truncation keeps its bootstrap, env columns independent,
  buffer fill/overfill/clear mechanics). Scope note: the planned "actor-critic net" file evaporated
  — for CartPole it's two `mlp()` calls inside the PPO agent (chunk 3); orthogonal init lands there
  too, as a PPO implementation detail (37-details paper). 26 tests green. Next: chunk 3 — PPOAgent
  (clipped surrogate, epochs/minibatches, entropy bonus), cartpole_ppo config, overlay vs the
  REINFORCE run.
- 2026-07-25 (PPO chunk 1) — **Vectorized collection seam landed; scalar path untouched.**
  `make_vec_env` (SyncVectorEnv — sync not async, the torch_threads lesson applied to processes;
  sub-env i seeded seed + i), `num_envs` config field (default 1), `Agent.vectorized` class flag
  (collection mode is a property of the algorithm, not a config value; PPO will set it), `ALGOS`
  registry in train.py (make_agent construction + path choice share one source of truth), train loop
  split into `_scalar_loop` (existing body moved verbatim) and `_vector_loop` (batched act/update,
  per-env episode accumulators, update metrics logged when reported rather than per-episode). Key
  decision, probed empirically on gymnasium 1.3.0 before coding: the default NEXT_STEP autoreset
  inserts a dummy transition after every terminal step (action ignored, reward 0, and no info flag
  to detect it by) which every downstream consumer would have to mask; instead autoreset is DISABLED
  and the loop resets finished sub-envs via `reset(options={"reset_mask": ...})` — every transition
  handed to update() is a real env step, and next_obs at a terminal row is the episode's true final
  observation (exactly what GAE bootstrapping needs at truncations). Vector smoke test drives the
  path with a test-local vectorized RandomAgent subclass that documents the act() contract (batched
  during collection, single-obs when eval calls). Throughput probe (env-only, CartPole): scalar 323k
  steps/s vs vec-8 282k — SyncVectorEnv's Python fan-out costs ~13% on a free env; the win arrives
  when a policy forward dominates the step. 20 tests green. Next: chunk 2 — rollout buffer + GAE +
  actor-critic net.
- 2026-07-25 (Phase 2 start) — **REINFORCE on-ramp landed: the policy-gradient core works end-to-end
  on the unchanged harness.** `rl/agents/reinforce.py` (categorical policy over `mlp()` logits;
  per-episode update on discounted reward-to-go, normalized within the episode as the baseline
  stand-in; log-probs recomputed in one batched forward at episode end — exact, the policy is frozen
  mid-episode; truncation treated as an episode end, the documented Monte-Carlo bias PPO's critic
  later removes), `configs/cartpole_reinforce.yaml`, REINFORCE smoke in `test_harness.py`, `algo:
  reinforce` in `make_agent` (watch/record/eval_checkpoint inherit it). Scoped out on purpose: no
  rollout buffer (episode storage is three lists; the real buffer lands with PPO's vectorized GAE
  needs), no batching knob, no learned V(s) baseline. Train-loop fix PPO also needs: `loss/*`
  metrics now average over the steps that reported them, not `ep_length` (DQN unchanged — it reports
  every step; REINFORCE's once-per-episode loss was being divided by episode length). Single-seed
  150k-step run (~12 s wall at ~12k steps/s): eval 500 ± 0 from 55k, one mid-run churn window
  (85k–100k, dips to ~320), then 500 ± 0 held for the final 45k; de-biased 100-episode re-eval of
  the *final* checkpoint 499.98; entropy glides 0.69 → 0.51 (max ln 2 ≈ 0.693) — no premature
  collapse. 19 tests green. Next: PPO proper — vectorized envs (the `make_env` seam becomes real),
  GAE + critic, clipped surrogate, rollout buffer.
- 2026-07-25 — **Phase 1 benchmark complete: 5/5 games replicate the paper.** Asterix probe closed
  the last gap: centered RMSprop at lr 1e-4 → **16.81 ± 1.18** (paper ~16.5), still climbing at 5M,
  mid-run decay gone (every seed's final bucket is its highest); Adam lr 1e-4 improved but fell
  short (13.63 ± 0.83). Full replication table + figure (`assets/minatar_dqn_campaign.png`) +
  findings now in README Results. Campaign total: 63 completed 5M-step runs (~60 core-hours). Ops
  note: launching 6 simultaneous wandb-offline inits hit a ~15s service-startup timeout and each
  process retried, leaving an abandoned ~45KB stub `offline-run-*` dir beside the real one in all 6
  run dirs — `extract_history.py`'s exactly-one guard caught it; stubs deleted before the post-pass.
  Remaining Phase-1 nicety (not a blocker): `--record`/`scripts/record.py` annotated-GIF rollouts
  for the README. Next: Phase 2 PPO — decide the REINFORCE/VPG on-ramp at phase start; vectorized
  envs make the `make_env` seam real.
