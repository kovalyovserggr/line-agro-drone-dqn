<!--
Article 3 — draft, section 2 (Materials and Methods). Working language: English.
Status: DRAFT v0.1, 2026-10-21. Written before results are frozen (WORKFLOW_CHECKLIST E): this section
describes the model and protocol only; numbers that come from runs are marked with their run_id and are
re-checked by script before submission.
Markers: [REF: …] — citation to add (primary source, DOI); [CHECK: …] — fact to verify against code/logs;
[AUTHOR: …] — decision or wording for Serhii. Decision IDs (Р…) refer to docs/article3/A3_DECISIONS.md and are
removed in the final text.
-->

# 2. Materials and Methods

## 2.1. Production line model

The study uses the discrete-event simulation model of an agrodrone production line introduced in Part 1 of
this series [REF: Article 1]. Version 0.3 of the model, developed for this study, adds equipment ageing,
imperfect maintenance and age-dependent quality and speed losses (Section 2.2). The Part 1 code is left
unchanged, so Part 1 results remain reproducible. The v0.3 classes are built on top of it.

The line produces one product, an agricultural drone, from 23 item types arranged in a bill of materials
(BOM). In the base configuration it has 30 processing nodes in six equipment classes (Table 1):

- five filament extruders (class C1), which recycle polymer into filament;
- fifteen FDM printers (C2), which produce the 15 printed part types;
- five industrial assembly robots (C3);
- three manual workstations (C4): wiring harness, sub-assembly and final assembly;
- a software installation bench (C5);
- a test-and-calibration stand (C6), whose output is a finished drone.

Each node executes the operations assigned to it. An operation consumes its BOM inputs and occupies the node
for a nominal duration. A short changeover (180 s for printers, 120 s for other nodes) is added whenever the
node switches between item types. When a node becomes free, the D17 dispatching rule of Part 1 selects its
next operation.

Material flow between nodes is governed by kanban limits (Р22): an intermediate item may be started only
while its stock plus work in progress is below a buffer cap. For discrete items the cap equals two drones'
worth of the item. For filament it is the BOM filament requirement of two drones, 76 740 g (Р38). Finished
drones are not capped.

At the nominal durations, the BOM implies 62.7 h of printing plus 5.8 h of changeovers per drone, and 19.2 h of
extrusion plus 1.9 h of changeovers. Without any downtime this caps throughput at 36.8 drones/week for the
printers (15 nodes) and 39.8 drones/week for the extruders. Printing is therefore the bottleneck in the base
configuration.

To study how the bottleneck position affects maintenance decisions, the number of printers is varied over
{12, 15, 17, 19} (Р38). With 17 or more printers the printing limit (41.7 drones/week at 17) exceeds the
extrusion limit, and extrusion becomes the bottleneck. This shift was confirmed by the time decomposition of
nodes in simulation (capacity_diag, a3-dt-printers-L). Extra printers are copies of a base printer with their
own random streams (Section 2.2.4), so all other nodes keep identical random histories across configurations.

Maintenance resources are limited. A single technician performs both preventive maintenance (PM) and
corrective repairs (crew = 1, Р30). A failed node waits for the technician if they are busy. Preventive
maintenance is only started when the technician is free (Section 2.4.2).

Time advances in steps of Δt = 100 s. The time step is carry-over-exact (Р39): when an operation, changeover,
PM or repair ends inside a step, the remainder of the step is passed on to the node's next state, and resources
released in a step can be allocated within the same step. As a result, outputs do not depend on the step size:
the throughput spread across Δt ∈ {25, 50, 100, 200} s was 0.06% (a3-dt-printers-L) [CHECK: confirm 0.06% and
the run id against RESULTS_LOG].

## 2.2. Reliability and degradation model

### 2.2.1. Failure modes and imperfect maintenance

Each node is modelled as a series system of independent failure modes, or components. Each mode c has its own
virtual age v_c, which grows only while the node is processing, and a Weibull time-to-failure distribution
with shape β_c and scale η_c. The hazard rate is

  h_c(v) = (β_c/η_c) · (v/η_c)^(β_c − 1).                                                    (1)

The node fails as soon as any one of its modes fails.

Interventions follow the Kijima type II model [REF: Kijima 1989, J. Appl. Probab.]:

- after preventive maintenance, v_c ← q_pm,c · v_c;
- after a repair of the failed mode, v_c ← q_cm,c · v_c.

In this study, wear modes are restored "as good as new" (q_pm = q_cm = 0). Electronic modes (β = 1) are not
affected by PM (q_pm = 1).

After each intervention, the next failure age is sampled conditionally on the age already reached, by
inverting the cumulative hazard:

  v_f = η · [ (v/η)^β − ln U ]^(1/β),   U ~ U(0, 1).                                          (2)

The model was verified against renewal theory on single nodes (tests (a)–(d) in tests/test_reliability_v3.py):

- the simulated mean time between failures matches η·Γ(1 + 1/β);
- under age-replacement policies, simulated availability matches the analytical
  A(T) = ∫₀ᵀR / (∫₀ᵀR + d_pm·R(T) + d_cm·F(T)) to within 0.004 across the T grid;
- the simulated optimum agrees with the analytical T*;
- for β = 1, preventive maintenance never improves availability.

At line level, the simulated MTBF of each class agrees with the series-system value
1/Σ_c [η_c Γ(1 + 1/β_c)]⁻¹ to within 4–11% (test L4).

### 2.2.2. Age-dependent quality and speed losses

Wear affects production through two efficiency channels as well as through failures. Let r = max_c v_c/η_c be
the normalised age of the node's wear modes, taken over the modes that PM restores. Then:

- **Scrap.** The probability that a finished item is scrapped is p_def(r) = p₀ + (p_max − p₀)·(1 − e^(−r)).
- **Speed.** Processing slows by the factor s(r) = 1 + κ·min(r, 1).

A scrapped item is withdrawn from stock together with its embodied inputs, and the kanban logic automatically
releases replacement work.

An operation interrupted by a failure is also lost:

- discrete inputs are scrapped;
- bulk inputs (filament, raw polymer) are lost in proportion to the progress made (Р10).

Because these channels depend on r rather than on β, maintenance can pay off through quality and speed even when
failures are memoryless. Test L6 on the line confirms this: with β = 1 but active degradation, periodic PM
reduces availability but raises throughput.

### 2.2.3. Intervention durations

PM durations follow a normal distribution truncated at 0.2 of the mean, with coefficient of variation (CV)
0.1 (Р8). Repair durations follow a lognormal distribution with CV 0.4, the usual model for maintenance task
times [REF: MIL-HDBK-338B, primary source to be verified]. The class means are listed in Table 1.

### 2.2.4. Random streams

All randomness comes from independent generators, one for each pair of node and process (failure, quality,
PM duration, repair duration). The generators are spawned from one master seed with NumPy's
SeedSequence (Р6). Different policies evaluated on the same seed therefore see the same failure-age draws,
scrap draws and duration draws. These common random numbers (CRN) make paired comparisons between policies
far more precise than independent replications.

## 2.3. Parameterisation and scenarios

Table 1 lists the class parameters.

The scale and shape ranges follow published component reliability data:

- bearings, belts, DC/AC motors, gears and mechanical seals have β of 1.2–2 and η of 10⁴–10⁵ h for wear
  mechanisms [REF: Bloch & Geitner, Practical Machinery Management for Process Plants, Vol. 2, 1994 — via
  Barringer Weibull database; verify against the book];
- solid-state electronics have β ≤ 1.1 [REF: same; MIL-HDBK-217F].

The robot class C3 has an MTBF of 13 086 h. This lies between the lowest field MTBF reported for painting-line
robots (5 280 h) and manufacturers' recommendations of 60 000–80 000 h [REF: Abd Majid & Fudzin 2017, ARPN JEAS
12(1)].

For FDM printers (C2), no peer-reviewed field reliability data were found. The quality parameters rest on
open-workshop studies:

- failed prints accounted for 19% of material use [REF: Song & Telenko 2016, SFF Symp.];
- 14.7% of all prints failed because of the machine, mainly through nozzle clogging
  [REF: Song & Telenko 2019, Procedia CIRP 80:494–499 — verify DOI].

These values bound p_max of class C2 from above (Table 1). The wear scale of the printer (η = 1 500 h) and the
tooling parameters of the manual workstations (C4) are assumptions. Their influence is quantified in a
sensitivity study (Section 2.7.4).

**Table 1.** Equipment classes (base, "moderate" degradation scenario). Times in hours of operation; MTBF is the
series-system value of all modes; T* is the analytical availability-optimal age-replacement interval of the
wear mode (Section 2.6).

| Class | Nodes | Wear mode β / η | Random mode β / η | MTBF | PM mean (CV) | Repair mean (CV) | p₀ → p_max | κ | T* |
|---|---|---|---|---|---|---|---|---|---|
| C1 extrusion | 5 | 1.8 / 4 000 | 1.0 / 30 000 | 3 180 | 1.0 (0.1) | 6.0 (0.4) | 0.005 → 0.05 | 0.10 | 1 899 |
| C2 FDM printer | 15 (12–19) | 1.5 / 1 500 † | 1.0 / 20 000 | 1 268 | 0.5 (0.1) | 4.0 (0.4) | 0.01 → 0.10 | 0.20 | 677 |
| C3 assembly robot | 5 | 2.0 / 20 000 | 1.0 / 50 000 | 13 086 | 2.0 (0.1) | 8.0 (0.4) | 0.002 → 0.02 | 0.10 | 11 878 |
| C4 manual workstation | 3 | 1.2 / 5 000 † | — | 4 703 | 0.25 (0.1) | 1.0 (0.4) | 0.005 ‡ | 0 | 11 362 |
| C5 software bench | 1 | — | 1.0 / 20 000 | 20 000 | 0.5 (0.1) | 2.0 (0.4) | 0.002 | 0 | — |
| C6 test stand | 1 | 1.2 / 15 000 | — | 14 110 | 0.5 (0.1) | 3.0 (0.4) | 0.001 | 0 | 18 621 |

† Assumption without a field source; varied in the sensitivity study (Section 2.7.4).
‡ The wiring-harness workstation (node 22) has p_max = 0.03, because its quality degrades with tool wear (Р14).
The other two manual stations have no quality degradation.

Two scenario axes are studied:

- **Degradation strength** (Р28). The quality range p_max − p₀ and the slowdown κ of every class are multiplied
  by 0.5 ("weak"), 1 ("moderate", base) or 2 ("strong"). The strong scenario puts p_max of the printers at 0.19,
  the upper bound reported by Song and Telenko [REF].
- **Bottleneck position.** The number of printers is varied over {12, 15, 17, 19} (Section 2.1).

The main experiment covers six scenarios: moderate degradation with 12, 15, 17 and 19 printers, plus weak and
strong degradation with 15 printers.

**Economics** (Р32). Line performance is measured by annual profit:

  Profit = n_drones · V_d − Σ PM cost − Σ repair cost − c_f · (filament lost to scrap and interruptions),   (3)

where:

- V_d = 4 000 USD is the value of a good drone;
- the PM cost of a node is the maintenance cost field of the Part 1 node parameters (400–700 USD);
- a repair costs k_r = 5 times the PM cost of the same node;
- c_f = 0.02 USD/g is the filament cost.

Only cost items that depend on the maintenance policy are included. [AUTHOR: confirm the wording "V_d is the
contribution per good drone net of nominal material and labour"; k_r and c_f are assumptions — consider
listing them with ranges.]

## 2.4. Maintenance decisions as a semi-Markov decision process

### 2.4.1. Decision epochs and actions

Maintenance is decided node by node, at events. A node reaches a decision epoch:

- **E1:** when the dispatching rule has chosen its next operation, but before resources are requested;
- **E2:** when it starts waiting for missing inputs or is blocked by a full downstream buffer;
- **E3:** every Δ_w = 1 h while that wait continues.

The action set is {CONTINUE, PM}:

- **CONTINUE:** at E1, request inputs and process; at E2/E3, keep waiting.
- **PM:** start maintenance. At E2/E3 the pending resource request is withdrawn first.

PM is masked out in two cases: when the node is practically new (r < 0.01), and when the technician is busy.
In the second case, planned maintenance does not queue behind repairs (Р30).

Because the time between epochs is variable, the problem is a semi-Markov decision process (SMDP)
[REF: Puterman 1994; Bradtke & Duff 1995]. Discounting is continuous: a transition of duration Δt is
discounted by γ = exp(−Δt/τ), with τ = 300 h (Р43).

### 2.4.2. Decision grid

If a node is queried at every operation boundary, consecutive decisions are only about 3·10⁻⁴ η of operating
time apart. At that spacing the difference between Q(PM) and Q(CONTINUE) is far below the estimation noise.
In a pilot run without a grid the agent learned never to perform PM (a3-dqn-np15-L, Р43).

We therefore query a node only when its normalised age r crosses into a new cell of width 0.025. The cell index
resets after PM. Between grid crossings, the action is CONTINUE.

### 2.4.3. Observation

The observation of node i is a 26-dimensional vector computed deterministically from the simulation state and
the known distribution parameters; no random samples are used. It contains three groups of features.

**Node features:**

- equipment class, one-hot (6 features);
- normalised age r, clipped at 3;
- failure probability over the next operation, p_f(v, d) = 1 − exp(−Σ_c [H_c(v_c + d) − H_c(v_c)]);
- failure probability up to the next grid crossing;
- current quality degradation, 1 − e^(−r);
- current slowdown, s(r) − 1.

**Context of the decision:**

- epoch type: E1, waiting for inputs, or blocked;
- time already spent waiting;
- input coverage of the next operation by stock;
- fill level of the node's output buffer;
- log ratios of the mean PM duration and the mean repair duration to the operation duration.

**Line features:**

- share of all nodes, and of the node's own class, that are under PM or repair;
- share of idle printers and of idle extruders;
- whether the technician is free;
- PM cost and repair cost of the node, normalised by the drone value.

[CHECK: list against observe() in DQNLineV3.py before submission — 26 = 6 + 20.]

### 2.4.4. Reward: profit decomposed over nodes, with throughput accounting

The objective is line profit (3), but decisions are taken per node. We therefore decompose profit into local
node accounts r_i that add up to profit. Each node's account (Р41) is:

- credited with the value added by each good operation it completes;
- debited for:
  - the embodied value of inputs lost to scrap or to an interrupted operation;
  - filament lost;
  - its own PM cost;
  - its own repair cost.

The value added by an operation is its share of the embodied value of the drone, so the credits along the BOM of
one drone sum to V_d. Hence Σ_i r_i equals profit (3) up to the change in work-in-progress value (verified to
0.13% over 2 000 h).

When value added is credited in proportion to processing time, a non-bottleneck node is paid for output that
the bottleneck cannot absorb. In the 17-printer configuration an agent trained this way under-maintained the
extruders, which were the bottleneck, and lost 55 k USD/year against the best classical policy (Section 3).

We therefore credit value in the spirit of throughput accounting [REF: Goldratt; e.g. Goldratt & Cox 1984 or
a peer-reviewed TOC accounting source] (Р45). The processing time of each operation is weighted by a class
factor

  w_c = exp(s · (u_c − max_k u_k)),   s = 20,                                                (4)

where u_c is the busy share of class c: the fraction of time its nodes are processing, changing over, under PM
or under repair. The weights are then renormalised, so that the identity Σ_i r_i = profit still holds. The
bottleneck class receives a weight close to 1, and classes with spare capacity are credited little.

The utilisations u_c are estimated in a calibration episode without PM. They are then updated after every
training episode by exponential smoothing with weight 0.5.

**Policy-invariant baseline correction (Р41).** The transition reward of node i between epochs k and k+1 is

  R_k = 10⁻³ · [ r_i(t_{k+1}) − r_i(t_k) − ρ_c τ (1 − e^(−Δt_k/τ)) ],                        (5)

where ρ_c is the average reward rate per node of class c, estimated from the preceding episodes.

Subtracting the discounted integral of a constant rate shifts the return of every trajectory by the same
amount, −ρ_c τ, so the optimal policy is unchanged. What the correction removes is the spurious attraction of
long transitions. Without it, PM was always preferred, because after PM the next decision of the node comes
only after a long interval, which carries a large positive reward.

**Table 2.** SMDP specification (summary).

| Element | Specification |
|---|---|
| Agent | One Q-network shared by all nodes; decisions decentralised, training centralised (shared replay buffer) |
| Decision epochs | E1 (operation boundary), E2 (start of waiting or blocking), E3 (every 1 h of waiting), filtered by the age grid of width 0.025 |
| Actions | CONTINUE, PM; PM masked if r < 0.01 or the technician is busy |
| Observation | 26 deterministic features (Section 2.4.3) |
| Reward | Increment of the node's throughput-accounting profit account, minus the policy-invariant baseline term (5), in k USD |
| Discount | γ = exp(−Δt/τ), τ = 300 h |

## 2.5. Agent and training

The agent is a Double DQN [REF: van Hasselt et al. 2016] with a dueling head [REF: Wang et al. 2016] and
proportional prioritised experience replay [REF: Schaul et al. 2016] (α = 0.6, β annealed from 0.4).

- **Network.** Two hidden layers of 128 units, each with layer normalisation and ReLU, followed by value and
  advantage streams of 64 units each.
- **Training updates.** Huber loss, Adam optimiser, gradient-norm clipping at 10, soft target updates
  (τ_soft = 0.005), batch size 64, buffer of 2·10⁵ transitions, one gradient step per decision after a warm-up
  of 1 000 transitions.
- **SMDP discounting.** Each transition stores its own discount factor γ_k.
- **Exploration.** ε decays linearly per episode from 1 to 0.05 over the first 40% of episodes and then stays
  at 0.05.
- **Learning rate.** Decays linearly from 3·10⁻⁴ to 3·10⁻⁵ over the estimated total number of training steps.

**Training protocol.**

- Each scenario is trained separately, as is each classical rule tuned separately.
- Each agent trains for 40 episodes of 26 weeks of simulated time.
- Every episode starts from a new line with all ages at zero, on an environment seed drawn from {0, …, 9999}.
- Before the first episode there are two calibration episodes of 13 weeks each without PM: one to estimate u_c
  for (4), the other to estimate ρ_c for (5).
- For each scenario, three agents are trained with agent seeds 1, 2 and 3. The agent seed controls network
  initialisation, exploration and replay sampling.

Training one agent takes about 0.5 h on one CPU core (Section 2.8). The agent is not tuned per scenario: the
hyperparameters above were fixed during development on the 15-printer moderate scenario, and the same values
are used everywhere.

[AUTHOR: state the development budget honestly — the grid width, τ ∈ {300, 1 000} h (a3-dqn-grid-*) and
s = 20 were chosen on that scenario; list it as the tuning budget, alongside the baseline budget (Table 3).]

**Table 3.** Agent hyperparameters. [TODO: generate from config.json of the dqn_ta_* runs by script.]

## 2.6. Baseline policies

All baselines decide only at E1 (operation boundaries), except the opportunistic rule. They use the same
technician constraint as the agent.

- **B0 — corrective only.** PM is never performed.
- **B1 — periodic PM, analytical interval.** For the wear mode of each class we compute the classical
  availability-optimal age-replacement interval T*_c [REF: Barlow & Proschan 1965], treating the node in
  isolation with its mean PM and repair durations (Table 1). This is how the interval would be set in
  engineering practice. PM is performed when the operating time since the last PM reaches m·T*_c, with one
  multiplier m for all classes. The multiplier is tuned over m ∈ {0.25, 0.35, 0.5, 0.75, 1.0, 1.5}.
- **B1\* — periodic PM with class-specific multipliers.** This is the strongest classical baseline. Multipliers
  m_c are tuned by one pass of coordinate descent, first over C2 and then over C1, on the grid
  {0.25, 0.35, 0.5, 0.75, 1.0}, starting from the best common m. Classes C3, C4 and C6 do not reach their T*_c
  within the evaluation horizon, so they receive no PM (Р35).
- **B1a / B1a\* — age replacement on the component's virtual age** (Р52). These are identical to B1 / B1\*,
  except that the clock is the virtual age of the wear component. That age is reset both by PM and by the repair
  of that component, as in classical age replacement [REF: Barlow & Proschan 1965]. B1 and B1\* count operating
  time since the last PM, and a corrective repair does not reset that count. [TODO Р52: tuning budget and results.]
- **B2 — risk threshold.** PM is performed when the failure probability over the next operation exceeds θ.
  The threshold is tuned over θ ∈ {3·10⁻⁴, 10⁻³, 3·10⁻³}.
- **B3+ — opportunistic.** PM is performed during waiting or blocking (E2/E3) if r > x. At operation
  boundaries the rule behaves like periodic PM. Two versions were tuned:
  - with the analytical interval (m = 1), over x ∈ {0.1, 0.2, 0.3, 0.5};
  - on top of the best periodic multiplier (m = 0.5), over x ∈ {0.2, 0.3}; this is the version reported as B3+,
    with x = 0.3.

  The rule embodies the hypothesis that maintenance is cheap while a node is idle anyway, which a learned policy
  might also discover.

All baselines are tuned by maximising mean profit on the five tuning seeds 1000–1004, using the same warm-up and
window as the evaluation (Section 2.7). Baselines are tuned per scenario, just as the agent is trained per
scenario. The tuning budget, i.e. the number of configurations × 5 seeds per scenario, is listed in Table 3b
[TODO: generate from tune_*_L.csv, b3plus_v13_L.csv and b1star_*_L.csv by script].

## 2.7. Experimental protocol

### 2.7.1. Seeds

The four seed sets are disjoint:

| Purpose | Environment seeds |
|---|---|
| Agent training | {0, …, 9999}, drawn by each agent's own generator |
| Baseline tuning | 1000–1004 |
| Validation (agent selection) | 200000–200002 |
| Testing | 100000–100019 (20 seeds) |

The baseline tuning seeds may coincide with agent training seeds; both belong to the "development" side.

### 2.7.2. Evaluation run

Each evaluation run simulates a 26-week warm-up followed by a one-year measurement window (8 760 h) (Р26).
Policies are greedy, with no exploration. Metrics are computed on the measurement window only.

### 2.7.3. Agent selection and reporting

Three agents per scenario differ in quality: in the strong-degradation scenario one agent lost about 100 k USD/year
against the other two. We therefore fix a selection rule before looking at the test seeds (Р47): the agent with
the highest mean profit on the three validation seeds is used.

We report both:

- the selected agent, which is the practical protocol;
- the mean of the three agents, which measures the stability of the method.

### 2.7.4. Metrics and statistics

**Primary metric:** annual profit (3).

**Secondary metrics:**

- throughput (good drones per week);
- OEE = A·P·Q, where:
  - availability A counts PM, repair and changeover time as losses, while starvation and blocking are external
    and are not counted (Р23);
  - performance P is ideal over actual processing time;
  - quality Q is good over completed items;
  - line OEE is weighted by run time;
- MTBF;
- numbers of failures and of PM interventions.

Because all policies share common random numbers, policies are compared through paired differences on the same
test seed. We report the mean paired difference to B1\* and its standard error over the 20 test seeds.
Significance is assessed with the two-sided Wilcoxon signed-rank test on the paired differences
[REF: Wilcoxon 1945]. For each policy, p-values are adjusted for the six scenarios with the Holm procedure
[REF: Holm 1979].

### 2.7.5. Robustness and sensitivity studies

**Model misspecification (Р49).** All policies, and the agent's observation, rely on Weibull models of
failures. To test robustness, the true time to failure of every wear mode is replaced by a lognormal distribution
with the same mean and coefficient of variation; random modes stay exponential. The lognormal hazard rises and
then falls, and the distribution has a lighter left tail. Policies are frozen:

- baselines keep their Weibull-tuned parameters;
- agents are not retrained;
- the agent's observation still uses Weibull-based failure probabilities.

As a reference, B1 is re-tuned on the lognormal truth ("informed" B1). Failure ages are sampled from the same
random streams as in the Weibull case, by inverse transform from the same exponential draw, so CRN pairing is
preserved. Scenarios: moderate degradation with 15 and 17 printers, and strong degradation with 15 printers;
20 test seeds.

**Unsourced parameters (Р50).**

- For the printer wear scale η_C2 ∈ {750, 3 000} h (base 1 500 h), every policy is re-tuned or re-trained, with
  the same selection rules fixed in advance. The B2 threshold grid is widened to θ ∈ {10⁻⁴, 3·10⁻⁴, 10⁻³,
  3·10⁻³, 10⁻²}, because the per-operation failure risk, and hence the optimal θ, scales with η.
- For the manual-station tooling scale η_C4 ∈ {2 500, 10 000} h (base 5 000 h), the base policies are frozen,
  since C4 does not receive PM within the horizon.

Both studies use moderate degradation, 15 printers and 20 test seeds.

## 2.8. Implementation and reproducibility

The simulator, agent and experiment scripts are written in Python 3.14, with NumPy 2.4, SciPy 1.18 and
PyTorch 2.10 (CPU). Official runs were performed on a desktop computer: Intel Core i5-9600K, 6 cores at 3.7 GHz,
Windows 10.

**Determinism.** Simulations are deterministic for a given seed. Re-running an evaluation reproduces the
logged results bit for bit (checked on test seeds after each code change). Network training is not bitwise
reproducible across processors and library versions, so reproducibility of training is assessed at the level of
conclusions.

**Traceability.**

- Every run is recorded with its full command, seeds and git tag.
- Every table and figure is generated by script from the raw result files.
- Code, configuration, raw results and the event-log format are available in the project repository (tag
  [TODO]) and archived on Zenodo (DOI [TODO]).
