# Agent Architecture Roadmap

Purpose: replace the current prompt-heavy, heuristic-accumulating control loop
with a more token-efficient, generalizable, and environment-agnostic runtime.

This document is intentionally biased toward architectural changes, not tactical
heuristic patches. It assumes the current system in
[src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
already contains many useful abstractions, but that too much control remains in:
- long system prompts
- per-turn multi-agent chat
- open-loop action sequencing
- task-family-specific controller branches

The goal is not to make the agent simpler. The goal is to move complexity out of
prompt text and into reusable runtime structure.

## Principles

1. Optimize for a Pareto frontier, not raw success.
   The primary target is: success under lower tokens, fewer chat rounds, and
   lower environment-specific coupling.

2. Keep environment semantics outside the language loop where possible.
   The model should reason over compact structured state, not raw logs plus
   prompt prose plus environment-local action clutter.

3. Spend language-model compute only when information value is high.
   Deliberation is a scarce resource, not the default step function.

4. Prefer generic control abstractions over task-family branches.
   New capabilities should enter as reusable state, option, uncertainty, and
   progress interfaces, not as more handwritten if-else control in one task
   family.

5. Treat prompt growth as technical debt.
   If a new behavior requires substantial prompt text, it is a sign that a
   runtime abstraction is missing.

## Current Bottlenecks

The main performance issues visible in recent runs are architectural:

- Too many chat rounds for too little environment progress.
- Large model-facing percepts assembled every step.
- Deliberation triggered too often by control flow instead of by information
  value.
- Sequence execution that remains too open-loop in exploratory environments.
- Action selection still overly mediated by prompt rules rather than a compact
  operator interface.
- Evaluation that reports cost and tokens, but does not yet use them as first-
  class gating metrics for architecture changes.

These are more important than any individual action-choice bug.

## Target Architecture

The target architecture has five layers.

### 1. World State Layer

Owner:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)

Replace ad hoc model-facing snapshots with a typed latent state object that
captures only reusable decision state:

- goal stack
- grounded entities
- grounded relations
- unresolved referents
- active hypotheses
- recent state deltas
- progress signals
- uncertainty signals
- active option
- option-local memory

This state should become the canonical interface between runtime and model.

### 2. Option Layer

Owner:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)

Replace raw action bursts with interruptible options:

- `ExploreFrontier`
- `InspectTarget`
- `TestMechanism`
- `ManipulateTarget`
- `VerifyOutcome`
- `RecoverFromFailure`

Each option has:

- entry conditions
- expected information/change profile
- termination conditions
- interruption conditions
- compact option-local context

The model chooses an option, not an entire open-loop trajectory.

### 3. Meta-Control Layer

Owner:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml)

Replace turn-by-turn multi-agent deliberation with a generic controller that
decides:

- whether to deliberate
- whether to continue the current option
- whether to terminate the option
- whether to revise belief state
- whether to allocate higher-cost reasoning

This decision should be based on structured signals, not task-family prompt
instructions.

### 4. Learned Policy Layer

Owner:
- new module under `src/agent/`
- offline analysis path under `scripts/`

Add a small learned controller over structured runtime features to predict:

- deliberation necessity
- option family choice
- expected stall risk
- expected information gain

This is the right place for complexity if we want generalization without
hand-authoring more rules.

### 5. Evaluation Layer

Owner:
- [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)
- [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py)

Make efficiency and transfer explicit optimization targets:

- success rate
- tokens per successful episode
- chat rounds per successful episode
- actions per successful episode
- repeated-state density
- repeated-action density
- option interruption rate
- cross-environment variance

## Phase Plan

## Phase 0: Instrumentation Before Refactor

Goal: establish architecture-facing metrics before changing behavior.

Files:
- [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)
- [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py)
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)

Add:

- per-episode repeated-state density
- per-episode repeated-action density
- number of deliberation invocations
- mean tokens per deliberation
- mean tokens per executed action
- option/burst length distribution
- option interruption reason distribution
- observation novelty rate
- grounded-entity graph growth rate

Do not change policy yet.

Gate:

- We can explain where tokens are spent and where progress stalls for each
  environment family without reading raw traces.

## Phase 1: Introduce a Typed Decision State

Goal: compress what the model sees.

Files:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- new file: `src/agent/decision_state.py`

Work:

- Define `DecisionState`, `GoalState`, `OptionState`, `ProgressState`,
  `UncertaintyState`, and `GroundingState`.
- Build a single state-construction path after `update_percept`.
- Stop exposing free-form runtime snapshots directly to the action agent.
- Keep analyst-trace richness unchanged; only shrink model-facing state.

Concrete seam:

- Replace the ad hoc payload assembled in `_get_action_agent_runtime_snapshots`
  with a serializer from `DecisionState`.
- Keep existing internal trackers initially; wrap them rather than deleting them.

Why first:

- This is the highest token-efficiency win.
- It also creates the feature surface needed for later learned control.

Gate:

- Model-facing token payload drops materially.
- No major success regression on random mixed-environment smoke runs.
- Analyst trace remains fully detailed.

## Phase 2: Uncertainty-Triggered Deliberation

Goal: stop paying for reasoning every time control flow cycles.

Files:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml)

Work:

- Replace the current belief/thinking routing logic with a generic
  `DeliberationPolicy`.
- Compute deliberation need from structured signals:
  - observation novelty
  - contradiction score
  - unresolved ambiguity mass
  - option stall probability
  - progress stagnation
  - model confidence proxy
- Convert belief updates for low-uncertainty transitions into deterministic
  runtime updates.
- Reserve LLM deliberation for option selection, ambiguity resolution, and
  hypothesis revision.

Concrete seam:

- The routing block around the current Thinking_Agent / Belief_State_Agent
  selection is the first implementation target.
- Existing `Belief_State_Agent` can be kept temporarily, but called only when
  the controller says the state merits language revision.

Gate:

- Chat rounds per episode decline significantly.
- Tokens per action decline.
- No environment-specific compensating prompt growth is needed.

## Phase 3: Replace Open-Loop Bursts With Interruptible Options

Goal: preserve efficiency gains from sequences without blind overcommitment.

Files:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- new file: `src/agent/options.py`

Work:

- Introduce an option executor around the current `execute_action_sequence`
  path.
- Options output primitive actions incrementally and stop on generic interrupt
  signals:
  - novelty
  - contradiction
  - invalid outcome
  - repeated state
  - unexpected affordance change
  - target reached
- Make pending continuation belong to `OptionState`, not raw leftover action
  strings.
- Keep sequences as an execution mechanism, but no longer as the planning unit.

Concrete seam:

- Refactor `execute_action_sequence` so it executes the active option policy
  instead of a literal user/model-authored action list.
- The model should emit either:
  - option selection, or
  - option revision
  rather than large explicit movement bundles.

Gate:

- Exploratory environments no longer regress mainly because of long blind
  movement/search loops.
- Average open-loop length becomes adaptive rather than prompt-determined.

## Phase 4: Move Action Choice to Operator Grounding

Goal: reduce dependence on prompt wording and lexical action ranking.

Files:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- possibly new file: `src/agent/operators.py`

Work:

- Define a compact operator schema:
  - operator family
  - target slots
  - preconditions
  - expected effects
  - information gain estimate
  - reversibility
- Map admissible actions into operator instances.
- Have the model rank grounded operator candidates, not raw strings.
- Let runtime realize operators as exact admissible actions.

Why this matters:

- It is far more environment-agnostic than rules about concrete commands.
- It allows learned components to operate over stable features.

Gate:

- Action-selection prompt shrinks.
- Fewer prompt rules are needed to get grounded action selection.
- Cross-environment portability improves on held-out tasks.

## Phase 5: Learned Meta-Control

Goal: use experience without baking in benchmark scripts.

Files:
- new module under `src/agent/`
- new offline tooling under `scripts/`

Work:

- Train a small model over structured runtime features to predict:
  - deliberate vs continue
  - option family choice
  - stall risk
  - interruption value
- Train only on abstract features:
  - counts
  - novelty scores
  - graph-growth stats
  - uncertainty mass
  - option history
  - outcome signatures
- Do not use environment-local lexical tokens as primary features.

Training data source:

- existing runs and analyst traces
- per-step runtime state snapshots added in earlier phases

Role:

- This component should bias control allocation, not replace the main model's
  general reasoning.

Gate:

- Better efficiency frontier than the hand-authored controller on random mixed
  runs.
- No obvious collapse on unseen tasks/environments.

## Phase 6: Consolidate and Delete Heuristic Control

Goal: reduce branch surface after the new architecture proves itself.

Files:
- [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- [docs/heuristic_inventory.md](/home/eduardo/Projects/cogAgent_tracker/docs/heuristic_inventory.md)

Work:

- Use the heuristic inventory as a deletion queue.
- Remove heuristics that become redundant under:
  - typed decision state
  - option control
  - operator grounding
  - learned meta-control
- Keep only mechanisms that remain clearly general and architecture-level.

Gate:

- Total controller complexity decreases while performance and efficiency hold.

## File-Level Ownership

### [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)

Primary refactor target.

Responsibilities after roadmap:

- maintain environment interface
- construct `DecisionState`
- manage option execution
- expose compact model-facing state
- own generic progress and uncertainty metrics

Responsibilities that should shrink here over time:

- prompt-encoded policy
- task-family-specific branch logic
- direct raw-action scoring as the main planner

### [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml)

Turn from behavior warehouse into thin interface contract.

Desired future role:

- define output schema
- define role boundaries
- define compact reasoning protocol

Undesired future role:

- carrying ever more environment-conditional action policy

### [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)

Upgrade from experiment runner to architecture evaluator.

Add:

- first-class efficiency summaries
- architecture-phase tags
- option and deliberation metrics
- random mixed-environment benchmark slices

### [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py)

Likely schema additions:

- deliberation_count
- tokens_per_action
- repeated_state_density
- repeated_action_density
- option_interruptions
- option_type histogram or serialized summary

Schema changes should stay minimal and justified by actual use in evaluation.

## Evaluation Strategy

Use the same principle throughout: every architecture change must justify itself
on efficiency and transfer, not just on success.

### Primary Evaluation Slices

1. Random mixed-environment single-episode runs.
2. Small repeated-seed batches for NetHack-like exploratory environments.
3. Small repeated-task batches for science or recipe environments.
4. Periodic held-out regression slice not used during active iteration.

### Required Metrics

- success rate
- tokens per successful episode
- tokens per action
- chat rounds per episode
- actions to success
- invalid action rate
- repeated-state density
- repeated-action density
- mean deliberation count per episode
- mean option interruption count per episode
- variance across environments

### Decision Rule

Reject changes that:

- improve only one environment while harming mixed-environment efficiency
- improve success by materially increasing token cost without offsetting gains
- require prompt growth that cannot be defended as schema, contract, or role
  clarification

## Things Not To Do

Do not spend the next iteration cycle on:

- more environment-specific prompt clauses
- more benchmark-local lexical hints
- more special-case task-family scoring branches
- more handcrafted action penalties or bonuses
- more sequence rules tied to particular environment phenomena

Those may produce local wins, but they move the architecture away from the
stated objective.

## Recommended Execution Order

If work begins immediately, the order should be:

1. Phase 0 instrumentation
2. Phase 1 typed decision state
3. Phase 2 uncertainty-triggered deliberation
4. Phase 3 interruptible options
5. Phase 4 operator grounding
6. Phase 5 learned meta-control
7. Phase 6 heuristic deletion

This ordering matters. Do not start with learned control before the state and
option interfaces exist.

## Success Condition

This roadmap is successful when:

- the agent reasons less often but more profitably
- model-facing context stays compact as capabilities grow
- new environments do not require prompt accretion to remain competitive
- the controller can improve by learning over abstract runtime features rather
  than by accumulating benchmark-conditioned rules
- heuristics become removable instead of foundational
