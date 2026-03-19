# V2 General Cognitive Runtime Roadmap

Purpose: define a concrete reset path from the current prompt-routed runtime to
a general cognitive adapter that can be dropped into new environments, perform
reasonably well initially, and improve through memory and experience.

This document is intentionally not a NetHack roadmap. NetHack, ScienceWorld,
WebArena, and future real-world settings are development signals for the same
core architecture. The standard for success is transfer.

## Why A V2 Reset

Recent iterations improved local controller mechanics:

- fewer blind open-loop bursts
- better interruption observability
- better token efficiency per action
- more structured option state

But they did not produce reliable task-level gains. That is a sign that the
current architecture in
[src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
is approaching a local maximum:

- too much control still lives in prompt text and chat orchestration
- the runtime still infers too much from environment-local text churn
- state representation, planning, execution, and memory are still too coupled
- adding more control patches mostly moves failure modes around

V2 should preserve the useful artifacts from the current work, but stop
evolving the old loop as the primary path.

## Product Goal

Build a reusable cognitive runtime with these properties:

1. Environment-agnostic core.
   The same planner, memory system, and controller should run across NetHack,
   ScienceWorld, WebArena, and future environments.

2. Strong initial competence.
   The agent should behave reasonably on first contact in a new environment by
   relying on generic exploration, grounding, option control, and memory
   formation rather than benchmark-specific heuristics.

3. Online adaptation.
   The agent should improve within and across episodes through episodic memory,
   semantic abstraction, and operator reliability estimates.

4. Sparse language-model usage.
   LLM calls should be reserved for semantic interpretation, planning under
   uncertainty, compression into memory, and recovery from failure.

5. Transfer-first evaluation.
   Architecture changes should be accepted only if they improve the efficiency /
   transfer frontier, not just an in-domain benchmark score.

## Architectural Target

V2 should have five stable layers.

### 1. Environment Adapter Layer

Owner:
- [src/agent/env_adapter.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/env_adapter.py)

Responsibility:
- normalize environment observations into a common state-update event stream
- normalize raw actions into operator candidates
- expose environment structure such as:
  - entities
  - relations
  - UI elements
  - spatial neighborhoods
  - navigation frontiers
  - status / reward deltas
- preserve raw environment artifacts for analyst traces

This layer should be the only place where NetHack- or WebArena-specific parsing
is allowed.

### 2. World Model Layer

New modules:
- `src/agent/world_model.py`
- `src/agent/world_state_types.py`

Responsibility:
- maintain the canonical typed internal state
- represent:
  - entities and aliases
  - relations and containment
  - affordances and action operators
  - uncertainty and unresolved referents
  - frontier and revisitation state
  - active goals and subgoals
  - progress and novelty metrics
  - current option and option outcome history
- support deterministic updates after each environment step

This should supersede prompt-centric belief maintenance as the primary runtime
state.

### 3. Planner / Controller Layer

New modules:
- `src/agent/planner.py`
- `src/agent/controller.py`
- `src/agent/options_v2.py`

Responsibility:
- choose abstract option families
- allocate reasoning budget
- decide when to continue, revise, or abandon an option
- request LLM planning only when uncertainty or value-of-information is high

Core option families should stay generic:
- `explore_frontier`
- `inspect_novelty`
- `pursue_reward`
- `engage_or_avoid_threat`
- `manipulate_target`
- `commit_transition`
- `verify_outcome`
- `recover_from_failure`

The planner should output structured option objects, not raw action strings.

### 4. Deterministic Executor Layer

New modules:
- `src/agent/executor.py`
- `src/agent/operators.py`

Responsibility:
- map abstract options to executable operator sequences
- step one primitive action at a time
- update the world model after every step
- stop when:
  - the option succeeds
  - the option fails
  - option value decays
  - a contradiction appears
  - a high-value new opportunity appears

This is the main token-efficiency lever. Normal execution should not require a
full LLM round-trip.

### 5. Memory / Learning Layer

New modules:
- `src/agent/memory/runtime_memory.py`
- `src/agent/memory/episodic_store.py`
- `src/agent/memory/semantic_store.py`
- optional later: `src/agent/learned_control.py`

Responsibility:
- store episodic traces keyed by abstract state / option / outcome signatures
- compress repeated episodes into semantic knowledge
- track operator reliability and context applicability
- retrieve relevant prior abstractions at planning time
- support online adaptation without benchmark-specific scripts

If any learned component is added later, it should operate on abstract runtime
features rather than environment-local tokens.

## What To Keep, Freeze, And Replace

### Keep

These are good foundations and should carry forward:

- environment adapters in
  [src/agent/env_adapter.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/env_adapter.py)
- typed decision-state work in
  [src/agent/decision_state.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/decision_state.py)
- architecture metrics persistence in
  [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)
  and
  [src/storage/models.py](/home/eduardo/Projects/cogAgent_tracker/src/storage/models.py)
- analyst traces and run artifact persistence in
  [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)

### Freeze

Treat these as legacy baseline behavior unless needed for compatibility:

- the current multi-agent prompt routing loop in
  [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
- prompt-centric belief maintenance in
  [src/agent/configs/prompts.yaml](/home/eduardo/Projects/cogAgent_tracker/src/agent/configs/prompts.yaml)
- further environment-specific controller patches inside the legacy runtime

### Replace

These are the main V2 replacement targets:

- chat-mediated control flow
- raw action-string planning as the main policy interface
- option continuation based mainly on prompt output
- memory represented primarily as free-form text files
- environment progress inferred from textual churn rather than typed state

## Core Interfaces

V2 should define a small set of stable interfaces early.

### Adapter Event

Every environment step should yield a normalized event with fields such as:

- `raw_observation`
- `state_delta`
- `entity_updates`
- `relation_updates`
- `operator_candidates`
- `reward_delta`
- `status_delta`
- `frontier_delta`
- `novelty_signals`
- `ui_or_spatial_context`

### World Model Snapshot

The planner should see only compact structured state:

- active goal stack
- candidate option families
- grounded entities and unresolved aliases
- local frontier summary
- recent reward / status changes
- repeated-state and revisitation metrics
- uncertainty summary
- memory retrieval summary
- current option contract

### Option Contract

Every option should include:

- `family`
- `objective`
- `target_signature`
- `expected_outcomes`
- `progress_budget`
- `failure_budget`
- `termination_conditions`
- `interrupt_conditions`
- `reasoning_budget`

### Memory Record

Every important outcome should be storable as:

- `state_signature`
- `option_signature`
- `outcome_signature`
- `value_estimate`
- `reusability_scope`
- `source_environment`

This allows mixed-environment learning without turning the memory into a set of
environment-specific scripts.

## Training And Learning Policy

Training is optional and secondary to the runtime design. If added, it must be
scalable and transfer-oriented.

### Acceptable Learning Targets

- option-value prediction
- option-family ranking
- progress prediction
- novelty estimation
- operator reliability estimation
- memory retrieval relevance
- uncertainty calibration

### Unacceptable Learning Targets

- benchmark-local action mimicry
- environment-specific lexical memorization
- hardcoded object priors disguised as learned control
- policies trained only to exploit one benchmark's scoring quirks

### Data Policy

If we learn from traces, the dataset should be built from mixed-environment
structured trajectories:

- NetHack
- ScienceWorld
- WebArena
- future environments

The feature surface should be abstract:

- counts
- graph growth
- frontier change
- uncertainty mass
- option history
- reward / status deltas
- revisitation
- operator success rates

The goal is transfer of control, not transfer of benchmark content.

## Implementation Plan

### Phase A: Carve Out V2 Seams Without Breaking V1

Goal: create a clean parallel path.

Files:
- new package under `src/agent/v2/` or equivalent standalone modules
- minimal compatibility changes in
  [src/agent/gwt_agent.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/gwt_agent.py)
  only where needed for reuse

Work:
- create `V2RuntimeConfig`
- define adapter event types, world model types, and option contract types
- isolate any reusable helper logic from the legacy runtime
- do not change legacy policy behavior

Acceptance gate:
- V1 still runs unchanged
- V2 modules import cleanly and can be unit-tested independently

### Phase B: Build The Common Adapter Protocol

Goal: make environment differences explicit and local.

Files:
- [src/agent/env_adapter.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/env_adapter.py)
- new adapter protocol / types module

Work:
- define a common adapter protocol
- refactor the NetHack adapter to emit normalized adapter events
- make the protocol capable of representing WebArena-style UI state without
  changing the core controller

Acceptance gate:
- NetHack can produce structured events without going through prompt text
- adapter output is rich enough to support both grid and UI environments

### Phase C: Build The World Model

Goal: move belief maintenance out of prompts.

Files:
- new world model modules
- bridge code from
  [src/agent/decision_state.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/decision_state.py)

Work:
- define canonical world-state structures
- implement deterministic update rules from adapter events
- define compact planner-facing serialization
- make analyst traces read from world-model state instead of bespoke ad hoc
  runtime dictionaries

Acceptance gate:
- the planner can consume world state without prompt-heavy reconstruction
- state transitions can be tested deterministically

### Phase D: Build Sparse Planning + Deterministic Execution

Goal: replace chat-loop control.

Files:
- planner / controller / executor modules

Work:
- planner selects option family and target
- executor carries out primitive actions with local stepwise updates
- reasoning budget is allocated only for:
  - option selection
  - ambiguity resolution
  - recovery from failure
  - memory compression

Acceptance gate:
- token usage drops materially on exploratory tasks
- the executor can complete multi-step behaviors without chat-driven action
  selection every step

### Phase E: Add Memory-Driven Adaptation

Goal: improve with experience in a transferable way.

Files:
- new episodic / semantic memory modules
- compatibility bridge to
  [src/agent/memory](/home/eduardo/Projects/cogAgent_tracker/src/agent/memory)

Work:
- log episodic state-option-outcome tuples
- add semantic abstraction over repeated patterns
- add retrieval into planner context
- keep memory scoped by abstract applicability, not environment names alone

Acceptance gate:
- repeated episodes in the same environment improve
- memory retrieval remains small, interpretable, and abstract
- no benchmark-specific scripts are needed to realize improvement

### Phase F: WebArena As Transfer Gate

Goal: validate the architecture on a qualitatively different environment.

Files:
- WebArena adapter modules
- evaluation harness updates in
  [scripts/run_agent.py](/home/eduardo/Projects/cogAgent_tracker/scripts/run_agent.py)

Work:
- implement a WebArena adapter using the same adapter protocol
- keep planner, executor, and memory unchanged
- compare transfer behavior against NetHack / ScienceWorld

Acceptance gate:
- the same core runtime works with only adapter-layer specialization
- the main failures in WebArena are adapter or representation gaps, not prompt
  rewrites

## Evaluation Policy

The primary evaluation target should become a transfer frontier, not one
environment's success rate.

Track at least:

- success rate by environment
- tokens per successful episode
- chat rounds per successful episode
- actions per successful episode
- first-episode competence in new environments
- within-environment improvement from memory
- cross-environment variance
- failure-mode diversity

Architecture decisions should be accepted only when they improve one of:

- initial general competence
- adaptation speed
- token efficiency
- transfer robustness

without a large regression in the others.

## Immediate Next Steps On This Branch

1. Add a `src/agent/v2/` package with typed interfaces only.
2. Refactor
   [src/agent/env_adapter.py](/home/eduardo/Projects/cogAgent_tracker/src/agent/env_adapter.py)
   toward a formal adapter protocol and normalized event output.
3. Implement the first `WorldModel` and `OptionContract` types.
4. Keep the current runtime as a baseline path in parallel.
5. Do not add more prompt/controller heuristics to the legacy loop unless they
   are required for safety or compatibility.

## Non-Goals For This Branch

- squeezing extra benchmark score out of the legacy loop
- adding more NetHack- or ScienceWorld-specific controller logic
- training a policy directly on one environment's traces
- moving more state back into prompts

The point of V2 is to make the architecture more reusable, not more tuned.
