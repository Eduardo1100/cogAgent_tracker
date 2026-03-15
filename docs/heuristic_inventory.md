# Heuristic Inventory — `gwt_agent.py`

Input to the ablation sweep. Every heuristic, behavioral patch, and task-specific
control in `GWTAutogenAgent` is listed here with a one-line description and a tag.

Tags:
- `[general]` — useful for any text-adventure / interactive fiction environment
- `[maybe-general]` — probably transfers but was tuned on ScienceWorld tasks
- `[scienceworld-specific]` — hardcodes ScienceWorld domain concepts
- `[unknown]` — tag unclear without more data

Line numbers are approximate anchors into `src/agent/gwt_agent.py`.

---

## 1. Task-Type Detection (episode start, `_build_task_contract`)

These run once per episode to classify the task and populate the `task_contract`
dict that drives every downstream controller.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 1 | **Explicit search mode** — detects "find", "locate", "identify", "discover", "determine whether" in task text; activates `search_mode`. | 1223–1226 | `[general]` |
| 2 | **Inferred search mode** — if primary_targets parsed but no explicit search/candidate/lifecycle/growth/specialist task, infer search is needed. | 1308–1319 | `[general]` |
| 3 | **Procedural sequence** — detects "first", "then", "and then", "next"; sets `procedural_sequence` to order steps without implying a lifecycle. | 1219–1222 | `[maybe-general]` |
| 4 | **Ordered sequence cue** — detects "earliest", "latest", "before", "after", "order", "sequence" etc.; sets `ordering_cues = ["ordered_sequence"]`. | 1380–1384 | `[maybe-general]` |
| 5 | **Lifecycle task** — detects "life stage", "life stages", "lifecycle", "life cycle" keywords; enables stage-aware focus controller. | 1215–1218 | `[scienceworld-specific]` |
| 6 | **Growth task** — detects "grow", "pollinate", "produce fruit", "bear fruit" etc.; activates precursor-generation phase before branch/commit. | 1295–1307 | `[scienceworld-specific]` |
| 7 | **Growth-from-conditional-branch** — if a conditional branch task has evidence targets matching growth tokens, also activates growth_task path. | 1282–1289 | `[scienceworld-specific]` |
| 8 | **State-change task** — detects "melt", "freeze", "boil", "heat", "cool" etc. in task; activates substance search and transformation phases. | 1228–1232 | `[scienceworld-specific]` |
| 9 | **Artifact creation task** — detects "create", "make", "produce", "synthesize" in task; activates ingredient-gap and combine/verify phases. | 1263–1266 | `[maybe-general]` |
| 10 | **Measurement task** — detects property + target + measurement verb pattern; activates instrument search, measurement, and branch gating. | 1233–1240 | `[scienceworld-specific]` |
| 11 | **Comparison task** — detects "which of … has most/least/highest/lowest" pattern with ≥ 2 targets; keeps targets separate until evidence resolves winner. | 1244–1273 | `[scienceworld-specific]` |
| 12 | **Conditional branch task** — detects "if …, [action]" pattern with explicit branch mappings; gates branch-target actions until evidence resolves branch. | 1250–1281 | `[maybe-general]` |
| 13 | **Relation mechanism task** — detects "connect", "electrical circuit", "anode/cathode" etc.; activates relation frontier and pruned mechanism space. | (via `_TASK_FAMILY_HINTS["relation"]`, 1347–1367) | `[scienceworld-specific]` |
| 14 | **Required action families** — task verbs (focus, inspect, connect, move, place, heat, open …) map to protected families that cannot be deprioritized. | 1341–1367 | `[general]` |
| 15 | **Support families** — adds "inspect" as support family whenever any search/growth/change/comparison/branch mode is active (unless inspect already required). | 1369–1378 | `[general]` |
| 16 | **Generic tool preamble suppression** — "use chemistry to create/make/produce/synthesize" → suppresses tool_application from required_families to avoid scoring noise. | 1343–1356 | `[scienceworld-specific]` |
| 17 | **Measurement property type** — classifies measured property as "instantaneous" (temperature reading) vs "stable threshold" (melting point); controls whether one measurement suffices for branch resolution. | ~1241–1243 | `[scienceworld-specific]` |
| 18 | **Transformation direction** — maps task verb (melt→warm, freeze/chill→cool) to a `transformation_direction` used to score thermal devices. | 1683–1691 (`_TASK_STATE_CHANGE_HINTS`) | `[scienceworld-specific]` |
| 19 | **Measurement branch extraction** — parses "if X is above/below N degrees Celsius, focus on Y" rules from task text; caps at 2 branches. | ~2069–2143 | `[scienceworld-specific]` |
| 20 | **Comparison target extraction** — parses "determine which of A, B, … has the most/least X" from task text. | ~1244–1249 | `[scienceworld-specific]` |
| 21 | **Conditional branch target extraction** — parses "if [condition], focus on / place [target]" from task text. | ~1250–1257 | `[maybe-general]` |
| 22 | **Artifact creation contract** — extracts artifact type, descriptor tokens, intermediate targets, and final targets from task text; separates type from adjective distractors. | ~1258–1265 | `[maybe-general]` |
| 23 | **Role-phrase extraction** — regex extracts primary_targets ("focus on X", "activate X", "move X to", "check/read/examine X") and supporting_targets ("using Y", "move X to Y"). | 662–684 | `[general]` |
| 24 | **Candidate class extraction** — extracts generic class labels ("living thing", "non-living thing") from search tasks; lets agent pivot between candidates without a hardcoded list. | ~1251 | `[scienceworld-specific]` |

---

## 2. Phase Machine (`_get_current_phase`)

22+ named phases govern family priority and score bonuses. The phase is determined
each timestep from grounding state, focus completion, and task type.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 25 | **General fallback phase** — "explore_and_gather_evidence" when no specialist state machine applies; falls through to "inspect_mechanism_progress" when a mechanism target is grounded. | ~9244–9280 | `[general]` |
| 26 | **State-change phases** — `locate_substance → probe_sources → confirm_referent → test_transformation → verify_outcome`; advances on substance grounding and observation evidence. | ~9126–9147 | `[scienceworld-specific]` |
| 27 | **Artifact creation phases** — `locate_base_artifact → find_missing_ingredient_or_reagent → combine_or_transform → verify_intermediate → verify_final`; advances on artifact grounding. | ~9148–9187 | `[scienceworld-specific]` |
| 28 | **Relation mechanism phases** — `locate_primary_target → confirm_primary_target → locate_supporting_source → inspect_target_mechanism → integrate_control_or_verify`; triggered by relation_mechanism_task + "off" target status signal. | ~9079–9101 | `[scienceworld-specific]` |
| 29 | **Measurement phases** — `locate_instrument → locate_measured_target → measure_target → induce_property_change → verify_transition → resolve_branch → execute_branch`; branches based on property type and event observation. | ~9102–9125 | `[scienceworld-specific]` |
| 30 | **Comparison phases** — `locate_primary_target → gather_branch_evidence → execute_branch`; advances when all comparison targets observed. | ~9060–9069 | `[scienceworld-specific]` |
| 31 | **Conditional branch phases** — same three phases; advances on branch_ready flag. | ~9070–9078 | `[maybe-general]` |
| 32 | **Growth conditional branch phases** — extended machine adds `commit_to_goal` and `test_mechanism` sub-phases; precursor signal gates branch entry. | ~9025–9059 | `[scienceworld-specific]` |
| 33 | **Lifecycle phases** — `observe_lifecycle → focus_lifecycle`; driven by visible stage labels and focused stage list. | ~9188–9211 | `[scienceworld-specific]` |
| 34 | **Growth phases** — mirror of growth-conditional without branch: `locate_growth_target → grow_and_observe → commit_to_goal`. | ~9212–9243 | `[scienceworld-specific]` |
| 35 | **Candidate search "gather_evidence" sub-phase** — activated when rejected_candidates is non-empty; prompts search rather than repeated focus. | ~9023–9024 | `[general]` |

---

## 3. Shortlist Scoring (`_score_action_for_shortlist`)

Multi-factor scoring applied to every candidate action before quota selection.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 36 | **Keyword hit score** — task keyword tokens matched in action content: +5 per hit. | ~9427 | `[general]` |
| 37 | **Grounded token score** — observation-grounded tokens matched in action: +6 per hit. | ~9428 | `[general]` |
| 38 | **Target entity score** — task `target_entities` matched in action: +5 per hit. | ~9429 | `[general]` |
| 39 | **Primary role full-match bonus** — all primary role tokens present in action: +12. | ~9431 | `[general]` |
| 40 | **Primary role hit score** — per-token hit on primary role: +3 each. | ~9433 | `[general]` |
| 41 | **Relation role full-match bonus** — all `required_relations` tokens present: +9. | ~9435 | `[scienceworld-specific]` |
| 42 | **Support role hit score** — per-token hit on supporting role: +2 each. | ~9439 | `[general]` |
| 43 | **Phase family priority** — lookup family priority from phase table (range −6 to +9). | ~9441 | `[general]` |
| 44 | **Required family bonus** — families explicitly required by task get +4 to +8 depending on family type. | ~9447–9455 | `[general]` |
| 45 | **Support family bonus** — inspect (and other support families) get +4. | ~9459 | `[general]` |
| 46 | **Invalid exact action penalty** — per action string normalized: −12 − count × 4. | ~9527 | `[general]` |
| 47 | **Recent repeat penalty** — if same (family, referent) pair repeated ≥ 2 times in last 6: −10; ≥ 1 time: −4. | ~9528–9531 | `[general]` |
| 48 | **State-change substance role boost** — +8 per target-substance token hit, +6 for full match. | ~7287–7294 | `[scienceworld-specific]` |
| 49 | **State-change non-target substance penalty** — −10 to −18 for actions on grounded substances that are not the target. | ~7295–7303 | `[scienceworld-specific]` |
| 50 | **State-change phase-specific scoring** — separate score tables per phase (locate_substance, probe_sources, confirm_referent, test_transformation, verify_outcome). | ~7305–7401 | `[scienceworld-specific]` |
| 51 | **State-change stalled room boost** — when room_search_stalled, boost open-door/navigation actions; penalize local inspect/device loops. | ~7403–7432 | `[scienceworld-specific]` |
| 52 | **State-change unsupported substance penalty** — penalise actions that reference substance tokens not yet grounded in observation. | ~7434–7464 | `[scienceworld-specific]` |
| 53 | **Artifact creation type/target boost** — +10–+12 for actions matching artifact type, intermediate target, or final target tokens. | ~7518–7533 | `[maybe-general]` |
| 54 | **Artifact creation phase scoring** — separate scoring per phase; "find_missing_ingredient" phase applies −78/−62 penalty for premature relocation. | ~7535–7654 | `[maybe-general]` |
| 55 | **Artifact descriptor-only penalty** — −18 if action matches descriptor tokens (e.g., color) but not artifact type. | ~7656–7662 | `[maybe-general]` |
| 56 | **Artifact single-artifact relocation penalty** — −10 if only one artifact grounded and action is relocation/transfer. | ~7664–7671 | `[maybe-general]` |
| 57 | **Artifact stalled room boost** — same door/navigation boost when artifact_room_search_stalled. | ~7673–7710 | `[maybe-general]` |
| 58 | **Relation mechanism phase scoring** — focus: +22 in locate phase; large ±36 bonuses for relation family in inspect/integrate phases. | ~7797–7962 | `[scienceworld-specific]` |
| 59 | **Relation invalid referent penalty** — −8 to −14 minus attempts×4 for repeated invalid referent attempts on same (family, referent) pair. | ~7964–7970 | `[general]` |
| 60 | **Measurement phase scoring** — full scoring table per phase (locate_instrument through execute_branch) including +22 instrument-target pair bonus and +34–+62 control candidate boosts in induce_property_change. | ~8705–9017 | `[scienceworld-specific]` |
| 61 | **Measurement hidden target penalty** — −24 to −72 for actions on measurement targets that are inside a closed container (not yet accessible). | ~8957–9017 | `[scienceworld-specific]` |
| 62 | **Comparison target focus penalty/boost** — in gather_branch_evidence: −120 for focus on any comparison target; in execute_branch: +150 for focus on selected target, −90 otherwise. | ~8079–8146 | `[scienceworld-specific]` |
| 63 | **Comparison stalled room boost** — same navigation boost when comparison_room_search_stalled. | ~8027–8078 | `[scienceworld-specific]` |
| 64 | **Growth precursor penalty** — −10 if precursor is focused but action is not on primary match. | ~8193–8200 | `[scienceworld-specific]` |
| 65 | **Growth stalled room boost** — boost navigation/door actions when growth_room_search_stalled. | ~8202–8240 | `[scienceworld-specific]` |
| 66 | **Conditional branch target suppression** — −120 for branch target actions in locate_primary_target phase; −180 to −120 in growth-branch tasks. | ~8318–8445 | `[maybe-general]` |
| 67 | **Conditional branch execute_branch boost** — +160 for relocation commit, ±140 for focus on selected branch target. | ~8500–8549 | `[maybe-general]` |

---

## 4. Shortlist Family Quotas (`_get_shortlist_family_quotas` + overrides)

Each phase has a base quota table (1–4 slots per family). Quotas are then
conditionally overridden at summary-build time.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 68 | **Phase-based quota table** — 22+ named phases each specifying per-family slot counts; drives which action families appear in the shortlist. | ~6242–6448 | `[general]` |
| 69 | **Ordered sequence quota override** — when ordered_sequence active and targets visible: focus +1 (min 2), inspect −1. | ~10193–10201 | `[general]` |
| 70 | **Relation focus-completed quota override** — when primary target focused and relation is required: relation min 2, focus −1. | ~10202–10208 | `[scienceworld-specific]` |
| 71 | **Relation commit-ready quota override** — relation min 3, device_control min 2, inspect max 1. | ~10209–10213 | `[scienceworld-specific]` |
| 72 | **Candidate search inspect quota boost** — inspect min 2; when rejected candidates exist, also focus min 2 and relocation min 2. | ~10214–10219 | `[general]` |
| 73 | **Remote room signal relocation quota boost** — when remote_room_signal present: relocation min 1, device_control min 1, focus −1. | ~10220–10225 | `[general]` |
| 74 | **Lifecycle search mode relocation quota** — when lifecycle targets not visible: relocation = 1 (navigation only) or 0. | ~10231–10263 | `[scienceworld-specific]` |

---

## 5. Action Blocking (`blocked_actions` in `_summarize_admissible_actions_uncached`)

Actions removed from shortlist selection entirely regardless of score.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 75 | **Lifecycle search mode: block non-navigation relocations** — when lifecycle targets not visible, only "go to" / "enter" relocation actions are allowed; "pick up", "take", "move" etc. are blocked. | ~10255–10263 | `[scienceworld-specific]` |
| 76 | **Unresolved conditional branch: block branch-target actions** — any action targeting a conditional branch target is blocked until branch_ready. | ~10264–10274 | `[maybe-general]` |
| 77 | **Gather-branch-evidence phase: block non-support relocation/relation/tool actions** — blocks high-primary-role, zero-support-role relocation/relation/tool_application actions during evidence gathering. | ~10275–10283 | `[maybe-general]` |
| 78 | **Relation commit-ready: block primary relation re-inspection** — once commit candidates exist, suppress redundant inspect actions on the primary mechanism target. | ~10284–10293 | `[scienceworld-specific]` |

---

## 6. Action Canonicalization (`_canonicalize_suggested_action`)

Maps the model's suggested action string to the closest admissible command.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 79 | **Exact normalized match** — first tries exact string equality after normalization. | ~6769–6774 | `[general]` |
| 80 | **Token subset greedy matching** — score = (suggested_tokens_subset × 10) − (extra_tokens × 2); requires score ≥ 18 to accept. | ~6776–6820 | `[general]` |
| 81 | **Score gap fallback** — if best_score − second_best < 4, treat match as ambiguous and try room-transition or substance fallbacks before accepting. | ~6834 | `[general]` |
| 82 | **Room transition canonicalization** — "go to {room}" / "enter {room}" mapped to "open {room} door" with +4 bonus for "open" prefix; sorted by (−bonus − extra, extra, len). | ~4125–4167 | `[general]` |
| 83 | **Unsupported substance fallback** — for focus/inspect/tool/transfer actions: strips non-grounded substance tokens from suggested action and re-matches the reduced token set. | ~4065–4123 | `[scienceworld-specific]` |

---

## 7. Stagnation & Repetition Detection

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 84 | **Stale action counter** — ticks once per echo_agent relay cycle when `num_actions_taken` has not changed; at ≥ 8 stale cycles, force-caps `max_round` to terminate episode. | ~12361–12385 | `[general]` |
| 85 | **Echo agent stale relay counter** — independent counter in echo_agent: after 6 consecutive relays with no tool message (Action_Agent outputting text instead of JSON), fires STRAWBERRY termination. | ~12368–12370, `create_echo_agent` in helpers | `[general]` |
| 86 | **Consecutive identical belief guard** — if Belief_State_Agent emits identical content twice in a row while uncertainty is signalled, skip Thinking_Agent and route directly to Action_Agent. | ~12420–12432 | `[general]` |
| 87 | **Task-done conversation cap** — once task_success or task_failed, if conversation exceeds `_task_done_msg_count + 12` messages, force cap max_round to terminate. | ~12449–12454 | `[general]` |
| 88 | **Action/observation signature deduplication** — `_action_observation_signatures` tracks (family, action_token_sig, obs_token_sig) triple; used to compute `stalled_attempts` in hypothesis ledger. | ~1188 | `[general]` |
| 89 | **Invalid exact action memory** — `_invalid_exact_actions` counts per normalized action string; feeds −12 − count×4 penalty. Capped by hypothesis ledger deprioritization. | ~1201 | `[general]` |
| 90 | **Invalid referent attempt memory** — `_invalid_referent_attempts` counts per (family, referent) pair; used in relation scoring penalty. | ~1202 | `[general]` |
| 91 | **Recent referent repeat counter** — counts (family, referent) hits in last 6 hypothesis tests; −10 penalty at ≥ 2, −4 at ≥ 1. | ~4584–4598 | `[general]` |
| 92 | **Local exploration counter per room** — counts inspect/device_control family actions in current room that do not target a frontier/navigation; used by all "room_search_stalled" guards. | ~2663–2876, 3286–3347 | `[general]` |
| 93 | **State-change room search stalled** — fires when: visible_doors ≥ 2 AND current_room set AND local_exploration ≥ 2 AND target substance not yet grounded. Boosts navigation actions. | ~3907–3929 | `[scienceworld-specific]` |
| 94 | **Measurement instrument search stalled** — fires when: visible_doors ≥ 1 AND local_exploration ≥ 2. | ~3931–3953 | `[scienceworld-specific]` |
| 95 | **Artifact creation room search stalled** — fires when: visible_doors ≥ 2 AND local_exploration ≥ 2 AND no grounded artifacts. | ~3955–3973 | `[scienceworld-specific]` |
| 96 | **Growth room search stalled** — fires when: (visible_doors ≥ 1 AND local_exploration ≥ 2 AND visible stages only "seed") OR (nonproductive_growth_tests ≥ 3 AND stalled_growth_tests ≥ 2 AND precursor signal present). | ~3975–4035 | `[scienceworld-specific]` |
| 97 | **Comparison room search stalled** — fires when: visible_doors ≥ 1 AND local_exploration ≥ 2. | ~4037–4063 | `[scienceworld-specific]` |

---

## 8. Hypothesis Ledger (`episode_hypothesis_ledger`)

Per-family confidence accounting that drives deprioritization.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 98 | **Per-family confidence update** — +0.1 for observable_change outcome, −0.2 for invalid attempt, −0.15 for stalled attempt. `_DIRECT_EFFECT_RE` classifies outcomes; covers state-change verbs plus cooking/manipulation verbs (roasted, grilled, fried, baked, chopped, sliced, diced, cooked, cut, prepared, seasoned, dropped, taken, removed). | ~7111–7179, 1082–1092 | `[general]` |
| 99 | **Deprioritization rule** — family is deprioritized when: (invalid_attempts ≥ 2 OR stalled_attempts ≥ 2) AND observable_change_attempts == 0 AND family not in required_families. | ~7150–7157 | `[general]` |
| 100 | **Required family protection** — families explicitly named in `required_families` are never deprioritized. | ~7150 | `[general]` |
| 101 | **Eligible family set** — only `relation`, `relocation`, `transfer_or_transform`, `device_control`, `tool_application` may be deprioritized; `inspect` and `focus` are immune. | 1048–1054 | `[general]` |
| 102 | **Recent hypothesis test buffer** — keeps last 6 tests (family, action, outcome, evidence); feeds repeat-penalty and ledger updates. | ~7165–7173 | `[general]` |
| 103 | **Phase family priority table** — separate priority scores by phase for 8 families (range −6 to +9); prioritized families get higher shortlist scores. | ~6451–6753 | `[general]` |

---

## 9. Candidate Tracking (search-and-place tasks)

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 104 | **Active candidate state machine** — tracks `focused`, `relocated`, `support_confirmed`, `container_confirmed`, `room_confirmed` per candidate object. | ~5611–5639 | `[general]` |
| 105 | **Candidate retirement rule** — retire (mark as rejected) when: `post_goal_confirmations ≥ 1` OR `stalled_confirmations ≥ 2`. | ~6013–6015 | `[general]` |
| 106 | **Rejected candidates cap** — rejected candidate list capped at 4. | ~6020 | `[general]` |
| 107 | **Candidate support entity separation** — destination rooms/containers are supporting targets, not primary candidates; prevents double-counting focus targets. | 669–676 | `[general]` |

---

## 10. Lifecycle & Growth Mechanics

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 108 | **Lifecycle stage pattern matching** — 9 named stages (egg/seed, germinating, seedling/juvenile, flowering, fruiting, adult, dead) detected by regex in observations. | 910–951 | `[scienceworld-specific]` |
| 109 | **Lifecycle stage ordering** — `_LIFECYCLE_STAGE_ORDER` assigns ordinal ranks; "next expected stage" is first observed label with rank > highest focused rank. | 952–962, 4355–4368 | `[scienceworld-specific]` |
| 110 | **Stage evidence by referent** — tracks stage labels per object referent (8 recent per referent); used to distinguish container names from true lifecycle evidence bearers. | 1183, ~4219–4260 | `[scienceworld-specific]` |
| 111 | **Observed stage labels buffer** — global list of observed stage labels capped at 8; feeds visible_lifecycle_stage_labels for phase and quota decisions. | 1182, ~4261–4271 | `[scienceworld-specific]` |
| 112 | **Focused stage labels tracking** — separate list of stages the agent has explicitly focused on; prevents re-focusing on completed stages. | 1181, ~6123–6134 | `[scienceworld-specific]` |
| 113 | **Ordered target progress** — for focus/ordered tasks: tracks which primary targets have been focused (capped at 6); includes lifecycle stages for lifecycle tasks. | ~6107–6174 | `[scienceworld-specific]` |
| 114 | **Growth precursor signal** — precursor considered present if: primary target grounded OR support role grounded OR visible stage labels OR grounded tokens match `_GROWTH_TASK_TARGET_HINTS`. | ~4728–4757 | `[scienceworld-specific]` |
| 115 | **Genetic trait hints** — "dominant", "recessive", "trait", "inherit" etc. flagged for conditional branch subject extraction in phenotype/genotype tasks. | 634–643 | `[scienceworld-specific]` |

---

## 11. State-Change / Substance Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 116 | **Target substance extraction** — extracts substance names from task text using state-change stopwords; filters domain-specific noise ("combusting", "boiling", "will"). | ~1228–1231, 712–720 | `[scienceworld-specific]` |
| 117 | **Grounded substances tracking** — `_grounded_substances` accumulates substance labels explicitly seen in observations; prevents inventing ungrounded substance names. | 1190 | `[scienceworld-specific]` |
| 118 | **Source candidates list** — objects/fixtures that look like plausible substance sources, used to probe when target substance still missing. | ~3907, `_get_substance_search_snapshot` | `[scienceworld-specific]` |
| 119 | **Exhausted container targets** — containers fully probed without finding target; list capped at 8; fed to phase logic to avoid re-probing. | 1191, ~2891–2893 | `[scienceworld-specific]` |
| 120 | **Thermal device direction matching** — `_THERMAL_CONTROL_DIRECTION_HINTS` maps warm/cool direction to specific device names (boiler, burner, freezer, fridge …); devices matching the task direction get boosted. | 692–711 | `[scienceworld-specific]` |

---

## 12. Artifact Creation Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 121 | **Grounded artifacts tracking** — `_grounded_artifacts` accumulates artifact labels seen in observations; anchors creation logic to real objects. | 1189 | `[maybe-general]` |
| 122 | **Artifact identity recovery from generic focus confirmations** — "You focus on the wood cup." collapsed to container name; recovers original artifact identity from successful referent. Guards: success-only, non-ambiguous. | ~(AGENTS.md line 182) | `[scienceworld-specific]` |
| 123 | **Artifact type vs descriptor separation** — descriptor tokens (colors, adjectives) stored separately from artifact type; descriptor-only actions penalised −18 to prevent wrong-type focus. | 1259–1264, ~7656 | `[maybe-general]` |
| 124 | **Room frontier pressure after stalled search** — after repeated non-productive local search with no base artifact grounded, boost open-door/navigation into shortlist before more local inspect/device loops. | ~(AGENTS.md line 184), ~7673–7710 | `[maybe-general]` |

---

## 13. Measurement Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 125 | **Measurement observations buffer** — records (action, observation, value) for each measurement step; capped at 8. | ~3375–3419 | `[scienceworld-specific]` |
| 126 | **Branch selection by threshold crossing** — measurement branch target selected when observed numeric value is above/below the branch threshold; operator parsed from task ("above", "below"). | ~3428–3436 | `[scienceworld-specific]` |
| 127 | **Branch readiness flag** — `branch_ready = True` only once `_selected_measurement_branch_target` is set; blocks execute_branch phase and branch target actions before that. | ~3505–3506 | `[scienceworld-specific]` |
| 128 | **Measurement property event flag** — `_measurement_property_event_observed` gates "induce_property_change" → "verify_transition" phase transition. | ~1194 | `[scienceworld-specific]` |
| 129 | **Control candidates for measurement** — thermal/control component hints identify objects that can mediate temperature change; capped at 4; boosted in induce_property_change phase. | ~1975–2031 | `[scienceworld-specific]` |
| 130 | **Celsius/numeric token stopwords** — "celsius", "degrees", "melting", "point" stripped from measurement target tokens to avoid numeric noise. | 721–731 | `[scienceworld-specific]` |

---

## 14. Comparison Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 131 | **Comparison target from action/observation** — resolves observed comparison target by subset-matching action tokens (limit 24) or observation tokens (limit 36) against candidate list. | ~3592–3641 | `[scienceworld-specific]` |
| 132 | **Comparison readiness rule** — branch ready when all targets have been observed with the same metric type. | ~3731–3734 | `[scienceworld-specific]` |
| 133 | **Friction/percent metric parsing** — special-cases "percent_down_plane" metric from friction-related observation text. | ~3624–3641 | `[scienceworld-specific]` |
| 134 | **Comparison rank guard** — only ranks if ≥ 2 targets have numeric values and values differ; avoids spurious winner selection. | ~3759 | `[scienceworld-specific]` |

---

## 15. Conditional Branch Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 135 | **Branch selection from single-token match** — resolves branch by matching a single exact condition token from observation; returns early if `len(matches) != 1` to avoid ambiguous commits. | ~3819–3823 | `[maybe-general]` |
| 136 | **Growth precursor requirement for growth-conditional branch** — if conditional branch evidence target contains growth token hints, gates branch entry on precursor signal. | ~1282–1289 | `[scienceworld-specific]` |

---

## 16. Relation Mechanism Tracking

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 137 | **Relation frontier referents** — tracks primary and secondary referents from relation actions (connect/disconnect); capped at 8; forms the local mechanism graph. | ~5188–5190, 1203 | `[scienceworld-specific]` |
| 138 | **Power source/sink/bridge classification** — `_POWER_SOURCE_HINTS`, `_POWER_SINK_HINTS`, `_RELATION_BRIDGE_HINTS` classify objects as source/sink/connector; drives source-search in locate_supporting_source phase. | 805–860 | `[scienceworld-specific]` |
| 139 | **Renewable vs non-renewable source hints** — separate hint sets for renewable (solar, wind, hydro) vs non-renewable (coal, diesel, fossil) sources; used for abstract support role inference. | 821–840 | `[scienceworld-specific]` |
| 140 | **Abstract support role grounding** — if task mentions "renewable power source" (abstract), infer concrete candidates from observation-backed objects matching power source hints. | ~795–804, `_ABSTRACT_SUPPORT_ROLE_HINTS` | `[scienceworld-specific]` |
| 141 | **Control component hints** — "switch", "button", "lever", "dial", "knob", "toggle", "trigger" identified as control candidates; boosted in integrate_control_or_verify phase. | 786–794 | `[maybe-general]` |
| 142 | **Relation commit-ready signal** — `commit_ready = True` when primary target is focused AND primary relation component is grounded AND at least one high-scoring relation/device_control action exists. | ~10164–10186 | `[scienceworld-specific]` |
| 143 | **Relation commit candidates cap** — at most 4 commit candidates exposed in shortlist summary. | ~10184 | `[scienceworld-specific]` |
| 144 | **Primary relation inspect suppression post-commit** — once commit candidates exist, actions that would re-inspect the primary mechanism target are blocked. | ~10284–10293 | `[scienceworld-specific]` |

---

## 17. Remote Room Signal

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 145 | **Cross-room evidence memory** — when an observation mentions task-relevant tokens in a room description, stores the room + signal tokens as a remote_room_signal. | ~5410–5481, 1204 | `[general]` |
| 146 | **Signal timestamp priority** — if multiple rooms are candidates, keeps the one with the most recent timestamp, then most tokens, then alphabetical room name. | ~5450–5470 | `[general]` |
| 147 | **Containment by object tracking** — `_containment_by_object` records which container an object was placed into (from relocation/transfer actions); used to route hidden-target actions through the enclosing referent. | ~3116–3138, 1200 | `[general]` |

---

## 18. Cognitive Loop Controls (`_select_speaker`)

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 148 | **Thinking_Agent gated on uncertainty signal** — only invokes Thinking_Agent when Belief_State_Agent output matches `_UNCERTAINTY_RE` or "no observation". | ~12400–12433, 1055–1061 | `[general]` |
| 149 | **Thinking_Agent blocked on unchanged belief** — if same belief content fires uncertainty twice in a row, routes directly to Action_Agent to avoid token waste. | ~12420–12427 | `[general]` |
| 150 | **Belief_State_Agent format repair** — if BSA output lacks "BELIEF STATE:" prefix, synthesize a fallback belief state before routing; resets consecutive count. Format rule is placed at the very top of the BSA system message to prevent format drift. | ~12410–12415 | `[general]` |
| 151 | **Thinking_Agent output validation** — validates Thinking_Agent output for required prefix/format (`_THINKING_PREFIXES`); synthesizes fallback if malformed, then routes to Action_Agent. | ~12435–12443 | `[general]` |
| 152 | **Learning_Agent gated on task outcome** — only invokes Learning_Agent after task_success or task_failed. | ~12390–12398 | `[general]` |
| 153 | **Belief_State_Agent post-success grace period** — BSA may speak 2 more times after success (allows Learning_Agent one cycle) before termination. | ~11239–11255 | `[general]` |
| 154 | **Message history limit: 60 messages** — all cognitive agents except Action_Agent have MessageHistoryLimiter(max_messages=60). | ~11732–11742 | `[general]` |
| 155 | **Action_Agent excluded from history limiter** — MessageHistoryLimiter would strip tool_call/response pairs causing Action_Agent to output text instead of JSON tool calls; excluded deliberately. | ~11744–11748 | `[general]` |
| 156 | **LLM temperature: 0.0 standard / 1.0 reasoner** — standard and support agents at temperature 0.0; reasoner config at 1.0 (required for R1/o1 style models). | ~11476–11499 | `[general]` |
| 157 | **LLM config priority: Gemini → Chat → Reasoner (standard); reversed for reasoner** — standard config tries Gemini first; reasoner config reverses list to try Reasoner first. | ~11473–11498 | `[general]` |
| 158 | **Support config: non-Google models first** — support_config deprioritises Google/Gemini by moving it to last position in config_list. | ~11479–11494 | `[general]` |
| 159 | **Candidate search task prompt injection** — injects hint "Use focus on the candidate only when required by the task…" into Action_Agent runtime context for candidate tasks. | ~(search "treat the destination room") | `[general]` |
| 160a | **`action_executed` percept field** — `update_percept()` now accepts `executed: bool` and writes `action_executed` into the percept JSON. `True` when the environment accepted and ran the action; `False` when it was inadmissible or rejected. BSA must check this field together with `resulting_observation` before marking an action as successful. | ~11373–11384 | `[general]` |
| 160b | **BSA completion grounding rule** — prompt-level rule in BSA system message: an action is only successful if (1) `action_executed` is `true` AND (2) `resulting_observation` positively confirms the change. Prevents hallucinated success claims when the environment silently rejects an action. | `src/agent/configs/prompts.yaml` | `[general]` |

---

## 19. Provider Fallback

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 160 | **Quota error detection by provider URL** — detects quota/rate errors by matching "generativelanguage"/"gemini" or "deepseek" in error message string. | ~11139–11148 | `[general]` |
| 161 | **Runtime provider deprioritisation** — on quota error, moves the offending provider's config entries to end of config_list (once per provider per episode); retries the in-flight chat. | ~11151–11208 | `[general]` |
| 162 | **Per-episode fallback guard** — `_provider_fallbacks_applied` set prevents re-applying the same fallback twice in one episode. | ~1178 | `[general]` |

---

## 20. Dynamic Task Contract Updates

Mid-episode mutations to the task contract cache, triggered by structured observations.

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 175 | **Recipe/instruction target injection** — when a `read|check|examine` action produces an observation matching an "ingredients … directions" section structure, extracts ingredient nouns as new `primary_targets` and cooking verbs as evidence for `tool_application` family. Injects discovered targets into the live task contract cache in-place. Does not fire in environments without recipe-structured text. | ~4458–4489, ~11437–11439 | `[maybe-general]` |

---

## 21. Episode Memory & Misc

| # | Heuristic | Lines | Tag |
|---|-----------|-------|-----|
| 163 | **Memory directory routing by adapter type** — memory stored under `memory/scienceworld/` or `memory/alfworld/` depending on adapter instance; falls back to `env_type` arg; defaults to "alfworld". | ~1160–1170 | `[general]` |
| 164 | **RAG cache cleared per game** — `_episodic_rag_cache` and `_concept_rag_cache` cleared on `set_environment`; prevents stale cross-game embedding hits. | ~11302–11303 | `[general]` |
| 165 | **Cluster cache NOT cleared per game** — cluster cache is keyed by content hash of memory file; intentionally preserved across games, auto-invalidates only when memory changes. | ~11298–11302 | `[general]` |
| 166 | **Shortlist cached per (actions, limit) pair** — `_admissible_summary_cache` avoids re-scoring same action set multiple times per timestep; invalidated on each `update_percept`. | ~10356–10364 | `[general]` |
| 167 | **Shortlist limit: 12 (internal) / 20 (percept/context) / 8 (agent-facing)** — different limits for internal ranking (12), full percept storage (20), and agent-visible snapshot (8). | ~10104, 11392, ~10458 | `[general]` |
| 168 | **Salient entities: 8 internal / 6 agent-facing** — grounded tokens capped at 8 in summary, 6 in agent-facing percept. | ~10336, ~10415 | `[general]` |
| 169 | **Compact task contract: strips measurement_branches and conditional_branches** — agent-facing percept omits the full branch rule lists to reduce prompt size. | ~10400–10402 | `[general]` |
| 170 | **Newly/no-longer relevant actions capped at 4** — delta between consecutive admissible sets capped for prompt efficiency. | ~10460–10462 | `[general]` |
| 171 | **GroupChat and GroupChatManager reinitialized per game** — prevents stale history from game N bleeding into game N+1; agent `_oai_messages` cleared explicitly. | ~11288–11297 | `[general]` |
| 172 | **Non-candidate referent token filter** — tokens like "agent", "air", "inventory", "door", "studio" filtered out when extracting candidate referents to avoid spurious grounding. | 881–888 | `[general]` |
| 173 | **Generic primary target token filter** — tokens "thing", "object", "item", "target", "entity" suppressed from primary target extraction to avoid vacuous matches. | 874–880 | `[general]` |
| 174 | **Referent signature stopwords** — "self", "watering" stripped from referent signature keys to prevent aliasing. | 994–997 | `[scienceworld-specific]` |

---

## Summary: Tag Distribution

| Tag | Count |
|-----|-------|
| `[general]` | 87 |
| `[maybe-general]` | 25 |
| `[scienceworld-specific]` | 61 |
| `[unknown]` | 0 |
| **Total** | **177** |

### Key Ablation Candidates

**High-value ablations (well-isolated, may generalise):**
- #84 Stale action counter (8 cycles → force terminate)
- #85 Echo agent stale relay counter (6 cycles)
- #86 Consecutive identical belief guard
- #98–103 Hypothesis ledger / deprioritization
- #79–83 Action canonicalization
- #145–147 Remote room signal

**Environment-specific; likely safe to stub out for non-ScienceWorld envs:**
- #5–7 Lifecycle/growth task detection
- #8 State-change task detection
- #10–11 Measurement and comparison tasks
- #13 Relation mechanism task
- #108–115 Lifecycle & growth mechanics
- #116–120 Substance tracking
- #125–130 Measurement tracking
- #137–144 Relation mechanism tracking
