from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.v2.types import (
    AdapterEvent,
    MemoryCue,
    OptionContract,
    WorldModelSnapshot,
)


def _unique_ordered(values: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for value in values:
        if value:
            seen.setdefault(value, None)
    return list(seen.keys())


def state_signature(snapshot: WorldModelSnapshot) -> str:
    observation_mode = snapshot.observation_mode
    goal = snapshot.goals[0].description if snapshot.goals else "unknown-goal"
    operator_families = sorted(
        {candidate.family for candidate in snapshot.operator_candidates}
    )
    if snapshot.metadata.get("ui_context"):
        region = snapshot.metadata["ui_context"].get("page_title", "ui")
    else:
        region = snapshot.frontier_summary.get("current_region", "generic")
    return f"{observation_mode}|{region}|{goal}|{','.join(operator_families)}"


def option_signature(option: OptionContract | None) -> str:
    if option is None:
        return "no-option"
    target = option.target_signature or "no-target"
    return f"{option.family}|{target}"


def outcome_signature(
    snapshot: WorldModelSnapshot,
    event: AdapterEvent,
    option: OptionContract | None,
) -> str:
    parts = [event.action_result.operator_family or "unknown-family"]
    if event.reward_delta:
        parts.append("reward")
    if event.status_delta:
        parts.append("status")
    if event.frontier_delta.opened:
        parts.append("frontier-opened")
    if event.frontier_delta.closed:
        parts.append("frontier-closed")
    if event.novelty_signals:
        parts.append("novelty")
    if event.action_result.failure_reason:
        parts.append("failure")
    if option is not None:
        parts.append(option.family)
    if snapshot.metadata.get("ui_context"):
        parts.append("ui")
    return ",".join(_unique_ordered(parts))


@dataclass(frozen=True)
class EpisodicMemoryRecord:
    step_index: int
    state_signature: str
    option_signature: str
    outcome_signature: str
    reward_delta: float
    action_label: str | None = None
    option_family: str | None = None
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class EpisodicMemoryStore:
    records: list[EpisodicMemoryRecord] = field(default_factory=list)

    def add(self, record: EpisodicMemoryRecord) -> None:
        self.records.append(record)

    def recent(self, limit: int = 10) -> list[EpisodicMemoryRecord]:
        if limit <= 0:
            return []
        return self.records[-limit:]


@dataclass(frozen=True)
class SemanticMemoryRecord:
    key: str
    summary: str
    source_scope: str
    relevance: float
    applicability: list[str] = field(default_factory=list)

    def to_cue(self) -> MemoryCue:
        return MemoryCue(
            cue_id=self.key,
            summary=self.summary,
            source_scope=self.source_scope,
            relevance=self.relevance,
            applicability=list(self.applicability),
        )


@dataclass
class SemanticMemoryStore:
    records: dict[str, SemanticMemoryRecord] = field(default_factory=dict)

    def update_from_episode(self, episodic_store: EpisodicMemoryStore) -> None:
        grouped: dict[tuple[str, str], list[EpisodicMemoryRecord]] = {}
        for record in episodic_store.records:
            key = (record.option_signature, record.outcome_signature)
            grouped.setdefault(key, []).append(record)
        new_records: dict[str, SemanticMemoryRecord] = {}
        for (option_sig, outcome_sig), records in grouped.items():
            reward_total = sum(record.reward_delta for record in records)
            count = len(records)
            option_family = records[-1].option_family or option_sig.split("|", 1)[0]
            reward_bias = reward_total / count if count else 0.0
            relevance = min(1.0, 0.4 + abs(reward_bias) + min(count, 5) * 0.08)
            key = f"{option_sig}:{outcome_sig}"
            new_records[key] = SemanticMemoryRecord(
                key=key,
                summary=(
                    f"Option family `{option_family}` previously produced outcome "
                    f"`{outcome_sig}` across {count} similar episodes."
                ),
                source_scope="semantic",
                relevance=relevance,
                applicability=[option_family, outcome_sig],
            )
        self.records = new_records

    def retrieve(
        self,
        snapshot: WorldModelSnapshot,
        *,
        limit: int = 6,
        preferred_option: OptionContract | None = None,
    ) -> list[MemoryCue]:
        current_state = state_signature(snapshot)
        preferred_family = (
            preferred_option.family if preferred_option is not None else None
        )
        scored: list[tuple[float, MemoryCue]] = []
        for record in self.records.values():
            score = record.relevance
            if preferred_family and preferred_family in record.applicability:
                score += 0.4
            if snapshot.observation_mode in record.applicability:
                score += 0.2
            if any(
                candidate.family in record.applicability
                for candidate in snapshot.operator_candidates
            ):
                score += 0.2
            if current_state.split("|", 1)[0] in record.applicability:
                score += 0.1
            scored.append((score, record.to_cue()))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [cue for _, cue in scored[:limit]]


@dataclass
class RuntimeMemoryManager:
    episodic_store: EpisodicMemoryStore = field(default_factory=EpisodicMemoryStore)
    semantic_store: SemanticMemoryStore = field(default_factory=SemanticMemoryStore)

    def record_event(
        self,
        snapshot: WorldModelSnapshot,
        event: AdapterEvent,
        *,
        current_option: OptionContract | None = None,
    ) -> EpisodicMemoryRecord:
        record = EpisodicMemoryRecord(
            step_index=event.step_index,
            state_signature=state_signature(snapshot),
            option_signature=option_signature(current_option),
            outcome_signature=outcome_signature(snapshot, event, current_option),
            reward_delta=float(event.reward_delta or 0.0),
            action_label=event.action_result.action_text,
            option_family=current_option.family if current_option is not None else None,
            metadata={
                "observation_mode": snapshot.observation_mode,
                "operator_count": len(snapshot.operator_candidates),
            },
        )
        self.episodic_store.add(record)
        self.semantic_store.update_from_episode(self.episodic_store)
        return record

    def retrieve_cues(
        self,
        snapshot: WorldModelSnapshot,
        *,
        preferred_option: OptionContract | None = None,
        limit: int = 6,
    ) -> list[MemoryCue]:
        semantic_cues = self.semantic_store.retrieve(
            snapshot,
            limit=limit,
            preferred_option=preferred_option,
        )
        if semantic_cues:
            return semantic_cues
        recent_records = self.episodic_store.recent(limit=limit)
        return [
            MemoryCue(
                cue_id=f"episode:{record.step_index}",
                summary=(
                    f"Recent episode used `{record.option_signature}` and produced "
                    f"`{record.outcome_signature}`."
                ),
                source_scope="episodic",
                relevance=0.5,
                applicability=[record.option_family or "no-option"],
            )
            for record in recent_records
        ]
