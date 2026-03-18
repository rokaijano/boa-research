from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schema import CandidateMetadata, CandidatePlan, IncumbentRecord, PatchDescriptor, SearchToolCall, StageRunResult, TrialSummary


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_candidate_plan(text: str | None) -> CandidatePlan | None:
    if not text:
        return None
    data = json.loads(text)
    return CandidatePlan(**data)


def _load_candidate(text: str | None) -> CandidateMetadata | None:
    if not text:
        return None
    data = json.loads(text)
    return CandidateMetadata(**data)


def _load_descriptor(text: str | None) -> PatchDescriptor | None:
    if not text:
        return None
    data = json.loads(text)
    return PatchDescriptor(**data)


def _load_search_trace(text: str | None) -> list[SearchToolCall]:
    if not text:
        return []
    payload = json.loads(text)
    calls: list[SearchToolCall] = []
    for item in list(payload):
        calls.append(SearchToolCall(**item))
    return calls


class ExperimentStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_trial_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(trials)").fetchall()}
        for name in ("candidate_plan_json", "search_trace_json"):
            if name not in columns:
                conn.execute(f"ALTER TABLE trials ADD COLUMN {name} TEXT")

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    run_tag TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    parent_branch TEXT NOT NULL,
                    parent_trial_id TEXT,
                    acceptance_status TEXT NOT NULL,
                    canonical_stage TEXT,
                    canonical_score REAL,
                    candidate_plan_json TEXT,
                    candidate_json TEXT,
                    descriptor_json TEXT,
                    search_trace_json TEXT,
                    diff_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stage_results (
                    trial_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    primary_metric REAL,
                    adjusted_score REAL,
                    metrics_json TEXT NOT NULL,
                    resource_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    artifact_dir TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY (trial_id, stage_name)
                );

                CREATE TABLE IF NOT EXISTS incumbents (
                    stage_name TEXT PRIMARY KEY,
                    trial_id TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    adjusted_score REAL NOT NULL,
                    primary_metric REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_trial_columns(conn)

    def has_trial(self, trial_id: str) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
        return row is not None

    def next_trial_sequence(self, *, run_tag: str) -> int:
        pattern = re.compile(rf"^{re.escape(run_tag)}-(\d+)$")
        next_value = 1
        with self.connect() as conn:
            rows = conn.execute("SELECT trial_id FROM trials WHERE run_tag = ?", (run_tag,)).fetchall()
        for row in rows:
            match = pattern.fullmatch(str(row["trial_id"]))
            if not match:
                continue
            next_value = max(next_value, int(match.group(1)) + 1)
        return next_value

    def create_trial(
        self,
        *,
        run_tag: str,
        trial_id: str,
        branch_name: str,
        parent_branch: str,
        parent_trial_id: str | None,
        candidate_plan: CandidatePlan | None,
        candidate: CandidateMetadata | None,
        descriptor: PatchDescriptor | None,
        search_trace: list[SearchToolCall],
        diff_path: Path | None,
        acceptance_status: str = "pending",
    ) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO trials (
                    trial_id, run_tag, branch_name, parent_branch, parent_trial_id,
                    acceptance_status, canonical_stage, canonical_score, candidate_plan_json,
                    candidate_json, descriptor_json, search_trace_json,
                    diff_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    run_tag,
                    branch_name,
                    parent_branch,
                    parent_trial_id,
                    acceptance_status,
                    None if candidate_plan is None else json.dumps(candidate_plan, default=_json_default, sort_keys=True),
                    None if candidate is None else json.dumps(candidate, default=_json_default, sort_keys=True),
                    None if descriptor is None else json.dumps(descriptor, default=_json_default, sort_keys=True),
                    json.dumps(search_trace, default=_json_default, sort_keys=True),
                    None if diff_path is None else str(diff_path),
                    now,
                    now,
                ),
            )

    def update_trial_descriptor(self, *, trial_id: str, descriptor: PatchDescriptor, diff_path: Path) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE trials
                SET descriptor_json = ?, diff_path = ?, updated_at = ?
                WHERE trial_id = ?
                """,
                (
                    json.dumps(descriptor, default=_json_default, sort_keys=True),
                    str(diff_path),
                    utc_now(),
                    trial_id,
                ),
            )

    def set_acceptance(
        self,
        *,
        trial_id: str,
        acceptance_status: str,
        canonical_stage: str | None,
        canonical_score: float | None,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE trials
                SET acceptance_status = ?, canonical_stage = ?, canonical_score = ?, updated_at = ?
                WHERE trial_id = ?
                """,
                (acceptance_status, canonical_stage, canonical_score, utc_now(), trial_id),
            )

    def record_stage_result(self, *, trial_id: str, result: StageRunResult) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stage_results (
                    trial_id, stage_name, branch_name, status, primary_metric, adjusted_score,
                    metrics_json, resource_json, reason, artifact_dir, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    result.stage_name,
                    result.branch_name,
                    result.status,
                    result.primary_metric,
                    result.adjusted_score,
                    json.dumps(result.metrics, sort_keys=True),
                    json.dumps(
                        {
                            "command_results": [asdict(command) for command in result.command_results],
                            "resource_metadata": result.resource_metadata,
                        },
                        default=_json_default,
                        sort_keys=True,
                    ),
                    result.reason,
                    str(result.artifact_dir),
                    result.started_at,
                    result.completed_at,
                ),
            )

    def upsert_incumbent(
        self,
        *,
        stage_name: str,
        trial_id: str,
        branch_name: str,
        adjusted_score: float,
        primary_metric: float,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO incumbents (stage_name, trial_id, branch_name, adjusted_score, primary_metric, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(stage_name) DO UPDATE SET
                    trial_id = excluded.trial_id,
                    branch_name = excluded.branch_name,
                    adjusted_score = excluded.adjusted_score,
                    primary_metric = excluded.primary_metric,
                    updated_at = excluded.updated_at
                """,
                (stage_name, trial_id, branch_name, adjusted_score, primary_metric, utc_now()),
            )

    def get_incumbent(self, stage_name: str) -> IncumbentRecord | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT stage_name, trial_id, branch_name, adjusted_score, primary_metric, updated_at FROM incumbents WHERE stage_name = ?",
                (stage_name,),
            ).fetchone()
        if row is None:
            return None
        return IncumbentRecord(
            stage_name=row["stage_name"],
            trial_id=row["trial_id"],
            branch_name=row["branch_name"],
            adjusted_score=float(row["adjusted_score"]),
            primary_metric=float(row["primary_metric"]),
            updated_at=row["updated_at"],
        )

    def recent_trials(self, *, limit: int = 20, run_tag: str | None = None) -> list[TrialSummary]:
        query = """
            SELECT trial_id, run_tag, branch_name, parent_branch, parent_trial_id, acceptance_status,
                   canonical_stage, canonical_score, candidate_plan_json, candidate_json,
                   descriptor_json, search_trace_json, created_at, updated_at
            FROM trials
        """
        params: list[Any] = []
        if run_tag is not None:
            query += " WHERE run_tag = ?"
            params.append(run_tag)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
            stage_rows = conn.execute(
                "SELECT trial_id, stage_name, adjusted_score FROM stage_results ORDER BY completed_at DESC"
            ).fetchall()
        stage_scores: dict[str, dict[str, float]] = {}
        for row in stage_rows:
            trial_id = row["trial_id"]
            stage_scores.setdefault(trial_id, {})
            if row["adjusted_score"] is not None:
                stage_scores[trial_id][row["stage_name"]] = float(row["adjusted_score"])
        summaries: list[TrialSummary] = []
        for row in rows:
            summaries.append(
                TrialSummary(
                    trial_id=row["trial_id"],
                    run_tag=row["run_tag"],
                    branch_name=row["branch_name"],
                    parent_branch=row["parent_branch"],
                    parent_trial_id=row["parent_trial_id"],
                    acceptance_status=row["acceptance_status"],
                    canonical_stage=row["canonical_stage"],
                    canonical_score=None if row["canonical_score"] is None else float(row["canonical_score"]),
                    candidate_plan=_load_candidate_plan(row["candidate_plan_json"]),
                    candidate=_load_candidate(row["candidate_json"]),
                    descriptor=_load_descriptor(row["descriptor_json"]),
                    search_trace=_load_search_trace(row["search_trace_json"]),
                    stage_scores=stage_scores.get(row["trial_id"], {}),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return summaries
