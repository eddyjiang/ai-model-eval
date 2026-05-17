import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "data" / "results.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_conn():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def create_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS responses (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT NOT NULL,
                model_key         TEXT NOT NULL,
                model_name        TEXT NOT NULL,
                group_category    TEXT NOT NULL,
                group_name        TEXT NOT NULL,
                template_id       TEXT NOT NULL,
                template_category TEXT NOT NULL,
                template_valence  TEXT NOT NULL,
                trial             INTEGER NOT NULL,
                prompt            TEXT NOT NULL,
                response          TEXT,
                error             TEXT,
                response_time_ms  INTEGER,
                input_tokens      INTEGER,
                output_tokens     INTEGER,
                created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(run_id, model_key, group_name, template_id, trial)
            );

            CREATE TABLE IF NOT EXISTS scores (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id           INTEGER NOT NULL REFERENCES responses(id),
                judge_model_key       TEXT NOT NULL,
                judge_model_name      TEXT NOT NULL,
                compliance            INTEGER,
                compliance_level      TEXT,
                caveat_count          INTEGER,
                caveat_density        REAL,
                tone_softening        REAL,
                unsolicited_balance   INTEGER,
                refusal_type          TEXT,
                explanation_offered   INTEGER,
                sharpness             REAL,
                notes                 TEXT,
                raw_judge_response    TEXT,
                error                 TEXT,
                judged_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)


def response_exists(run_id: str, model_key: str, group_name: str, template_id: str, trial: int) -> bool:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT id FROM responses
               WHERE run_id=? AND model_key=? AND group_name=? AND template_id=? AND trial=?
               AND error IS NULL AND response IS NOT NULL""",
            (run_id, model_key, group_name, template_id, trial),
        ).fetchone()
        return row is not None


def delete_errored_responses(run_id: Optional[str] = None, model_key: Optional[str] = None) -> int:
    """Delete API-error response rows so they will be retried on the next run."""
    sql = "DELETE FROM responses WHERE error IS NOT NULL"
    params: list = []
    if run_id:
        sql += " AND run_id = ?"
        params.append(run_id)
    if model_key:
        sql += " AND model_key = ?"
        params.append(model_key)
    with get_conn() as conn:
        return conn.execute(sql, params).rowcount


def save_response(
    run_id: str,
    model_key: str,
    model_name: str,
    group_category: str,
    group_name: str,
    template_id: str,
    template_category: str,
    template_valence: str,
    trial: int,
    prompt: str,
    response: Optional[str],
    error: Optional[str],
    response_time_ms: Optional[int],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT OR REPLACE INTO responses
               (run_id, model_key, model_name, group_category, group_name,
                template_id, template_category, template_valence, trial,
                prompt, response, error, response_time_ms, input_tokens, output_tokens)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (run_id, model_key, model_name, group_category, group_name,
             template_id, template_category, template_valence, trial,
             prompt, response, error, response_time_ms, input_tokens, output_tokens),
        )
        return cur.lastrowid


def delete_errored_scores(run_id: Optional[str] = None, model_key: Optional[str] = None) -> int:
    """Delete scores that have errors so they can be re-judged. Returns count deleted."""
    sql = """
        DELETE FROM scores
        WHERE error IS NOT NULL
        AND response_id IN (
            SELECT id FROM responses WHERE 1=1
    """
    params: list = []
    if run_id:
        sql += " AND run_id = ?"
        params.append(run_id)
    if model_key:
        sql += " AND model_key = ?"
        params.append(model_key)
    sql += ")"

    with get_conn() as conn:
        cur = conn.execute(sql, params)
        return cur.rowcount


def get_unscored_responses(run_id: Optional[str] = None, model_key: Optional[str] = None) -> list[dict]:
    sql = """
        SELECT r.* FROM responses r
        LEFT JOIN scores s ON s.response_id = r.id
        WHERE s.id IS NULL AND r.response IS NOT NULL
    """
    params: list = []
    if run_id:
        sql += " AND r.run_id = ?"
        params.append(run_id)
    if model_key:
        sql += " AND r.model_key = ?"
        params.append(model_key)

    with get_conn() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def save_score(
    response_id: int,
    judge_model_key: str,
    judge_model_name: str,
    score: dict,
    raw_judge_response: str,
    error: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO scores
               (response_id, judge_model_key, judge_model_name,
                compliance, compliance_level, caveat_count, caveat_density,
                tone_softening, unsolicited_balance, refusal_type,
                explanation_offered, sharpness, notes, raw_judge_response, error)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                response_id, judge_model_key, judge_model_name,
                score.get("compliance"), score.get("compliance_level"),
                score.get("caveat_count"), score.get("caveat_density"),
                score.get("tone_softening"), score.get("unsolicited_balance"),
                score.get("refusal_type"), score.get("explanation_offered"),
                score.get("sharpness"), score.get("notes"),
                raw_judge_response, error,
            ),
        )


def get_scored_data(run_id: Optional[str] = None) -> list[dict]:
    sql = """
        SELECT
            r.run_id, r.model_key, r.model_name, r.group_category,
            r.group_name, r.template_id, r.template_category,
            r.template_valence, r.trial, r.prompt, r.response, r.error,
            s.compliance, s.compliance_level, s.caveat_count,
            s.caveat_density, s.tone_softening, s.unsolicited_balance,
            s.refusal_type, s.explanation_offered, s.sharpness,
            s.notes, s.judge_model_key
        FROM responses r
        LEFT JOIN scores s ON s.response_id = r.id
    """
    params: list = []
    if run_id:
        sql += " WHERE r.run_id = ?"
        params.append(run_id)

    with get_conn() as conn:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]


def get_run_ids() -> list[str]:
    with get_conn() as conn:
        rows = conn.execute("SELECT DISTINCT run_id FROM responses ORDER BY run_id").fetchall()
        return [r["run_id"] for r in rows]


def get_run_stats(run_id: Optional[str] = None) -> dict:
    params: list = []
    where = ""
    if run_id:
        where = "WHERE r.run_id = ?"
        params.append(run_id)

    with get_conn() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM responses r {where}", params).fetchone()[0]
        scored = conn.execute(
            f"SELECT COUNT(*) FROM responses r JOIN scores s ON s.response_id=r.id {where}", params
        ).fetchone()[0]
        errors = conn.execute(
            f"SELECT COUNT(*) FROM responses r {where} {'AND' if where else 'WHERE'} r.error IS NOT NULL",
            params,
        ).fetchone()[0]
        return {"total": total, "scored": scored, "errors": errors}
