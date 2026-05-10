import sqlite3
import json
from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
from config.settings import settings


class Database:
    def __init__(self):
        self.pool = None
        self.is_sqlite = False

    async def connect(self):
        """Connect to database - tries PostgreSQL, falls back to SQLite"""
        # Check if it's a file-based SQLite URL
        if "sqlite" in settings.DATABASE_URL or ".db" in settings.DATABASE_URL:
            await self._connect_sqlite()
        else:
            try:
                await self._connect_postgres()
            except Exception as e:
                print(f"PostgreSQL failed: {e}, falling back to SQLite")
                await self._connect_sqlite()

    async def _connect_sqlite(self):
        """Connect to SQLite"""
        self.db_path = "agenticflow.db"
        self.is_sqlite = True
        print(f"Using SQLite database: {self.db_path}")
        await self.init_schema()

    async def _connect_postgres(self):
        """Connect to PostgreSQL"""
        import asyncpg
        self.pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=5,
            max_size=20
        )

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
        elif self.is_sqlite:
            pass  # SQLite connection is stateless

    async def init_schema(self):
        """Initialize database schema"""
        conn = self._get_connection()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    user_query TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    result TEXT,
                    context TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    agent_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    input_hash TEXT,
                    output_hash TEXT,
                    latency_ms REAL DEFAULT 0,
                    token_count INTEGER DEFAULT 0,
                    policy_violations TEXT,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_runs (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    test_results TEXT NOT NULL,
                    summary TEXT,
                    prompts_used TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_cases (
                    id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    expected_answer TEXT,
                    category TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_rewrites (
                    id TEXT PRIMARY KEY,
                    agent_name TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    proposed_prompt TEXT NOT NULL,
                    justification TEXT NOT NULL,
                    diff TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    approved_at TIMESTAMP,
                    approved_by TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    agent_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_data TEXT NOT NULL,
                    output_data TEXT,
                    latency_ms REAL NOT NULL,
                    accepted INTEGER NOT NULL DEFAULT 1,
                    retry_count INTEGER DEFAULT 0,
                    error TEXT,
                    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _get_connection(self):
        """Get SQLite connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def create_job(self, job_id: UUID, query: str) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO jobs (id, user_query, status) VALUES (?, ?, 'pending')",
                (str(job_id), query)
            )
            conn.commit()
        finally:
            conn.close()

    async def update_job(self, job_id: UUID, status: str, result: Optional[Dict] = None) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE jobs SET status = ?, result = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (status, json.dumps(result) if result else None, str(job_id))
            )
            conn.commit()
        finally:
            conn.close()

    async def get_job(self, job_id: UUID) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (str(job_id),))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    async def log_event(self, job_id: UUID, event: Dict) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO execution_logs
                (job_id, agent_id, event_type, input_hash, output_hash, latency_ms, token_count, policy_violations, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(job_id), event.get("agent_id"), event.get("event_type"),
                 event.get("input_hash"), event.get("output_hash"),
                 event.get("latency_ms", 0), event.get("token_count", 0),
                 json.dumps(event.get("policy_violations", [])),
                 json.dumps(event.get("metadata", {})))
            )
            conn.commit()
        except Exception as e:
            print(f"Failed to log event: {e}")
        finally:
            conn.close()

    async def get_execution_trace(self, job_id: UUID) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM execution_logs WHERE job_id = ? ORDER BY timestamp",
                (str(job_id),)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    async def log_tool_call(self, job_id: UUID, agent_id: str, tool_name: str,
                            input_data: Dict, output_data: Optional[Dict],
                            latency_ms: float, accepted: bool, retry_count: int,
                            error: Optional[str] = None) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO tool_calls
                (id, job_id, agent_id, tool_name, input_data, output_data, latency_ms, accepted, retry_count, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (str(UUID.uuid4()), str(job_id), agent_id, tool_name,
                 json.dumps(input_data), json.dumps(output_data) if output_data else None,
                 latency_ms, 1 if accepted else 0, retry_count, error)
            )
            conn.commit()
        finally:
            conn.close()

    async def save_eval_run(self, run_id: UUID, test_results: List[Dict],
                            summary: Dict, prompts_used: Dict) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "INSERT INTO eval_runs (id, test_results, summary, prompts_used) VALUES (?, ?, ?, ?)",
                (str(run_id), json.dumps(test_results), json.dumps(summary), json.dumps(prompts_used))
            )
            conn.commit()
        finally:
            conn.close()

    async def get_latest_eval_run(self) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()

    async def get_eval_runs(self, limit: int = 10) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM eval_runs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    async def save_prompt_rewrite(self, rewrite: Dict) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                """INSERT INTO prompt_rewrites
                (id, agent_name, original_prompt, proposed_prompt, justification, diff, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rewrite["id"], rewrite["agent_name"], rewrite["original_prompt"],
                 rewrite["proposed_prompt"], rewrite["justification"], rewrite["diff"],
                 rewrite["status"])
            )
            conn.commit()
        finally:
            conn.close()

    async def get_pending_rewrites(self) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM prompt_rewrites WHERE status = 'pending' ORDER BY created_at DESC"
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    async def update_rewrite_status(self, rewrite_id: UUID, status: str,
                                    approved_by: Optional[str] = None) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE prompt_rewrites SET status = ?, approved_at = CURRENT_TIMESTAMP, approved_by = ? WHERE id = ?",
                (status, approved_by, str(rewrite_id))
            )
            conn.commit()
        finally:
            conn.close()

    async def get_rewrite(self, rewrite_id: UUID) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT * FROM prompt_rewrites WHERE id = ?",
                (str(rewrite_id),)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None
        finally:
            conn.close()


db = Database()


async def get_db() -> Database:
    return db