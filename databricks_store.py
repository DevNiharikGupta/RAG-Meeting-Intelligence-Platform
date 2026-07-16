"""
Task 5 Part 1 - Databricks Integration

This module handles saving transcript chunks and extracted insights
to Databricks Delta tables. It works ALONGSIDE the local ChromaDB
storage — so local still works even if Databricks is not connected.

Tables created:
  - transcript_chunks: stores each chunk with its source and embedding
  - meeting_insights: stores extracted insights (action items, decisions, etc.)

How it works:
  1. We connect to Databricks using a SQL connector (like connecting to any database)
  2. We create tables if they don't exist (Delta tables)
  3. When ingesting a transcript, we save chunks to the transcript_chunks table
  4. When extracting insights, we save the JSON to the meeting_insights table
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---- Databricks config from .env ----
DB_HOST = os.getenv("DATABRICKS_HOST", "")
DB_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
DB_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
DB_CATALOG = os.getenv("DATABRICKS_CATALOG", "")
DB_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "default")


def get_table_prefix():
    """Build table prefix like 'catalog.schema' or just 'schema' for Community Edition."""
    if DB_CATALOG:
        return f"{DB_CATALOG}.{DB_SCHEMA}"
    return DB_SCHEMA


def is_databricks_configured():
    """Check if Databricks credentials are actually filled in."""
    if not DB_HOST or not DB_TOKEN or not DB_HTTP_PATH:
        return False
    if DB_TOKEN == "your-token-here":
        return False
    return True


def get_connection():
    """Open a connection to Databricks SQL warehouse."""
    from databricks import sql

    connection = sql.connect(
        server_hostname=DB_HOST.replace("https://", ""),
        http_path=DB_HTTP_PATH,
        access_token=DB_TOKEN,
    )
    return connection


def create_tables():
    """Create the Delta tables if they don't exist yet.
    Safe to call multiple times — it uses IF NOT EXISTS."""

    if not is_databricks_configured():
        print("Databricks not configured, skipping table creation.")
        return False

    conn = get_connection()
    cursor = conn.cursor()

    prefix = get_table_prefix()

    cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {prefix}")

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.transcript_chunks (
            chunk_id STRING,
            source_file STRING,
            chunk_index INT,
            total_chunks INT,
            chunk_text STRING,
            ingested_at TIMESTAMP
        )
    """)

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {prefix}.meeting_insights (
            insight_id STRING,
            source_file STRING,
            action_items STRING,
            decisions STRING,
            key_discussion_points STRING,
            participant_contributions STRING,
            extracted_at TIMESTAMP
        )
    """)

    cursor.close()
    conn.close()
    print("Databricks tables ready.")
    return True


def save_chunks_to_databricks(chunks, source_name):
    """Save transcript chunks to the Databricks Delta table.

    Args:
        chunks: list of text strings (the chunked transcript)
        source_name: filename of the transcript (e.g. "sprint_planning.txt")

    Returns:
        number of chunks saved, or 0 if Databricks is not configured
    """
    if not is_databricks_configured():
        print("Databricks not configured, skipping chunk upload.")
        return 0

    conn = get_connection()
    cursor = conn.cursor()

    table = f"{get_table_prefix()}.transcript_chunks"
    now = datetime.now().isoformat()
    saved = 0

    for i, chunk in enumerate(chunks):
        chunk_id = f"{source_name}_{i}_{now}"
        safe_text = chunk.replace("'", "''")

        cursor.execute(f"""
            INSERT INTO {table}
            VALUES ('{chunk_id}', '{source_name}', {i}, {len(chunks)},
                    '{safe_text}', '{now}')
        """)
        saved += 1

    cursor.close()
    conn.close()
    print(f"  {saved} chunks saved to Databricks.")
    return saved


def save_insights_to_databricks(insights, source_name):
    """Save extracted insights (JSON) to the Databricks Delta table.

    Args:
        insights: dict with keys like action_items, decisions, etc.
        source_name: filename of the transcript

    Returns:
        True if saved, False if Databricks not configured
    """
    if not is_databricks_configured():
        print("Databricks not configured, skipping insights upload.")
        return False

    conn = get_connection()
    cursor = conn.cursor()

    table = f"{get_table_prefix()}.meeting_insights"
    now = datetime.now().isoformat()
    insight_id = f"{source_name}_{now}"

    def to_json_str(key):
        data = insights.get(key, [])
        return json.dumps(data).replace("'", "''")

    cursor.execute(f"""
        INSERT INTO {table}
        VALUES (
            '{insight_id}',
            '{source_name}',
            '{to_json_str("action_items")}',
            '{to_json_str("decisions")}',
            '{to_json_str("key_discussion_points")}',
            '{to_json_str("participant_contributions")}',
            '{now}'
        )
    """)

    cursor.close()
    conn.close()
    print(f"  Insights saved to Databricks for {source_name}.")
    return True


def get_all_chunks(source_name=None):
    """Read chunks back from Databricks.

    Args:
        source_name: optional filter by source file

    Returns:
        list of dicts, or empty list if not configured
    """
    if not is_databricks_configured():
        return []

    conn = get_connection()
    cursor = conn.cursor()

    table = f"{get_table_prefix()}.transcript_chunks"

    if source_name:
        safe = source_name.replace("'", "''")
        cursor.execute(f"SELECT * FROM {table} WHERE source_file = '{safe}' ORDER BY chunk_index")
    else:
        cursor.execute(f"SELECT * FROM {table} ORDER BY ingested_at DESC")

    columns = [desc[0] for desc in cursor.description]
    rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

    cursor.close()
    conn.close()
    return rows


def get_all_insights(source_name=None):
    """Read insights back from Databricks.

    Args:
        source_name: optional filter by source file

    Returns:
        list of dicts with parsed JSON fields, or empty list
    """
    if not is_databricks_configured():
        return []

    conn = get_connection()
    cursor = conn.cursor()

    table = f"{get_table_prefix()}.meeting_insights"

    if source_name:
        safe = source_name.replace("'", "''")
        cursor.execute(f"SELECT * FROM {table} WHERE source_file = '{safe}' ORDER BY extracted_at DESC")
    else:
        cursor.execute(f"SELECT * FROM {table} ORDER BY extracted_at DESC")

    columns = [desc[0] for desc in cursor.description]
    rows = []
    for row in cursor.fetchall():
        record = dict(zip(columns, row))
        for key in ["action_items", "decisions", "key_discussion_points", "participant_contributions"]:
            if key in record and isinstance(record[key], str):
                try:
                    record[key] = json.loads(record[key])
                except json.JSONDecodeError:
                    pass
        rows.append(record)

    cursor.close()
    conn.close()
    return rows


# ---- run directly to test ----
if __name__ == "__main__":
    if not is_databricks_configured():
        print("Databricks is NOT configured.")
        print("Fill in DATABRICKS_HOST, DATABRICKS_TOKEN, and DATABRICKS_HTTP_PATH in .env")
        print("Then run this file again.")
    else:
        print("Databricks is configured. Creating tables...")
        create_tables()
        print("Done! Tables are ready.")
