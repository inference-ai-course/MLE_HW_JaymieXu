import sqlite3
from typing import List, Dict, Any
from rag2_config import cfg, get_rag2_config_path

    
def init_chunks_db():
    """Initialize chunks SQLite database with schema"""
    conn = sqlite3.connect(get_rag2_config_path("proc") / cfg.files.chunks_file)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            text TEXT
        )
    ''')
    
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id,
            doc_id,
            text,
            content='chunks',
            content_rowid='rowid'
        )
    ''')
    
    conn.commit()
    conn.close()
    
    
def init_chunk_meta_db():
    """Initialize chunk metadata SQLite database with schema"""
    conn = sqlite3.connect(get_rag2_config_path("indexed") / cfg.files.side_car)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunk_meta (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            title TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_chunks_db()
init_chunk_meta_db()