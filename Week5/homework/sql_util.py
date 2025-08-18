import sqlite3
import settings as cfg

def init_documents_db():
    """Initialize documents SQLite database with schema"""
    conn = sqlite3.connect(cfg.DOCS_OUT)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            title TEXT,
            authors TEXT, -- JSON string of author list
            published TEXT,
            primary_category TEXT,
            n_pages INTEGER,
            source_pdf TEXT,
            text TEXT,
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    
def init_chunks_db():
    """Initialize chunks SQLite database with schema"""
    conn = sqlite3.connect(cfg.CHUNKS_OUT)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT,
            page INTEGER,
            text TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    
    
def init_chunk_meta_db():
    """Initialize chunk metadata SQLite database with schema"""
    conn = sqlite3.connect(cfg.SIDE_CAR_PATH)
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