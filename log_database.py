import sqlite3, os
from datetime import datetime, timezone
from smolagents import Tool
DB_PATH = os.environ.get("UNANSWERED_QUERIES_DB_PATH", "unanswered_queries.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''  
            CREATE TABLE IF NOT EXISTS  unanswered_queries (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 query TEXT NOT NULL,
                 timestamp TEXT NOT NULL,
                 session_id TEXT,
                 statusTEXT DEFAULT unresolved)
                 ''')
    conn.commit()
    conn.close()



#tool

class Log_unanswered_query(Tool):
    name = "log_unanswered_query"
    description = """
    Call this tool whenever the retrieved documents do not provide a satisfactory answer to the user's query, BEFORE replying with
    the fallback message. Pass the user's original query as is. 
    This does not return any output, but will log the query so the knowledge base gap can be improved later.
    """
    
    inputs={
        "query":{
            "type":"string",
            "description":"The original user query that was not answered satisfactorily."
        },
        "session_id":{
            "type":"string",
            "description":"The session ID of the user query.", "nullable": True
        }
    }
    output_type = "string"

    
    def __init__(self):
        super().__init__()
        init_db()

    def forward(self, query: str, session_id: str = None) ->str:
        try:
            conn = sqlite3.connect(DB_PATH)
            timestamp = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO unanswered_queries (query, timestamp, session_id) VALUES (?, ?, ?)",
                (query, timestamp, session_id)
            )
            conn.commit()
            conn.close()
            return "Query logged for future improvement"
        except Exception as e:
            return f"Error logging query: {str(e)}"