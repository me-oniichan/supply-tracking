import json
import os
import re
import sqlite3
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import AsyncOpenAI
import asyncio

load_dotenv()

# FastAPI app setup
app = FastAPI(title="Supply Chain Chatbot API")
app.mount("/assets", StaticFiles(directory="dist/assets"), name="static")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DEFAULT_MODEL = "kimi-k2.5"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    user_input: str

class ChatResponse(BaseModel):
    response: str

# Database tools
def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default

class SQLiteDataTools:
    def __init__(self) -> None:
        self.db_path = os.getenv("SQLITE_DB_PATH", "supply_chain.db")
        self.max_rows = get_env_int("MAX_QUERY_ROWS", 100)

    def connect(self) -> sqlite3.Connection:
        # uri=True allows read-only mode if desired, but here we just connect standard
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_database_schema(self) -> dict[str, Any]:
        """Return tables, columns, and data types from the SQLite database."""
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        tables_dict = {}

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query)
            tables = [row["name"] for row in cur.fetchall()]

            for table in tables:
                cur.execute(f"PRAGMA table_info('{table}');")
                columns = cur.fetchall()
                tables_dict[table] = [
                    {
                        "column": col["name"],
                        "type": col["type"],
                        "nullable": not col["notnull"]
                    }
                    for col in columns
                ]

        return {"tables": tables_dict}

    def describe_table(self, table_name: str) -> dict[str, Any]:
        """Return columns, indexes, and foreign keys for one table."""
        self._validate_identifier(table_name, "table_name")

        with self.connect() as conn:
            cur = conn.cursor()
            
            cur.execute(f"PRAGMA table_info('{table_name}');")
            columns = cur.fetchall()
            
            cur.execute(f"PRAGMA index_list('{table_name}');")
            indexes = cur.fetchall()
            
            cur.execute(f"PRAGMA foreign_key_list('{table_name}');")
            foreign_keys = cur.fetchall()

        return {
            "table": table_name,
            "columns": [
                {
                    "column": c["name"],
                    "type": c["type"],
                    "nullable": not c["notnull"],
                    "default": c["dflt_value"]
                }
                for c in columns
            ],
            "indexes": [
                {"name": i["name"]} for i in indexes
            ],
            "foreign_keys": [
                {
                    "column": fk["from"],
                    "referenced_table": fk["table"],
                    "referenced_column": fk["to"]
                }
                for fk in foreign_keys
            ],
        }

    def sample_table(self, table_name: str, limit: int = 5) -> dict[str, Any]:
        """Return a small sample from a table."""
        self._validate_identifier(table_name, "table_name")
        limit = min(max(int(limit), 1), 20)
        query = f'SELECT * FROM "{table_name}" LIMIT ?'

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []

            return {
                "columns": columns,
                "rows": [dict(row) for row in rows]
            }

    def run_read_only_query(self, sql: str) -> dict[str, Any]:
        """Run a safe read-only SQL query and return capped results."""
        clean_sql = self._validate_read_only_sql(sql)
        limited_sql = f"SELECT * FROM ({clean_sql}) LIMIT ?"

        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(limited_sql, (self.max_rows,))
            rows = cur.fetchall()
            columns = [desc[0] for desc in cur.description] if cur.description else []

            return {
                "columns": columns,
                "rows": [dict(row) for row in rows],
                "max_rows": self.max_rows,
                "row_count": len(rows)
            }

    def _validate_identifier(self, value: str, name: str) -> None:
        if not IDENTIFIER_RE.match(value):
            raise ValueError(f"Invalid {name}: {value!r}")

    def _validate_read_only_sql(self, sql: str) -> str:
        clean_sql = sql.strip()
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()
        if ";" in clean_sql:
            raise ValueError("Only one SQL statement is allowed")
        if not re.match(r"^(select|with)\b", clean_sql, re.IGNORECASE):
            raise ValueError("Only SELECT or WITH queries are allowed")
        return clean_sql

# Tools definition
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_database_schema",
            "description": "Get the available SQLite tables, columns, data types, and nullability for the supply-chain dataset.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Get detailed metadata for one table, including columns, indexes, and foreign keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Table name, for example shipments or inventory.",
                    }
                },
                "required": ["table_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sample_table",
            "description": "Read a small sample of rows from a table to understand its values and shape.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Table name."},
                    "limit": {"type": "integer", "description": "Number of rows to sample, from 1 to 20."},
                },
                "required": ["table_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_read_only_query",
            "description": "Run one read-only SQL query against SQLite. Use only SELECT or WITH queries. Prefer explicit table aliases and include business-friendly aggregates for answers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT or WITH query. Do not include destructive statements.",
                    }
                },
                "required": ["sql"],
                "additionalProperties": False,
            },
        },
    },
]

SYSTEM_PROMPT = """
You are a supply-chain analytics chatbot connected to a local SQLite database.

You have access to the following table:
- **supply_chain_data** with columns:
  - product_type VARCHAR(100)
  - sku VARCHAR(50)
  - price DECIMAL
  - availability INT
  - number_of_products_sold INT
  - revenue_generated DECIMAL
  - customer_demographics VARCHAR(50)
  - stock_levels INT
  - lead_times INT
  - order_quantities INT
  - shipping_times INT
  - shipping_carrier VARCHAR(100)
  - shipping_costs DECIMAL
  - supplier_name VARCHAR(100)
  - location VARCHAR(100)
  - lead_time_alt INT
  - production_volumes INT
  - manufacturing_lead_time INT
  - manufacturing_costs DECIMAL
  - inspection_results VARCHAR(50)
  - defect_rates DECIMAL
  - transportation_modes VARCHAR(50)
  - routes VARCHAR(50)
  - costs DECIMAL

Answer using the database tools whenever the user asks about data, metrics, trends, rankings, exceptions, suppliers, orders, inventory, shipments, demand, warehouses, lead times, costs, fulfillment, production, manufacturing, defects, transportation, or forecasts.

Rules:
- Use read-only SQL only. Query the supply_chain_data table directly for analysis.
- Prefer concise answers with the important numbers, rankings, and business interpretation.
- If data is missing or the schema does not support the question, say so and explain what column would be needed.
- Do not invent column names, metrics, or results.
""".strip()

def call_tool(tool_name: str, arguments: dict[str, Any], data_tools: SQLiteDataTools) -> dict[str, Any]:
    if tool_name == "get_database_schema":
        return data_tools.get_database_schema()
    if tool_name == "describe_table":
        return data_tools.describe_table(arguments["table_name"])
    if tool_name == "sample_table":
        return data_tools.sample_table(arguments["table_name"], arguments.get("limit", 5))
    if tool_name == "run_read_only_query":
        return data_tools.run_read_only_query(arguments["sql"])
    raise ValueError(f"Unknown tool: {tool_name}")

async def stream_agent_response(
    client: AsyncOpenAI,
    data_tools: SQLiteDataTools,
    messages: list[dict[str, Any]]
) -> AsyncGenerator[str, None]:
    """Stream the agent response token by token."""
    model = os.getenv("DIGITALOCEAN_MODEL", DEFAULT_MODEL)
    
    for iteration in range(8):
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            stream=True,
        )
        
        # Collect the full message to check for tool calls
        message_content = ""
        tool_calls = []
        async for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                message_content += content
                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        # Check if we need to handle tool calls
        # Re-fetch the full message to get tool calls
        if iteration < 7:  # Not the last iteration
            full_response = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
                stream=False,
            )
            message = full_response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            
            if not message.tool_calls:
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            
            # Process tool calls
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    result = call_tool(function_name, args, data_tools)
                    yield f"data: {json.dumps({'type': 'tool_call', 'tool': function_name, 'args': args})}\n\n"
                except Exception as exc:
                    result = {"error": str(exc)}
                
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )
        else:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    """Chat endpoint with streaming support."""
    try:
        do_api_key = os.getenv("DIGITALOCEAN_API_KEY")
        do_base_url = os.getenv("DIGITALOCEAN_BASE_URL")
        
        if not do_api_key or not do_base_url:
            raise HTTPException(
                status_code=500,
                detail="DigitalOcean API configuration missing. Set DIGITALOCEAN_API_KEY and DIGITALOCEAN_BASE_URL in .env"
            )
        
        client = AsyncOpenAI(api_key=do_api_key, base_url=do_base_url)
        data_tools = SQLiteDataTools()
        
        # Build messages
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.user_input})
        
        return StreamingResponse(
            stream_agent_response(client, data_tools, messages),
            media_type="text/event-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/{full_path:path}")
async def serve_react_app(full_path: str):
    # Check if the file exists in the dist folder
    file_path = os.path.join("dist", full_path)
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # Fallback to index.html for SPA routing
    return FileResponse("dist/index.html")

@app.post("/chat/simple")
async def chat_simple(request: ChatRequest):
    """Non-streaming chat endpoint (fallback)."""
    try:
        do_api_key = os.getenv("DIGITALOCEAN_API_KEY")
        do_base_url = os.getenv("DIGITALOCEAN_BASE_URL")
        
        if not do_api_key or not do_base_url:
            raise HTTPException(
                status_code=500,
                detail="DigitalOcean API configuration missing"
            )
        
        client = AsyncOpenAI(api_key=do_api_key, base_url=do_base_url)
        data_tools = SQLiteDataTools()
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": request.user_input})
        
        # Non-streaming version
        for _ in range(8):
            response = await client.chat.completions.create(
                model=os.getenv("DIGITALOCEAN_MODEL", DEFAULT_MODEL),
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            
            if not message.tool_calls:
                return ChatResponse(response=message.content or "I could not produce an answer.")
            
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                    result = call_tool(function_name, args, data_tools)
                except Exception as exc:
                    result = {"error": str(exc)}
                
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, default=str),
                    }
                )
        
        return ChatResponse(response="I reached the tool-calling limit while analyzing the data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)