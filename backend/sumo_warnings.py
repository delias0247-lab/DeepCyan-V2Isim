import asyncio
from collections import deque
from typing import Deque, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, JSONResponse

router = APIRouter()

WARN_BUFFER_MAX = 5000
warnings_buffer: Deque[str] = deque(maxlen=WARN_BUFFER_MAX)
warnings_queue: "asyncio.Queue[str]" = asyncio.Queue()
_loop: Optional[asyncio.AbstractEventLoop] = None

def init(loop: asyncio.AbstractEventLoop):
    global _loop
    _loop = loop

def push_line(line: str) -> None:
    if not line:
        return
    line = line.rstrip("\n")
    if not line:
        return

    # Keep only SUMO warnings (change if you want everything)
    if "Warning:" not in line and not line.startswith("Warning:"):
        return

    warnings_buffer.append(line)

    # push to SSE queue from thread
    try:
        if _loop:
            _loop.call_soon_threadsafe(warnings_queue.put_nowait, line)
    except Exception:
        pass

@router.get("/warnings/recent")
def recent(limit: int = 200):
    limit = max(1, min(int(limit), 2000))
    items = list(warnings_buffer)[-limit:]
    return JSONResponse({"count": len(items), "items": items})

@router.get("/warnings/stream")
async def stream():
    async def event_gen():
        yield "event: status\ndata: connected\n\n"
        while True:
            line = await warnings_queue.get()
            yield f"data: {line}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")