"""A tiny fake calendar as an MCP server: three tools, one JSON file.

Every call is appended to calls.jsonl with the model's raw arguments, before any
validation, so the presentation site can show exactly what the model sent.
Started by an MCP client (Claude Code, Copilot) via .mcp.json.
"""

import json
import os
from datetime import date as Date, datetime
from pathlib import Path
from typing import Annotated

from pydantic import Field
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# CALENDAR_DATA lets test runs use their own copy instead of the one the site shows
DATA = Path(os.environ.get("CALENDAR_DATA", Path(__file__).resolve().parent.parent / "site" / "data" / "calendar"))
CALENDAR = DATA / "calendar.json"
CALLS = DATA / "calls.jsonl"


class LoggingServer(MCPServer):
    """An MCPServer that records every tools/call: tool name, raw arguments, result or error."""

    async def call_tool(self, name, arguments, context=None):
        entry = {"at": datetime.now().isoformat(timespec="seconds"), "tool": name, "arguments": arguments}
        try:
            result = await super().call_tool(name, arguments, context)
            # what the model gets back: structured content if there is any, else the text blocks
            data = result.structured_content
            if data is None:
                data = [json.loads(c.text) for c in result.content]
                data = data[0] if len(data) == 1 else data
            entry["result"] = data.get("result", data) if isinstance(data, dict) and list(data) == ["result"] else data
        except ToolError as e:
            entry["error"] = str(e)
            raise
        finally:
            with CALLS.open("a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return result


mcp = LoggingServer("calendar")

# What the model may send. These become the input schema.
Day = Annotated[str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$",
                           description="The day, as YYYY-MM-DD")]
Time = Annotated[str, Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$",
                            description="Start time, 24-hour HH:MM")]
Title = Annotated[str, Field(min_length=1, max_length=80,
                             description="Short title of the appointment")]
Minutes = Annotated[int, Field(ge=15, le=480,
                               description="Length in minutes")]
Email = Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
                             description="Recipient address")]


def minutes(hhmm):
    h, m = map(int, hhmm.split(":"))
    return h * 60 + m


def check_day(day):
    try:
        Date.fromisoformat(day)
    except ValueError:
        raise ToolError(f"{day} is not a real date")


def get_appointments(date: Day) -> list[dict]:
    """List all appointments on one day, sorted by start time, with their notes."""
    check_day(date)
    return sorted((a for a in json.loads(CALENDAR.read_text()) if a["date"] == date), key=lambda a: a["time"])


def add_appointment(date: Day, time: Time, title: Title, duration_minutes: Minutes = 60) -> dict:
    """Book an appointment on one day. Fails if it overlaps an existing appointment."""
    check_day(date)
    calendar = json.loads(CALENDAR.read_text())
    start, end = minutes(time), minutes(time) + duration_minutes
    for a in calendar:
        if a["date"] == date and start < minutes(a["time"]) + a["duration_minutes"] and minutes(a["time"]) < end:
            raise ToolError(f"Overlaps with '{a['title']}' at {a['time']} on {date}")
    new = {"date": date, "time": time, "title": title, "duration_minutes": duration_minutes, "added_by": "agent"}
    calendar.append(new)
    CALENDAR.write_text(json.dumps(calendar, ensure_ascii=False, indent=1))
    return new


def send_email(to: Email, subject: str, body: str) -> dict:
    """Send an email from the user's account."""
    # Fake: nothing leaves this machine. The call log is the only record.
    return {"status": "sent", "to": to}


# Register the functions as tools: name, docstring and type hints become what the model sees in tools/list.
mcp.add_tool(get_appointments)
mcp.add_tool(add_appointment)
mcp.add_tool(send_email)

if __name__ == "__main__":
    mcp.run()