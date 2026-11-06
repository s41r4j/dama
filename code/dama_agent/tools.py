"""The Memory Agent's tool surface for the Primary LLM (MCP-shaped: name, description, JSON-schema input).

`packet` is not a tool: the runtime pushes it into the prompt before the Primary LLM starts. These are the pulls
and explicit writes the Primary LLM may call while answering. `dispatch` maps a tool call to the agent.
"""
from __future__ import annotations

TOOLS = [
    {"name": "recall", "description": "Search long-term memory for more than the pushed packet contained. Returns time-ordered "
                                      "memories with [m:ID] citations, or NOT_FOUND.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "default": 16}},
                      "required": ["query"]}},
    {"name": "timeline", "description": "How a fact changed over time: matching facts, including superseded and retracted ones, oldest first.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "note", "description": "Store something the user explicitly asked to be remembered (bypasses the salience gate).",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "observe", "description": "Record one conversation turn (normally done by the runtime, not the model).",
     "input_schema": {"type": "object", "properties": {"text": {"type": "string"}, "speaker": {"type": "string"},
                                                       "ts": {"type": "string"}}, "required": ["text"]}},
    {"name": "commit", "description": "Process queued turns through the write path (the runtime calls this after each reply).",
     "input_schema": {"type": "object", "properties": {}}},
]


def dispatch(agent, scope: str, name: str, args: dict):
    if name == "recall":
        r = agent.recall(scope, args["query"], k=int(args.get("k", 16)))
        return {"text": r["text"], "ids": r["ids"]}
    if name == "timeline":
        return agent.timeline(scope, args["query"])
    if name == "note":
        return agent.note(scope, args["text"], speaker=args.get("speaker", "user"))
    if name == "observe":
        return agent.observe(scope, args["text"], speaker=args.get("speaker", "user"), ts=args.get("ts"), defer=True)
    if name == "commit":
        return agent.commit()
    raise KeyError(f"unknown tool {name}")
