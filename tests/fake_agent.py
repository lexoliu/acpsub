#!/usr/bin/env python3
"""Fake ACP agent over stdio for acpsub tests.

Answers initialize / session/new / session/load / session/set_mode /
session/set_config_option. On session/prompt it streams a thought chunk, a
plan, message chunks, and a tool_call with tool_call_update updates, then:

- "PERMISSION" in the prompt -> a session/request_permission request whose
  outcome is recorded into the reply.
- "READ <path>" in the prompt -> an fs/read_text_file request; the content's
  first 40 chars (or the error) are recorded into the reply.
- "RUN <cmd>" in the prompt -> terminal/create, wait_for_exit, output; the
  output and exit code are recorded into the reply.
- "wait" -> the prompt never completes until session/cancel.
- "die" -> the process exits mid-turn with status 3.

session/cancel answers the pending prompt with stopReason cancelled.
FAKE_NO_LOAD=1 in the environment makes the agent not advertise loadSession.
"""

import json
import os
import shlex
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def respond(request_id, result):
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def update(session_id, update_payload):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update_payload},
        }
    )


pending_prompt = None
session_id = ""
session_count = 0
next_req = 0
outbound = {}       # request id -> kind: perm | fs | term-create | term-wait | term-output
terminal_id = None
collected = []      # markers appended to the final reply


def new_request(method, params, kind):
    global next_req
    request_id = f"agent-{next_req}"
    next_req += 1
    outbound[request_id] = kind
    send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def finish_prompt():
    global pending_prompt
    if pending_prompt is None or outbound:
        return
    update(
        session_id,
        {"sessionUpdate": "tool_call_update", "toolCallId": "tc-1", "status": "completed"},
    )
    text = "world" + (" " + " ".join(collected) if collected else "")
    update(
        session_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": text},
        },
    )
    respond(pending_prompt, {"stopReason": "end_turn"})
    pending_prompt = None


def on_prompt(request_id, params):
    global pending_prompt, session_id, collected
    session_id = params.get("sessionId", session_id)
    text = (params.get("prompt") or [{}])[0].get("text", "")
    if text == "die":
        sys.exit(3)
    pending_prompt = request_id
    collected = []
    update(
        session_id,
        {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"type": "text", "text": "thinking about it"},
        },
    )
    update(
        session_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "Hello "},
        },
    )
    update(
        session_id,
        {
            "sessionUpdate": "plan",
            "entries": [
                {"content": "do the thing", "status": "in_progress"},
                {"content": "report back", "status": "pending"},
            ],
        },
    )
    update(
        session_id,
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-1",
            "title": "fake-tool",
            "kind": "execute",
            "status": "pending",
            "locations": [{"path": "/tmp/fake.py", "line": 1}],
            "rawInput": {"cmd": "fake"},
        },
    )
    update(
        session_id,
        {"sessionUpdate": "tool_call_update", "toolCallId": "tc-1", "status": "in_progress"},
    )
    if text == "wait":
        return
    if "PERMISSION" in text:
        new_request(
            "session/request_permission",
            {
                "sessionId": session_id,
                "toolCall": {
                    "toolCallId": "tc-1",
                    "title": "fake-tool",
                    "status": "in_progress",
                },
                "options": [
                    {"optionId": "allow-1", "name": "Allow once", "kind": "allow_once"},
                    {"optionId": "allow-2", "name": "Allow always", "kind": "allow_always"},
                    {"optionId": "deny-1", "name": "Deny", "kind": "reject_once"},
                ],
            },
            "perm",
        )
    if text.startswith("READ "):
        path = text[5:].strip()
        new_request(
            "fs/read_text_file",
            {"sessionId": session_id, "path": path},
            "fs",
        )
    if text.startswith("RUNLINE "):
        new_request(
            "terminal/create",
            {
                "sessionId": session_id,
                "command": text[8:].strip(),
                "outputByteLimit": 4096,
            },
            "term-create",
        )
    elif text.startswith("RUN "):
        argv = shlex.split(text[4:].strip())
        new_request(
            "terminal/create",
            {
                "sessionId": session_id,
                "command": argv[0],
                "args": argv[1:],
                "outputByteLimit": 4096,
            },
            "term-create",
        )
    update(
        session_id,
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "ab"},
        },
    )
    finish_prompt()


def on_response(msg):
    global terminal_id
    request_id = msg.get("id")
    kind = outbound.pop(request_id, None)
    if kind is None:
        return
    if kind == "perm":
        if "result" in msg:
            outcome = msg["result"].get("outcome", {})
            collected.append("perm:" + str(outcome.get("optionId", "cancelled")))
        else:
            collected.append("perm:error")
    elif kind == "fs":
        if "result" in msg:
            collected.append("fs:" + msg["result"].get("content", "")[:40].replace("\n", "\\n"))
        else:
            collected.append("fserr:" + msg.get("error", {}).get("message", "?")[:60])
    elif kind == "term-create":
        if "result" in msg:
            terminal_id = msg["result"]["terminalId"]
            new_request(
                "terminal/wait_for_exit",
                {"sessionId": session_id, "terminalId": terminal_id},
                "term-wait",
            )
        else:
            collected.append("termerr:" + msg.get("error", {}).get("message", "?"))
    elif kind == "term-wait":
        code = msg.get("result", {}).get("exitCode")
        collected.append("exit:" + str(code))
        new_request(
            "terminal/output",
            {"sessionId": session_id, "terminalId": terminal_id},
            "term-output",
        )
    elif kind == "term-output":
        output = msg.get("result", {}).get("output", "")
        collected.append("term:" + output.strip()[:40])
        new_request(
            "terminal/release",
            {"sessionId": session_id, "terminalId": terminal_id},
            "term-release",
        )
    elif kind == "term-release":
        pass
    finish_prompt()


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue

    if "method" in msg and "id" in msg:
        method, request_id, params = msg["method"], msg["id"], msg.get("params") or {}
        if method == "initialize":
            respond(
                request_id,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": os.environ.get("FAKE_NO_LOAD") != "1",
                    },
                    "agentInfo": {"name": "fake-agent-py", "version": "0.1.0"},
                },
            )
        elif method == "session/new":
            session_count += 1
            session_id = f"sess-{session_count}"
            respond(
                request_id,
                {
                    "sessionId": session_id,
                    "modes": {
                        "currentModeId": "default",
                        "availableModes": [
                            {"id": "default", "name": "Default"},
                            {"id": "bypass", "name": "Bypass"},
                        ],
                    },
                    "configOptions": [
                        {
                            "id": "model",
                            "name": "Model",
                            "type": "select",
                            "currentValue": "a",
                            "options": [
                                {"value": "a", "name": "A"},
                                {"value": "b", "name": "B"},
                            ],
                        }
                    ],
                },
            )
        elif method == "session/load":
            session_id = params.get("sessionId", session_id)
            update(
                session_id,
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "loaded user message"},
                },
            )
            respond(
                request_id,
                {
                    "modes": {
                        "currentModeId": "default",
                        "availableModes": [{"id": "default", "name": "Default"}],
                    }
                },
            )
        elif method == "session/set_mode":
            respond(request_id, {})
        elif method == "session/set_config_option":
            respond(
                request_id,
                {
                    "configOptions": [
                        {
                            "id": params["configId"],
                            "name": "Model",
                            "type": "select",
                            "currentValue": params["value"],
                            "options": [],
                        }
                    ]
                },
            )
        elif method == "session/prompt":
            on_prompt(request_id, params)
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
    elif "method" in msg:
        if msg["method"] == "session/cancel" and pending_prompt is not None:
            respond(pending_prompt, {"stopReason": "cancelled"})
            pending_prompt = None
            outbound.clear()
    elif "id" in msg:
        on_response(msg)
