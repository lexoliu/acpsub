#!/usr/bin/env python3
"""Fake ACP agent over stdio for acpsub tests.

Answers initialize / session/new / session/load / session/set_mode /
session/set_config_option. On session/prompt it streams a thought chunk, a
plan, message chunks, and a tool_call with tool_call_update updates, then:

- "PERMISSION" in the prompt -> a session/request_permission request whose
  outcome is recorded into the reply.
- "READ <path>" in the prompt -> an fs/read_text_file request; the content's
  first 40 chars (or the error) are recorded into the reply.
- "WRITE <path>" in the prompt -> an fs/write_text_file request; "fsw:ok" or
  the error is recorded into the reply.
- "RUN <cmd>" / "RUNLINE <cmd>" in the prompt -> terminal/create,
  wait_for_exit, output; the output, exit code, signal, and truncation flag
  are recorded into the reply.
- "KILL <cmd>" -> terminal/create, terminal/kill, then wait_for_exit and
  output; the kill's signal exit is recorded into the reply.
- "wait" -> the prompt never completes until session/cancel.
- "gather" -> the prompt completes once a steered prompt arrives; the
  steered text is recorded into the reply.
- "die" -> the process exits mid-turn with status 3.

A session/prompt that arrives while a prompt is pending is a steer: its
request resolves with the same result as the turn's own prompt, and its
text joins the turn via "steer:<text>" in the reply. FAKE_NO_STEER=1 makes
the agent reject a concurrent prompt instead. FAKE_QUEUE_PROMPTS=1 makes
the agent park a concurrent prompt and run it as its own turn after the
current one ends (agents that serialize prompts instead of injecting).

- "hold" -> the turn stays open until a queued prompt arrives
  (FAKE_QUEUE_PROMPTS mode only).

session/cancel answers the pending prompt (and every steer) with
stopReason cancelled.
FAKE_NO_LOAD=1 in the environment makes the agent not advertise loadSession.
Session ids are `sess-<pid>-<n>` so concurrently spawned agents never
collide.
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
pending_text = ""
steered = []        # request ids of prompts injected mid-turn
queued = []         # (request_id, text) parked in FAKE_QUEUE_PROMPTS mode
session_id = ""
session_count = 0
next_req = 0
outbound = {}       # request id -> kind: perm | fs | fsw | term-create |
                    #   term-kill-create | term-kill | term-wait |
                    #   term-output | term-release
terminal_id = None
collected = []      # markers appended to the final reply


def new_request(method, params, kind):
    global next_req
    request_id = f"agent-{next_req}"
    next_req += 1
    outbound[request_id] = kind
    send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})


def answer_all(result):
    """Resolve the pending prompt and every steer with the same result, then
    start the next agent-side queued prompt, if any."""
    global pending_prompt
    for steer_id in steered:
        respond(steer_id, result)
    steered.clear()
    if pending_prompt is not None:
        respond(pending_prompt, result)
        pending_prompt = None
    if pending_prompt is None and queued:
        request_id, text = queued.pop(0)
        begin_turn(request_id, text)


def finish_prompt():
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
    answer_all({"stopReason": "end_turn"})


def begin_turn(request_id, text):
    """Open a turn for request_id and stream the standard updates."""
    global pending_prompt, pending_text, collected
    pending_prompt = request_id
    pending_text = text
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
    if text in ("wait", "gather", "hold"):
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
    if text.startswith("WRITE "):
        new_request(
            "fs/write_text_file",
            {
                "sessionId": session_id,
                "path": text[6:].strip(),
                "content": "written-by-fake-agent",
            },
            "fsw",
        )
    if text.startswith("KILL "):
        argv = shlex.split(text[5:].strip())
        new_request(
            "terminal/create",
            {
                "sessionId": session_id,
                "command": argv[0],
                "args": argv[1:],
                "outputByteLimit": 4096,
            },
            "term-kill-create",
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


def on_prompt(request_id, params):
    global session_id
    session_id = params.get("sessionId", session_id)
    text = (params.get("prompt") or [{}])[0].get("text", "")
    if text == "die":
        sys.exit(3)
    if pending_prompt is not None:
        # A prompt arriving while a turn runs is a steer injection — unless
        # FAKE_NO_STEER=1 rejects it, or FAKE_QUEUE_PROMPTS=1 parks it
        # agent-side to run as its own turn after this one ends.
        if os.environ.get("FAKE_NO_STEER") == "1":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32600, "message": "prompt in flight"},
                }
            )
            return
        if os.environ.get("FAKE_QUEUE_PROMPTS") == "1":
            queued.append((request_id, text))
            if pending_text == "hold":
                finish_prompt()
            return
        steered.append(request_id)
        collected.append("steer:" + text)
        update(
            session_id,
            {
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": text},
            },
        )
        if pending_text == "gather":
            finish_prompt()
        return
    begin_turn(request_id, text)


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
    elif kind == "fsw":
        if "result" in msg:
            collected.append("fsw:ok")
        else:
            collected.append("fswerr:" + msg.get("error", {}).get("message", "?"))
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
    elif kind == "term-kill-create":
        if "result" in msg:
            terminal_id = msg["result"]["terminalId"]
            new_request(
                "terminal/kill",
                {"sessionId": session_id, "terminalId": terminal_id},
                "term-kill",
            )
        else:
            collected.append("termerr:" + msg.get("error", {}).get("message", "?"))
    elif kind == "term-kill":
        if "result" in msg:
            new_request(
                "terminal/wait_for_exit",
                {"sessionId": session_id, "terminalId": terminal_id},
                "term-wait",
            )
        else:
            collected.append("killerr:" + msg.get("error", {}).get("message", "?"))
    elif kind == "term-wait":
        result = msg.get("result", {})
        collected.append("exit:" + str(result.get("exitCode")))
        if result.get("signal") is not None:
            collected.append("signal:" + str(result["signal"]))
        new_request(
            "terminal/output",
            {"sessionId": session_id, "terminalId": terminal_id},
            "term-output",
        )
    elif kind == "term-output":
        result = msg.get("result", {})
        output = result.get("output", "")
        collected.append("term:" + output.strip()[:40])
        if result.get("truncated"):
            collected.append("trunc")
            collected.append("tail:" + output.strip()[-40:])
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
            session_id = f"sess-{os.getpid()}-{session_count}"
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
            answer_all({"stopReason": "cancelled"})
            outbound.clear()
    elif "id" in msg:
        on_response(msg)
