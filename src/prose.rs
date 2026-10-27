//! Prose renderings of the tools' structured results — the default
//! output of `acpsub serve`, `acpsub daemon`, and the client commands.
//!
//! A tool call's `json` argument (the CLI's `--json`) returns the
//! structured `Value` instead; every function here renders exactly that
//! value, so the two forms can never disagree. `wait` also takes the
//! call's `expect_secs` — its still-running paragraph compares the
//! elapsed time against it — because the view does not carry it.

use std::fmt::Write as _;

use serde_json::Value;

/// The string `view` carries at `key`. Every renderer reads only the
/// fields its tool's view always writes; a missing one is a bug in the
/// view, not a case to render around.
fn str_field<'a>(view: &'a Value, key: &str) -> &'a str {
    view[key]
        .as_str()
        .unwrap_or_else(|| panic!("the view carries {key:?} as a string"))
}

/// `key`'s string when the view carries it.
fn opt_str<'a>(view: &'a Value, key: &str) -> Option<&'a str> {
    view.get(key).and_then(Value::as_str)
}

/// `""` for one, `"s"` for more.
const fn plural(n: u64) -> &'static str {
    if n == 1 { "" } else { "s" }
}

/// `list`'s id array joined for prose.
fn ids(list: &Value) -> Vec<&str> {
    list.as_array()
        .expect("the view carries an array")
        .iter()
        .filter_map(Value::as_str)
        .collect()
}

/// `wait`/`wait_any`: a turn that ended is one outcome line — session,
/// turn and outcome — followed by the turn's final assistant message
/// verbatim.
///
/// A turn still running (`overrun`, or a `max_wait_secs` expiry's
/// `running`) is one short paragraph naming the session and turn, the
/// elapsed time against `expect_secs`, and `acpsub transcript` as the
/// place to look.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn wait(view: &Value, expect_secs: u64) -> String {
    let session = str_field(view, "session_id");
    let state = str_field(view, "state");
    let turn = view["turn"].as_u64();
    if matches!(state, "overrun" | "running") {
        let elapsed = view["elapsed_secs"]
            .as_f64()
            .expect("a running view carries elapsed_secs");
        let turn = turn.map_or_else(String::new, |n| format!(" turn {n}"));
        let mut text = format!(
            "Session {session}{turn} is still running after {elapsed:.1}s \
             against an --expect of {expect_secs}s — `acpsub transcript \
             {session}` shows what it is doing."
        );
        // A parked session's report carries where the limit stands.
        if let Some(resume_at) = opt_str(view, "resume_at") {
            let _ = write!(
                text,
                " It is rate-limited until {resume_at}: {}.",
                str_field(view, "reason")
            );
        }
        text.push('\n');
        return text;
    }
    let mut text = turn.map_or_else(
        || format!("Session {session}: {state}"),
        |n| format!("Session {session}, turn {n}: {state}"),
    );
    if let Some(error) = opt_str(view, "error") {
        let _ = write!(text, " — {error}");
    } else if let Some(reason) = opt_str(view, "stop_reason").filter(|r| *r != state) {
        let _ = write!(text, " ({reason})");
    }
    if state == "rate_limited" {
        let _ = write!(text, " — resumes at {}", str_field(view, "resume_at"));
        if let Some(reason) = opt_str(view, "reason") {
            let _ = write!(text, ": {reason}");
        }
    }
    if let Some(pending) = view.get("pending_permission") {
        let request = str_field(pending, "request_id");
        let options = pending["options"]
            .as_array()
            .expect("the request carries its options")
            .iter()
            .map(|option| {
                format!(
                    "{} ({})",
                    str_field(option, "name"),
                    str_field(option, "option_id")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(
            text,
            " — {}\npermission request {request}: answer with `acpsub \
             permit {session} --request {request} --option OPT` — \
             options: {options}",
            str_field(&pending["tool_call"], "title"),
        );
    }
    let reply = str_field(view, "reply");
    if reply.is_empty() {
        text.push('\n');
    } else {
        let _ = write!(text, "\n\n{reply}");
        if !reply.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// `spawn`/`adopt`: the live session's handle, agent and state, where it
/// runs and where its transcript lands.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn spawned(view: &Value) -> String {
    let mut text = format!(
        "Session {} ({}) is {} — cwd {}",
        str_field(view, "session_id"),
        str_field(view, "agent"),
        str_field(view, "state"),
        str_field(view, "cwd"),
    );
    for key in ["model", "mode"] {
        if let Some(value) = opt_str(view, key) {
            let _ = write!(text, ", {key} {value}");
        }
    }
    if let Some(resume_at) = opt_str(view, "resume_at") {
        let _ = write!(
            text,
            ", rate-limited until {resume_at} ({})",
            str_field(view, "reason")
        );
    }
    let _ = write!(text, "; transcript {}", str_field(view, "transcript"));
    text.push('\n');
    text
}

/// `send`: what the prompt did — started a turn, parked on the queue, or
/// went in as a steer.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn send(view: &Value) -> String {
    let session = str_field(view, "session_id");
    match str_field(view, "state") {
        "queued" => format!(
            "Prompt queued on session {session} at position {}.\n",
            view["position"]
                .as_u64()
                .expect("a queued send carries position")
        ),
        "submitted" => format!(
            "Steer {} submitted to session {session}, targeting turn {}.\n",
            view["steer_id"].as_u64().expect("a steer carries its id"),
            view["turn"]
                .as_u64()
                .expect("a steer carries its target turn"),
        ),
        state => format!("Prompt sent to session {session}; the turn is {state}.\n"),
    }
}

/// `status`: a live session's state, where it runs and what it has done;
/// a closed one's registry facts.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn status(view: &Value) -> String {
    let mut text = format!(
        "Session {} ({}) — {}",
        str_field(view, "session_id"),
        str_field(view, "agent"),
        str_field(view, "state"),
    );
    if let Some(error) = opt_str(view, "error") {
        let _ = write!(text, " — {error}");
    }
    if let Some(resume_at) = opt_str(view, "resume_at") {
        let _ = write!(
            text,
            ", resumes at {resume_at} ({})",
            str_field(view, "reason")
        );
    }
    let turns = view["turns"].as_u64().expect("status carries turns");
    if view["live"].as_bool() == Some(true) {
        let mut facts = format!("cwd {}", str_field(view, "cwd"));
        for key in ["model", "mode"] {
            if let Some(value) = opt_str(view, key) {
                let _ = write!(facts, "; {key} {value}");
            }
        }
        if let Some(permission) = opt_str(view, "permission") {
            let _ = write!(facts, "; permission {permission}");
        }
        let _ = write!(facts, "; transcript {}", str_field(view, "transcript"));
        let _ = write!(text, "\n{facts}\n{turns} turn{}", plural(turns));
        if let Some(stop) = opt_str(view, "last_stop_reason") {
            let _ = write!(text, ", last ended {stop}");
        }
        let queued = view["queued"].as_array().map_or(0, Vec::len);
        if queued > 0 {
            let _ = write!(text, ", {queued} queued prompt{}", plural(queued as u64));
        }
        let steers = view["steers"].as_array().map_or(0, Vec::len);
        if steers > 0 {
            let _ = write!(text, ", {steers} steer{} in flight", plural(steers as u64));
        }
        let pending = ids(&view["pending_permissions"]);
        if !pending.is_empty() {
            let _ = write!(text, ", pending permissions: {}", pending.join(", "));
        }
    } else {
        let _ = write!(
            text,
            "\n{turns} turn{}, created {}, last turn {}, cwd {}",
            plural(turns),
            str_field(view, "created"),
            str_field(view, "last_turn"),
            str_field(view, "cwd"),
        );
    }
    text.push('\n');
    text
}

/// `result`: the turn's number and stop reason, then its reply verbatim.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn result(view: &Value) -> String {
    let mut text = format!(
        "Session {}, turn {}",
        str_field(view, "session_id"),
        view["turn"].as_u64().expect("result carries the turn")
    );
    if let Some(reason) = opt_str(view, "stop_reason") {
        let _ = write!(text, " ({reason})");
    }
    text.push(':');
    let reply = str_field(view, "reply");
    if reply.is_empty() {
        text.push('\n');
    } else {
        let _ = write!(text, "\n\n{reply}");
        if !reply.ends_with('\n') {
            text.push('\n');
        }
    }
    text
}

/// `cancel`: the running turn was cancelled.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn cancel(view: &Value) -> String {
    format!(
        "Session {}: turn cancelled.\n",
        str_field(view, "session_id")
    )
}

/// `permit`: the queued permission request was answered.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn permit(view: &Value) -> String {
    format!(
        "Session {}: permission request {} answered {}.\n",
        str_field(view, "session_id"),
        str_field(view, "request_id"),
        str_field(view, "option_id"),
    )
}

/// `close`: the session's process ended; its registry entry stays
/// adoptable.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn close(view: &Value) -> String {
    let mut text = format!("Session {} closed", str_field(view, "session_id"));
    if view["launch_in_flight"].as_bool() == Some(true) {
        text.push_str(" (its launch was still in flight)");
    } else if view["was_live"].as_bool() == Some(false) {
        text.push_str(" (it was not live)");
    }
    text.push_str(".\n");
    text
}

/// `forget`: the registry entry is gone, and the process with it.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn forget(view: &Value) -> String {
    let mut text = format!("Session {} forgotten", str_field(view, "session_id"));
    if view["launch_in_flight"].as_bool() == Some(true) {
        text.push_str(" (its launch was still in flight)");
    }
    text.push_str(".\n");
    text
}

/// `list`: one line per subagent — live ones and registered (closed but
/// resumable) ones.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn list(view: &Value) -> String {
    let subs = view["subagents"]
        .as_array()
        .expect("list carries subagents");
    if subs.is_empty() {
        return "No subagents live or registered.\n".to_string();
    }
    let mut text = String::new();
    for sub in subs {
        let turns = sub["turns"].as_u64().expect("an entry carries turns");
        let _ = write!(
            text,
            "{} ({}) — {}, {turns} turn{}, cwd {}",
            str_field(sub, "session_id"),
            str_field(sub, "agent"),
            str_field(sub, "state"),
            plural(turns),
            str_field(sub, "cwd"),
        );
        if let Some(resume_at) = opt_str(sub, "resume_at") {
            let _ = write!(
                text,
                "; resumes at {resume_at} ({})",
                str_field(sub, "reason")
            );
        }
        if let Some(error) = opt_str(sub, "error") {
            let _ = write!(text, "; {error}");
        }
        text.push('\n');
    }
    text
}

/// `agents`: each configured agent — its command, whether it is the
/// default, its live subagents, and what an initialised process reported
/// (agent info, modes, config options).
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn agents(view: &Value) -> String {
    let agents = view["agents"].as_array().expect("agents is an array");
    if agents.is_empty() {
        return "No agents configured.\n".to_string();
    }
    let mut text = String::new();
    for agent in agents {
        let args = agent["args"]
            .as_array()
            .expect("an agent carries its args")
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .join(" ");
        let _ = write!(
            text,
            "{}: {}{}",
            str_field(agent, "name"),
            str_field(agent, "command"),
            if args.is_empty() {
                String::new()
            } else {
                format!(" {args}")
            },
        );
        if agent["default"].as_bool() == Some(true) {
            text.push_str(" (default)");
        }
        if agent["allow_outside_cwd"].as_bool() == Some(true) {
            text.push_str(", allows outside cwd");
        }
        text.push('\n');
        let subs = ids(&agent["subagents"]);
        if !subs.is_empty() {
            let _ = writeln!(text, "  live subagents: {}", subs.join(", "));
        }
        if let Some(info) = agent.get("agent_info") {
            let _ = write!(
                text,
                "  agent: {} {}",
                str_field(info, "name"),
                str_field(info, "version"),
            );
            if let Some(title) = opt_str(info, "title") {
                let _ = write!(text, " ({title})");
            }
            text.push('\n');
        }
        if let Some(modes) = agent.get("modes") {
            let available = modes["available"]
                .as_array()
                .expect("modes carries the available list")
                .iter()
                .map(|mode| format!("{} ({})", str_field(mode, "id"), str_field(mode, "name")))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(
                text,
                "  modes: {}; available: {available}",
                str_field(modes, "current"),
            );
        }
        let options = agent["config_options"]
            .as_array()
            .map_or(&[][..], Vec::as_slice);
        if !options.is_empty() {
            let options = options
                .iter()
                .map(|option| {
                    let current = match &option["current_value"] {
                        Value::String(id) => id.clone(),
                        other => other.to_string(),
                    };
                    format!("{} = {current}", str_field(option, "id"))
                })
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(text, "  config options: {options}");
        }
    }
    text
}

/// `daemon/resumed`: what a restart's resume pass did — who came back,
/// who failed, and whose turn the previous generation still runs.
/// # Panics
///
/// Panics when the view is missing a field its tool always writes — a bug in the view, not caller input.
#[must_use]
pub fn resumed(view: &Value) -> String {
    let resumed = ids(&view["resumed"]);
    let failed = view["failed"]
        .as_array()
        .expect("the report carries failed");
    let pending = ids(&view["pending"]);
    if resumed.is_empty() && failed.is_empty() && pending.is_empty() {
        return "No sessions to resume.\n".to_string();
    }
    let mut text = String::new();
    if !resumed.is_empty() {
        let _ = writeln!(
            text,
            "Resumed {} session{}: {}.",
            resumed.len(),
            plural(resumed.len() as u64),
            resumed.join(", "),
        );
    }
    if !failed.is_empty() {
        let failed = failed
            .iter()
            .map(|failure| {
                format!(
                    "{} ({})",
                    str_field(failure, "session_id"),
                    str_field(failure, "error")
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let _ = writeln!(text, "Failed to resume: {failed}.");
    }
    if !pending.is_empty() {
        let _ = writeln!(text, "Still handing over: {}.", pending.join(", "));
    }
    text
}
