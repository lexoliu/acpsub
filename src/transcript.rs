//! Transcript files: one JSONL file per subagent, plus the human-readable
//! renderer shared by the `transcript` tool and `acpsub transcript`.

use std::fmt::Write;
use std::io;
use std::path::{Path, PathBuf};

use aither_acp::{ContentBlock, SessionUpdate, ToolCallContent, ToolCallStatus, ToolKind};
use serde_json::Value;
use tokio::io::AsyncWriteExt;

/// Records longer than this are clipped unless `full` is requested.
const CLIP: usize = 400;

/// Append-only JSONL transcript writer.
///
/// Each line is one record: `{"ts", "turn", "prompt"}` for prompts,
/// `{"ts", "turn", "update"}` for `session/update` notifications, and
/// `{"ts", "turn", "stop_reason"}` for turn ends.
#[derive(Debug)]
pub struct TranscriptWriter {
    file: tokio::fs::File,
    path: PathBuf,
}

impl TranscriptWriter {
    /// Open (creating) the transcript file at `path`, creating its parent
    /// directory first.
    ///
    /// # Errors
    ///
    /// Returns an error if the parent directory or file cannot be created.
    pub async fn create(path: &Path) -> io::Result<Self> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            file,
            path: path.to_path_buf(),
        })
    }

    /// The file this writer appends to.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Append one JSON record followed by a newline.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or the write fails.
    pub async fn append(&mut self, record: &Value) -> io::Result<()> {
        let mut line = serde_json::to_string(record).map_err(io::Error::other)?;
        line.push('\n');
        self.file.write_all(line.as_bytes()).await?;
        self.file.flush().await
    }
}

/// Options controlling [`render`].
#[derive(Debug, Clone, Copy, Default)]
pub struct RenderOptions {
    /// Skip the first `from` records.
    pub from: usize,
    /// Show only the last `tail` records.
    pub tail: Option<usize>,
    /// Do not clip long values.
    pub full: bool,
    /// Include `agent_thought_chunk` records.
    pub thinking: bool,
}

/// Render a JSONL transcript as human-readable text.
///
/// Unparsable lines render as `--- UPDATE unparsable` rather than failing the
/// whole render: a transcript is evidence, and a partial view beats none.
#[must_use]
pub fn render(text: &str, options: &RenderOptions) -> String {
    let records: Vec<Value> = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap_or_else(|_| Value::String(line.to_string())))
        .collect();
    let records = &records[options.from.min(records.len())..];
    let records = options.tail.map_or(records, |tail| {
        &records[records.len().saturating_sub(tail)..]
    });

    let mut out = String::new();
    // Consecutive chunk updates of one kind merge into a single section.
    let mut pending: Option<(&'static str, String)> = None;
    for record in records {
        let chunk = record
            .get("update")
            .and_then(|u| serde_json::from_value::<SessionUpdate>(u.clone()).ok());
        // Thought chunks are hidden entirely unless `thinking` is set.
        if matches!(chunk, Some(SessionUpdate::AgentThoughtChunk(_))) && !options.thinking {
            flush(&mut pending, &mut out, options.full);
            continue;
        }
        let label = match &chunk {
            Some(SessionUpdate::AgentMessageChunk(_)) => Some("--- ASSISTANT"),
            Some(SessionUpdate::AgentThoughtChunk(_)) => Some("--- THINKING"),
            Some(SessionUpdate::UserMessageChunk(_)) => Some("--- USER"),
            _ => None,
        };
        if let (Some(label), Some(update)) = (label, &chunk)
            && let Some(text) = chunk_text(update)
        {
            match &mut pending {
                Some((pending_label, buf)) if *pending_label == label => buf.push_str(&text),
                _ => {
                    flush(&mut pending, &mut out, options.full);
                    pending = Some((label, text));
                }
            }
            continue;
        }
        flush(&mut pending, &mut out, options.full);
        render_record(record, chunk.as_ref(), &mut out, options);
    }
    flush(&mut pending, &mut out, options.full);
    out
}

/// Emit a pending merged chunk section.
fn flush(pending: &mut Option<(&'static str, String)>, out: &mut String, full: bool) {
    if let Some((label, text)) = pending.take() {
        let _ = writeln!(out, "{label}");
        let _ = writeln!(out, "{}", clip(&text, full));
    }
}

/// Extract the text of a chunk update, when it is a text block.
fn chunk_text(update: &SessionUpdate) -> Option<String> {
    let (SessionUpdate::AgentMessageChunk(chunk)
    | SessionUpdate::AgentThoughtChunk(chunk)
    | SessionUpdate::UserMessageChunk(chunk)) = update
    else {
        return None;
    };
    match &chunk.content {
        ContentBlock::Text(text) => Some(text.text.clone()),
        ContentBlock::Image(_) => Some("[image]".to_string()),
        ContentBlock::Resource(resource) => Some(format!("[resource {}]", resource.resource.uri)),
    }
}

/// Render one non-chunk record.
fn render_record(
    record: &Value,
    update: Option<&SessionUpdate>,
    out: &mut String,
    options: &RenderOptions,
) {
    let turn = record.get("turn").and_then(Value::as_u64).unwrap_or(0);
    if let Some(prompt) = record.get("prompt").and_then(Value::as_str) {
        let _ = writeln!(
            out,
            "=== [turn {turn}] USER\n{}",
            clip(prompt, options.full)
        );
        return;
    }
    if let Some(stop) = record.get("stop_reason") {
        let reason = stop.as_str().unwrap_or("unknown");
        let _ = writeln!(out, "=== [turn {turn}] END {reason}");
        if let Some(error) = record.get("error").and_then(Value::as_str) {
            let _ = writeln!(out, "    error: {}", clip(error, options.full));
        }
        return;
    }
    let Some(update) = update else {
        // Unparsable line or an update shape with no record structure.
        let raw = record
            .get("update")
            .map_or_else(|| record.to_string(), ToString::to_string);
        let _ = writeln!(
            out,
            "--- UPDATE unparsable\n    {}",
            clip(&raw, options.full)
        );
        return;
    };
    match update {
        SessionUpdate::ToolCall(call) => {
            let kind = call.kind.map_or("other", kind_str);
            let status = call.status.map_or("pending", status_str);
            let _ = writeln!(
                out,
                "--- CALL {kind} {} ({}) [{status}]",
                clip(&call.title, options.full),
                call.tool_call_id
            );
            for location in &call.locations {
                match location.line {
                    Some(line) => {
                        let _ = writeln!(out, "    {}:{line}", location.path.display());
                    }
                    None => {
                        let _ = writeln!(out, "    {}", location.path.display());
                    }
                }
            }
            if let Some(input) = &call.raw_input {
                let _ = writeln!(out, "    input: {}", clip(&input.to_string(), options.full));
            }
            render_tool_content(&call.content, out, options);
        }
        SessionUpdate::ToolCallUpdate(call) => {
            match call.status {
                Some(status) => {
                    let _ = writeln!(
                        out,
                        "--- RESULT ({}) [{}]",
                        call.tool_call_id,
                        status_str(status)
                    );
                }
                None => {
                    let _ = writeln!(out, "--- RESULT ({})", call.tool_call_id);
                }
            }
            if let Some(content) = &call.content {
                render_tool_content(content, out, options);
            }
            if let Some(output) = &call.raw_output {
                let _ = writeln!(
                    out,
                    "    output: {}",
                    clip(&output.to_string(), options.full)
                );
            }
        }
        SessionUpdate::Plan(plan) => {
            let _ = writeln!(out, "--- PLAN");
            for entry in &plan.entries {
                let status = match entry.status {
                    aither_acp::PlanEntryStatus::Pending => "pending",
                    aither_acp::PlanEntryStatus::InProgress => "in_progress",
                    aither_acp::PlanEntryStatus::Completed => "completed",
                };
                let _ = writeln!(out, "    [{status}] {}", clip(&entry.content, options.full));
            }
        }
        SessionUpdate::CurrentModeUpdate(mode) => {
            let _ = writeln!(out, "--- MODE {}", mode.current_mode_id);
        }
        other => {
            let value = serde_json::to_value(other).unwrap_or_default();
            let tag = value
                .get("sessionUpdate")
                .and_then(Value::as_str)
                .unwrap_or("unknown");
            let _ = writeln!(
                out,
                "--- UPDATE {tag}\n    {}",
                clip(&value.to_string(), options.full)
            );
        }
    }
}

/// Render tool call content blocks, one clipped line each.
fn render_tool_content(content: &[ToolCallContent], out: &mut String, options: &RenderOptions) {
    for item in content {
        let text = match item {
            ToolCallContent::Content { content } => match content {
                ContentBlock::Text(text) => text.text.clone(),
                ContentBlock::Image(_) => "[image]".to_string(),
                ContentBlock::Resource(resource) => {
                    format!("[resource {}]", resource.resource.uri)
                }
            },
            ToolCallContent::Diff(diff) => format!("diff {}", diff.path.display()),
            ToolCallContent::Terminal { terminal_id } => format!("terminal {terminal_id}"),
        };
        let _ = writeln!(out, "    {}", clip(&text, options.full));
    }
}

/// Clip `text` to [`CLIP`] chars unless `full`.
fn clip(text: &str, full: bool) -> String {
    if full || text.chars().count() <= CLIP {
        return text.to_string();
    }
    let head: String = text.chars().take(CLIP).collect();
    let rest = text.chars().count() - CLIP;
    format!("{head}\n… ({rest} more chars)")
}

const fn kind_str(kind: ToolKind) -> &'static str {
    match kind {
        ToolKind::Read => "read",
        ToolKind::Edit => "edit",
        ToolKind::Delete => "delete",
        ToolKind::Move => "move",
        ToolKind::Search => "search",
        ToolKind::Execute => "execute",
        ToolKind::Think => "think",
        ToolKind::Fetch => "fetch",
        ToolKind::SwitchMode => "switch_mode",
        ToolKind::Other => "other",
    }
}

const fn status_str(status: ToolCallStatus) -> &'static str {
    match status {
        ToolCallStatus::Pending => "pending",
        ToolCallStatus::InProgress => "in_progress",
        ToolCallStatus::Completed => "completed",
        ToolCallStatus::Failed => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(json: &str) -> String {
        format!("{json}\n")
    }

    #[test]
    fn renders_prompt_turn_and_merges_chunks() {
        let text = concat!(
            r#"{"ts":"t","turn":1,"prompt":"do it"}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"Hel"}}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"lo"}}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"tool_call","toolCallId":"tc-1","title":"fake-tool","kind":"execute","status":"pending","locations":[{"path":"/tmp/x","line":3}],"rawInput":{"cmd":"ls"}}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"tool_call_update","toolCallId":"tc-1","status":"completed","content":[{"type":"content","content":{"type":"text","text":"done ok"}}]}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"plan","entries":[{"content":"step one","status":"completed"},{"content":"step two","status":"in_progress"}]}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"current_mode_update","currentModeId":"bypass"}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"acme.weird","x":1}}"#,
            "\n",
            r#"{"ts":"t","turn":1,"stop_reason":"end_turn"}"#,
            "\n"
        );
        let rendered = render(text, &RenderOptions::default());
        assert!(rendered.contains("=== [turn 1] USER\ndo it\n"));
        assert!(rendered.contains("--- ASSISTANT\nHello\n"));
        assert!(
            rendered.contains("--- CALL execute fake-tool (tc-1) [pending]"),
            "{rendered}"
        );
        assert!(rendered.contains("    /tmp/x:3\n"), "{rendered}");
        assert!(rendered.contains("    input: {\"cmd\":\"ls\"}\n"));
        assert!(rendered.contains("--- RESULT (tc-1) [completed]\n    done ok\n"));
        assert!(
            rendered.contains("--- PLAN\n    [completed] step one\n    [in_progress] step two\n")
        );
        assert!(rendered.contains("--- MODE bypass\n"));
        assert!(rendered.contains("--- UPDATE acme.weird\n"), "{rendered}");
        assert!(rendered.contains("=== [turn 1] END end_turn\n"));
        assert!(!rendered.contains("THINKING"));
    }

    #[test]
    fn thinking_hidden_unless_requested() {
        let text = record(
            r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"agent_thought_chunk","content":{"type":"text","text":"hmm"}}}"#,
        );
        assert!(!render(&text, &RenderOptions::default()).contains("hmm"));
        let opts = RenderOptions {
            thinking: true,
            ..RenderOptions::default()
        };
        assert!(render(&text, &opts).contains("--- THINKING\nhmm\n"));
    }

    #[test]
    fn clips_long_values_unless_full() {
        let long = "x".repeat(500);
        let text = format!(
            "{}\n",
            serde_json::json!({"ts":"t","turn":1,"prompt": long})
        );
        let rendered = render(&text, &RenderOptions::default());
        assert!(rendered.contains("… (100 more chars)"), "{rendered}");
        let opts = RenderOptions {
            full: true,
            ..RenderOptions::default()
        };
        assert!(render(&text, &opts).contains(&long));
    }

    #[test]
    fn from_and_tail_slice_records() {
        let mut text = String::new();
        for i in 0..5 {
            text.push_str(&record(&format!(
                r#"{{"ts":"t","turn":1,"update":{{"sessionUpdate":"agent_message_chunk","content":{{"type":"text","text":"c{i}"}}}}}}"#
            )));
        }
        // Non-consecutive chunks must not merge: interleave a mode record.
        let mut text2 = String::new();
        for i in 0..5 {
            text2.push_str(&record(&format!(
                r#"{{"ts":"t","turn":1,"update":{{"sessionUpdate":"agent_message_chunk","content":{{"type":"text","text":"c{i}"}}}}}}"#
            )));
            text2.push_str(&record(
                r#"{"ts":"t","turn":1,"update":{"sessionUpdate":"current_mode_update","currentModeId":"m"}}"#,
            ));
        }
        let opts = RenderOptions {
            tail: Some(2),
            ..RenderOptions::default()
        };
        let rendered = render(&text2, &opts);
        assert!(rendered.contains("c4"));
        assert!(!rendered.contains("c3"));
        let opts = RenderOptions {
            from: 8,
            ..RenderOptions::default()
        };
        let rendered = render(&text2, &opts);
        assert!(rendered.contains("c4"));
        assert!(!rendered.contains("c3"));
        // The merged run renders as one ASSISTANT section.
        assert_eq!(
            render(&text, &RenderOptions::default())
                .matches("--- ASSISTANT")
                .count(),
            1
        );
    }
}
