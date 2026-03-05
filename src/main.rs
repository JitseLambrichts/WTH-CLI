use colored::Colorize;
use dialoguer::{Select, theme::ColorfulTheme};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::thread;

// ── Ollama types ─────────────────────────────────────────────────────
#[derive(Serialize)]
#[allow(dead_code)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    think: bool,
}

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: Option<OllamaMessage>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct OllamaMessage {
    content: Option<String>,
}

// ── OpenAI-compatible types (used by OpenAI and OpenRouter) ─────────
#[derive(Serialize)]
struct ApiChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
}

#[derive(Deserialize)]
struct ApiChatResponse {
    choices: Option<Vec<ApiChatChoice>>,
}

#[derive(Deserialize)]
struct ApiChatChoice {
    message: Option<ApiChatMessage>,
}

#[derive(Deserialize)]
struct ApiChatMessage {
    content: Option<String>,
}

// ── Gemini types ─────────────────────────────────────────────────────
#[derive(Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
}

#[derive(Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
}

#[derive(Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiResponseContent>,
}

#[derive(Deserialize)]
struct GeminiResponseContent {
    parts: Option<Vec<GeminiResponsePart>>,
}

#[derive(Deserialize)]
struct GeminiResponsePart {
    text: Option<String>,
}

// ── Shared ───────────────────────────────────────────────────────────
#[derive(Serialize, Deserialize, Clone)]
struct ChatMessage {
    role: String,
    content: String,
}

// ── Main ─────────────────────────────────────────────────────────────
fn main() {
    // Load .env (silently ignore if missing)
    let _ = dotenvy::dotenv();

    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() {
        println!(
            "\n  {} – run any command; if it fails, AI tells you why.\n\n  {}  wth <command>\n  {} wth npm run build\n  {} wth --setup\n",
            "wth".bold().cyan(),
            "Usage:".dimmed(),
            "Example:".dimmed(),
            "Setup:".dimmed(),
        );
        std::process::exit(0);
    }

    // ── Setup mode ───────────────────────────────────────────────────
    if args.len() == 1 && args[0] == "--setup" {
        run_setup();
        std::process::exit(0);
    }

    // ── Run the command ──────────────────────────────────────────────
    let command = &args[0];
    let command_args = &args[1..];
    let full_command = args.join(" ");

    let mut child = match Command::new(command)
        .args(command_args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(err) => {
            eprintln!(
                "{}",
                format!("\n✖ Could not run \"{}\": {}", full_command, err).red()
            );
            std::process::exit(1);
        }
    };

    // ── Capture stderr while streaming it live ───────────────────────
    let stderr_handle = child.stderr.take().expect("stderr was piped");
    let stderr_thread = thread::spawn(move || {
        let reader = BufReader::new(stderr_handle);
        let mut buffer = String::new();
        for line in reader.lines() {
            match line {
                Ok(line) => {
                    eprintln!("{}", line);
                    buffer.push_str(&line);
                    buffer.push('\n');
                }
                Err(_) => break,
            }
        }
        buffer
    });

    let status = child.wait().expect("failed to wait on child process");
    let stderr_output = stderr_thread.join().unwrap_or_default();
    let exit_code = status.code().unwrap_or(1);

    if exit_code == 0 || stderr_output.trim().is_empty() {
        std::process::exit(exit_code);
    }

    // ── AI analysis ──────────────────────────────────────────────────
    let spinner = ProgressBar::new_spinner();
    spinner.set_style(
        ProgressStyle::with_template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏ "),
    );
    spinner.set_message("wth is analyzing the error…".cyan().to_string());
    spinner.enable_steady_tick(std::time::Duration::from_millis(80));

    let prompt = build_prompt(&full_command, &stderr_output);
    let answer = get_ai_response(&prompt);

    spinner.finish_and_clear();

    if let Some((answer, provider)) = answer {
        let separator = "─".repeat(50).yellow();
        println!(
            "\n{}\n{} {}\n",
            separator,
            "🤖 wth says".bold().cyan(),
            format!("({}):", provider).dimmed(),
        );
        termimad::print_text(&answer);
        println!("\n{}\n", separator);
    }

    std::process::exit(exit_code);
}

// ── Prompt construction ──────────────────────────────────────────────
fn build_prompt(cmd: &str, stderr: &str) -> String {
    let os_type = env::consts::OS;
    let os_arch = env::consts::ARCH;
    let shell = env::var("SHELL")
        .or_else(|_| env::var("COMSPEC"))
        .unwrap_or_else(|_| "unknown".to_string());

    format!(
        "You are a helpful terminal assistant. A command just failed on this machine.\n\
         \n\
         OS: {} ({})\n\
         Shell: {}\n\
         Command: {}\n\
         \n\
         stderr output:\n\
         ```\n\
         {}\n\
         ```\n\
         \n\
         Provide a structured output formatted in Markdown, using these exact headings:\n\
         ### What is wrong\n (A short explanation of the cause)\n\
         ### What can be done to help\n (The exact steps / code to fix it)\n\
         ### File to change\n (Specify which file needs to be changed, if any)",
        os_type,
        os_arch,
        shell,
        cmd,
        stderr.trim()
    )
}

// ── Setup ────────────────────────────────────────────────────────────
fn run_setup() {
    println!(
        "\n{}",
        "🔧 wth setup".bold().cyan()
    );
    println!(
        "{}\n",
        "Select your preferred AI provider:".dimmed()
    );

    let providers = &[
        "Ollama    – Local, free, and private (requires Ollama installed)",
        "OpenAI    – Cloud (requires OPENAI_API_KEY)",
        "Gemini    – Cloud (requires GEMINI_API_KEY)",
        "OpenRouter – Cloud (requires OPENROUTER_API_KEY)",
    ];

    let selection = Select::with_theme(&ColorfulTheme::default())
        .items(providers)
        .default(0)
        .interact();

    let selection = match selection {
        Ok(s) => s,
        Err(_) => {
            println!("\n{}", "Setup cancelled.".yellow());
            return;
        }
    };

    let provider_name = match selection {
        0 => "ollama",
        1 => "openai",
        2 => "gemini",
        3 => "openrouter",
        _ => unreachable!(),
    };

    // Read existing .env or start fresh
    let env_path = ".env";
    let mut contents = fs::read_to_string(env_path).unwrap_or_default();

    // Update or add WTH_PROVIDER
    if contents.contains("WTH_PROVIDER=") {
        let new_contents: Vec<String> = contents
            .lines()
            .map(|line| {
                if line.starts_with("WTH_PROVIDER=") {
                    format!("WTH_PROVIDER={}", provider_name)
                } else {
                    line.to_string()
                }
            })
            .collect();
        contents = new_contents.join("\n");
        if !contents.ends_with('\n') {
            contents.push('\n');
        }
    } else {
        if !contents.is_empty() && !contents.ends_with('\n') {
            contents.push('\n');
        }
        contents.push_str(&format!("\nWTH_PROVIDER={}\n", provider_name));
    }

    if let Err(e) = fs::write(env_path, &contents) {
        eprintln!(
            "\n{} {}",
            "✖ Could not write .env:".red(),
            e.to_string().red()
        );
        return;
    }

    let display_name = match selection {
        0 => "Ollama",
        1 => "OpenAI",
        2 => "Gemini",
        3 => "OpenRouter",
        _ => unreachable!(),
    };

    println!(
        "\n{} {} {}",
        "✔".green().bold(),
        "Provider set to".bold(),
        display_name.cyan().bold()
    );

    // Show provider-specific instructions
    match selection {
        0 => {
            println!(
                "\n  {}\n  {}\n",
                "Make sure Ollama is running and you have a model pulled:".dimmed(),
                "ollama pull qwen3.5:9b".cyan()
            );
        }
        1 => {
            if env::var("OPENAI_API_KEY").is_err() {
                println!(
                    "\n  {}\n  {}\n",
                    "Add your API key to .env:".dimmed(),
                    "OPENAI_API_KEY=sk-...".cyan()
                );
            }
        }
        2 => {
            if env::var("GEMINI_API_KEY").is_err() {
                println!(
                    "\n  {}\n  {}\n",
                    "Add your API key to .env:".dimmed(),
                    "GEMINI_API_KEY=AI...".cyan()
                );
            }
        }
        3 => {
            if env::var("OPENROUTER_API_KEY").is_err() {
                println!(
                    "\n  {}\n  {}\n",
                    "Add your API key to .env:".dimmed(),
                    "OPENROUTER_API_KEY=sk-or-...".cyan()
                );
            }
        }
        _ => {}
    }
}

// ── AI provider selection ────────────────────────────────────────────
fn get_ai_response(prompt: &str) -> Option<(String, String)> {
    let provider = env::var("WTH_PROVIDER")
        .unwrap_or_default()
        .to_lowercase();

    // If a provider is explicitly configured, use only that one
    match provider.as_str() {
        "ollama" => {
            if let Some(answer) = try_ollama(prompt) {
                return Some((answer, "Ollama".to_string()));
            }
            eprintln!(
                "\n{}",
                "✖ Ollama failed. Is it running? Try: ollama serve".red()
            );
            return None;
        }
        "openai" => {
            if let Some(answer) = try_openai(prompt) {
                return Some((answer, "OpenAI".to_string()));
            }
            eprintln!(
                "\n{}",
                "✖ OpenAI failed. Check your OPENAI_API_KEY in .env".red()
            );
            return None;
        }
        "gemini" => {
            if let Some(answer) = try_gemini(prompt) {
                return Some((answer, "Gemini".to_string()));
            }
            eprintln!(
                "\n{}",
                "✖ Gemini failed. Check your GEMINI_API_KEY in .env".red()
            );
            return None;
        }
        "openrouter" => {
            if let Some(answer) = try_openrouter(prompt) {
                return Some((answer, "OpenRouter".to_string()));
            }
            eprintln!(
                "\n{}",
                "✖ OpenRouter failed. Check your OPENROUTER_API_KEY in .env".red()
            );
            return None;
        }
        _ => {
            // No provider configured – try all in order (auto-detect)
            if let Some(answer) = try_ollama(prompt) {
                return Some((answer, "Ollama".to_string()));
            }
            if let Some(answer) = try_openai(prompt) {
                return Some((answer, "OpenAI".to_string()));
            }
            if let Some(answer) = try_gemini(prompt) {
                return Some((answer, "Gemini".to_string()));
            }
            if let Some(answer) = try_openrouter(prompt) {
                return Some((answer, "OpenRouter".to_string()));
            }
        }
    }

    // No provider available
    println!(
        "\n{}\n\n  {}\n\n  {}\n  Install from {} then run:\n  {}\n\n  {}\n  Create a {} file with:\n  {}\n\n  {}\n  Create a {} file with:\n  {}\n\n  {}\n  Create a {} file with:\n  {}\n",
        "✖ No AI provider available.".red(),
        "Run 'wth --setup' to configure a provider.".bold().cyan(),
        "Option 1 – Ollama (local, free, private)".bold(),
        "https://ollama.com".underline(),
        "ollama pull qwen3.5:4b".cyan(),
        "Option 2 – OpenAI (cloud)".bold(),
        ".env".cyan(),
        "OPENAI_API_KEY=your_key_here".cyan(),
        "Option 3 – Gemini (cloud)".bold(),
        ".env".cyan(),
        "GEMINI_API_KEY=your_key_here".cyan(),
        "Option 4 – OpenRouter (cloud)".bold(),
        ".env".cyan(),
        "OPENROUTER_API_KEY=your_key_here".cyan(),
    );

    None
}

// ── Ollama ───────────────────────────────────────────────────────────
fn try_ollama(prompt: &str) -> Option<String> {
    let model = env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3.5:9b".to_string());
    let base_url = env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://localhost:11434".to_string());

    let body = OllamaChatRequest {
        model,
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        stream: false,
        think: false,
    };

    let res = ureq::post(&format!("{}/api/chat", base_url))
        .timeout(std::time::Duration::from_secs(600))
        .send_json(&body)
        .ok()?;

    let data: OllamaChatResponse = res.into_json().ok()?;
    let content = data.message?.content?;
    let trimmed = content.trim().to_string();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

// ── OpenAI ───────────────────────────────────────────────────────────
fn try_openai(prompt: &str) -> Option<String> {
    let api_key = env::var("OPENAI_API_KEY").ok()?;
    let base_url =
        env::var("OPENAI_API_BASE").unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let body = ApiChatRequest {
        model: env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4.1-mini".to_string()),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let res = ureq::post(&format!("{}/chat/completions", base_url))
        .timeout(std::time::Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_key))
        .send_json(&body);

    match res {
        Ok(response) => {
            let data: ApiChatResponse = response.into_json().ok()?;
            let content = data.choices?.into_iter().next()?.message?.content?;
            let trimmed = content.trim().to_string();

            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        Err(e) => {
            if let ureq::Error::Status(code, response) = e {
                let error_body = response.into_string().unwrap_or_else(|_| "unknown error".to_string());
                eprintln!(
                    "\n{} (Status {}): {}",
                    "⚠ OpenAI API Error".yellow().bold(),
                    code,
                    error_body.red()
                );
            } else {
                eprintln!("\n{} {}", "⚠ Connection Error:".yellow().bold(), e.to_string().red());
            }
            None
        }
    }
}

// ── Gemini ───────────────────────────────────────────────────────────
fn try_gemini(prompt: &str) -> Option<String> {
    let api_key = env::var("GEMINI_API_KEY").ok()?;
    let model =
        env::var("GEMINI_MODEL").unwrap_or_else(|_| "gemini-2.0-flash".to_string());

    let body = GeminiRequest {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart {
                text: prompt.to_string(),
            }],
        }],
    };

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let res = ureq::post(&url)
        .timeout(std::time::Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .send_json(&body);

    match res {
        Ok(response) => {
            let data: GeminiResponse = response.into_json().ok()?;
            let content = data
                .candidates?
                .into_iter()
                .next()?
                .content?
                .parts?
                .into_iter()
                .next()?
                .text?;
            let trimmed = content.trim().to_string();

            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed)
            }
        }
        Err(e) => {
            if let ureq::Error::Status(code, response) = e {
                let error_body =
                    response.into_string().unwrap_or_else(|_| "unknown error".to_string());
                eprintln!(
                    "\n{} (Status {}): {}",
                    "⚠ Gemini API Error".yellow().bold(),
                    code,
                    error_body.red()
                );
            } else {
                eprintln!(
                    "\n{} {}",
                    "⚠ Connection Error:".yellow().bold(),
                    e.to_string().red()
                );
            }
            None
        }
    }
}

// ── OpenRouter ───────────────────────────────────────────────────────
fn try_openrouter(prompt: &str) -> Option<String> {
    let api_key = env::var("OPENROUTER_API_KEY").ok()?;

    let body = ApiChatRequest {
        model: env::var("OPENROUTER_MODEL")
            .unwrap_or_else(|_| "arcee-ai/trinity-mini:free".to_string()),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
    };

    let res = ureq::post("https://openrouter.ai/api/v1/chat/completions")
        .timeout(std::time::Duration::from_secs(30))
        .set("Content-Type", "application/json")
        .set("Authorization", &format!("Bearer {}", api_key))
        .send_json(&body)
        .ok()?;

    let data: ApiChatResponse = res.into_json().ok()?;
    let content = data.choices?.into_iter().next()?.message?.content?;
    let trimmed = content.trim().to_string();

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}
