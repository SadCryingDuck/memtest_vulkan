//! Wrapping long messages at word boundaries, so consoles don't break them mid-word.

use std::io::IsTerminal;

/// Widest line used, also when the output is a log file or a pipe
const MAX_WIDTH: usize = 100;

/// The width messages are wrapped to: the console width (at most 100 columns),
/// or 100 when the output isn't a console
pub fn line_width() -> usize {
    if std::io::stdout().is_terminal()
        && let Ok((columns, _)) = crossterm::terminal::size()
    {
        // one column less: a full-width line makes some consoles add an empty line
        return (columns as usize).saturating_sub(1).clamp(40, MAX_WIDTH);
    }
    MAX_WIDTH
}

/// Wraps `text` for the current output, see `wrap_to`
pub fn wrap(prefix: &str, text: &str) -> String {
    wrap_to(prefix, text, line_width())
}

/// Wraps `text` at word boundaries to lines of at most `width` characters. The first line
/// starts with `prefix`, the next ones are indented to line up after it. Line breaks in `text`
/// are treated as spaces, so messages can be written in any shape. A word longer than a line
/// (like a path) gets a line of its own rather than being cut.
pub fn wrap_to(prefix: &str, text: &str, width: usize) -> String {
    let indent = " ".repeat(prefix.chars().count());
    let mut result = prefix.to_owned();
    let mut line_length = prefix.chars().count();
    let mut line_has_words = false;
    for word in text.split_whitespace() {
        let word_length = word.chars().count();
        if line_has_words && line_length + 1 + word_length > width {
            result.push('\n');
            result.push_str(&indent);
            line_length = indent.len();
            line_has_words = false;
        }
        if line_has_words {
            result.push(' ');
            line_length += 1;
        }
        result.push_str(word);
        line_length += word_length;
        line_has_words = true;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAUSE: &str = "scattered single-bit flips - typical of unstable memory: memory overclock or \
                         undervolt, overheating, or weak memory cells";

    #[test]
    fn wraps_at_word_boundaries_with_aligned_continuation() {
        let wrapped = wrap_to("Likely cause: ", CAUSE, 60);
        let lines: Vec<&str> = wrapped.lines().collect();
        assert!(lines.len() > 1, "{wrapped}");
        for line in &lines {
            assert!(line.chars().count() <= 60, "{line}");
        }
        assert!(lines[0].starts_with("Likely cause: scattered"), "{wrapped}");
        assert!(lines[1].starts_with("              ") && !lines[1].starts_with("               "));
        // nothing lost or cut: the words are the same
        let words: Vec<&str> = wrapped.split_whitespace().collect();
        let expected: Vec<&str> = "Likely cause:"
            .split_whitespace()
            .chain(CAUSE.split_whitespace())
            .collect();
        assert_eq!(words, expected);
    }

    #[test]
    fn existing_line_breaks_are_reflowed() {
        assert_eq!(
            wrap_to("", "one two\n   three\nfour", 100),
            "one two three four"
        );
    }

    #[test]
    fn short_text_and_long_words() {
        assert_eq!(wrap_to("Result:     ", "PASSED", 100), "Result:     PASSED");
        assert_eq!(wrap_to("Note: ", "", 100), "Note: ");
        let path = "C:\\a\\very\\long\\path\\to\\memtest_vulkan.log";
        assert_eq!(
            wrap_to("Log: ", &format!("see {path} now"), 20),
            format!("Log: see\n     {path}\n     now")
        );
    }

    #[test]
    fn width_is_capped() {
        assert!(line_width() <= MAX_WIDTH);
    }
}
