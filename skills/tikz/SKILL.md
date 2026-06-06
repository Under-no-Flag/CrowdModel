---
name: tikz
description: "Use TikZ to create high-quality vector diagrams and visualizations rendered to SVG with instant browser preview. Use this skill whenever the user asks to draw, visualize, or create diagrams — flowcharts, architecture diagrams, neural networks, mathematical figures, circuit diagrams, graphs, trees, state machines, timelines, or any structured visual. Also triggers on mentions of TikZ, LaTeX drawing, or pgfplots. Even if the user just says 'draw me a diagram' without mentioning TikZ, use this skill if TikZ is a good fit."
user-invocable: true
argument-hint: "<description of the diagram to create>"
---

# TikZ Diagram Skill

## How to Use

**Step 1.** Write a `.tex` file in the project directory using the Write tool:

```latex
\usetikzlibrary{arrows.meta, positioning}
\begin{tikzpicture}
  \node[circle, draw] (a) {A};
  \node[circle, draw, right=of a] (b) {B};
  \draw[->] (a) -- (b);
\end{tikzpicture}
```

**Step 2.** Pass the file path to the render script:

```bash
python skills/tikz/scripts/render_tikz.py <path-to-tex-file>
```

Output is two lines: the URL and the local PNG file path. Each render gets a unique timestamp, so re-running the same file produces a new URL.

**Step 3.** Spawn a subagent to verify the rendering result and iterate if needed:

```
Use the Agent tool to spawn a subagent with this prompt:

"Read the PNG image at <png-path> and the TikZ source at <tex-path>.
Verify the rendering quality:
- Is any text overlapping or cut off?
- Are elements too crowded or poorly spaced?
- Are arrows and connections clear?
- Is the overall layout readable?

If issues are found, edit the .tex file to fix them (increase node distance,
adjust font size, reposition elements, etc.), then re-render:
  python skills/tikz/scripts/render_tikz.py <tex-path>
Read the new PNG and verify again. Repeat until the result looks good.
Return the final URL."
```

**Step 4.** Return the final URL from the subagent to the user.

**Do NOT:**
- Call `ensure_server.py` — the render script handles server internally
- Open the URL with `open` command or browser tools

## Embedding in Plan Documents

When creating diagrams for a plan or markdown document, render the diagram first, then use the relative URI as a markdown image:

```markdown
![Pipeline Overview](/diagram_20260316_120000_123456.svg)
```

Use relative URI (e.g. `/name.svg`), not absolute URL or local file path. The plan viewer will display the image inline.

## TikZ Tips

- Write partial code (just `\begin{tikzpicture}...\end{tikzpicture}` + `\usetikzlibrary` / `\usepackage` lines) — auto-wrapped in `standalone` document
- Or write a full document with `\documentclass` — used as-is
- Use `positioning` library for `right=of` syntax
- Use `arrows.meta` for modern arrow styles

## Temp Markdown Preview

The preview server hosts a `/temp/` directory for viewing arbitrary markdown files (translations, notes, etc.).

Submit content via the script (similar to `render_tikz.py`):

```bash
# Copy a file into temp preview
python skills/tikz/scripts/submit_temp.py <path-to-file>

# Symlink instead of copy (good for large files or live-editing)
python skills/tikz/scripts/submit_temp.py --link <path>

# Symlink a directory
python skills/tikz/scripts/submit_temp.py --link /path/to/docs/

# Custom name
python skills/tikz/scripts/submit_temp.py --name translated.md <path>
```

Output is two lines: the preview URL and the path in the temp directory. The script ensures the server is running automatically.

- `.md` files are rendered with full markdown viewer (e-ink / normal / eyecare modes, TOC, auto-refresh)
- Directories get an auto-generated index page
- Other files are served as static downloads

**Do NOT** manipulate the temp directory directly — always use the script.

## Error Recovery

Compilation errors are printed to stderr. Read the error, fix the `.tex` file, re-render.

## Prerequisites

Before first use, read [setup.md](setup.md) and follow both steps:
1. Install dependencies (tectonic + pdf2svg)
2. Ask the user if they want to configure host, port, or default view mode

## Server Management

Server runs automatically. To stop (only when user explicitly asks):

```bash
python skills/tikz/scripts/ensure_server.py --stop
```
