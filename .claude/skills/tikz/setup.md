# TikZ Skill Setup

## Step 1: Install Dependencies

### Windows (PowerShell)

```powershell
where.exe pdflatex
where.exe pdftocairo
```

Install TeX Live or MiKTeX for `pdflatex`, and Poppler for `pdftocairo`.
The renderer can also use `tectonic` as the TeX engine and `pdf2svg` as the SVG converter if those are available.

### macOS (Homebrew)

```bash
which tectonic || brew install tectonic
which pdf2svg || which pdftocairo || brew install pdf2svg
```

### Verify

```bash
tectonic --version
pdflatex --version || tectonic --version
pdf2svg || pdftocairo -h  # Should show usage info
```

## Step 2: Post-Install Configuration

After installing dependencies, ask the user if they want to customize the following settings. Explain each option clearly:

1. **Host IP** (`host` in config.json, default `127.0.0.1`)
   - This is the IP address used in the preview URLs returned to the user
   - Default `127.0.0.1` means only the local machine can access the preview
   - If the user wants to access from other devices on the same network (e.g. a tablet, e-ink reader, or another computer), they should set this to the machine's LAN IP (e.g. `192.168.1.100` or `10.8.0.2`)
   - The server always listens on all interfaces (`0.0.0.0`) regardless of this setting

2. **Port** (`port` in config.json, default `8073`)
   - The port the preview server listens on
   - Change if the default port conflicts with another service

3. **Default view mode** (`default_view_mode` in config.json, default `normal`)
   - `normal` — standard web reading: scrollable, system font, full-width layout
   - `eink` — optimized for e-ink screens: paginated, large serif font, high contrast, click-to-turn-page
   - Users can always toggle between modes in the browser; this just sets the initial default

If the user wants to change any of these, edit `config.json` in the skill directory:

```bash
# Find the config file
Get-Content .claude/skills/tikz/config.json
```

Example config:

```json
{
  "host": "192.168.1.100",
  "port": 9094,
  "default_view_mode": "eink"
}
```

If the user says "keep defaults" or doesn't need to change anything, skip this step.
