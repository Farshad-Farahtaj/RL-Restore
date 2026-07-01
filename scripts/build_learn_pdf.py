import re, pathlib, markdown
from playwright.sync_api import sync_playwright

LEARN = pathlib.Path("docs/learn")
OUT = LEARN / "RL-Restore-Complete-Guide.pdf"
HTMLF = LEARN / "_build.html"
CHAPTERS = ["00-foundations.md", "01-the-paper.md", "02-the-toolbox-and-agent.md",
            "03-training-and-experiments.md", "04-findings-and-generative.md", "05-the-app-and-defense.md"]

combined = "\n\n".join((LEARN / c).read_text() for c in CHAPTERS)

# stash mermaid so markdown doesn't escape it
blocks = []
def _stash(m):
    blocks.append(m.group(1))
    return f"\n\n@@MERMAID{len(blocks)-1}@@\n\n"
combined = re.sub(r"```mermaid\s*\n(.*?)```", _stash, combined, flags=re.S)

md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists", "toc", "md_in_html"],
                       extension_configs={"toc": {"toc_depth": "1-2"}})
body = md.convert(combined)
toc = md.toc

body = re.sub(r"<p>@@MERMAID(\d+)@@</p>", lambda m: f'<pre class="mermaid">{blocks[int(m.group(1))]}</pre>', body)

# color-code the three teaching levels for scannability
for pat, cls in [(r"<p>(<strong>Technical\.</strong>)", "tech"),
                 (r"<p>(<strong>Example\.</strong>)", "ex"),
                 (r"<p>(<strong>Like you[’']re 5\.</strong>)", "eli5")]:
    body = re.sub(pat, lambda m, c=cls: f'<p class="lvl {c}">{m.group(1)}', body)

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'

CSS = """
:root{--accent:#4f7cff;--ink:#1b2230;--muted:#5b6678}
*{box-sizing:border-box}
body{font-family:"Source Serif 4",Georgia,serif;font-size:10.8pt;line-height:1.58;color:var(--ink);margin:0}
h1,h2,h3,h4{font-family:Inter,system-ui,sans-serif;line-height:1.2;color:#10141d}
body>h1{font-size:23pt;font-weight:700;page-break-before:always;margin:0 0 .1em;padding-bottom:.18em;border-bottom:3px solid var(--accent);letter-spacing:-.01em}
h2{font-size:15pt;font-weight:600;margin:1.5em 0 .4em;padding-bottom:.15em;border-bottom:1px solid #e2e6ee}
h3{font-size:12pt;font-weight:600;margin:1.25em 0 .3em;color:#26304a}
h4{font-size:10.8pt;font-weight:600;margin:1em 0 .2em}
p{margin:.55em 0}
a{color:var(--accent);text-decoration:none}
strong{color:#0e1320}
.lvl{padding:.35em 0 .35em .9em;border-left:3px solid #d6dbe6;margin:.45em 0}
.lvl.tech{border-color:#378add}.lvl.tech>strong{color:#185fa5}
.lvl.eli5{border-color:#1d9e75}.lvl.eli5>strong{color:#0f6e56}
.lvl.ex{border-color:#9b6bff}.lvl.ex>strong{color:#5b3aa6}
blockquote{background:#f5f8fd;border-left:4px solid var(--accent);margin:.7em 0;padding:.5em .9em;color:#2c3647;border-radius:0 4px 4px 0}
blockquote p{margin:.3em 0}
code{font-family:"JetBrains Mono",monospace;font-size:.86em;background:#eef1f6;padding:.08em .35em;border-radius:3px;color:#3a2a6b}
pre{background:#f6f8fb;border:1px solid #e2e6ee;border-radius:6px;padding:.7em .9em;overflow:hidden;font-family:"JetBrains Mono",monospace;font-size:8.6pt;line-height:1.4;white-space:pre-wrap}
pre code{background:none;padding:0;color:#1b2230}
table{border-collapse:collapse;width:100%;margin:.8em 0;font-size:9.4pt}
th,td{border:1px solid #dde2ec;padding:5px 9px;text-align:left;vertical-align:top}
th{background:#eef2fb;font-family:Inter,sans-serif;font-weight:600}
tr:nth-child(even) td{background:#fafbfe}
img{max-width:100%;height:auto;display:block;margin:1em auto;border:1px solid #e6e9f1;border-radius:6px}
em{color:#3a4456}
ul,ol{margin:.5em 0 .5em 0;padding-left:1.4em}
li{margin:.2em 0}
hr{border:0;border-top:1px solid #e2e6ee;margin:1.4em 0}
.mermaid{margin:1.1em auto;text-align:center;background:#fbfcfe;border:1px solid #eef1f6;border-radius:8px;padding:.8em}
.mermaid svg{max-width:100%!important;height:auto}
.titlepage{height:247mm;display:flex;flex-direction:column;justify-content:center;page-break-after:always;text-align:center}
.titlepage .kicker{font-family:Inter,sans-serif;font-size:11pt;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-weight:600}
.titlepage h1{font-family:Inter,sans-serif;font-size:38pt;font-weight:700;border:0;margin:.25em 0;letter-spacing:-.02em;page-break-before:avoid}
.titlepage .sub{font-size:14pt;color:var(--muted);max-width:30em;margin:.3em auto 0;line-height:1.5}
.titlepage .meta{margin-top:2.4em;font-family:Inter,sans-serif;font-size:10pt;color:var(--muted)}
.titlepage .rule{width:64px;height:4px;background:var(--accent);margin:1.3em auto;border-radius:2px}
.front{page-break-after:always}
.front h1.notoc{font-size:20pt;border:0;page-break-before:avoid;margin:0 0 .5em}
.toc{font-family:Inter,sans-serif;font-size:10.5pt}
.toc ul{list-style:none;padding-left:1.1em;margin:.2em 0}
.toc>ul{padding-left:0}
.toc>ul>li{margin:.35em 0;font-weight:600}
.toc>ul>li>ul>li{font-weight:400;color:#42506a}
.toc a{color:#26304a}
.foreword p{font-size:11pt}
"""

TITLE = """
<section class="titlepage">
  <div class="kicker">A teaching companion</div>
  <h1>RL-Restore</h1>
  <div class="kicker" style="letter-spacing:.1em;color:#8a93a6">the complete guide, from zero</div>
  <div class="rule"></div>
  <p class="sub">Reimplementing and extending a deep reinforcement-learning toolchain for image restoration — every idea explained at three levels, assuming no prior background.</p>
  <div class="meta">Computer-vision course project · 2026</div>
</section>
"""

FOREWORD = """
<section class="front foreword">
  <h1 class="notoc">How to read this</h1>
  <p>This guide teaches the whole project — the paper it builds on, the science we did, the models we trained, and the app we shipped — to someone who knows <strong>nothing</strong> about computer vision, machine learning, or reinforcement learning.</p>
  <p>Every concept appears at <strong>three levels</strong>, so you can read at whatever depth you need:</p>
  <p class="lvl tech"><strong>Technical.</strong> the rigorous version — real definitions, the math, the actual numbers.</p>
  <p class="lvl eli5"><strong>Like you're 5.</strong> the same idea in plain words, with an everyday analogy.</p>
  <p class="lvl ex"><strong>Example.</strong> a concrete instance, usually from this very project.</p>
  <p>Diagrams, plots, and before/after images appear throughout. Read the chapters in order — each builds on the last. The live demo is at <strong>parhamkhoshsolat-rl-restore.hf.space</strong>.</p>
</section>
"""

TOC = f'<section class="front"><h1 class="notoc">Contents</h1>{toc}</section>'

SCRIPTS = """
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true,theme:"neutral",flowchart:{useMaxWidth:true},securityLevel:"loose"});</script>
"""

full = f"<!doctype html><html lang=en><head><meta charset=utf-8>{FONTS}<style>{CSS}</style></head><body>{TITLE}{FOREWORD}{TOC}{body}{SCRIPTS}</body></html>"
HTMLF.write_text(full)
print("html written; mermaid blocks:", len(blocks))

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto(HTMLF.resolve().as_uri(), wait_until="networkidle")
    try:
        pg.wait_for_function(f"document.querySelectorAll('.mermaid svg').length >= {len(blocks)}", timeout=90000)
        print("all mermaid rendered")
    except Exception:
        n = pg.evaluate("document.querySelectorAll('.mermaid svg').length")
        print(f"WARN mermaid rendered {n}/{len(blocks)}")
    pg.evaluate("() => document.fonts.ready")
    pg.wait_for_timeout(1200)
    pg.pdf(path=str(OUT), format="A4", print_background=True,
           margin={"top": "16mm", "bottom": "16mm", "left": "17mm", "right": "17mm"},
           display_header_footer=True, header_template="<div></div>",
           footer_template='<div style="font-size:8px;color:#9aa3b2;width:100%;text-align:center;font-family:sans-serif">RL-Restore — the complete guide &nbsp;·&nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    b.close()
print("PDF:", OUT, round(OUT.stat().st_size/1024/1024, 2), "MB")
