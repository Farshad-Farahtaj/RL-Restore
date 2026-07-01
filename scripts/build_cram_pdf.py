import re, pathlib, markdown
from playwright.sync_api import sync_playwright

CRAM = pathlib.Path("docs/learn/cram")
OUT = CRAM / "RL-Restore-Exam-Study-Guide.pdf"
HTMLF = CRAM / "_build.html"
CHAPTERS = ["cheatsheet.md", "00-foundations.md", "01-the-paper.md", "02-the-toolbox-and-agent.md",
            "03-training-and-experiments.md", "04-findings-and-generative.md", "05-the-app-and-defense.md"]

combined = "\n\n".join((CRAM / c).read_text() for c in CHAPTERS)

blocks = []
def _stash(m):
    blocks.append(m.group(1)); return f"\n\n@@MERMAID{len(blocks)-1}@@\n\n"
combined = re.sub(r"```mermaid\s*\n(.*?)```", _stash, combined, flags=re.S)

md = markdown.Markdown(extensions=["tables", "fenced_code", "attr_list", "sane_lists", "toc", "md_in_html"],
                       extension_configs={"toc": {"toc_depth": "1-2"}})
body = md.convert(combined)
toc = md.toc
body = re.sub(r"<p>@@MERMAID(\d+)@@</p>", lambda m: f'<pre class="mermaid">{blocks[int(m.group(1))]}</pre>', body)
for pat, cls in [(r"<p>(<strong>Technical\.</strong>)", "tech"), (r"<p>(<strong>Example\.</strong>)", "ex"),
                 (r"<p>(<strong>Like you[’']re 5\.</strong>)", "eli5")]:
    body = re.sub(pat, lambda m, c=cls: f'<p class="lvl {c}">{m.group(1)}', body)

FONTS = '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">'

CSS = """
:root{--accent:#4f7cff;--ink:#1b2230;--muted:#5b6678}
*{box-sizing:border-box}
body{font-family:"Source Serif 4",Georgia,serif;font-size:10.6pt;line-height:1.5;color:var(--ink);margin:0}
h1,h2,h3,h4{font-family:Inter,system-ui,sans-serif;line-height:1.2;color:#10141d}
body>h1{font-size:21pt;font-weight:700;page-break-after:avoid;margin:1.7em 0 .15em;padding-bottom:.18em;border-bottom:3px solid var(--accent);letter-spacing:-.01em}
body>h1:first-of-type{margin-top:0}
h2{font-size:13.5pt;font-weight:600;page-break-after:avoid;margin:1.2em 0 .35em;padding-bottom:.12em;border-bottom:1px solid #e2e6ee}
h3{font-size:11.3pt;font-weight:600;page-break-after:avoid;margin:1em 0 .25em;color:#26304a}
h4{font-size:10.6pt;font-weight:600;margin:.8em 0 .2em}
p{margin:.45em 0}
a{color:var(--accent);text-decoration:none}
strong{color:#0e1320}
.lvl{padding:.3em 0 .3em .8em;border-left:3px solid #d6dbe6;margin:.4em 0}
.lvl.tech>strong{color:#185fa5}.lvl.eli5>strong{color:#0f6e56}.lvl.ex>strong{color:#5b3aa6}
blockquote{background:#f5f8fd;border-left:4px solid var(--accent);margin:.6em 0;padding:.45em .85em;color:#2c3647;border-radius:0 4px 4px 0}
blockquote p{margin:.25em 0}
code{font-family:"JetBrains Mono",monospace;font-size:.85em;background:#eef1f6;padding:.06em .32em;border-radius:3px;color:#3a2a6b}
pre{background:#f6f8fb;border:1px solid #e2e6ee;border-radius:6px;padding:.6em .8em;overflow:hidden;font-family:"JetBrains Mono",monospace;font-size:8.4pt;line-height:1.4;white-space:pre-wrap}
pre code{background:none;padding:0;color:#1b2230}
table{border-collapse:collapse;width:100%;margin:.7em 0;font-size:9.2pt}
th,td{border:1px solid #dde2ec;padding:4px 8px;text-align:left;vertical-align:top}
th{background:#eef2fb;font-family:Inter,sans-serif;font-weight:600}
tr:nth-child(even) td{background:#fafbfe}
img{max-width:84%;max-height:96mm;height:auto;display:block;margin:.9em auto;border:1px solid #e6e9f1;border-radius:6px;page-break-inside:avoid}
em{color:#3a4456}
ul,ol{margin:.4em 0;padding-left:1.35em}
li{margin:.15em 0}
hr{border:0;border-top:1px solid #e2e6ee;margin:1.2em 0}
.mermaid{margin:1em auto;text-align:center;background:#fbfcfe;border:1px solid #eef1f6;border-radius:8px;padding:.7em;page-break-inside:avoid}
.mermaid svg{max-width:100%!important;max-height:88mm;height:auto}
.titlepage{height:247mm;display:flex;flex-direction:column;justify-content:center;page-break-after:always;text-align:center}
.titlepage .kicker{font-family:Inter,sans-serif;font-size:11pt;letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-weight:600}
.titlepage h1{font-family:Inter,sans-serif;font-size:36pt;font-weight:700;border:0;margin:.25em 0;letter-spacing:-.02em;page-break-before:avoid}
.titlepage .sub{font-size:13.5pt;color:var(--muted);max-width:30em;margin:.3em auto 0;line-height:1.5}
.titlepage .meta{margin-top:2.2em;font-family:Inter,sans-serif;font-size:10pt;color:var(--muted)}
.titlepage .rule{width:64px;height:4px;background:var(--accent);margin:1.2em auto;border-radius:2px}
.front{page-break-after:always}
.front h1.notoc{font-size:18pt;border:0;page-break-before:avoid;margin:0 0 .5em}
.toc{font-family:Inter,sans-serif;font-size:10pt}
.toc ul{list-style:none;padding-left:1em;margin:.15em 0}
.toc>ul{padding-left:0}
.toc>ul>li{margin:.3em 0;font-weight:600}
.toc>ul>li>ul>li{font-weight:400;color:#42506a}
.toc a{color:#26304a}
"""

TITLE = """
<section class="titlepage">
  <div class="kicker">Exam study guide · cram edition</div>
  <h1>RL-Restore</h1>
  <div class="kicker" style="letter-spacing:.1em;color:#8a93a6">everything you need, fast</div>
  <div class="rule"></div>
  <p class="sub">The condensed companion: every concept, number, and finding from the full guide — distilled for a 3-day study sprint.</p>
  <div class="meta">Computer-vision course project · 2026 · the full 118-page guide remains for depth</div>
</section>
"""

FOREWORD = """
<section class="front">
  <h1 class="notoc">How to use this</h1>
  <p>This is the <strong>cram edition</strong>. It keeps every concept, key number, and finding from the full guide, but cuts the slow on-ramp (the long analogies and the three-level repetition). Start with the <strong>cheat sheet</strong> (next page) for the whole project at a glance, then read each chapter's <strong>"Must know"</strong> box and skim the rest. Each chapter is the same as in the full guide, just tighter. For deeper explanations of any single idea, open the matching chapter in the full 118-page guide.</p>
</section>
"""

TOC = f'<section class="front"><h1 class="notoc">Contents</h1>{toc}</section>'
SCRIPTS = '<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script><script>mermaid.initialize({startOnLoad:true,theme:"neutral",flowchart:{useMaxWidth:true},securityLevel:"loose"});</script>'

full = f"<!doctype html><html lang=en><head><meta charset=utf-8>{FONTS}<style>{CSS}</style></head><body>{TITLE}{FOREWORD}{TOC}{body}{SCRIPTS}</body></html>"
HTMLF.write_text(full)
print("html written; mermaid blocks:", len(blocks))

with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto(HTMLF.resolve().as_uri(), wait_until="networkidle")
    try:
        pg.wait_for_function(f"document.querySelectorAll('.mermaid svg').length >= {len(blocks)}", timeout=90000)
        print("all mermaid rendered")
    except Exception:
        print("WARN mermaid rendered", pg.evaluate("document.querySelectorAll('.mermaid svg').length"), "/", len(blocks))
    pg.evaluate("() => document.fonts.ready"); pg.wait_for_timeout(1000)
    pg.pdf(path=str(OUT), format="A4", print_background=True,
           margin={"top": "15mm", "bottom": "15mm", "left": "16mm", "right": "16mm"},
           display_header_footer=True, header_template="<div></div>",
           footer_template='<div style="font-size:8px;color:#9aa3b2;width:100%;text-align:center;font-family:sans-serif">RL-Restore — exam study guide (cram) &nbsp;·&nbsp; <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    b.close()
print("PDF:", OUT, round(OUT.stat().st_size/1024/1024, 2), "MB")
