"""Build study-guide.html: every slide rendered next to its speaker script, plus the Q&A section.

Run after build_slides.py. Reads seminar-slides.html and speaker-script.md.
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).parent
SLIDES = HERE / "seminar-slides.html"
SCRIPT = HERE / "speaker-script.md"
OUT = HERE / "study-guide.html"


def md_inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*\((.+?)\)\*", r'<span class="act">(\1)</span>', text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def md_block(chunk: str) -> str:
    out = []
    for para in re.split(r"\n\s*\n", chunk.strip()):
        lines = para.strip().splitlines()
        if all(l.lstrip().startswith("- ") for l in lines):
            out.append("<ul>" + "".join(f"<li>{md_inline(l.lstrip()[2:])}</li>" for l in lines) + "</ul>")
        else:
            out.append(f"<p>{md_inline(' '.join(l.strip() for l in lines))}</p>")
    return "\n".join(out)


def main() -> None:
    src = SLIDES.read_text(encoding="utf-8")
    style = re.search(r"<style>(.*?)</style>", src, re.S).group(1)
    sections = re.findall(r"<section class=\"slide[^\"]*\">.*?</section>", src, re.S)
    assert len(sections) == 19, len(sections)

    script = SCRIPT.read_text(encoding="utf-8")
    intro, rest = script.split("---", 1)
    body, qa = rest.rsplit("---", 1)
    parts = re.split(r"^## Slide (\d+) — (.+?)$", body, flags=re.M)
    # parts: [pre, num, title, chunk, num, title, chunk, ...]
    notes = {}
    for i in range(1, len(parts), 3):
        notes[int(parts[i])] = (parts[i + 1].strip(), parts[i + 2])

    intro_html = md_block(intro.split("\n", 1)[1])
    qa_title, qa_body = qa.split("\n", 1)
    qa_html = md_block(qa_body)

    rows = []
    for n, sec in enumerate(sections, start=1):
        title, chunk = notes.get(n, ("", ""))
        rows.append(
            f'<div class="row" id="s{n}">'
            f'<div class="frame">{sec}</div>'
            f'<div class="talk"><h3>Slide {n} — {md_inline(title)}</h3>{md_block(chunk)}</div>'
            f"</div>"
        )

    toc = "".join(f'<a href="#s{n}">{n}. {md_inline(notes[n][0].split(" (")[0])}</a>' for n in range(1, 20))

    page = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Học seminar HFR-MKGC</title>
<style>
{style}
/* study-guide overrides */
html,body{{background:#f4f2ee;height:auto;overflow:auto}}
.wrap{{max-width:1400px;margin:0 auto;padding:24px 28px 80px}}
.frame{{position:relative;width:640px;height:360px;overflow:hidden;border:1px solid #d8d4cc;border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.08);flex:none;background:#fff}}
.frame .slide{{display:block;left:0;top:0;transform:scale(.5);transform-origin:top left}}
.row{{display:flex;gap:26px;align-items:flex-start;margin:26px 0;padding-bottom:26px;border-bottom:1px solid #e0ddd6}}
.talk{{flex:1;min-width:0;font-size:17px;line-height:1.6;color:#1f2328}}
.talk h3{{margin:0 0 10px;font-size:20px;color:#2c5361}}
.talk p{{margin:8px 0;font-size:17px}}
.talk ul{{margin:6px 0;padding-left:22px}}
.talk li{{font-size:17px;line-height:1.5}}
.talk .act{{color:#a33f2b;font-style:italic}}
.talk code{{background:#eee;padding:1px 5px;border-radius:4px;font-size:15px}}
.top h1{{font-size:30px;margin:0 0 6px}}
.top p{{font-size:17px}}
.toc{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:14px;margin:12px 0 6px}}
.toc a{{color:#2c5361;text-decoration:none;background:#fff;border:1px solid #e0ddd6;border-radius:6px;padding:3px 8px}}
.qa{{margin-top:40px}}
.qa h2{{font-size:26px;border:0;padding:0}}
.qa p{{font-size:17px;margin:12px 0;background:#fff;border:1px solid #e0ddd6;border-radius:8px;padding:10px 14px;line-height:1.55}}
.qa p b{{color:#2c5361;display:block;margin-bottom:4px}}
@media (max-width:1100px){{.row{{flex-direction:column}}.frame{{width:100%;height:auto;aspect-ratio:16/9}}.frame .slide{{transform:scale(calc(100% / 1280));width:1280px}}}}
@media print{{.frame{{box-shadow:none}}.row{{page-break-inside:avoid}}}}
</style></head><body><div class="wrap">
<div class="top"><h1>Học seminar HFR-MKGC — slide kèm giải thích</h1>
<p>Bên trái là slide đúng như khi trình chiếu, bên phải là phần nói cho slide đó. Đọc từ trên xuống. Chữ đỏ nghiêng là hành động (chỉ hình, chuyển slide).</p>
{intro_html}
<div class="toc">{toc}</div></div>
{''.join(rows)}
<div class="qa"><h2>{md_inline(qa_title.strip('# ').strip())}</h2>{qa_html}</div>
</div></body></html>"""
    OUT.write_text(page, encoding="utf-8")
    print(OUT, OUT.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
