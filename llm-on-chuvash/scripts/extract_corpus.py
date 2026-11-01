#!/usr/bin/env python3
"""Extract plain text from a MediaWiki XML dump (e.g. cvwiki) for LM pretraining.

Usage: python -I extract_corpus.py <dump.xml.bz2> <out.txt> [min_chars]
"""
import bz2
import re
import sys
import xml.etree.ElementTree as ET


def strip_templates(text: str) -> str:
    """Remove {{...}} templates (iteratively, handles nesting)."""
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r"\{\{[^{}]*\}\}", "", text)
    return text


def clean_wikitext(raw: str) -> str:
    t = raw
    # HTML comments and refs
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>/]*?/>", "", t)
    t = re.sub(r"<ref.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<gallery.*?</gallery>", "", t, flags=re.S)
    # Tables (line-based)
    t = re.sub(r"^\{\|.*?\|\}", "", t, flags=re.S | re.M)
    # Templates after refs (some wrap refs)
    t = strip_templates(t)
    # File/image links away entirely
    t = re.sub(r"\[\[(?:File|Image|Файл|Ывăс):[^\[\]]*\]\]", "", t)
    # Categories away
    t = re.sub(r"\[\[(?:Category|Категорисем):[^\[\]]*\]\]", "", t)
    # Internal links: [[target|label]] -> label, [[target]] -> target
    t = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]", r"\1", t)
    # External links: [http://... label] -> label
    t = re.sub(r"\[https?://\S+\s+([^\]]+)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    # Headings -> plain line
    t = re.sub(r"^=+\s*(.*?)\s*=+\s*$", r"\1", t, flags=re.M)
    # Bold/italic quotes
    t = t.replace("'''", "").replace("''", "")
    # List/indent markers at line start
    t = re.sub(r"^\s*[#*:;]+\s*", "", t, flags=re.M)
    # Leftover HTML tags
    t = re.sub(r"<[^>]+>", "", t)
    # Magic words
    t = re.sub(r"__\w+__", "", t)
    # Interwiki leftovers
    t = re.sub(r"\[\[[a-z\-]{2,10}:[^\[\]]*\]\]", "", t)
    # Normalize whitespace, keep paragraph breaks
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def local(tag: str) -> str:
    """Strip XML namespace: '{http://...}page' -> 'page'."""
    return tag.rsplit("}", 1)[-1]


def main() -> None:
    dump_path, out_path = sys.argv[1], sys.argv[2]
    min_chars = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    n_pages = n_kept = 0
    total_chars = 0
    with bz2.open(dump_path, "rt", encoding="utf-8") as fin, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for _, elem in ET.iterparse(fin, events=("end",)):
            if local(elem.tag) != "page":
                continue
            n_pages += 1
            fields = {local(c.tag): c for c in elem}
            revision = fields.get("revision")
            text_el = fields.get("text")
            if revision is not None and text_el is None:
                text_el = next((c for c in revision if local(c.tag) == "text"), None)
            raw = (text_el.text or "") if text_el is not None else ""
            ns = fields.get("ns")
            ns_val = (ns.text or "0") if ns is not None else "0"

            is_article = ns_val == "0"
            is_redirect = raw.lstrip().lower().startswith("#redirect")
            text = clean_wikitext(raw) if (is_article and not is_redirect) else ""

            if len(text) >= min_chars:
                n_kept += 1
                total_chars += len(text)
                fout.write(text + "\n\n")
            elem.clear()
    print(
        f"pages={n_pages} kept={n_kept} chars={total_chars:,} "
        f"(~{total_chars / 1_000_000:.1f}M chars)"
    )


if __name__ == "__main__":
    main()
