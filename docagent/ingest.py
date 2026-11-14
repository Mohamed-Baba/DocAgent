"""PDF -> list[Page] with Docling (layout, reading order, tables, page images)."""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .schema import Page, append_jsonl, load_pages

log = logging.getLogger(__name__)


def make_converter(images_scale: float = 2.0, do_ocr: bool = False):
    """Docling converter for born-digital PDFs (arXiv PDFs have a text layer: OCR off)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions(do_ocr=do_ocr, do_table_structure=True,
                              generate_page_images=True, images_scale=images_scale)
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def doc_to_pages(doc, doc_id: str, image_dir: Path | None = None) -> list[Page]:
    """Split a DoclingDocument into one Page per physical page (numbers from 1).

    doc.pages: dict page_no -> PageItem (PageItem.image.pil_image is the rendering)
    doc.tables: list of TableItem (TableItem.prov[0].page_no, TableItem.export_to_markdown(doc=doc))
    doc.export_to_markdown(page_no=p): Markdown of page p only
    """
    # Group the tables by the page of their first provenance.
    tables_by_page = defaultdict(list)
    for table in doc.tables:
        if table.prov:
            tables_by_page[table.prov[0].page_no].append(table.export_to_markdown(doc=doc))

    pages = []
    for page_no, page_item in sorted(doc.pages.items()):
        text = doc.export_to_markdown(page_no=page_no)
        tables = tables_by_page.get(page_no, [])

        image_path = None                      # reset at EVERY page, inside the loop
        img = page_item.image
        if image_dir is not None and img is not None and img.pil_image is not None:
            image_dir = Path(image_dir)
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f"{doc_id}_p{page_no:03d}.png"
            img.pil_image.save(image_path)
            image_path = str(image_path)

        pages.append(Page(doc_id=doc_id, page=page_no, text=text, tables=tables, image_path=image_path))
    return pages



def pdfium_pages(pdf_path, doc_id: str) -> list[Page]:
    """Fallback: raw text layer with pypdfium2 (no layout, no tables, no image)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return [Page(doc_id=doc_id, page=i + 1, text=pdf[i].get_textpage().get_text_range())
                for i in range(len(pdf))]
    finally:
        pdf.close()


def ingest_corpus(pdf_dir, out_path, image_dir=None, converter=None) -> dict:
    """Ingest every PDF of pdf_dir into out_path (JSONL, appended), skipping cached documents.

    Returns statistics: docs, pages, empty_pages (< 20 words), fallback, failed, seconds.
    """
    pdf_dir, out_path = Path(pdf_dir), Path(out_path)
    done = {p.doc_id for p in load_pages(out_path)} if out_path.exists() else set()
    stats = {"docs": 0, "pages": 0, "empty_pages": 0, "fallback": [], "failed": [], "seconds": 0.0}
    t0 = time.perf_counter()
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        doc_id = pdf.stem
        if doc_id in done:
            continue
        try:                                   # Docling first (the converter is built once, lazily)
            if converter is None:
                converter = make_converter()
            result = converter.convert(pdf)    # a Path, not a str
            pages = doc_to_pages(result.document, doc_id, image_dir)
        except Exception as docling_error:     # one bad PDF must not stop the ingestion
            log.warning("Docling failed for %s: %s", doc_id, docling_error)
            try:
                pages = pdfium_pages(pdf, doc_id)
                stats["fallback"].append(doc_id)
            except Exception as pdfium_error:
                log.error("PDFium also failed for %s: %s", doc_id, pdfium_error)
                stats["failed"].append(doc_id)
                continue
        append_jsonl(out_path, (asdict(p) for p in pages))
        done.add(doc_id)
        stats["docs"] += 1
        stats["pages"] += len(pages)
        stats["empty_pages"] += sum(len(p.text.split()) < 20 for p in pages)
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    return stats