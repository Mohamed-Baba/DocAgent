"""Pages -> overlapping word windows (chunks never cross a page boundary)."""
from __future__ import annotations

from .schema import Chunk, Page


def chunk_pages(pages: list[Page], size: int = 200, overlap: int = 50) -> list[Chunk]:
    """Sliding windows of `size` words; two consecutive windows share `overlap` words.

    Windows start at 0, s, 2s, ... with stride s = size - overlap, while start < n - overlap
    (n = number of words of the page), and a non-empty page always gets at least one window.
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    stride = size - overlap
    chunks: list[Chunk] = []
    for page in pages:
        words = page.text.split()
        n = len(words)

        # Empty page -> no chunk.
        if n == 0:
            continue

        # Generate the starting position of each window.
        for start in range(0, max(n - overlap, 1), stride):

            # Extract at most `size` words.
            chunk_words = words[start:start + size]

            # Reconstruct the text.
            chunk_text = " ".join(chunk_words)

            chunks.append(Chunk(
                chunk_id=len(chunks),
                doc_id=page.doc_id,
                page=page.page,
                text=chunk_text,
                start=start,
            ))

    return chunks
