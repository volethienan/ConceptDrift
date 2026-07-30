from __future__ import annotations

import json
import sys
from pathlib import Path

from docx import Document


def main() -> None:
    path = Path(sys.argv[1])
    document = Document(path)
    payload = {
        "path": str(path),
        "sections": len(document.sections),
        "paragraphs": [
            {
                "index": index,
                "style": paragraph.style.name if paragraph.style else None,
                "text": paragraph.text,
            }
            for index, paragraph in enumerate(document.paragraphs)
            if paragraph.text.strip()
        ],
        "tables": [
            {
                "index": table_index,
                "rows": [
                    [cell.text for cell in row.cells]
                    for row in table.rows
                ],
            }
            for table_index, table in enumerate(document.tables)
        ],
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
