"""Session date strings of any source -> datetime (LoCoMo '1:56 pm on 8 May, 2023', LongMemEval '2023/05/20 (Sat) 02:21')."""
from __future__ import annotations
import re
from datetime import datetime


def parse_any_date(s):
    s = str(s or "")
    m = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})(?:\D+(\d{1,2}):(\d{2}))?", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return datetime(y, mo, d, int(m.group(4) or 0), int(m.group(5) or 0))
    for fmt in ("%I:%M %p on %d %B, %Y", "%d %B, %Y", "%d %B %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", s)
    if m:
        try:
            return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%d %B %Y")
        except ValueError:
            return None
    return None
