import io
import zipfile
from pathlib import Path
from datetime import datetime

def make_zip_bytes(file_paths, zip_name_prefix="ollie_clips"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in file_paths:
            p = str(p)
            if Path(p).exists():
                zf.write(p, arcname=Path(p).name)
    buf.seek(0)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return buf.getvalue(), f"{zip_name_prefix}_{ts}.zip"