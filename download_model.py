"""Step 1 - Download the pre-trained YOLOv4-tiny model (COCO, 80 classes) and a sample video.

* Works with slow / unstable internet: downloads in chunks, retries and RESUMES.
* Files already downloaded are skipped, so you can run it again safely.

Manual alternative (if downloads keep failing): download the files from the URLs below
in your browser and copy them into the  models/  folder (video -> data/).
"""
import os
import sys
import time
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# (url, destination, minimum valid size in bytes, required?)
FILES = [
    ("https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
     os.path.join(MODELS_DIR, "yolov4-tiny.cfg"), 1000, True),
    ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
     os.path.join(MODELS_DIR, "coco.names"), 500, True),
    ("https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
     os.path.join(MODELS_DIR, "yolov4-tiny.weights"), 20_000_000, True),
    ("https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi",
     os.path.join(DATA_DIR, "sample_video.avi"), 5_000_000, False),
]


def _remote_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            length = r.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download(url, dest, min_size, retries=30):
    """Download with resume support. Returns True on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) >= min_size:
        print(f"[ok]   already downloaded: {os.path.basename(dest)}")
        return True

    part = dest + ".part"
    total = _remote_size(url)
    print(f"[get]  {os.path.basename(dest)}" + (f"  ({total / 1e6:.1f} MB)" if total else ""))

    for attempt in range(1, retries + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if total and done >= total:
            break
        headers = {"User-Agent": "Mozilla/5.0"}
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                mode = "ab" if (done and status == 206) else "wb"
                if mode == "wb":
                    done = 0
                with open(part, mode) as f:
                    last_print = 0.0
                    while True:
                        chunk = resp.read(64 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if time.time() - last_print > 1.0:
                            last_print = time.time()
                            if total:
                                print(f"       {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="\r")
                            else:
                                print(f"       {done / 1e6:6.1f} MB", end="\r")
            if (not total) or os.path.getsize(part) >= total:
                break
        except (urllib.error.URLError, OSError, ConnectionError) as exc:
            print(f"\n       connection problem ({exc}); retry {attempt}/{retries} ...")
            time.sleep(min(3 * attempt, 15))

    if os.path.exists(part) and os.path.getsize(part) >= min_size:
        os.replace(part, dest)
        print(f"\n[ok]   saved: {dest}")
        return True
    print(f"\n[fail] could not download {os.path.basename(dest)}")
    return False


def main():
    ok_required = True
    for url, dest, min_size, required in FILES:
        ok = download(url, dest, min_size)
        if not ok and required:
            ok_required = False
            print(f"       Manual download: {url}\n       Save it as: {dest}")
    if not ok_required:
        sys.exit("\nSome required files are missing. Run this script again "
                 "(it resumes) or download them manually.")
    print("\nAll set! Now run:  python main.py --source data/sample_video.avi")


if __name__ == "__main__":
    main()
