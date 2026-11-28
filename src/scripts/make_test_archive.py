"""A small encrypted archive laid out like final_encrypted.bin, to rehearse the final run before 13:00.

Made from training images (plus a 1x1 and a very wide one). It holds competition data: write it under data/ and
upload it only to a PRIVATE repository. Password: prova-generale (a test password, not a secret).

  python src/scripts/make_test_archive.py --out data/rehearsal/test_final.7z   # needs 7z, 7zz or py7zr
"""

import argparse
import csv
import shutil
import subprocess
import tempfile
from pathlib import Path

PASSWORD = "prova-generale"
ROOT = Path(__file__).resolve().parents[2]


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--images", type=int, default=126)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with open(args.data_dir / "dev.csv", newline="", encoding="utf-8") as handle:
        columns = next(csv.reader(handle))
    with open(args.data_dir / "train.csv", newline="", encoding="utf-8") as handle:
        rows = [row for _, row in zip(range(args.images), csv.DictReader(handle))]
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        folder = Path(scratch) / "final"
        (folder / "final_images").mkdir(parents=True)
        table = []
        for number, row in enumerate(rows):
            source = args.data_dir / row["image_path"]
            target = Path("final_images") / f"test_{number:04d}{source.suffix}"
            shutil.copy(source, folder / target)
            table.append({**{c: row.get(c, "") for c in columns}, "image_id": f"test_{number:04d}",
                          "image_path": target.as_posix()})
        for number, size in enumerate([(1, 1), (2048, 32)], start=len(rows)):
            target = Path("final_images") / f"test_{number:04d}.png"
            Image.new("RGB", size, (200, 200, 200)).save(folder / target)
            table.append({**{c: "" for c in columns}, "image_id": f"test_{number:04d}",
                          "image_path": target.as_posix()})
        with open(folder / "final.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(table)
        out.unlink(missing_ok=True)
        tool = shutil.which("7z") or shutil.which("7zz")
        if tool:
            subprocess.run([tool, "a", "-t7z", f"-p{PASSWORD}", "-mhe=on", str(out), "final"], cwd=scratch,
                           check=True, stdout=subprocess.DEVNULL)
        else:
            import py7zr

            with py7zr.SevenZipFile(out, "w", password=PASSWORD) as archive:
                archive.writeall(folder, "final")
    print(f"{out}: {len(table)} images, password {PASSWORD}")


if __name__ == "__main__":
    main()
