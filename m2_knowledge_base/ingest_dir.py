import argparse
import pathlib
from app import store, loaders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default="data/docs")
    ap.add_argument("--urls", help="text file with lines: topic url")
    args = ap.parse_args()

    total = 0
    for path in sorted(pathlib.Path(args.folder).rglob("*")):
        if not path.is_file():
            continue
        topic = path.parent.name if path.parent != pathlib.Path(args.folder) else "general"
        try:
            doc_type, pages = loaders.load_file(path.name, path.read_bytes())
            n = store.add_document(path.name, doc_type, pages, topic)
            print(f"OK   {path.name} [{topic}] -> {n} chunks")
            total += n
        except Exception as e:
            print(f"SKIP {path.name}: {e}")

    if args.urls:
        for line in pathlib.Path(args.urls).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            topic, url = line.split(maxsplit=1)
            try:
                n = store.add_document(url, "html", loaders.load_url(url), topic)
                print(f"OK   {url} [{topic}] -> {n} chunks")
                total += n
            except Exception as e:
                print(f"SKIP {url}: {e}")
    print(f"Done. {total} chunks added. Total in store: {store.collection().count()}")


if __name__ == "__main__":
    main()