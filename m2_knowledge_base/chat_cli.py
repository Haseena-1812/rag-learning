import json
import requests

URL = "http://localhost:8000/chat"
sid, filters = None, {}
print("Commands: /filter topic quantum | /filter source x.pdf | /clear | exit")

while True:
    q = input("\nYou: ").strip()
    if q in ("exit", "quit"):
        break
    if q == "/clear":
        filters = {}
        print("filters cleared")
        continue
    if q.startswith("/filter "):
        _, key, value = q.split(maxsplit=2)
        filters[key] = value
        print("filters:", filters)
        continue

    sources = []
    with requests.post(URL, json={"question": q, "session_id": sid, **filters}, stream=True) as r:
        print("Claude-RAG: ", end="")
        for line in r.iter_lines():
            if not line:
                continue
            ev = json.loads(line)
            if ev["type"] == "meta":
                sid, sources = ev["session_id"], ev["sources"]
            elif ev["type"] == "token":
                print(ev["text"], end="", flush=True)
            elif ev["type"] == "error":
                print("\nERROR:", ev["message"])
    print("\n\nSources:")
    for s in sources:
        page = f" p.{s['page']}" if s["page"] else ""
        print(f"  [{s['n']}] {s['source']}{page} (score {s['score']}, topic {s['topic']})")