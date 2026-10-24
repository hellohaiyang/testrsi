import hashlib, json
from pathlib import Path

WS   = Path("/tmp/rsi_v3_e1")
CONF = WS / "cfg/upstream.conf"
CONF.parent.mkdir(parents=True, exist_ok=True)

BODY = [
    "# upstream map for gateway",
    "# NOTE: example only -- proxy_pass http://127.0.0.1:8080; keep as doc",
    " ",
    "upstream legacy {",
    "    server 10.0.0.5:9000;",
    "}",
    " ",
    "server {",
    "    listen 8080;",
    "    location /healthz {",
    "        proxy_pass http://127.0.0.1:8080;",
    "    }",
    "    location /api/ {",
    "        # proxy_pass http://127.0.0.1:8080;",
    "        proxy_pass http://127.0.0.1:8080;",
    "    }",
    "}",
]
WS_LINES = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17}
src = "".join(
    (l + " " if i in WS_LINES else l) + "\r\n"
    for i, l in enumerate(BODY, start=1)
)
CONF.write_bytes(src.encode("utf-8"))

lines = src.split("\r\n")
assert lines[14] == "        proxy_pass http://127.0.0.1:8080;"
lines[14] = lines[14].replace("http://127.0.0.1:8080", "http://127.0.0.1:9099", 1)
dst = "\r\n".join(lines)

truth = {
    "task_id": "E1",
    "file": str(CONF),
    "original_sha256": hashlib.sha256(src.encode()).hexdigest(),
    "expected_final_sha256": hashlib.sha256(dst.encode()).hexdigest(),
    "eol": "\\r\\n", "bom": False, "ends_with_newline": True, "total_lines": 17,
    "changed_line_no": 15,
    "before_line": "        proxy_pass http://127.0.0.1:8080;",
    "after_line":  "        proxy_pass http://127.0.0.1:9099;",
    "counts_after": {
        "http://127.0.0.1:8080": dst.count("http://127.0.0.1:8080"),
        "http://127.0.0.1:9099": dst.count("http://127.0.0.1:9099"),
        "8080": dst.count("8080"),
    },
    "literal_matches_before_edit": src.count("http://127.0.0.1:8080"),
    "trailing_ws_line_count": sum(1 for l in src.split("\r\n") if l != l.rstrip()),
    "read_tool_only_lists_first": 10,
    "allowed_paths": ["read:2", "edit:1", "total:3"],
}
Path(WS / "eval_context.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2))
print(json.dumps(truth, ensure_ascii=False, indent=2))
