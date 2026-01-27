from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser

import uvicorn


def _pick_port(*, host: str, preferred: int, tries: int) -> int:
    for port in range(preferred, preferred + max(1, tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    return preferred


def main() -> None:
    host = os.getenv("HOST", "127.0.0.1")
    preferred_port = int(os.getenv("PORT", "8002"))
    tries = int(os.getenv("PORT_TRIES", "20"))
    port = _pick_port(host=host, preferred=preferred_port, tries=tries)
    url = f"http://{host}:{port}/"

    if os.getenv("OPEN_BROWSER", "1") not in {"0", "false", "False"}:
        def _open() -> None:
            time.sleep(0.6)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "1") not in {"0", "false", "False"},
        log_level=os.getenv("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
