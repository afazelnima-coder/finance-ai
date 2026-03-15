"""
MCP HTTP integration test — connects via SSE, runs the MCP handshake, and lists tools.

Usage:
    # Start the server first:
    uvicorn mcp_http_server:app --port 8000

    # Then run this script:
    python test_mcp_tools.py
"""
import json
import threading
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 15.0  # seconds to wait for each SSE response


def run_test():
    # Collect SSE events here so the background reader thread can share them
    events = []
    done = threading.Event()

    def read_sse(client: httpx.Client):
        """Background thread: keep the GET /sse connection open and collect events."""
        with client.stream("GET", f"{BASE_URL}/sse", timeout=None) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line.startswith("data:"):
                    data = line[len("data:"):].strip()
                    events.append(data)
                    done.set()  # signal that a new event arrived

    with httpx.Client() as client:
        # ── Step 1: Open the SSE stream ────────────────────────────────
        print("Connecting to SSE stream …")
        t = threading.Thread(target=read_sse, args=(client,), daemon=True)
        t.start()

        # ── Step 2: Wait for the endpoint event ────────────────────────
        # The server sends the session URL as plain text, e.g.:
        #   data: /messages/?session_id=<uuid>
        if not done.wait(timeout=TIMEOUT):
            raise TimeoutError("Timed out waiting for SSE endpoint event")

        endpoint_path = events[0]  # plain text, NOT JSON
        print(f"  Session endpoint: {endpoint_path}")

        session_url = BASE_URL + endpoint_path

        # ── Step 3: initialize ─────────────────────────────────────────
        print("\nSending initialize …")
        done.clear()

        init_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "0.1.0"},
            },
        }
        r = client.post(session_url, json=init_payload, timeout=TIMEOUT)
        r.raise_for_status()

        if not done.wait(timeout=TIMEOUT):
            raise TimeoutError("Timed out waiting for initialize response")

        init_response = json.loads(events[-1])
        print(f"  Server info: {init_response.get('result', {}).get('serverInfo', {})}")

        # ── Step 4: initialized notification ──────────────────────────
        client.post(
            session_url,
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            timeout=TIMEOUT,
        )

        # ── Step 5: tools/list ─────────────────────────────────────────
        print("\nSending tools/list …")
        done.clear()

        list_payload = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
        r = client.post(session_url, json=list_payload, timeout=TIMEOUT)
        r.raise_for_status()

        if not done.wait(timeout=TIMEOUT):
            raise TimeoutError("Timed out waiting for tools/list response")

        tools_response = json.loads(events[-1])
        tools = tools_response.get("result", {}).get("tools", [])

        print(f"\n{'='*50}")
        print(f"  Found {len(tools)} MCP tool(s):")
        print(f"{'='*50}")
        for tool in tools:
            name = tool.get("name", "?")
            desc = tool.get("description", "").split(".")[0]  # first sentence
            required = tool.get("inputSchema", {}).get("required", [])
            print(f"\n  Tool: {name}")
            print(f"  Desc: {desc}")
            if required:
                print(f"  Args: {', '.join(required)}")
            else:
                print("  Args: (none)")

        print(f"\n{'='*50}")
        print("All tests passed!")


if __name__ == "__main__":
    run_test()
