# -*- coding: utf-8 -*-
"""
Live-flight control panel: a separate tkinter process that talks to the sim over a socket.

WHY A SEPARATE PROCESS (this is not an accident)
------------------------------------------------
The live flight program runs under `mjpython`: on macOS the Cocoa main thread belongs to
the MuJoCo viewer, and the script itself runs on a secondary thread. Every GUI toolkit -
tkinter, Qt, matplotlib windows - refuses to own a window from that thread. The pad
reader already sidesteps this with its own IOKit thread; a *window* cannot.

So the panel runs as its OWN process (a normal `python`, own main thread, own Tk), and the
two processes talk over a loopback TCP socket with newline-delimited JSON. Neither side
blocks the other: the sim polls for commands once per frame (non-blocking) and pushes a
status record at ~10 Hz, and the panel reconnects by itself if the sim restarts.

WHAT THE PANEL CONTROLS
-----------------------
    modes        MANUAL (acro) | MANUAL (assisted) | POLICY (your sticks)
    manoeuvres   flip / orbit / figure-8 / lissajous / slalom / waypoints / hover
                 - each is a training-shaped trajectory the policy flies from wherever
                 the vehicle currently is (see live_target.ShiftedTrajectory)
    pad          guided 4-step CALIBRATION ("push the left stick left", ...), layout
                 selector (Outer Wilds roles vs Mode 2), per-axis invert checkboxes and a
                 live readout of the four axes - this is the fix for "left acts like
                 right": measure it instead of guessing flags
    flight       respawn, stop (close the sim), save calibration

Run it through the sim (it launches automatically) or standalone to watch a running sim:

    .venv/bin/python Simulation/flight_gui.py [--port 51234]

`--selftest` exercises the protocol and the widget construction without a display loop.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from typing import Any, Dict, List, Optional

DEFAULT_PORT: int = 51234
PROTOCOL_VERSION: int = 1


# ======================================================================================
# protocol (shared by both ends - imported by the flight program as well)
# ======================================================================================
def encode(message: Dict[str, Any]) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


class LineDecoder:
    """Incremental newline-JSON decoder; tolerant of partial reads and junk lines."""

    def __init__(self) -> None:
        self._buffer = b""
        self.bad_lines = 0

    def feed(self, data: bytes) -> List[Dict[str, Any]]:
        self._buffer += data
        out: List[Dict[str, Any]] = []
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
                if isinstance(obj, dict):
                    out.append(obj)
                else:
                    self.bad_lines += 1
            except Exception:
                self.bad_lines += 1
        if len(self._buffer) > 1 << 20:          # never let a dead peer grow forever
            self._buffer = b""
            self.bad_lines += 1
        return out


class GuiServer:
    """
    The sim-side end: accepts one panel, queues its commands, broadcasts status.

    Threading: a daemon thread accepts connections and reads lines; the flight loop calls
    `poll()` (non-blocking) for commands and `status()` to publish. All state crossing the
    boundary is plain JSON-able values.
    """

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        self.port = int(port)
        self.host = host
        self._commands: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._conn: Optional[socket.socket] = None
        self._conn_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._server: Optional[socket.socket] = None
        self.clients_seen = 0
        self.error: Optional[str] = None

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> bool:
        try:
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._server.bind((self.host, self.port))
            self._server.listen(1)
            self._server.settimeout(0.5)
        except Exception as exc:
            self.error = str(exc)
            return False
        self._thread = threading.Thread(target=self._run, name="flight-gui-server", daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        with self._conn_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass

    @property
    def connected(self) -> bool:
        with self._conn_lock:
            return self._conn is not None

    # -- server loop ---------------------------------------------------------------
    def _run(self) -> None:
        decoder = LineDecoder()
        while not self._stop.is_set():
            conn = None
            try:
                conn, _addr = self._server.accept()      # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(0.5)
            with self._conn_lock:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                self._conn = conn
            self.clients_seen += 1
            try:
                while not self._stop.is_set():
                    try:
                        data = conn.recv(4096)
                    except socket.timeout:
                        continue
                    except OSError:
                        break
                    if not data:
                        break
                    with self._lock:
                        self._commands.extend(decoder.feed(data))
            finally:
                with self._conn_lock:
                    if self._conn is conn:
                        self._conn = None
                try:
                    conn.close()
                except Exception:
                    pass

    # -- sim side ------------------------------------------------------------------
    def poll(self) -> List[Dict[str, Any]]:
        with self._lock:
            commands, self._commands = self._commands, []
        return commands

    def publish(self, status: Dict[str, Any]) -> None:
        with self._conn_lock:
            conn = self._conn
        if conn is None:
            return
        try:
            conn.sendall(encode({"type": "status", "version": PROTOCOL_VERSION, "status": status}))
        except Exception:
            with self._conn_lock:
                if self._conn is conn:
                    self._conn = None


# ======================================================================================
# the panel
# ======================================================================================
MODES: List[Dict[str, str]] = [
    {"cmd": "mode:manual_game", "label": "MANUAL - game (left=move, right=turn)"},
    {"cmd": "mode:manual_acro", "label": "MANUAL - acro (your sticks)"},
    {"cmd": "mode:manual_assisted", "label": "MANUAL - assisted"},
    {"cmd": "mode:policy_human", "label": "POLICY - tracking your sticks"},
]
TRAJECTORIES: List[Dict[str, str]] = [
    {"cmd": "traj:hover", "label": "hover hold"},
    {"cmd": "traj:flip", "label": "FLIP"},
    {"cmd": "traj:orbit", "label": "orbit"},
    {"cmd": "traj:figure8", "label": "figure-8"},
    {"cmd": "traj:lissajous", "label": "lissajous"},
    {"cmd": "traj:slalom", "label": "slalom"},
    {"cmd": "traj:waypoints", "label": "waypoints"},
]


class FlightPanel:
    """Tkinter control panel. All widget callbacks just send one JSON command."""

    def __init__(self, port: int = DEFAULT_PORT, host: str = "127.0.0.1") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.port = port
        self.host = host
        self.sock: Optional[socket.socket] = None
        self.decoder = LineDecoder()
        self.status: Dict[str, Any] = {}
        self._last_status_time = 0.0
        self._closing = False

        self.root = tk.Tk()
        self.root.title("Live flight - control panel")
        self.root.geometry("470x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        pad = {"padx": 10, "pady": 4}
        header = tk.Label(self.root, text="LIVE FLIGHT", font=("Helvetica", 16, "bold"))
        header.pack(anchor="w", **pad)
        self.conn_label = tk.Label(self.root, text="connecting...", fg="#a15c00")
        self.conn_label.pack(anchor="w", padx=10)

        # -- modes -----------------------------------------------------------------
        box = ttk.LabelFrame(self.root, text="Flight mode")
        box.pack(fill="x", **pad)
        for item in MODES:
            ttk.Button(box, text=item["label"], width=40,
                       command=lambda c=item["cmd"]: self.send(c)).pack(fill="x", padx=8, pady=3)

        # -- trajectories ----------------------------------------------------------
        box = ttk.LabelFrame(self.root, text="Policy executes a trajectory (from your current hover)")
        box.pack(fill="x", **pad)
        grid = ttk.Frame(box)
        grid.pack(fill="x", padx=8, pady=4)
        for i, item in enumerate(TRAJECTORIES):
            ttk.Button(grid, text=item["label"], width=18,
                       command=lambda c=item["cmd"]: self.send(c)).grid(
                row=i // 2, column=i % 2, padx=3, pady=3, sticky="ew")

        # -- pad -------------------------------------------------------------------
        box = ttk.LabelFrame(self.root, text="Controller")
        box.pack(fill="x", **pad)
        self.cal_label = tk.Label(box, text="", fg="#7a0000", wraplength=420, justify="left")
        self.cal_label.pack(anchor="w", padx=8)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=2)
        ttk.Button(row, text="Calibrate (4 steps)", command=lambda: self.send("pad:calibrate")).pack(side="left")
        ttk.Button(row, text="Save", command=lambda: self.send("pad:save")).pack(side="left", padx=4)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=2)
        tk.Label(row, text="Stick layout:").pack(side="left")
        ttk.Button(row, text="Outer Wilds", command=lambda: self.send("pad:layout:ow")).pack(side="left", padx=4)
        ttk.Button(row, text="Mode 2", command=lambda: self.send("pad:layout:mode2")).pack(side="left")
        ttk.Button(row, text="swap L/R sticks", command=lambda: self.send("pad:swap_sticks")).pack(side="left", padx=4)

        self.inv_vars = {}
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=2)
        for ch, text in (("lx", "L-x"), ("ly", "L-y"), ("rx", "R-x"), ("ry", "R-y")):
            var = tk.BooleanVar(value=False)
            self.inv_vars[ch] = var
            ttk.Checkbutton(row, text=f"invert {text}", variable=var,
                            command=lambda c=ch: self.send(f"pad:invert:{c}:{int(self.inv_vars[c].get())}")
                            ).pack(side="left", padx=2)

        self.axes_label = tk.Label(box, text="sticks: -", font=("Courier", 11))
        self.axes_label.pack(anchor="w", padx=8, pady=2)

        # -- flight ----------------------------------------------------------------
        box = ttk.LabelFrame(self.root, text="Flight")
        box.pack(fill="x", **pad)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Button(row, text="Respawn", command=lambda: self.send("flight:respawn")).pack(side="left")
        ttk.Button(row, text="Stop (close sim)", command=lambda: self.send("flight:quit")).pack(side="left", padx=6)

        # -- status ----------------------------------------------------------------
        box = ttk.LabelFrame(self.root, text="Status")
        box.pack(fill="both", expand=True, **pad)
        self.status_label = tk.Label(box, text="", justify="left", anchor="nw", font=("Courier", 11))
        self.status_label.pack(fill="both", expand=True, padx=8, pady=4)

        self.root.after(50, self.poll_socket)
        self.root.after(200, self.refresh_status)

    # -- link ------------------------------------------------------------------------
    def connect(self) -> None:
        if self.sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            s.setblocking(False)
            self.sock = s
            self.conn_label.config(text=f"connected on port {self.port}", fg="#0a6b0a")
        except Exception:
            self.sock = None
            self.conn_label.config(text=f"waiting for the sim on port {self.port}...", fg="#a15c00")

    def send(self, command: str) -> None:
        if self.sock is None:
            self.connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall(encode({"type": "command", "command": command,
                                      "version": PROTOCOL_VERSION}))
        except Exception:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
            self.conn_label.config(text="sim disconnected - retrying...", fg="#a15c00")

    def poll_socket(self) -> None:
        if self._closing:
            return
        if self.sock is None:
            self.connect()
        if self.sock is not None:
            try:
                data = self.sock.recv(8192)
                if not data:
                    raise ConnectionError("closed")
                for msg in self.decoder.feed(data):
                    if msg.get("type") == "status":
                        self.status = dict(msg.get("status", {}))
                        self._last_status_time = time.time()
            except BlockingIOError:
                pass
            except Exception:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self.conn_label.config(text="sim disconnected - retrying...", fg="#a15c00")
        self.root.after(50, self.poll_socket)

    def refresh_status(self) -> None:
        if self._closing:
            return
        if self.sock is None or (time.time() - self._last_status_time > 2.0):
            self.status_label.config(text="no status from the sim yet")
        else:
            st = self.status
            lines = [
                f"mode      {st.get('mode', '?')}",
                f"trajectory{str(st.get('trajectory', '-')):>10s}  remaining {st.get('trajectory_left', 0.0):4.1f}s",
                f"policy    {'loaded' if st.get('policy') else 'NOT LOADED'}   encoder {'on' if st.get('encoder') else 'off'}",
                f"pad       {st.get('pad', 'none')}",
                "",
                f"altitude  {st.get('z', 0.0):5.2f} m      speed {st.get('speed', 0.0):5.2f} m/s",
                f"ref error {st.get('ref_err', 0.0):5.2f} m      tilt {st.get('tilt', 0.0):5.0f} deg",
                f"thrust    {st.get('thrust', 0.0):5.1f} %     rate {st.get('rate', 0.0):4.0f} dps",
                f"link      {st.get('link', 'sim-only')}",
                "",
                st.get("message", ""),
            ]
            self.status_label.config(text="\n".join(lines))
            self.axes_label.config(
                text=("sticks: " + "  ".join(f"{k}={st.get('axes', {}).get(k, 0.0):+.2f}"
                                             for k in ("lx", "ly", "rx", "ry")))
            )
            flags = st.get("inverts", {})
            for ch, var in self.inv_vars.items():
                var.set(bool(flags.get(ch, False)))
            layout = str(st.get("layout", "ow"))
            # Mirror the calibration state (prompt first, then the report line).
            cal = st.get("calibration", "not calibrated")
            failed = str(st.get("calibration_failed", ""))
            done = bool(st.get("calibration_done", False))
            colour = "#7a0000" if failed else ("#0a6b0a" if done else "#333333")
            prompt = str(st.get("calibration_prompt", ""))
            self.cal_label.config(text=(prompt + f"\n{cal}" if prompt else cal), fg=colour)
            self.root.title(f"Live flight - control panel  [{layout}]")
        self.root.after(200, self.refresh_status)

    def on_close(self) -> None:
        self._closing = True
        try:
            self.send("flight:quit")
        except Exception:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


# ======================================================================================
# entry points
# ======================================================================================
def selftest() -> int:
    """Construct the panel widgets and round-trip the protocol without a mainloop."""
    ok = True
    port = _free_port()
    server = GuiServer(port=port)
    ok &= server.start()

    # Protocol round trip: a client sends a command, the sim-side server must decode it.
    import socket as _socket
    client = _socket.create_connection(("127.0.0.1", port), timeout=2.0)
    client.sendall(encode({"type": "command", "command": "mode:policy_human"}))
    deadline = time.time() + 2.0
    commands: List[Dict[str, Any]] = []
    while time.time() < deadline and not commands:
        commands = server.poll()
        time.sleep(0.02)
    ok &= commands == [{"type": "command", "command": "mode:policy_human"}]

    # Status direction: the server publishes, the client must decode it.
    server.publish({"mode": "POLICY", "z": 1.2})
    decoder = LineDecoder()
    got: List[Dict[str, Any]] = []
    deadline = time.time() + 2.0
    while time.time() < deadline and not got:
        try:
            got += decoder.feed(client.recv(4096))
        except _socket.timeout:
            pass
    ok &= any(m.get("type") == "status" and m.get("status", {}).get("mode") == "POLICY" for m in got)

    # Partial-line tolerance.
    d2 = LineDecoder()
    ok &= d2.feed(b'{"a":') == []
    ok &= d2.feed(b'1}\n') == [{"a": 1}]
    d3 = LineDecoder()
    ok &= d3.feed(b"not json\n") == [] and d3.bad_lines == 1

    client.close()
    server.stop()

    # Widget construction (withdrawn - no window is left behind).
    panel = FlightPanel(port=port)
    panel.root.withdraw()
    n_buttons = 0

    def count(widget) -> None:
        nonlocal n_buttons
        for child in widget.winfo_children():
            if child.winfo_class() in ("TButton", "Button"):
                n_buttons += 1
            count(child)

    count(panel.root)
    ok &= n_buttons >= len(MODES) + len(TRAJECTORIES) + 4
    panel.root.destroy()

    print("selftest:", "OK" if ok else "FAILED", f"({n_buttons} buttons, protocol round trip)")
    return 0 if ok else 1


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def main() -> int:
    parser = argparse.ArgumentParser(description="Live-flight control panel")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    panel = FlightPanel(port=args.port)
    panel.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
