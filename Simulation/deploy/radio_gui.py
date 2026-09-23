#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Control panel for `radio_flight.py` - the real-vehicle twin of `flight_gui.py`.

Runs in its own process (tkinter cannot own a window under mjpython, and the flight loop
must not be blocked by a widget). Talks the SAME newline-JSON protocol over loopback:
commands go out as {"type":"command","command":...}, status comes back as
{"type":"status","status":{...}} - `LineDecoder`/`encode` are imported from flight_gui so
there is one implementation of the framing.

    .venv/bin/python Simulation/deploy/radio_gui.py [--port 47865]

Buttons:
    STOCK   take the vehicle back - the app is disarmed and your sticks fly it
    HOVER   hand it to the policy (appchannel 0x01) - it holds the hover from that moment
    FLIP / orbit / figure8 / lissajous / slalom / waypoints
            one button per BAKED manoeuvre, sent as appchannel 0x05 <kind>. The list is
            read from the drone (it comes out of the same generated header the firmware
            was built with), so this panel can never offer a table the drone lacks. Each
            one is relocated onto the current pose and heading at launch and hands back to
            a hold anchored where it ended. HOLD stops the running manoeuvre in place.
    KILL    motors off + disarm. Always available. Also on the pad (X / square).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from typing import Any, Dict, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SIM_DIR = os.path.join(_PROJECT_ROOT, "Simulation")
for _p in [_PROJECT_ROOT, _SIM_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from flight_gui import DEFAULT_PORT, LineDecoder, encode  # noqa: E402


class RadioPanel:
    def __init__(self, port: int, host: str = "127.0.0.1") -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.port = port
        self.host = host
        self.sock: Optional[socket.socket] = None
        self.decoder = LineDecoder()
        self.status: Dict[str, Any] = {}
        self.hover = 50.0
        self._kinds_built = False
        self.root = tk.Tk()
        self.root.title("RADIO FLIGHT - Crazyflie 2.1")
        self.root.geometry("470x780")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        pad = {"padx": 10, "pady": 4}
        tk.Label(self.root, text="RADIO FLIGHT", font=("Helvetica", 16, "bold")).pack(anchor="w", **pad)
        self.conn = tk.Label(self.root, text="connecting...", fg="#a15c00")
        self.conn.pack(anchor="w", padx=10)

        box = ttk.LabelFrame(self.root, text="Flight mode")
        box.pack(fill="x", **pad)
        ttk.Button(box, text="STOCK - my sticks fly it", width=40,
                   command=lambda: self.send("mode:stock")).pack(fill="x", padx=8, pady=3)
        ttk.Button(box, text="HOVER - hand over to the policy", width=40,
                   command=lambda: self.send("mode:hover")).pack(fill="x", padx=8, pady=3)

        # One button per baked table. The list arrives WITH the status packets, so the grid
        # is filled in on the first one - it is the drone telling us what it can do.
        self.traj_box = ttk.LabelFrame(self.root, text="Manoeuvre (policy flies it, then holds)")
        self.traj_box.pack(fill="x", **pad)
        self.traj_hint = tk.Label(self.traj_box, text="waiting for the drone's manoeuvre list...",
                                  fg="#555555")
        self.traj_hint.pack(anchor="w", padx=8, pady=4)

        kill = tk.Button(self.root, text="KILL  (motors off + disarm)", font=("Helvetica", 14, "bold"),
                         bg="#b00000", fg="white", activebackground="#ff0000",
                         command=lambda: self.send("kill"))
        kill.pack(fill="x", padx=10, pady=8, ipady=8)

        box = ttk.LabelFrame(self.root, text="Hover thrust (centre stick = this)")
        box.pack(fill="x", **pad)
        row = ttk.Frame(box)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Button(row, text="-5%", width=6, command=lambda: self.bump(-5)).pack(side="left")
        self.hover_label = tk.Label(row, text="50 %", font=("Courier", 12, "bold"))
        self.hover_label.pack(side="left", padx=10)
        ttk.Button(row, text="+5%", width=6, command=lambda: self.bump(5)).pack(side="left")
        tk.Label(box, text="calibrate: read policy thrust while it hovers\n"
                           "(policy thrust % on the right = the answer)",
                 justify="left", fg="#555555").pack(anchor="w", padx=8, pady=(0, 4))

        box = ttk.LabelFrame(self.root, text="Status")
        box.pack(fill="both", expand=True, **pad)
        self.text = tk.Label(box, text="(no status yet)", justify="left", anchor="nw",
                             font=("Courier", 11))
        self.text.pack(fill="both", expand=True, padx=8, pady=4)

        self.root.after(50, self.poll_socket)
        self.root.after(200, self.refresh)

    # -- link --------------------------------------------------------------------------
    def connect(self) -> None:
        if self.sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=0.5)
            s.setblocking(False)
            self.sock = s
            self.conn.config(text=f"connected on port {self.port}", fg="#0a6b0a")
        except Exception:
            self.conn.config(text=f"waiting for radio_flight on port {self.port}", fg="#a15c00")

    def send(self, command: str) -> None:
        if self.sock is None:
            self.connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall(encode({"type": "command", "command": command}))
        except Exception:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None

    def build_manoeuvres(self, kinds: Any) -> None:
        """Fill the grid once, from the kind list the drone publishes in its status."""
        if self._kinds_built or not kinds:
            return
        self._kinds_built = True
        self.traj_hint.destroy()
        grid = self.tk.Frame(self.traj_box)
        grid.pack(fill="x", padx=8, pady=4)
        for n, kind in enumerate(kinds):
            name = str(kind.get("name", "?"))
            self.tk.Button(grid, text=name, width=12,
                           command=lambda name=name: self.send(f"traj:{name}")
                           ).grid(row=n // 3, column=n % 3, padx=2, pady=2, sticky="ew")
        self.tk.Button(self.traj_box, text="HOLD  (stop the manoeuvre where it is)",
                       command=lambda: self.send("hold")).pack(fill="x", padx=8, pady=(2, 6))

    def bump(self, delta: float) -> None:
        self.hover = max(0.0, min(90.0, self.hover + delta))
        self.hover_label.config(text=f"{self.hover:.0f} %")
        self.send(f"hover:{self.hover:.1f}")

    # -- tk loop -----------------------------------------------------------------------
    def poll_socket(self) -> None:
        if self.sock is None:
            self.connect()
        if self.sock is not None:
            try:
                data = self.sock.recv(65536)
                if not data:
                    raise ConnectionError("closed")
                for msg in self.decoder.feed(data):
                    if msg.get("type") == "status":
                        self.status = dict(msg.get("status", {}))
            except BlockingIOError:
                pass
            except Exception:
                try:
                    self.sock.close()
                except Exception:
                    pass
                self.sock = None
                self.conn.config(text="link lost - is radio_flight still running?", fg="#b00000")
        self.root.after(50, self.poll_socket)

    def refresh(self) -> None:
        s = self.status
        if not s:
            self.text.config(text="(no status yet - the app pushes every 200 ms)")
        else:
            c = s.get("cmd", (0, 0, 0, 0))
            armed = s.get("armed")
            flag = "ARMED - policy" if armed else "disarmed (your sticks)"
            self.build_manoeuvres(s.get("kinds"))
            lines = [
                f"mode        : {s.get('mode', '?')}",
                f"state       : {flag}",
                f"manoeuvre   : {s.get('manoeuvre_name') or '-'}",
                f"app abort   : {s.get('app_abort', 0)}   shadow: {s.get('shadow', -1)}",
                "",
                f"z           : {s.get('z', 0.0):5.2f} m",
                f"tilt        : {s.get('tilt', 0.0):5.1f} deg",
                f"policy thrt : {s.get('policy_thrust_pct', 0.0):5.1f} %",
                f"battery     : {s.get('vbat', 0.0):4.2f} V",
                f"ref err     : {s.get('p_err', 0.0):5.3f}",
                "",
                f"my cmd      : roll {c[0]:+5.1f}  pitch {c[1]:+5.1f}",
                f"              yaw {c[2]:+6.1f}  thrust {c[3]:5.1f} %",
                f"pad         : {s.get('pad', 'none')}",
            ]
            if int(s.get("app_abort", 0)):
                lines.append("")
                lines.append("*** the app auto-disarmed: see abort_reason ***")
            self.text.config(text="\n".join(lines))
            msg = s.get("message", "")
            self.conn.config(text=(msg[:80] if msg else f"connected on port {self.port}"),
                             fg="#0a6b0a")
        self.root.after(200, self.refresh)

    def on_close(self) -> None:
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass
        self.root.destroy()


def main() -> int:
    ap = argparse.ArgumentParser(description="Control panel for radio_flight.py")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT + 1)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    RadioPanel(args.port, args.host).root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
