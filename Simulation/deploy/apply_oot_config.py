# -*- coding: utf-8 -*-
"""
Merge an out-of-tree app-config fragment into a kbuild `.config` - on macOS, reliably.

WHY THIS EXISTS (measured, not theoretical)
-------------------------------------------
The firmware's own `scripts/kconfig/merge_config.sh` is a GNU-toolchain script
(`readlink -m`, `sed -i`, `cp -T`). On macOS every one of those fails, and the script
then prints

    # merged configuration written to  (needs make)

with an EMPTY output path and **exit code 0** - so `make` happily continues with an
UNMERGED config. The symptom is silent and expensive: CONFIG_APP_ENABLE stays unset,
the app-layer is never compiled, `controllerOutOfTreeInit` never appears in the ELF,
there is no `appMain`, and the only clue is that the binary grew by ~3 KB instead of
~155 KB. Three compilers' worth of "where did my app go".

This tool does the same fragment merge with nothing but the standard library:

    python3 Simulation/deploy/apply_oot_config.py <build/.config> <app-config>

Semantics (matching merge_config.sh for the simple `key=value` fragments we use):
  * a line `CONFIG_X=y` (or `=n`, `=800`, ...) replaces the existing `CONFIG_X=...`
    line in place, or is appended if absent;
  * `# CONFIG_X is not set` behaves the same way;
  * everything else in the config is untouched. Conflicts are printed, because a
    fragment silently overridden by a later `oldconfig` is the exact failure mode
    this tool exists to prevent.
"""

from __future__ import annotations

import re
import sys

LINE_RE = re.compile(r"^(CONFIG_[A-Za-z0-9_]+)=(.+)$")
NOTSET_RE = re.compile(r"^# (CONFIG_[A-Za-z0-9_]+) is not set$")


def parse_fragment(path: str) -> dict[str, str]:
    out: dict[str, str] = {}
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if m:
                out[m.group(1)] = f"{m.group(1)}={m.group(2)}"
                continue
            m = NOTSET_RE.match(line)
            if m:
                out[m.group(1)] = f"# {m.group(1)} is not set"
    return out


def apply(config_path: str, fragment: dict[str, str]) -> list[str]:
    with open(config_path) as fh:
        lines = fh.read().splitlines()

    index: dict[str, int] = {}
    for i, line in enumerate(lines):
        m = LINE_RE.match(line) or NOTSET_RE.match(line)
        if m:
            index[m.group(1)] = i

    changes: list[str] = []
    appended: list[str] = []
    for key, new_line in fragment.items():
        if key in index:
            i = index[key]
            if lines[i] != new_line:
                changes.append(f"  {key}: {lines[i]!r} -> {new_line!r}")
                lines[i] = new_line
        else:
            appended.append(new_line)
            changes.append(f"  {key}: (new) -> {new_line!r}")

    if appended:
        lines.append("")
        lines.append("# merged by Simulation/deploy/apply_oot_config.py")
        lines.extend(appended)

    with open(config_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return changes


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    config_path, fragment_path = argv[1], argv[2]
    fragment = parse_fragment(fragment_path)
    if not fragment:
        print(f"apply_oot_config: no CONFIG_ lines found in {fragment_path}")
        return 2
    changes = apply(config_path, fragment)
    print(f"apply_oot_config: merged {len(fragment)} symbols from "
          f"{fragment_path.split('/')[-1]} into {config_path}")
    for c in changes:
        print(c)
    if not changes:
        print("  (no changes needed - config already carries the fragment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
