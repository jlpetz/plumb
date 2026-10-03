#!/usr/bin/env python3
"""asm_check.py: the codegen gate for plumb (TMR-APP CLAUDE.md: "read the asm, don't infer
from timing").

Emits Intel-syntax asm (separate target dir, so the normal build is untouched),
then for every `k_*` kernel:
  * collects its body plus every local function it calls or tail-jumps to (with `#[simd]` and
    `kernel!` the hot loop is in a target-feature helper, not in the named wrapper);
  * finds the loops (backward branches) and reports the vector width used inside them;
  * lists the expected special instructions (NT stores, clflushopt, prefetch, movdir64b,
    fences, 64-bit multiplies) and where they sit (in a loop or not);
  * lists calls that survive *inside loops* (an out-of-line intrinsic or op is a call per
    element) and any memset/memcpy call anywhere;
  * counts vector spills/reloads to the stack inside loops;
  * checks each kernel against its expectation (EXPECT below) and prints PASS/FAIL.

Usage:
  python asm_check.py               # build the bench binary + table
  python asm_check.py --no-build    # reuse the last .s
  python asm_check.py --dump k_verify4_fs_512   # print that kernel's loops (and callees)
  python asm_check.py --package plumb_tiles --example asm_kernels   # check a crate's example
Extra expectations and special instructions are loaded from expect/*.json (one file per
module), so modules can be added without editing this script:
  {"special": ["tileloadd", ...], "expect": [["^k_tile_", {"loop_need": ["tileloadd"]}], ...]}
Exit code 1 if any expectation fails, so it can gate a fearless_simd version bump.
"""

import glob
import os
import re
import subprocess
import sys

import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # the plumb workspace
TARGET = os.path.join(ROOT, "target", "asm")

# --------------------------------------------------------------------------------------------
# Expectations. Each entry: regex on the kernel name -> dict of checks.
#   width:   the widest vector register class that must appear in the loops ("xmm"/"ymm"/"zmm")
#            and no wider one may.
#   need:    instructions that must appear (anywhere in the kernel's function set)
#   loop_need: instructions that must appear inside a loop
#   canary:  byte-uniform fill; report whether it collapsed to memset, don't fail either way
#   loop_calls_ok: True = calls inside loops are expected (the footgun variants)
#   known:   expected to fail today, with the reason; not counted as a failure
# Default for every kernel: no memset/memcpy, and no calls or vector spills in innermost loops
# (a call in an outer loop, like TMR's flush once per chunk, is fine).
# --------------------------------------------------------------------------------------------
EXPECT = [
    # Byte-uniform canary: reported, never failed. A memset here is the known trap; a surviving
    # loop is pass-order luck, not protection (see FINDINGS.md), so it is flagged as a NOTE.
    (r"^k_filluni_", {"canary": True}),
    (r"_128$", {"width": "xmm"}),
    (r"_256$", {"width": "ymm"}),
    (r"_512$", {"width": "zmm"}),
    (r"_auto$", {"width": "zmm"}),  # this box: Avx512 level; on Avx2-only CPUs expect ymm
    (r"^k_ntw_", {"loop_need": ["movntdq"], "need": ["sfence"]}),
    (r"^k_copynt_", {"loop_need": ["movntdq"], "need": ["sfence"]}),
    (r"^k_copymd_", {"loop_need": ["movdir64b"], "need": ["sfence"]}),
    (r"^k_flush_", {"loop_need": ["clflushopt"], "need": ["mfence"]}),
    (r"^k_wflush_", {"loop_need": ["clflushopt"], "need": ["mfence"]}),
    # k_*_fscap* (cap.rs capability token) fall under the two rules above: clflushopt inside the
    # loop and, by default, no calls in it, i.e. the stdarch intrinsic really inlined.
    (r"^k_pfv_", {"loop_need": ["prefetcht0"]}),
    (r"helper_512$", {"loop_calls_ok": True}),
    # Known failure today: no fearless_simd token lists `clflushopt`, so the stdarch intrinsic
    # can't inline into a fearless kernel and becomes a call per line. Reported as KNOWN; if it
    # ever passes, the verdict says so (a fearless_simd change worth reading about).
    (r"_fsintr", {"known": "no fearless token enables clflushopt; intrinsic is a call per line"}),
    (r"^k_fence_seqcst$", {"need_any": ["mfence", "lock"]}),
    (r"^k_fence_sfence$", {"need": ["sfence"]}),
]

SPECIAL = [
    "vmovntdq", "movntdq", "movnti", "clflushopt", "clflush", "clwb", "prefetcht0",
    "prefetchnta", "movdir64b", "sfence", "mfence", "lfence", "vpmullq", "vpmuludq",
    "vpternlogq", "vzeroupper",
]
LIBCALLS = ("memset", "memcpy", "memmove")


def load_expect_files():
    """Merge expect/*.json into EXPECT and SPECIAL (module-owned expectations)."""
    for path in sorted(glob.glob(os.path.join(HERE, "expect", "*.json"))):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for sp in data.get("special", []):
            if sp not in SPECIAL:
                SPECIAL.append(sp)
        for pat, exp in data.get("expect", []):
            EXPECT.append((pat, exp))


def target_spec(args):
    """(cargo args, asm file stem) for the bench binary or a package's example."""
    pkg = args[args.index("--package") + 1] if "--package" in args else "plumb-bench"
    if "--example" in args:
        ex = args[args.index("--example") + 1]
        return ["-p", pkg, "--example", ex], ex.replace("-", "_")
    return ["-p", pkg, "--bin", pkg], pkg.replace("-", "_")


def build(cargo_target):
    env = dict(os.environ, CARGO_TARGET_DIR=TARGET)
    cmd = (["cargo", "rustc", "--release"] + cargo_target +
           ["--", "--emit", "asm", "-C", "llvm-args=-x86-asm-syntax=intel"])
    print("building:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def latest_s(stem):
    # Older cargo: release/deps/<stem>-HASH.s (examples: release/examples/); newer build-dir
    # layout: release/build/<pkg>/HASH/out/<stem>.s.
    files = glob.glob(os.path.join(TARGET, "release", "**", f"{stem}*.s"), recursive=True)
    if not files:
        sys.exit(f"no {stem}*.s found; run without --no-build")
    return max(files, key=os.path.getmtime)


def demangle_v0(s):
    """Rust v0 mangling (_R...): pull out the length-prefixed identifiers. A heuristic, not a
    full demangler, but enough to name call targets (core::core_arch::x86::...::_mm_clflushopt)."""
    parts, i = [], 2
    while i < len(s):
        m = re.match(r"(\d+)_?", s[i:])
        if not m:
            i += 1
            continue
        ln = int(m.group(1))
        start = i + len(m.group(0))
        ident = s[start:start + ln]
        if ln and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ident or "-"):
            parts.append(ident)
            i = start + ln
        else:
            i += len(m.group(1))
    return "::".join(parts) if parts else s


def demangle(sym):
    """Rust symbol to a readable path; enough for call-target labels."""
    s = sym.strip('"')
    if s.startswith("_R"):
        return demangle_v0(s)
    m = re.match(r"_?_ZN(.*)E$", s)
    if not m:
        return s
    body, parts, i = m.group(1), [], 0
    while i < len(body):
        n = re.match(r"\d+", body[i:])
        if not n:
            break
        ln = int(n.group(0))
        i += len(n.group(0))
        parts.append(body[i:i + ln])
        i += ln
    if parts and re.match(r"^h[0-9a-f]{16}$", parts[-1]):
        parts.pop()
    out = "::".join(parts)
    for a, b in (("$LT$", "<"), ("$GT$", ">"), ("$u20$", " "), ("$C$", ","), ("$RF$", "&"),
                 ("$BP$", "*"), ("$u7b$", "{"), ("$u7d$", "}"), ("..", "::"), ("$u27$", "'"),
                 ("$u5b$", "["), ("$u5d$", "]"), ("$u3b$", ";")):
        out = out.replace(a, b)
    return out


def parse(path):
    """Return {function_name: [lines]} using COFF .def blocks as function boundaries."""
    funcs, cur, name, pending = {}, None, None, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            d = re.match(r"\s*\.def\s+(.+?);", line)
            if d:
                if name is not None:
                    funcs[name] = cur
                name, cur, pending = None, None, d.group(1).strip('"')
                continue
            if pending is not None:
                lab = re.match(r'^("?[^\s:"]+"?):\s*(#.*)?$', line)
                if lab and lab.group(1).strip('"') == pending:
                    name, cur, pending = pending, [], None
                    continue
            if cur is not None:
                cur.append(line)
    if name is not None:
        funcs[name] = cur
    return funcs


def instrs(lines):
    """(label_or_None, instruction_text) pairs, comments and most directives removed."""
    out = []
    for line in lines:
        code = line.split("#", 1)[0].rstrip() if not line.lstrip().startswith("#APP") else line
        s = code.strip()
        if not s:
            if "#APP" in line or "#NO_APP" in line:
                out.append((None, line.strip()))
            continue
        m = re.match(r'^("?[\w.$@?]+"?):$', s)
        if m:
            out.append((m.group(1).strip('"'), None))
            continue
        if s.startswith("."):
            continue
        out.append((None, s))
    return out


def loops(ins):
    """Index ranges [start, end] of backward-branch loops (label .. jump back to it)."""
    pos = {}
    res = []
    for i, (lab, txt) in enumerate(ins):
        if lab:
            pos[lab] = i
        elif txt:
            m = re.match(r"^j\w*\s+(\S+)$", txt)
            if m and m.group(1) in pos and pos[m.group(1)] < i:
                res.append((pos[m.group(1)], i))
    return res


def call_targets(txt, funcs):
    m = re.match(r"^(call|jmp)\s+(?:qword ptr \[rip \+ )?\"?([^\s\]\"]+)\"?\]?$", txt)
    if not m:
        return None
    tgt = m.group(2)
    if tgt.startswith(".L"):
        return None
    return tgt


def closure(name, funcs, depth=4):
    seen, order, frontier = {name}, [name], [name]
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for _, txt in instrs(funcs.get(f, [])):
                if not txt:
                    continue
                t = call_targets(txt, funcs)
                if t and t in funcs and t not in seen:
                    seen.add(t)
                    order.append(t)
                    nxt.append(t)
        frontier = nxt
    return order


def analyse(name, funcs):
    fset = closure(name, funcs)
    rep = {"funcs": fset, "loop_regs": set(), "special": {}, "loop_special": set(),
           "loop_calls": set(), "libcalls": set(), "spills": 0, "nloops": 0, "loop_text": [], "loop_sizes": []}
    for f in fset:
        ins = instrs(funcs[f])
        lps = loops(ins)
        rep["nloops"] += len(lps)
        in_loop = set()
        for a, b in lps:
            in_loop.update(range(a, b + 1))
            body = [t for _, t in ins[a:b + 1] if t and not t.startswith("#")]
            rep["loop_text"].append((f, body))
            rep["loop_sizes"].append(len(body))
        # Innermost loops contain no other loop. A call there costs per element (the failure
        # this gate exists for); a call in an outer loop, e.g. TMR's flush once per chunk, doesn't.
        inner = set()
        for a, b in lps:
            if not any(a <= c and d <= b and (c, d) != (a, b) for c, d in lps):
                inner.update(range(a, b + 1))
        # Blocks that end in `ret` are exits, never loop iterations, even when a backward branch
        # to a shared exit makes them look like part of a loop. Their stack traffic is the
        # epilogue (Windows x64 restores callee-saved xmm6-15 there), not spills.
        exits = set()
        for r, (_, txt) in enumerate(ins):
            if txt and txt.split()[0] == "ret":
                j = r
                while j >= 0 and ins[j][0] is None and not (ins[j][1] or "").startswith("j"):
                    exits.add(j)
                    j -= 1
        inner -= exits
        for i, (_, txt) in enumerate(ins):
            if not txt:
                continue
            op = txt.split()[0]
            for sp in SPECIAL:
                if op == sp or (sp == "movntdq" and op == "vmovntdq") or (sp == "lock" and op == "lock"):
                    rep["special"][sp] = rep["special"].get(sp, 0) + 1
                    if i in in_loop:
                        rep["loop_special"].add(sp)
            if op == "lock":
                rep["special"]["lock"] = rep["special"].get("lock", 0) + 1
            t = call_targets(txt, funcs)
            if t:
                plain = demangle(t)
                if any(lc in plain for lc in LIBCALLS):
                    rep["libcalls"].add(plain)
                if i in inner and op == "call":
                    rep["loop_calls"].add(plain)
            if i in in_loop:
                for reg in re.findall(r"\b([xyz]mm)\d+\b", txt):
                    rep["loop_regs"].add(reg)
            if i in inner and re.search(r"\[rsp", txt) and re.search(r"\b[xyz]mm\d+\b", txt):
                rep["spills"] += 1
    return rep


def widest(regs):
    for r in ("zmm", "ymm", "xmm"):
        if r in regs:
            return r
    return "-"


def check(name, rep):
    exp = {}
    for pat, e in EXPECT:
        if re.search(pat, name):
            exp.update(e)
    fails = []
    if exp.get("canary"):
        return (["NOTE: collapsed to memset (the known trap)"] if rep["libcalls"]
                else ["NOTE: loop survived this build (luck, not protection)"])
    else:
        if rep["libcalls"]:
            fails.append("libcall " + ",".join(sorted(rep["libcalls"])))
        if "width" in exp:
            wid = widest(rep["loop_regs"])
            if wid != exp["width"]:
                fails.append(f"loop width {wid}, want {exp['width']}")
        if rep["loop_calls"] and not exp.get("loop_calls_ok"):
            fails.append("call in loop")
        if rep["spills"] and not exp.get("loop_calls_ok"):
            fails.append(f"{rep['spills']} vector stack ops in loop")
    for n in exp.get("need", []):
        if n not in rep["special"]:
            fails.append(f"missing {n}")
    for n in exp.get("loop_need", []):
        if n not in rep["loop_special"]:
            fails.append(f"{n} not in a loop")
    if "need_any" in exp and not any(n in rep["special"] for n in exp["need_any"]):
        fails.append("missing " + "/".join(exp["need_any"]))
    return fails


def main():
    args = sys.argv[1:]
    load_expect_files()
    cargo_target, stem = target_spec(args)
    if "--no-build" not in args:
        build(cargo_target)
    path = latest_s(stem)
    funcs = parse(path)
    kernels = sorted(n for n in funcs if n.startswith("k_"))
    if "--dump" in args:
        want = args[args.index("--dump") + 1]
        rep = analyse(want, funcs)
        print(f"{want}: functions {[demangle(f) for f in rep['funcs']]}")
        for f, body in rep["loop_text"]:
            print(f"\n--- loop in {demangle(f)} ({len(body)} instrs)")
            for t in body:
                print("   ", t)
        return 0
    print(f"asm: {os.path.relpath(path, HERE)}  ({len(funcs)} functions, {len(kernels)} kernels)\n")
    hdr = (f"{'kernel':<24} {'width':<5} {'loop sizes':<12} {'spill':>5}  {'special (*=in loop)':<40} "
           f"{'calls in loop / libcalls':<26} verdict")
    print(hdr)
    print("-" * len(hdr))
    nfail = 0
    for k in kernels:
        rep = analyse(k, funcs)
        sp = " ".join(f"{s}{'*' if s in rep['loop_special'] else ''}x{c}"
                      for s, c in sorted(rep["special"].items()))
        calls = sorted(rep["loop_calls"] | rep["libcalls"])
        calls_s = ", ".join(c.split("::")[-1] if "::" in c else c for c in calls)
        fails = check(k, rep)
        known = next((e["known"] for p, e in EXPECT if "known" in e and re.search(p, k)), None)
        if fails and fails[0].startswith("NOTE"):
            verdict = fails[0]
        elif known:
            verdict = ("KNOWN: " + known) if fails else "FIXED? passes now; was: " + known
        else:
            nfail += bool(fails)
            verdict = "PASS" if not fails else "FAIL: " + "; ".join(fails)
        sizes = ",".join(str(n) for n in sorted(rep["loop_sizes"], reverse=True)) or "-"
        print(f"{k:<24} {widest(rep['loop_regs']):<5} {sizes[:12]:<12} {rep['spills']:>5}  "
              f"{sp[:40]:<40} {calls_s[:26]:<26} {verdict}")
    print(f"\n{len(kernels) - nfail}/{len(kernels)} kernels match expectations")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
