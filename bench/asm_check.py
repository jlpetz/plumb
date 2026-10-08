#!/usr/bin/env python3
"""asm_check.py: the codegen gate for plumb (rule: read the asm, don't infer codegen from
timing).

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
  python asm_check.py --density     # also print instructions per key memory op (twin metric)
  python asm_check.py --package plumb_tiles --example tile_kernels   # check a crate's example
  python asm_check.py --package plumb_tiles --example tile_kernels --toolchain nightly --features nightly
  python asm_check.py --package plumb_lines --example asm_kernels --toolchain stable
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
#   loop_forbid: mnemonics that must not appear in any innermost loop
#   stack_ok: reason; vector stack operands in loops are intended (e.g. an L1 scratch buffer),
#            not spills
#   twin:    a re.sub template on the matched name giving the TMR-style twin kernel; for every
#            key memory op kind in the twin's innermost loops (load, store, nt, flush, movdir64b,
#            tile)
#            this kernel's best innermost loop may use at most `twin_tol` (default 0.25) more
#            instructions per op
#   max_per_op: {kind: n}: this kernel's best innermost loop may use at most n instructions per
#            op of that kind (for kernels with no twin, e.g. plumb_lines' own example)
# Default for every kernel: no memset/memcpy, and no calls or vector spills in innermost loops
# (a call in an outer loop, like TMR's flush once per chunk, is fine).
# --------------------------------------------------------------------------------------------
EXPECT = [
    # Byte-uniform canary: reported, never failed. A memset here is the known trap; a surviving
    # loop is pass-order luck, not protection (see FINDINGS-TODO84.md), so it is flagged as a NOTE.
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


def build(cargo_target, toolchain=None):
    env = dict(os.environ, CARGO_TARGET_DIR=TARGET)
    cmd = (["cargo"] + ([f"+{toolchain}"] if toolchain else []) + ["rustc", "--release"] + cargo_target +
           ["--", "--emit", "asm", "-C", "llvm-args=-x86-asm-syntax=intel"])
    print("building:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def latest_s(stem, pkg=None):
    # Older cargo (stable today): release/deps/<stem>-HASH.s, examples in release/examples/;
    # newer build-dir layout (nightly): release/build/<pkg>/HASH/out/<stem>.s. Both can exist at
    # once, and two packages can have an example of the same name (and so share
    # release/examples/<stem>.s), so keep the files whose symbols name the package, then take the
    # newest.
    files = glob.glob(os.path.join(TARGET, "release", "**", f"{stem}*.s"), recursive=True)
    if pkg:
        crate = pkg.replace("-", "_")
        own = []
        for f in files:
            with open(f, encoding="utf-8", errors="replace") as fh:
                if crate in fh.read():
                    own.append(f)
        files = own or files
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


def blocks(ins):
    """Basic blocks as (start, end) index pairs (end inclusive) and their successor lists.
    A block starts at a label or after a jump/ret/ud2; `call` doesn't end one."""
    bl, cur = [], None
    for i, (lab, txt) in enumerate(ins):
        if lab is not None:
            if cur is not None:
                bl.append((cur, i - 1))
            cur = i
            continue
        if cur is None:
            cur = i
        op = txt.split()[0] if txt else ""
        if op.startswith("j") or op in ("ret", "ud2", "int3"):
            bl.append((cur, i))
            cur = None
    if cur is not None:
        bl.append((cur, len(ins) - 1))
    at = {ins[a][0]: k for k, (a, _) in enumerate(bl) if ins[a][0] is not None}
    succ = []
    for k, (_, b) in enumerate(bl):
        txt = ins[b][1] or ""
        op = txt.split()[0] if txt else ""
        nxt = [k + 1] if k + 1 < len(bl) else []
        m = re.match(r"^j\w*\s+(\S+)$", txt)
        tgt = [at[m.group(1)]] if m and m.group(1) in at else []
        if op == "jmp":
            succ.append(tgt)  # a non-local target is a tail call: no successor here
        elif op.startswith("j"):
            succ.append(tgt + nxt)
        elif op in ("ret", "ud2", "int3"):
            succ.append([])
        else:
            succ.append(nxt)
    return bl, succ


def loops(ins):
    """Loops as (sorted instruction indices, innermost) pairs: a loop-nesting forest from
    recursive SCC decomposition of the block graph. Each strongly connected component with a cycle
    is a loop; cutting the edges into its entry blocks from inside it exposes the nested loops.
    This needs no dominators, so a guard that jumps into the middle of a rotated loop (LLVM does)
    doesn't make the inner loop swallow the outer one, and a backward branch whose target can't
    get back to it (block layout, e.g. into a block that falls through to `ret`) isn't a loop."""
    bl, succ = blocks(ins)
    pred = [[] for _ in bl]
    for k, ss in enumerate(succ):
        for t in ss:
            pred[t].append(k)

    def sccs(nodes, cut):
        """Tarjan, iterative, over `nodes` without the `cut` edges."""
        index, low, on, st, out, n = {}, {}, set(), [], [], [0]
        for root in sorted(nodes):
            if root in index:
                continue
            work = [(root, iter(succ[root]))]
            index[root] = low[root] = n[0]
            n[0] += 1
            st.append(root)
            on.add(root)
            while work:
                v, it = work[-1]
                for w in it:
                    if w not in nodes or (v, w) in cut:
                        continue
                    if w not in index:
                        index[w] = low[w] = n[0]
                        n[0] += 1
                        st.append(w)
                        on.add(w)
                        work.append((w, iter(succ[w])))
                        break
                    if w in on:
                        low[v] = min(low[v], index[w])
                else:
                    work.pop()
                    if work:
                        low[work[-1][0]] = min(low[work[-1][0]], low[v])
                    if low[v] == index[v]:
                        comp = set()
                        while True:
                            w = st.pop()
                            on.discard(w)
                            comp.add(w)
                            if w == v:
                                break
                        out.append(comp)
        return out

    res = []

    def nest(nodes, cut):
        found = False
        for comp in sccs(nodes, cut):
            if len(comp) == 1:
                (v,) = comp
                if v not in succ[v] or (v, v) in cut:
                    continue
            found = True
            entries = [v for v in comp if any(p not in comp for p in pred[v])] or [min(comp)]
            inner_cut = cut | {(p, e) for e in entries for p in pred[e] if p in comp}
            has_child = nest(comp, inner_cut)
            idx = sorted(i for k in comp for i in range(bl[k][0], bl[k][1] + 1))
            res.append((idx, not has_child))
        return found

    nest(set(range(len(bl))), frozenset())
    return res


# Key memory ops per kind, for the twin density comparison.
NT_OPS = ("movntdq", "vmovntdq", "movnti", "movntps", "vmovntps", "movntpd", "vmovntpd")
FLUSH_OPS = ("clflushopt", "clflush", "clwb")


def mem_kind(txt):
    """The key memory op an instruction is ('nt', 'store', 'load', 'flush', 'movdir64b', 'tile'),
    or None. Stack ([rsp]) and constant ([rip]) operands don't count: they're overhead, not
    traffic."""
    op = txt.split()[0]
    if op in NT_OPS:
        return "nt"
    if op in ("tileloadd", "tileloaddt1", "tilestored"):
        return "tile" if "[rsp" not in txt else None
    if op in FLUSH_OPS:
        return "flush"
    if op == "movdir64b":
        return "movdir64b"
    ops = txt[len(op):].split(",")
    if "ptr [" not in txt or "[rsp" in txt or "[rip" in txt or not re.search(r"\b[xyz]mm\d+\b", txt):
        return None
    if "ptr [" in ops[0] and op.startswith(("vmov", "mov")):
        return "store"
    return "load"


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
           "loop_calls": set(), "libcalls": set(), "spills": 0, "nloops": 0, "loop_text": [], "loop_sizes": [],
           "density": {}, "loop_ops": set()}
    for f in fset:
        ins = instrs(funcs[f])
        lps = loops(ins)
        rep["nloops"] += len(lps)
        in_loop = set()
        for lp, _ in lps:
            in_loop.update(lp)
            body = [ins[i][1] for i in lp if ins[i][1] and not ins[i][1].startswith("#")]
            rep["loop_text"].append((f, body))
            rep["loop_sizes"].append(len(body))
        # Innermost loops contain no other loop. A call there costs per element (the failure
        # this gate exists for); a call in an outer loop, e.g. TMR's flush once per chunk, doesn't.
        # Exit blocks (ending in ret) can't reach a latch, so they're never in a loop and their
        # stack traffic (the Windows x64 xmm6-15 restore) isn't counted as spills.
        inner = set()
        for lp, innermost in lps:
            if innermost:
                inner.update(lp)
                # Twin density: instructions per key memory op of each kind in this loop; the
                # kernel's figure is its densest innermost loop for that kind.
                body = [ins[i][1] for i in sorted(lp) if ins[i][1] and not ins[i][1].startswith("#")]
                counts = {}
                for t in body:
                    kd = mem_kind(t)
                    if kd:
                        counts[kd] = counts.get(kd, 0) + 1
                for kd, c in counts.items():
                    d = len(body) / c
                    rep["density"][kd] = min(rep["density"].get(kd, d), d)
        for i in inner:
            if ins[i][1]:
                rep["loop_ops"].add(ins[i][1].split()[0])
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


def check(name, rep, reps=None):
    exp = {}
    for pat, e in EXPECT:
        if re.search(pat, name):
            exp.update(e)
            if "twin" in e:
                exp["twin"] = re.sub(pat, e["twin"], name)
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
        if rep["spills"] and not exp.get("loop_calls_ok") and not exp.get("stack_ok"):
            fails.append(f"{rep['spills']} vector stack ops in loop")
    for n in exp.get("need", []):
        if n not in rep["special"]:
            fails.append(f"missing {n}")
    for n in exp.get("loop_need", []):
        if n not in rep["loop_special"]:
            fails.append(f"{n} not in a loop")
    if "need_any" in exp and not any(n in rep["special"] for n in exp["need_any"]):
        fails.append("missing " + "/".join(exp["need_any"]))
    for kd, mx in sorted(exp.get("max_per_op", {}).items()):
        d = rep["density"].get(kd)
        if d is None:
            fails.append(f"no {kd} loop")
        elif d > mx + 1e-9:
            fails.append(f"{kd} {d:.2f} instrs/op > {mx}")
    bad = sorted(rep["loop_ops"] & set(exp.get("loop_forbid", [])))
    if bad:
        fails.append("in innermost loop: " + ",".join(bad))
    # Twin: per key memory op kind the TMR-style twin has, this kernel's densest innermost loop
    # must be within `twin_tol` (default 25%) of the twin's instructions per op.
    if "twin" in exp and reps is not None:
        twin = reps.get(exp["twin"])
        if twin is None:
            fails.append(f"twin {exp['twin']} not found")
        else:
            tol = exp.get("twin_tol", 0.25)
            for kd, td in sorted(twin["density"].items()):
                d = rep["density"].get(kd)
                if d is None:
                    fails.append(f"no {kd} loop (twin {td:.2f}/op)")
                elif d > td * (1 + tol) + 1e-9:
                    fails.append(f"{kd} {d:.2f} instrs/op vs twin {td:.2f}")
    return fails


def twins_table(kernels, funcs):
    """Print every kernel with a `twin` rule next to its twin, as a markdown table: instructions
    per key memory op in the densest innermost loop, per op kind (lower is better)."""
    reps = {k: analyse(k, funcs) for k in kernels}
    print("| plumb kernel | TMR-style twin | op | plumb instrs/op | twin instrs/op | plumb vs twin |")
    print("|---|---|---|---:|---:|---|")
    for k in kernels:
        twin = None
        for pat, e in EXPECT:
            if "twin" in e and re.search(pat, k):
                twin = re.sub(pat, e["twin"], k)
        if not twin or twin not in reps:
            continue
        mine, theirs = reps[k]["density"], reps[twin]["density"]
        for kd in sorted(theirs):
            d, td = mine.get(kd), theirs[kd]
            if d is None:
                verdict = "missing"
            elif abs(d - td) < 1e-9:
                verdict = "equal"
            elif d < td:
                verdict = f"denser ({(1 - d / td) * 100:.0f}% fewer)"
            else:
                verdict = f"looser ({(d / td - 1) * 100:.0f}% more)"
            ds = f"{d:.2f}" if d is not None else "-"
            print(f"| `{k}` | `{twin}` | {kd} | {ds} | {td:.2f} | {verdict} |")
    return 0


def main():
    args = sys.argv[1:]
    load_expect_files()
    cargo_target, stem = target_spec(args)
    toolchain = args[args.index("--toolchain") + 1] if "--toolchain" in args else None
    features = ["--features", args[args.index("--features") + 1]] if "--features" in args else []
    if "--no-build" not in args:
        build(cargo_target + features, toolchain)
    path = latest_s(stem, cargo_target[1])
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
    if "--twins" in args:
        return twins_table(kernels, funcs)
    print(f"asm: {os.path.relpath(path, HERE)}  ({len(funcs)} functions, {len(kernels)} kernels)\n")
    hdr = (f"{'kernel':<24} {'width':<5} {'loop sizes':<12} {'spill':>5}  {'special (*=in loop)':<40} "
           f"{'calls in loop / libcalls':<26} verdict")
    print(hdr)
    print("-" * len(hdr))
    nfail = 0
    reps = {k: analyse(k, funcs) for k in kernels}
    for k in kernels:
        rep = reps[k]
        sp = " ".join(f"{s}{'*' if s in rep['loop_special'] else ''}x{c}"
                      for s, c in sorted(rep["special"].items()))
        calls = sorted(rep["loop_calls"] | rep["libcalls"])
        calls_s = ", ".join(c.split("::")[-1] if "::" in c else c for c in calls)
        fails = check(k, rep, reps)
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
        if "--density" in args and rep["density"]:
            print(f"{'':<24} instrs/op: " + " ".join(f"{kd} {d:.2f}" for kd, d in sorted(rep["density"].items())))
    print(f"\n{len(kernels) - nfail}/{len(kernels)} kernels match expectations")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
