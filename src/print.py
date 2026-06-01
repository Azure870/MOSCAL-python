import numpy as np
import os
import sys

from .deom import DEOM
# from .rem_gen import build_hop_sequence, apply_PhiF_powers_product_with_hops_record_root_trace


def collect_second_tier_traces(deom: DEOM, nind):
    """
    返回 shape=(nind,nind) 的矩阵 M，其中
    M[i,j] = Tr( second-tier ADO 对应 (i,j) 的 ddos )
    """
    M = np.zeros((nind, nind), dtype=np.complex128)
    c = deom.c
    rid = c.root_id

    for i in range(nind):
        id1 = c.up[rid, i]
        # 如果 lmax<1 会是 -1
        if id1 == -1:
            continue
        for j in range(nind):
            id2 = c.up[id1, j]  # 对 i==j 也成立：root->(i)->(i+i)
            if id2 == -1:
                continue
            M[i, j] = np.trace(deom.ddos[id2])

    return M


def print_moscal_banner():
    title = "MOSCAL 2.2"
    subtitle = "Open Quantum Systems Toolkit"
    authors = [
        ("Zi-Hao Chen", ["czh5@mail.ustc.edu.cn", "chenzihao@hkqai.hk"]),
        ("Yu Su",       ["suyupilemao@mail.ustc.edu.cn"]),
        ("Yao Wang",    ["wy2010@ustc.edu.cn"]),
        ("YiJing Yan",  ["yanyj@ustc.edu.cn"]),
    ]

    # --- feature detection ---
    def _supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if not sys.stdout.isatty():
            return False
        term = os.environ.get("TERM", "")
        if term in ("dumb", ""):
            return False
        return True

    def _supports_unicode_box() -> bool:
        if not sys.stdout.isatty():
            return False
        enc = (sys.stdout.encoding or "").lower()
        return "utf" in enc

    use_color = _supports_color()
    use_unicode = _supports_unicode_box()

    # --- styling helpers ---
    if use_color:
        C = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "dim": "\033[2m",
            "cyan": "\033[36m",
            "mag": "\033[35m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
        }
    else:
        C = {k: "" for k in ["reset", "bold", "dim",
                             "cyan", "mag", "green", "yellow", "blue"]}

    if use_unicode:
        TL, TR, BL, BR = "┏", "┓", "┗", "┛"
        H, V = "━", "┃"
        L, R = "┣", "┫"
        bullet = "•"
    else:
        TL, TR, BL, BR = "+", "+"
        H, V = "-", "|"
        L, R = "+", "+"
        bullet = "*"

    # --- build content lines ---
    name_w = max(len(name) for name, _ in authors)
    body = []
    for name, emails in authors:
        body.append(f"{C['bold']}{name:<{name_w}}{C['reset']}  ")
        # first email on same line
        body[-1] += f"{C['dim']}{bullet}{C['reset']} {emails[0]}"
        # remaining emails as indented bullets
        for e in emails[1:]:
            body.append(f"{'':<{name_w}}  {C['dim']}{bullet}{C['reset']} {e}")

    # header block
    header_lines = [
        f"{C['mag']}{C['bold']}{title}{C['reset']}",
        f"{C['dim']}{subtitle}{C['reset']}",
    ]

    # compute box width based on visible length (strip ansi)
    import re
    ansi_re = re.compile(r"\x1b\[[0-9;]*m")

    def vis_len(s: str) -> int:
        return len(ansi_re.sub("", s))

    content = []
    content.extend(header_lines)
    content.append("")  # blank line
    content.append(f"{C['cyan']}{C['bold']}Authors{C['reset']}")
    content.extend(body)

    content_w = max(vis_len(s) for s in content)
    box_w = content_w + 4  # padding

    top = TL + H * (box_w - 2) + TR
    mid = L + H * (box_w - 2) + R
    bot = BL + H * (box_w - 2) + BR

    def line(s: str) -> str:
        pad = content_w - vis_len(s)
        return f"{V} {s}{' ' * pad} {V}"

    # --- print ---
    print(top)
    # centered title line (vis-centered)
    t = header_lines[0]
    tpad = content_w - vis_len(t)
    left = tpad // 2
    right = tpad - left
    print(f"{V} {' ' * left}{t}{' ' * right} {V}")

    s = header_lines[1]
    spad = content_w - vis_len(s)
    left = spad // 2
    right = spad - left
    print(f"{V} {' ' * left}{s}{' ' * right} {V}")

    print(mid)
    for s in content[2:]:  # after header lines (includes blank + authors)
        print(line(s))
    print(bot)
