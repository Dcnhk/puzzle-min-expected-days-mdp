# 第 4 步：可运行参考实现（纯标准库，端到端复现 D* = 649/73）
# 用法：python3 ref_impl.py
# 输出：策略迭代各轮违反数；V*(0,3) = 576/73 ; D*(0,3) = 649/73
from fractions import Fraction as F

P_MAX, R_MAX = 100, 3


def act_values(p, r, V):
    """返回 {(动作): (价值, 后继或 None)}"""
    out = {}
    if r > 0:                                   # U 提升
        if p + 2 >= 100:
            out['U'] = (F(0), None)             # 进入吸收态
        else:
            out['U'] = (V[(p + 2, r - 1)], (p + 2, r - 1))
    if p > 0:                                   # T 尝试（p=0 时禁止！）
        out['T'] = (F(100 - p, 100) * V[(p // 2, r)], (p // 2, r))
    out['E'] = (1 + V[(p, R_MAX)], (p, R_MAX))  # E 结束当天
    return out


# ---------- 阶段 A：浮点值迭代提取候选策略 pi_0 ----------
Vf = {(p, r): 0.0 for p in range(P_MAX + 1) for r in range(R_MAX + 1)}
for _ in range(100000):
    N = dict(Vf)
    for p in range(P_MAX):                      # 仅非吸收态 p=0..99
        for r in range(R_MAX + 1):
            vals = {}
            if r > 0:
                vals['U'] = 0.0 if p + 2 >= 100 else Vf[(p + 2, r - 1)]
            if p > 0:
                vals['T'] = (1 - p / 100) * Vf[(p // 2, r)]
            vals['E'] = 1.0 + Vf[(p, R_MAX)]
            N[(p, r)] = min(vals.values())
    delta = max(abs(N[s] - Vf[s]) for s in Vf)
    Vf = N
    if delta < 1e-9:                            # 收敛判据（实测 958–959 次，随计数口径差 1）
        break
else:                                          # for-else：循环未 break 即未收敛
    raise RuntimeError('值迭代未在 100000 次内收敛：不得带着未收敛的 V 继续，请检查建模')

pi = {}
for p in range(P_MAX):
    for r in range(R_MAX + 1):
        vals = {}
        if r > 0:
            vals['U'] = 0.0 if p + 2 >= 100 else Vf[(p + 2, r - 1)]
        if p > 0:
            vals['T'] = (1 - p / 100) * Vf[(p // 2, r)]
        vals['E'] = 1.0 + Vf[(p, R_MAX)]
        m = min(vals.values())
        pi[(p, r)] = [a for a in vals if vals[a] <= m + 1e-11][0]   # 平局任取


# ---------- 阶段 B + C：精确求值 + 证书验证（策略迭代，直至违反数为 0）----------
def solve(pi):
    """给定候选策略，精确求 J_pi；若非适定则抛异常。返回 (J, 违反数)"""
    alpha = {}
    for r in range(R_MAX + 1):                  # r 升序
        for p in range(P_MAX + 1):              # p 升序（每条边指向更小的键）
            if p >= P_MAX:
                alpha[(p, r)] = {}
                continue
            a = pi[(p, r)]
            if a == 'E':
                alpha[(p, r)] = {p: F(1)}
            elif a == 'T':
                c = F(100 - p, 100)
                alpha[(p, r)] = {k: v * c for k, v in alpha[(p // 2, r)].items()}
            else:
                alpha[(p, r)] = {} if p + 2 >= 100 else dict(alpha[(p + 2, r - 1)])

    A = [[F(0)] * P_MAX for _ in range(P_MAX)]
    b = [F(0)] * P_MAX
    for p in range(P_MAX):
        row = alpha[(p, R_MAX)]
        for pp, c in row.items():
            A[p][pp] += c
        A[p][p] -= F(1)
        b[p] = -sum(row.values())

    M = [A[i][:] + [b[i]] for i in range(P_MAX)]
    for col in range(P_MAX):                    # Fraction 高斯消元
        piv = max(range(col, P_MAX), key=lambda i: abs(M[i][col]))
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        if pv == 0:                             # 奇异 ⇒ π 非适定
            raise RuntimeError("线性系统奇异：候选策略 π 非适定（在其诱导的日级链上吸收概率 < 1，即存在永不通关的情形）。注意这与「不执行 E」不同：日内无限执行是不可能的（U 至多 3 次、T 严格减 p），非适定只能源于日级——例如「p>0 尝试、p=0 结束当天」每天都执行 E，却因到达 p=0 后再不尝试而永不通关。请另取猜测。")
        M[col] = [x / pv for x in M[col]]
        for i in range(P_MAX):
            if i != col and M[i][col] != 0:
                f = M[i][col]
                for j in range(col, P_MAX + 1):
                    M[i][j] -= f * M[col][j]
    J3 = [M[i][P_MAX] for i in range(P_MAX)]

    # 说明：执行 E 后必落到 (pp, 3)，其最优值即 J3[pp]，已包含此后全部未来
    #       的 E 成本，故该项写作 1 + J3[pp]（而非 J[(pp, r)]）。
    #       r = R_MAX 时 (p,3) 本身即结算状态，直接取 J3[p]。
    #       若日内中途直接成功（吸收），alpha 中无该项，行和 < 1，自动体现『成功则不付未来天数』。
    J = {}
    for r in range(R_MAX + 1):
        for p in range(P_MAX):
            J[(p, r)] = (sum(c * (1 + J3[pp]) for pp, c in alpha[(p, r)].items())
                         if r < R_MAX else J3[p])
        J[(P_MAX, r)] = F(0)

    viol = 0
    improved = {}
    for r in range(R_MAX + 1):
        for p in range(P_MAX):
            vs = act_values(p, r, J)
            best = min(v for v, _ in vs.values())
            if J[(p, r)] != best:
                viol += 1
            improved[(p, r)] = [a for a, (v, _) in vs.items() if v == best][0]
    return J, viol, improved


it = 0
while True:
    it += 1
    J, viol, improved = solve(pi)
    print(f"策略迭代第 {it} 轮：Bellman 违反数 = {viol}/{P_MAX * (R_MAX + 1)}")
    if viol == 0:
        break
    if it > 50:
        raise RuntimeError("策略迭代未收敛")
    pi = improved                               # 贪心改进后重跑

print("V*(0,3) =", J[(0, 3)])
print("D*(0,3) =", 1 + J[(0, 3)], "=", float(1 + J[(0, 3)]))

# ---------- 复现正文所称的『全局最小非零动作间隙』与『唯一平局』 ----------
gaps = []
ties = []
for r in range(R_MAX + 1):
    for p in range(P_MAX):
        vs = sorted(v for v, _ in act_values(p, r, J).values())
        if len(vs) > 1:                          # 单动作状态（即 (0,0)，全状态唯一的单动作态）间隙无定义，跳过
            if vs[1] > vs[0]:
                gaps.append((vs[1] - vs[0], (p, r)))
            else:                                # vs[1] == vs[0]：零间隙（平局）
                ties.append((p, r))
gmin, s_arg = min(gaps)
print(f"全局最小非零动作间隙 = {gmin} ≈ {float(gmin):.3e}（在 {s_arg} 处）")
print(f"零间隙（平局）状态数 = {len(ties)}：{ties}")

# ---------- 复现正文所称的『r = 1 层 T 态连续块』（非逐点交替的证据） ----------
def _blocks(ps):
    out, s, pv = [], ps[0], ps[0]
    for p in ps[1:]:
        if p != pv + 1:
            out.append((s, pv)); s = p
        pv = p
    out.append((s, pv))
    return out

T1 = [p for p in range(P_MAX) if min(act_values(p, 1, J).items(), key=lambda kv: kv[1][0])[0] == 'T']
print(f"r = 1 层 T 态连续块 = {_blocks(T1)}（共 {len(T1)} 个 T 态，最高 p = {max(T1)}）")