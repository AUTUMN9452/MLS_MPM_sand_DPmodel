from math import pi
import taichi as ti
import numpy as np
ti.init(arch=ti.cpu)

# 计算设置
n_grid = 128                  # 粒子数 和 单个方向上的网格数
dx = 1 / n_grid               # 网格的间距
dt = 2e-4                     # 时间增量步
g = ti.Vector([0, -9.8])       #重力

# 砂土材质
n_particle = 10000                                       # 粒子数
rho_0 = 1500                                            # 粒子密度
V0_sand = (dx * 0.5) ** 2                               # 粒子体积
mass0_sand = rho_0 * V0_sand                            # 粒子质量

E,v = 18e6, 0.2                                         # 刚度、泊松比
G, K = int(E/(2*(1+v))), int(E/(3*(1-2*v)))             # 剪切刚度、体积模量
lambda_0 = int(E * v / ((1 + v) * (1 - 2 * v)))         # 计算拉梅常数，剪切模量
f_a, d_a, cohe = 30 * pi/180, 0.1 * pi/180, 1000          # 摩擦角、膨胀角和粘聚力
q_f, q_d = 3 * ti.tan(f_a) / ti.sqrt(9 + 12 * ti.tan(f_a)**2), \
           3 * ti.tan(d_a) / ti.sqrt(9 + 12 * ti.tan(d_a)**2)
k_f = 3 * cohe / ti.sqrt(9 + 12 * ti.tan(f_a)**2)
a_B = ti.sqrt(1 + q_f ** 2) - q_f
max_t_s = cohe / ti.tan(f_a)                            # 最大拉伸强度 maximum tensile strength


# 数据容器——质量点
e = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # 应变
e_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)              # 偏应变
e_v = ti.field(dtype=float, shape=n_particle)                           # 体积应变

delta_e = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)          # 应变增量
delta_e_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)        # 偏应变增量
delta_e_v = ti.field(dtype=float, shape=n_particle)                     # 体积应变增量

omiga = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)            # 自旋张量

sigma = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)            # 应力
S_s = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)              # 偏应力
S_m = ti.field(dtype=float, shape=n_particle)                           # 球应力——标量

C_e = ti.Matrix.field(3, 3, float, shape=1)                             # 弹性刚度矩阵，voigt标记
rho_sand = ti.field(dtype=float, shape=n_particle)                      # 粒子密度
x = ti.Vector.field(2, dtype=float, shape=n_particle)                   # 粒子坐标位置
v = ti.Vector.field(2, dtype=float, shape=n_particle)                   # 粒子速度
C = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # 仿射速度场 affine velocity field,=速度增量
F = ti.Matrix.field(2, 2, dtype=float, shape=n_particle)                # 形变梯度

# 数据容器——网格
grid_v = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))        # 网格点动量或速度 grid node momentum/velocity
grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))                  # 网格点质量 grid node mass
grid_f = ti.Vector.field(2, dtype=float, shape=(n_grid, n_grid))        # 网格点力


@ti.func
def project(p):                                                             # 塑性投影——将试探应力拉回至屈服面
        St_s = S_s[p] + (S_s[p] @ omiga[p].transpose() + omiga[p] @ S_s[p]) * dt + 2 * G * delta_e_s[p]  # 试探偏应力
        St_m = S_m[p] + K * delta_e_v[p]                                                                 # 试探球应力
        St_t = ti.sqrt(0.5 * (St_s[0,0]**2 + St_s[1,1]**2 + 2 * St_s[1,0]**2))                           # 等效剪切应力
        delta_lam = 0.0                                         # 塑性变形大小
        fs = St_t + q_f * St_m - k_f                            # 剪切屈服方程
        hs = St_t - a_B * (St_m - max_t_s)                      # 拉伸屈服方程
        # print(fs, hs, max_t_s)
        if St_m < max_t_s:
            if fs > 0:                                          # 剪切破坏
                # print("剪切破坏")
                delta_lam = fs / (G + K * q_f * q_d)
                S_m[p] = St_m - K * delta_lam * q_d             # 更新球应力
                S_t = k_f - q_f * S_m[p]                        # 更新剪切应力
                S_s[p] = S_t / St_t * St_s                      # 更新偏应力
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
            else:                                               # 未发生破坏
                # print("未发生破坏")
                S_m[p] = St_m                                   # 更新球应力
                S_s[p] = St_s                                   # 更新偏应力
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
        elif St_m >= max_t_s:
            if hs > 0:                                          # 剪切破坏
                # print("剪切破坏")
                delta_lam = fs / (G + K * q_f)
                S_m[p] = St_m - K * delta_lam * q_d             # 更新球应力
                S_t = k_f - q_f * S_m[p]                        # 更新剪切应力
                S_s[p] = S_t / St_t * St_s                      # 更新偏应力
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）
            else:                                               # 拉伸破坏
                # print("拉伸破坏")
                S_m[p] = max_t_s                                # 更新球应力
                S_s[p] = St_s                                   # 更新偏应力
                sigma[p] = S_s[p] + S_m[p] * ti.Matrix.identity(float, 2)  # 柯西应力（矢量）


@ti.kernel
def p2g():                                                  # 根据粒子信息更新网格节点的力、质量、动量
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]   # 重置网格动量
        grid_f[i, j] = [0, 0]   # 重置网格力
        grid_m[i, j] = 0        # 重置网格质量



    for p in x:                                                         # 历遍所有粒子
        F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]        # 形变梯度矩阵
        e_dot = 0.5 * (C[p] + C[p].transpose())                         # 应变率
        delta_e[p] = e_dot * dt                                         # 应变增量
        omiga[p] = 0.5 * (C[p] - C[p].transpose())                      # 旋转速率
        delta_e_v[p] = delta_e[p].trace()                               # 体积应变增量
        rho_sand[p] = rho_sand[p] / (1 + delta_e_v[p])                  # 密度变化 (根据体积应变修改密度)

        # print(delta_e_v[p])

        delta_e_s[p] = delta_e[p] - 0.5 * delta_e_v[p] * ti.Matrix.identity(float, 2)             # 偏应变增量
        project(p)
        base = (x[p] / dx - 0.5).cast(int)                              # 3*3网格基点坐标
        if base[0] < 0 or base[1] < 0 or base[0] >= n_grid - 2 or base[1] >= n_grid - 2:          # 保护机制，防止对错误位置的节点继续运算
            continue
        fx = x[p] / dx - base.cast(float)                                                         # 粒子与基点的 相对距离
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (1.5 - (2 - fx)) ** 2]            # 权重-对应于三个网格
        grad_w = [fx - 1.5, 2 - 2 * fx, fx - 0.5]                                                 # 权重梯度 todo 核函数梯度在更新网格节点力时应用
        for i, j in ti.static(ti.ndrange(3, 3)):                                                  # 对粒子周围3x3网格开始历遍
            offset = ti.Vector([i, j])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1]
            grad_weight = ti.Matrix([grad_w[i][0] * w[j][1] / dx, w[i][0] * grad_w[j][1] / dx])
            # print("权重：",weight)
            # print("权重梯度：",grad_weight)
            grid_v[base + offset] += weight * mass0_sand * (v[p] + C[p] @ dpos)              # 网格动量
            grid_m[base + offset] += weight * mass0_sand                                     # 网格质量
            grid_f[base + offset] += - mass0_sand /rho_sand[p] * sigma[p] @ grad_weight        # 内力 (重力在网格处更新)
    # print(grid_f[60,10])

@ti.kernel
def update_grid():                                                          # 根据节点力更新速度并施加边界条件
    for i, j in grid_m:                                                     # 历遍所有网格
        if grid_m[i, j] > 0:
            grid_v[i,j] = (grid_v[i,j] + grid_f[i,j] * dt) / grid_m[i,j]    # 更新节点速度
            grid_v[i,j] += dt * (g)           # 重力与节点力作用导致的速度改变

            # print("网格速度为", grid_v[i, j])
            # print("网格力为", grid_f[i, j])

            normal = ti.Vector.zero(float,2)                                # 判断在哪个边界,用以实现摩擦
            if i < 3 and grid_v[i, j][0] < 0:                                        # 取消掉下边界的网格粒子向下的速度，防止粒子穿透
                normal = ti.Vector([1, 0])
            if i > n_grid - 3 and grid_v[i, j][0] > 0:                      # 取消掉上边界的网格粒子向上的速度，防止粒子穿透
                normal = ti.Vector([-1, 0])
            if j < 3 and grid_v[i, j][1] < 0:
                normal = ti.Vector([0, 1])
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                normal = ti.Vector([0, -1])

            if not (normal[0] == 0 and normal[1] == 0):                       # 判断是否在边界
                s = grid_v[i, j].dot(normal)
                if s <= 0:
                    v_normal = s * normal
                    v_tangent = grid_v[i, j] - v_normal
                    vt = v_tangent.norm()
                    if vt > 1e-12: grid_v[i, j] = v_tangent - (vt if vt < -0.5 * s else -0.5 * s) * (v_tangent / vt)


@ti.kernel
def g2p():                                                                              # 通过节点更新粒子速度、仿射速度、位置
    for p in x:                                                                         # 历遍所有粒子
        base = (x[p] / dx - 0.5).cast(int)                                              #
        fx = x[p] / dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 2)                                                # 单个粒子新的速度容器
        new_C = ti.Matrix.zero(float, 2, 2)                                             # 单个粒子新的仿射速度容器
        for i, j in ti.static(ti.ndrange(3, 3)):
            dpos = (ti.Vector([i, j]).cast(float) - fx) * dx                            # 绝对距离
            g_v = grid_v[base + ti.Vector([i, j])]                                      # 3x3中每个网格点的速度
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v                                                       # 重新根据3x3网格的 权重 计算粒子速度
            new_C += 4 * weight * g_v.outer_product(dpos) / dx                          # 更新 C
        v[p], C[p] = new_v, new_C                                                       # 返回 粒子速度和C
        x[p] += dt * v[p]                                                               # 根据速度计算位移


@ti.kernel
def initialize():
    init_pos = ti.Vector([0.4, 0.3])
    cube_size = 0.2
    spacing = 0.002
    num_per_row = (int) (cube_size // spacing)             # 平行于 x 轴的一行粒子数

    for i in range(n_particle):
        floor = i // num_per_row                       # 粒子y坐标（重力方向）
        col = i % num_per_row                          # 粒子x坐标
        x[i] = ti.Vector([col*spacing, floor*spacing]) + init_pos
        v[i] = ti.Matrix([0, 0])
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        rho_sand[i] = rho_0
        sigma[i] = ti.Matrix([[0, 0], [0, 0]])                      # 柯西应力
        S_m[i] = (sigma[i][0,0] + sigma[i][1,1]) * 0.5              # 球应力
        S_s[i] = sigma[i] - S_m[i] * ti.Matrix.identity(float, 2)   # 偏应力


def main():
    initialize()
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color = 0xFFFFFF)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-3 // dt)):                                    # 总时间帧
            print("现在是第"+str(s)+"帧")
            p2g()
            update_grid()
            g2p()
        gui.circles(x.to_numpy(),radius=2,color = 0x000080,)
        print(v[10])
        for i in np.arange(0, 1, 1/n_grid):
            gui.line([0,i], [1,i], radius=0.8, color=0xD0D0D0)
            gui.line([i, 0], [i, 1], radius=0.8, color=0xD0D0D0)
        gui.show()

if __name__ == "__main__":
    main()
