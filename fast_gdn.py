"""
Regla delta con compuerta (Gated DeltaNet) en forma *chunked* y diferenciable,
escrita en MLX puro.

mlx_lm usa un kernel Metal (sin VJP) en inferencia y, en entrenamiento, un bucle
token a token muy lento. Esta versión reproduce la misma recurrencia

    S_t = g_t · S_{t-1} + k_t ⊗ β_t (v_t − g_t · S_{t-1}ᵀ k_t)
    y_t = S_t q_t

con el algoritmo por bloques (representación WY) de Yang et al. (2024), de modo
que el coste es O(T·C) matmuls en lugar de T pasos secuenciales. Se instala con
`patch_mlx_lm()` sustituyendo `mlx_lm.models.gated_delta.gated_delta_ops`.
"""
from __future__ import annotations

import mlx.core as mx

import mlx_lm.models.gated_delta as _gd

CHUNK = 64


def _unit_lower_inverse(M: mx.array, n_iter: int | None = None) -> mx.array:
    """Inversa exacta y numéricamente estable de una matriz unitriangular inferior
    [..., C, C] por inversión en bloques con duplicación (log2 C niveles):
        [[a, 0], [c, d]]^{-1} = [[a^{-1}, 0], [-d^{-1} c a^{-1}, d^{-1}]]
    (Newton-Schulz es inestable aquí: las potencias intermedias de A se disparan.)"""
    C = M.shape[-1]
    lead = M.shape[:-2]
    s = 1
    inv = mx.ones(lead + (C, 1, 1), dtype=M.dtype)  # inversas de los bloques diagonales 1x1
    while s < C:
        nb = C // (2 * s)
        # bloques diagonales de tamaño 2s de M: [..., nb, 2s, 2s]
        D = M.reshape(lead + (nb, 2 * s, nb, 2 * s))
        D = mx.diagonal(D, axis1=-4, axis2=-2)          # [..., 2s, 2s, nb]
        D = mx.moveaxis(D, -1, -3)                       # [..., nb, 2s, 2s]
        c = D[..., s:, :s]                               # bloque inferior-izquierdo
        a = inv[..., 0::2, :, :]
        d = inv[..., 1::2, :, :]
        zeros = mx.zeros(lead + (nb, s, s), dtype=M.dtype)
        top = mx.concatenate([a, zeros], axis=-1)
        bottom = mx.concatenate([-(d @ c @ a), d], axis=-1)
        inv = mx.concatenate([top, bottom], axis=-2)     # [..., nb, 2s, 2s]
        s *= 2
    return inv[..., 0, :, :]


def gated_delta_chunked(q, k, v, g, beta, state=None, mask=None, chunk: int = CHUNK):
    """Misma interfaz que mlx_lm.models.gated_delta.gated_delta_ops.
    q,k: [B,T,Hk,Dk]  v: [B,T,Hv,Dv]  g: [B,T,Hv] (decaimiento en (0,1])  beta: [B,T,Hv]
    state: [B,Hv,Dv,Dk]. Devuelve y [B,T,Hv,Dv] y state."""
    assert mask is None, "la versión chunked no soporta máscaras de padding (usar batch=1)"
    assert g.ndim == 3, "solo gating escalar por cabeza"
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    in_dtype = q.dtype
    if (rep := Hv // Hk) > 1:
        q = mx.repeat(q, rep, -2)
        k = mx.repeat(k, rep, -2)
    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)

    # a [B,H,T,D] en float32
    q = q.astype(mx.float32).transpose(0, 2, 1, 3)
    k = k.astype(mx.float32).transpose(0, 2, 1, 3)
    v = v.astype(mx.float32).transpose(0, 2, 1, 3)
    beta = beta.astype(mx.float32).transpose(0, 2, 1)
    logg = mx.log(mx.maximum(g.astype(mx.float32), 1e-30)).transpose(0, 2, 1)  # log-decaimiento ≤ 0

    pad = (chunk - T % chunk) % chunk
    if pad:
        q = mx.pad(q, [(0, 0), (0, 0), (0, pad), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, 0), (0, pad), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (0, pad), (0, 0)])
        beta = mx.pad(beta, [(0, 0), (0, 0), (0, pad)])
        logg = mx.pad(logg, [(0, 0), (0, 0), (0, pad)])
    L = T + pad
    N = L // chunk
    C = chunk

    vb = v * beta[..., None]
    kb = k * beta[..., None]
    q, k, vb, kb = (x.reshape(B, Hv, N, C, x.shape[-1]) for x in (q, k, vb, kb))
    logg = logg.reshape(B, Hv, N, C)
    gc = mx.cumsum(logg, axis=-1)  # decaimiento acumulado dentro del bloque

    tril = mx.tril(mx.ones((C, C), dtype=mx.bool_))
    strict = mx.tril(mx.ones((C, C), dtype=mx.bool_), k=-1)
    # D_ij = exp(gc_i - gc_j) para i >= j
    decay = mx.where(tril, mx.exp(mx.minimum(gc[..., :, None] - gc[..., None, :], 0.0)), 0.0)
    A = mx.where(strict, (kb @ k.swapaxes(-1, -2)) * decay, 0.0)  # estrictamente inferior
    Tinv = _unit_lower_inverse(mx.eye(C, dtype=A.dtype) + A)  # (I + A)^{-1}
    u = Tinv @ vb                                  # [B,H,N,C,Dv]
    w = Tinv @ (kb * mx.exp(gc)[..., None])        # [B,H,N,C,Dk]

    # estado en convención [Dk, Dv] para los productos
    S = state.astype(mx.float32).swapaxes(-1, -2)  # [B,H,Dk,Dv]
    outs = []
    for i in range(N):
        q_i, k_i, u_i, w_i, gc_i = q[:, :, i], k[:, :, i], u[:, :, i], w[:, :, i], gc[:, :, i]
        attn = mx.where(tril, (q_i @ k_i.swapaxes(-1, -2)) * decay[:, :, i], 0.0)
        v_new = u_i - w_i @ S
        o_intra = attn @ v_new
        o_inter = (q_i * mx.exp(gc_i)[..., None]) @ S
        outs.append(o_intra + o_inter)
        g_last = gc_i[..., -1:]
        S = S * mx.exp(g_last)[..., None] + (
            (k_i * mx.exp(g_last - gc_i)[..., None]).swapaxes(-1, -2) @ v_new
        )
    y = mx.concatenate(outs, axis=2)[:, :, :T]  # [B,H,T,Dv]
    y = y.transpose(0, 2, 1, 3).astype(in_dtype)
    return y, S.swapaxes(-1, -2)


_ORIGINAL = _gd.gated_delta_ops


def patch_mlx_lm(enable: bool = True):
    """Sustituye el bucle secuencial de mlx_lm por la versión chunked (o lo restaura)."""
    _gd.gated_delta_ops = gated_delta_chunked if enable else _ORIGINAL


if __name__ == "__main__":
    import time

    mx.random.seed(0)
    B, T, H, Dk, Dv = 1, 203, 16, 128, 128
    q = mx.random.normal((B, T, H, Dk)) * 0.1
    k = mx.random.normal((B, T, H, Dk)) * 0.1
    v = mx.random.normal((B, T, H, Dv))
    g = mx.sigmoid(mx.random.normal((B, T, H)) + 2)  # decaimientos en (0,1)
    beta = mx.sigmoid(mx.random.normal((B, T, H)))
    y_ref, s_ref = _ORIGINAL(q, k, v, g, beta)
    y_new, s_new = gated_delta_chunked(q, k, v, g, beta)
    mx.eval(y_ref, s_ref, y_new, s_new)
    print("max |y_ref - y_new| =", float(mx.abs(y_ref - y_new).max()), " escala y:", float(mx.abs(y_ref).max()))
    print("max |S_ref - S_new| =", float(mx.abs(s_ref - s_new).max()), " escala S:", float(mx.abs(s_ref).max()))

    # gradiente respecto a v comprobado contra la referencia
    def f_ref(v):
        return _ORIGINAL(q, k, v, g, beta)[0].sum()

    def f_new(v):
        return gated_delta_chunked(q, k, v, g, beta)[0].sum()

    g_ref = mx.grad(f_ref)(v)
    g_new = mx.grad(f_new)(v)
    mx.eval(g_ref, g_new)
    print("max |grad_ref - grad_new| =", float(mx.abs(g_ref - g_new).max()), " escala:", float(mx.abs(g_ref).max()))

    for name, fn in (("secuencial", f_ref), ("chunked", f_new)):
        gr = mx.grad(fn)
        mx.eval(gr(v))
        t = time.time()
        for _ in range(3):
            mx.eval(gr(v))
        print(f"{name}: {(time.time() - t) / 3:.3f} s por fwd+bwd (T={T})")
