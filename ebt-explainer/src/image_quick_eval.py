"""Quick sanity eval for the toy denoising runs (python3 -I src/image_quick_eval.py <npz> <run_dir> <mode>)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax
import jax.numpy as jnp
import image_ebt as M


def psnr(a, b):
    a = np.clip((np.asarray(a) + 1) * 127.5, 0, 255)
    b = np.clip((np.asarray(b) + 1) * 127.5, 0, 255)
    mse = np.mean((a - b) ** 2, axis=(1, 2, 3))
    return 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-8)), mse


def main(npz, run_dir, mode):
    tr, te, _ = M.load_data(npz)
    y = jnp.asarray(te[:200])
    p = {k: jnp.asarray(v) for k, v in np.load(os.path.join(run_dir, f"{mode}_params.npz")).items()}
    key = jax.random.PRNGKey(123)
    for sig in [0.1, 0.2]:
        t = M.t_of_sigma(sig)
        x, _ = M.add_noise(jax.random.fold_in(key, t), y, t)
        pn, _ = psnr(x / np.sqrt(M.abar_of_t(t)), y)
        print(f"sigma={sig} t={t} abar={float(M.abar_of_t(t)):.4f} noisy(rescaled) PSNR={pn.mean():.2f}")
        if mode == "ebt":
            y0 = jax.random.normal(jax.random.fold_in(key, 7), y.shape)
            g = jax.jit(M.energy_grad)
            en = jax.jit(M.energy)
            yh = y0
            for i in range(1, 17):
                yh = yh - M.HP["ebt_alpha"] * g(p, x, yh)
                if i in (1, 2, 3, 4, 6, 8, 12, 16):
                    ps, _ = psnr(yh, y)
                    print(f"  step {i:2d} PSNR={ps.mean():.2f} E={float(jnp.mean(en(p, x, yh))):.3f}")
        elif mode == "ff":
            ps, _ = psnr(M.trunk(p, x), y)
            print(f"  FF PSNR={ps.mean():.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
