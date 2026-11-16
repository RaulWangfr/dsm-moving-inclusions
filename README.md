# Direct sampling for moving inclusions — numerical code

Code that produces the numerical experiments of

> **Hongjie Li, Fengru Wang, Jun Zou**,
> *Direct sampling for moving inclusions: construction and conditional
> localization*, 2026.

The repository contains an explicit three-dimensional finite-element solver for
the acoustic wave equation with a **moving** material contrast, and the
time-domain **direct sampling method (DSM)** index function used to track the
support of the effective source, and hence the moving object.

Two pieces only: a forward solver and an imaging formula. Nothing is fitted.

---

## 1. What is computed

Forward problem, on the truncated ball $\Omega_{\rm FEM}=B(0,7)$:

```
rho(t,x) p_tt - div( mu(t,x) grad p ) + gamma(t,x) p_t + sigma(t,x) p = f(t,x)
```

* background `rho = mu = 1`, `gamma = sigma = 0` (wave speed `c0 = 1`);
* inside the moving inclusion `omega(t) = B(xi(t), 0.3)` the coefficients
  `rho_in, mu_in, gamma_in, sigma_in` differ (see Table 1 below);
* first-order absorbing condition `(d_n + d_t) p = 0` on `dOmega_FEM`;
* the object moves on `xi(t) = 0.5 (cos(pi t / 10), sin(pi t / 10), 0)`.

Inverse step, the index function of the paper:

```
I(s,z) = (1/4pi) int_Gamma (yhat . n) p^s( s + |y - z| / c0 , y ) / |y-z| dS(y)
```

evaluated on the plane `z = 0`, on a `101 x 101` grid in `[-2,2]^2`, at
`s = 2, 4, ..., 18`.

---

## 2. Layout

```
config.py             every number of the paper (mesh, time step, pulse train, cases)
run_case.py           run one case: MS, MI1, MI2, MI3
run_all.py            run the four cases in sequence
dsm/mesh.py           gmsh ball mesh, CFL estimate, Fibonacci sampling
dsm/interp.py         robust tetrahedral interpolation (KD-tree + CSR)
dsm/fem_base.py       reference explicit FEM (lumped mass, central differences)
dsm/fem_solver.py     multi-threaded numba kernels (used in production)
dsm/inversion.py      receiver sampling and the DSM index function
dsm/experiment.py     CommonParameters / UnifiedExperiment, one interface per case
dsm/driver.py         forward + inversion for the four examples
figures/make_indicator.py        cache |I| to npz (numpy + numba only)
figures/make_montage.py          draw the 3 x 3 binary montage (pure PIL)
figures/make_receiver_surfaces.py  the receiver-configuration figure
```

Data flow: `run_case.py` → `data/forward_<case>.npz` (recorded scattered field)
→ `figures/make_indicator.py` → `outputs/ind_<case>_n<noise>.npz`
→ `figures/make_montage.py` → PNG.

---

## 3. Installation

Python 3.10+ (developed with 3.14), and

```bash
pip install -r requirements.txt      # numpy, numba, scipy, pillow, gmsh
```

Gmsh is only needed to build the mesh; a cached mesh is reused afterwards.
On Windows the gmsh wheel bundles the library, on Linux/macOS it may also need
the `gmsh` system package.

---

## 4. Running

The default mesh (`lc = 0.1`, radius 7) has about 1.0e6 vertices and 6.1e6
tetrahedra; a single forward solve then takes hours. For a smoke test use a
coarser mesh:

```bash
python run_case.py --case MS --lc 0.3 --threads 4      # minutes
```

Paper configuration, one case at a time:

```bash
python run_case.py --case MS            # moving source, 30% noise
python run_case.py --case MI1            # restoring contrast, 10% noise, decimated time grid
python run_case.py --case MI2            # damping contrast, closed frame aperture
python run_case.py --case MI3            # inertia-stiffness contrast, 9x9 aperture
```

Stages can be run separately, so that the expensive solve is done once:

```bash
python run_case.py --case MI1 --stage forward
python run_case.py --case MI1 --stage invert  --noise 0.10
python run_case.py --case MI1 --stage invert  --noise 0.00     # noiseless variant
```

Everything (all four cases):

```bash
python run_all.py --threads 6
```

Figures:

```bash
python figures/make_indicator.py --case MS                     # -> outputs/ind_MS_n30.npz
python figures/make_montage.py outputs/ind_MS_n30.npz MS.png    # -> 3x3 montage
python figures/make_receiver_surfaces.py --data-dir data \
        --out-dir figures/out      # -> fig_{ms,mi1,mi2,mi3}.png (receiver coordinates only)
```

---

## 5. The four examples

| case | kind | contrast inside `omega(t)` | receivers | aperture | noise | time grid |
|------|------|----------------------------|-----------|----------|-------|-----------|
| MS | moving source | — (prescribed source) | 96, Fibonacci sphere `R=2sqrt3` | full, sparse | 30% | every `3h_t` |
| MI1 | moving inclusion | `sigma_in = 1` (restoring) | 768, cube edge 4 | full | 10% | decimated by 15 |
| MI2 | moving inclusion | `gamma_in = 1` (damping) | 448, closed frame `\|y3\| <= 0.75`, masked out of a 1538-point cube-face cloud | partial | 10% | every `3h_t` |
| MI3 | moving inclusion | `rho_in = mu_in = 1.2` (inertia-stiffness) | 81, `9x9` square at `y3 = 2.5`, edge 2 | minimal | 10% | every `3h_t` |

The three difficulties tested are therefore: large noise (MS), a coarse
temporal grid (MI1), and incomplete apertures (MI2, MI3).

The incident field repeats with period `1`: the train of (6.3) has pulse spacing
`1` in MI1-MI3, and the source signal of (6.2) has period `1` in MS. The recorded
interval is `3h_t = 0.0216435`, i.e. about 46 samples per period; for MI1 the
records are decimated by 15 to `45h_t = 0.324653`, leaving about 3 samples per
period.

---

## 6. Discretization (what the solver actually does)

**Space.** Conforming tetrahedral mesh of the ball, linear (`P1`) elements on a
gmsh mesh with target size `lc = 0.1`. Stiffness: element-wise,
`K_e = mu_e * S_e` with the constant-per-element gradient matrix; the matvec is
an element loop with a per-element `mu_e`.

**Mass.** Lumped: node `i` receives `sum_e rho_e * |T_e| / 4` over the tets that
contain it. No consistent mass matrix is assembled, so no linear solve is ever
performed.

**Time.** Explicit central differences,
`p^{n+1} = 2 p^n - p^{n-1} - dt^2 M^{-1} (K p^n + boundary)`.

**Absorbing boundary.** `(d_n + d_t) p = 0` is lumped through the boundary face
areas, `D_i = sum_{faces at i} (1/c) * |F| / 3`; the update at a boundary node is
the central difference of `M p_tt + D p_t + K p = 0`.

**Damping `gamma`.** Treated semi-implicitly:
`p^{n+1} = (v + (dt/2) gamma p^{n-1}) / (1 + (dt/2) gamma)`, so it adds no CFL
restriction.

**Restoring `sigma`.** Explicit, as the term `- dt^2 sigma p^n`. With
`dt = 7.2e-3` and `sigma = 1` this contributes `dt^2 sigma = 5.2e-5`, i.e. the
stability condition `dt^2 sigma < 2` is satisfied by four orders of magnitude.
The binding restriction is therefore the usual explicit one,
`dt <= 0.5 h_min / c_max`, with `c_max = max(1, sqrt(mu_in/rho_in))` and the
safety factor 0.5.

**Moving coefficients.** The inclusion is *element-wise constant*, not
geometrically cut: at every time step the tetrahedra whose **centroid** lies in
`B(xi(t), 0.3)` receive `rho_in, mu_in`, and the lumped mass and the element
`mu` array are rebuilt from them (`dsm/fem_solver.py: material_mass_kernel`).
There is no cut-cell quadrature and no mesh motion; the interface is resolved to
one element (`lc = 0.1` against inclusion radius 0.3, i.e. about six elements
across the diameter), and the coefficient update is first order in `dt`.

**Incident field.** The plane-wave pulse train

```
p^i(t,y) = sum_{m=0}^{25} R(t - y1 - 2 - m),   R(tau) = (1 - 2 a tau^2) exp(-a tau^2),
a = pi^2 f_p^2,  f_p = 0.5
```

is imposed analytically: successive pulses are shifted by **1** time unit, so
the repetition interval of the train is 1 while the peak frequency of a single
pulse is `f_p = 0.5`. Scattering is computed in scattered-field form with the
contrast source built from `p^i`, `d_t p^i`, `d_tt p^i` on the inclusion
elements.

**CFL.** `dt = 0.5 * h_min / c_max`, with `h_min` the minimum tet height of the
mesh and `c_max = max(1, sqrt(mu_in/rho_in))`. For the paper mesh this gives
`h_t = 0.007214509` and the recorded interval `3h_t`.

---

## 7. Data and run time

Forward fields are **not** stored in this repository: the production mesh is
2.6 GB and each forward solve is hour-scale. Generate them with `run_case.py`;
a coarse-mesh smoke test runs in minutes. The figure scripts read

```
data/forward_MS.npz   data/forward_MI1.npz   data/forward_MI2.npz   data/forward_MI3.npz
```

each holding the recorded scattered field `recv_s` (receivers x times), the
receiver coordinates, `dt`, and the case metadata.

Noise is reproducible: the realisation is drawn from
`np.random.default_rng(20260923)` (`config.Config.noise_seed`).

## 8. What the figures look like

The montage draws, for each of the nine times, the binary map of `|I|`
(green where `|I| > 0.75 max|I|` of that panel, white elsewhere), the orange
envelope of the swept support, the black circle and cross for the moving object,
and a red cross at the maximum of `|I|`. Panel frames, tick conventions and the
time ordering belong to the LaTeX caption, not to the image.

---

## 9. License and citation

MIT, see `LICENSE`. If you use this code, please cite the paper above.