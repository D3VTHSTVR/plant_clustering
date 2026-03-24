# Git branching workflow

Repository: [https://github.com/D3VTHSTVR/plant_clustering](https://github.com/D3VTHSTVR/plant_clustering)

| Branch | Purpose |
|--------|---------|
| **prod** | Production-ready code. Default protected branch. |
| **dev** | Integration branch. Merges from `vdev` and `ldev` land here before `prod`. |
| **vdev** | Your development branch. Push work here first. |
| **ldev** | Colleague’s development branch. Push work here first. |

## Flow

1. Do daily work on **vdev** or **ldev** (never commit directly to `prod` or `dev` unless agreed).
2. Open pull requests **into `dev`** from `vdev` or `ldev`.
3. After review and integration, merge **`dev` → `prod`** when releases are ready.

## First-time setup

```bash
git clone https://github.com/D3VTHSTVR/plant_clustering.git
cd plant_clustering
git checkout vdev   # or ldev
```

## Set GitHub default branch to `prod`

In the repo: **Settings → General → Default branch** → set to **prod**.
