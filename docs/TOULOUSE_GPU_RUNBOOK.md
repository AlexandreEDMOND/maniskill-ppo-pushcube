# Runbook — PC de Toulouse (RTX 3090)

Ce document décrit l'exécution de la reproduction officielle de PPO sur le PC Linux équipé de la RTX 3090. Ne pas lancer cette expérience sur le Mac ni dans WSL : elle nécessite la simulation GPU `physx_cuda` de ManiSkill.

## 1. Récupérer le dépôt

Sur le PC de Toulouse, dans un terminal Linux natif :

```bash
git clone https://github.com/AlexandreEDMOND/maniskill-ppo-pushcube.git
cd maniskill-ppo-pushcube
uv sync --frozen
```

Si le dépôt existe déjà :

```bash
cd maniskill-ppo-pushcube
git pull --ff-only origin main
uv sync --frozen
```

## 2. Vérifier CUDA et Vulkan

Ces commandes doivent toutes réussir avant de lancer PPO :

```bash
nvidia-smi
vulkaninfo --summary
uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__); print(torch.cuda.get_device_name(0))'
uv run python -m mani_skill.examples.demo_random_action -e PushCube-v1 --sim-backend physx_cuda
```

Attendu : `torch.cuda.is_available()` est vrai, la RTX 3090 apparaît, `vulkaninfo` trouve le pilote NVIDIA, et le dernier test termine un épisode aléatoire sans erreur.

Si l'une de ces commandes échoue, ne pas lancer les entraînements : conserver la sortie d'erreur et corriger CUDA ou Vulkan d'abord.

## 3. Connecter le suivi Weights & Biases

La commande de référence ManiSkill utilise le suivi W&B. Une seule connexion est nécessaire :

```bash
uv run wandb login
```

## 4. Lancer les trois seeds officielles

Lancer les seeds séquentiellement, une à la fois, sur la même GPU :

```bash
for seed in 9351 4796 1788; do
  uv run scripts/run_official_baseline.py --profile official --seed "$seed"
done
```

La configuration est figée : `PushCube-v1`, observations `state`, Panda, `pd_joint_delta_pos`, 4096 environnements, 50 millions d'interactions par seed, PPO avec CUDA graphs.

Dans un second terminal, le suivi GPU peut se faire avec :

```bash
watch -n 5 nvidia-smi
```

## 5. Évaluer chaque checkpoint final

Après la fin des trois entraînements, lancer l'évaluation déterministe et conserver un dossier distinct par seed :

```bash
for seed in 9351 4796 1788; do
  uv run scripts/evaluate.py \
    "runs/ppo-PushCube-v1-state-${seed}-walltime_efficient/final_ckpt.pt" \
    --output-dir "artifacts/evaluations/official-${seed}"
done
```

Chaque dossier contient :

- `evaluation.json` : métriques agrégées et résultats par seed d'évaluation ;
- `videos/` : une vidéo déterministe ;
- le hash SHA-256 du checkpoint évalué dans le rapport.

Les métriques à vérifier sont le taux de réussite, le retour, la longueur d'épisode, les étapes jusqu'à la réussite et la distance finale cube-cible.

## 6. Critère de fin

La baseline est considérée reproduite lorsque les trois entraînements sont terminés et que leurs évaluations déterministes atteignent au moins 90 % de réussite. Les résultats doivent ensuite être comparés sous la forme moyenne ± écart-type sur les trois seeds.

## En cas de problème

- `CUDA unavailable` : ne pas modifier les dépendances du dépôt à la main ; vérifier le pilote NVIDIA et la version de PyTorch CUDA installée.
- erreur Vulkan/SAPIEN : vérifier `nvidia-smi`, puis `vulkaninfo --summary` et la configuration du pilote NVIDIA.
- erreur W&B : relancer `uv run wandb login` avant de reprendre la seed concernée.
- interruption d'un run : conserver le dossier `runs/` ; ne pas supprimer les checkpoints sans avoir vérifié les possibilités de reprise.

La référence complète des paramètres et de la provenance upstream est disponible dans [OFFICIAL_BASELINE.md](OFFICIAL_BASELINE.md).
