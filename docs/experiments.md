# Main-table experiment guide

The repository exposes two self-contained experiment runners: one for the
Class-IL table and one for the Task-IL table. Each runner evaluates GBReplay on
all four datasets by default.

## Default protocol

- Seed: 1024
- Repetitions: 5
- Epochs per task: 300
- Classes per task: 2
- Learning rate: 0.005
- Weight decay: 0.0005
- Class-IL by default; Task-IL is enabled explicitly
- Per-task replay budgets: CoraFull 2, Arxiv 26, Reddit 34, Products 319

GBReplay defaults:

- Propagation residual coefficient: 0.2
- Label-dependent edge coefficient: 0.3
- Alignment iterations: 100
- Prototype learning rate: 0.001
- Auxiliary encoder hidden size: 512
- BallScore weights: size 0.4, information gain 0.4, purity 0.2
- Replay adjacency: identity

## Main-table mapping

| Paper item | Script |
| --- | --- |
| Class-IL main results | `run_class_il.py` |
| Task-IL main results | `run_task_il.py` |

Run scripts from the repository root, for example:

```bash
python experiments/run_class_il.py
```

Run only selected datasets or override the hardware and repetition count:

```bash
python experiments/run_task_il.py \
  --datasets corafull arxiv \
  --device cuda:0 \
  --repeat 5
```

Use `--output-root` to select an explicit result directory. Each dataset gets a
separate `run.log`, and the runner writes a tab-separated `summary.tsv` after
all requested datasets finish.
