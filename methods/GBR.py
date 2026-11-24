import torch
import math
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor,matmul
from methods.replay import Replay
from backbones.encoder import Encoder
import numpy as np
from collections import Counter
from typing import Dict, Tuple


class GBR(Replay):
    def __init__(self, model, tasks, budget, m_update, device, args):
        super().__init__(model, tasks, budget, m_update, device)
        self.n_encoders = args['n_encoders']
        self.feat_lr = args['feat_lr']
        self.hid_dim = args['hid_dim']
        self.emb_dim = args['emb_dim']
        self.n_layers = args['n_layers']
        self.hop = args['hop']
        self.activation = args['activation']
        self.a = args['a']
        self.budget = budget
        self.device = device
        self.args = args

    def _resolve_ball_hparams(self, num_nodes: int, num_classes: int):
        min_leaf_size = int(self.args.get("ball_min_size", 10))
        purity_stop = float(self.args.get("ball_purity_stop", 0.95))
        entropy_stop = float(self.args.get("ball_entropy_stop", 0.1))
        max_depth = int(self.args.get("ball_max_depth", 10))
        oversample = int(self.args.get("ball_oversample", 4))
        split_mode = "centroid"
        max_svd_samples = 4096
        train_root_only = True

        # Adapt tree capacity to task scale without dataset-specific rules.
        large_graph = int(num_nodes) >= 50000
        many_classes = int(num_classes) >= 20
        complexity = float(math.log2(max(2, int(num_classes))))

        if "ball_oversample" not in self.args and many_classes:
            oversample = int(min(8, max(4, round(math.log2(num_classes + 1)))))
        if "ball_min_size" not in self.args and large_graph:
            min_leaf_size = int(min(64, max(10, num_nodes // 2000)))
        if "ball_purity_stop" not in self.args:
            purity_stop = (
                float(max(0.86, min(0.95, 0.96 - 0.01 * complexity)))
                if large_graph or many_classes
                else 0.95
            )
        if "ball_entropy_stop" not in self.args:
            entropy_stop = (
                float(min(0.22, max(0.08, 0.08 + 0.015 * complexity)))
                if large_graph or many_classes
                else 0.10
            )
        if "ball_max_depth" not in self.args:
            max_depth = (
                int(min(16, max(11, round(10 + complexity))))
                if large_graph or many_classes
                else 10
            )

        max_total_balls = max(1, oversample * int(self.budget) * max(1, num_classes))
        max_total_balls = min(max_total_balls, max(1, num_nodes))

        return {
            "min_leaf_size": min_leaf_size,
            "purity_stop": purity_stop,
            "entropy_stop": entropy_stop,
            "max_depth": max_depth,
            "oversample": oversample,
            "split_mode": split_mode,
            "max_svd_samples": max_svd_samples,
            "train_root_only": train_root_only,
            "max_total_balls": max_total_balls,
        }

    def memorize(self, task, budgets):
        task = task.to(self.device)
        # --budget is a per-task total, split across classes by
        # Replay._assign_budget_per_class distributes the per-task capacity.
        if budgets:
            budget_map = {int(cls): int(b) for cls, b in zip(task.classes, budgets)}
        else:
            budget_map = {int(cls): int(self.budget) for cls in task.classes}
        task.x = self._ppr_diffusion(task)
        feat_emb, labels_emb, _, _ = self._generate_granular_balls(
            task, task.y, budget_map
        )
        return self._embedding(task, feat_emb, labels_emb)

    def _make_replay_adj_t(self, x: torch.Tensor) -> SparseTensor:
        """Build an identity adjacency matrix for edge-free prototypes."""
        n = int(x.size(0))
        return SparseTensor.eye(n).to(x.device)

    def _ppr_diffusion(self, task, n_iter=50, tol=1e-5):
        device = task.x.device
        num_nodes = task.num_nodes
        feat = task.x.to(device)
        labels = task.y.to(device)
        train_mask = task.train_mask.to(device)

        adj = SparseTensor(
            row=task.edge_index[0].to(device),
            col=task.edge_index[1].to(device),
            sparse_sizes=(num_nodes, num_nodes)
        ).set_diag()
        # Propagate along both edge directions, matching the training graph.
        adj = adj.to_symmetric()

        deg = adj.sum(dim=1).to_dense()
        deg_inv = deg.clamp(min=1e-12).pow(-1)
        D_inv = SparseTensor(
            row=torch.arange(num_nodes, device=device),
            col=torch.arange(num_nodes, device=device),
            value=deg_inv,
            sparse_sizes=(num_nodes, num_nodes)
        )
        adj_norm = matmul(D_inv, adj)

        row, col, val = adj_norm.coo()
        known_pair = train_mask[row] & train_mask[col]
        same_class = (labels[row] == labels[col]) & known_pair
        diff_class = (labels[row] != labels[col]) & known_pair
        class_factor = float(self.args.get('class_factor', 0.3))
        modulation = torch.ones_like(val)
        modulation[same_class] = 1.0 + class_factor
        modulation[diff_class] = 1.0 - class_factor
        # class_factor >= 1 would give cross-class edges negative weight and
        # can destabilize the power iteration; clamp to non-negative instead.
        modulation = modulation.clamp_min(0.0)
        new_vals = val * modulation

        adj_class_aware = SparseTensor(row=row, col=col, value=new_vals,
                                       sparse_sizes=(num_nodes, num_nodes))

        deg_new = adj_class_aware.sum(dim=1).to_dense()
        deg_new_inv = deg_new.clamp(min=1e-12).pow(-1)
        D_new_inv = SparseTensor(
            row=torch.arange(num_nodes, device=device),
            col=torch.arange(num_nodes, device=device),
            value=deg_new_inv,
            sparse_sizes=(num_nodes, num_nodes)
        )
        adj_class_norm = matmul(D_new_inv, adj_class_aware)

        alpha = float(self.a)
        feat_diffused = feat.clone()
        for _ in range(n_iter):
            next_feat = alpha * feat + (1 - alpha) * matmul(adj_class_norm, feat_diffused)
            if torch.max(torch.abs(next_feat - feat_diffused)) < tol:
                feat_diffused = next_feat
                break
            feat_diffused = next_feat

        return feat_diffused

    def _should_split(
        self,
        ball_nodes: list,
        labels_list: list,
        *,
        min_leaf_size: int,
        purity_stop: float,
        entropy_stop: float,
        train_mask_list: list = None,
    ) -> bool:
        if len(ball_nodes) < max(2, 2 * int(min_leaf_size)):
            return False

        # Purity/entropy are computed over train-labeled nodes only.
        if train_mask_list is not None:
            train_nodes = [n for n in ball_nodes if train_mask_list[n]]
        else:
            train_nodes = list(ball_nodes)
        if not train_nodes:
            return False

        counts = Counter(labels_list[n] for n in train_nodes)
        if not counts:
            return False

        total = float(len(train_nodes))
        majority_count = float(counts.most_common(1)[0][1])
        purity = majority_count / total
        if purity >= float(purity_stop):
            return False

        h = 0.0
        for cnt in counts.values():
            p = float(cnt) / total
            if p > 0.0:
                h -= p * math.log2(p)
        return h > float(entropy_stop)

    def _split_ball(
        self,
        ball_nodes: list,
        x_norm: np.ndarray,
        *,
        labels_list: list = None,
        train_mask_list: list = None,
        split_mode: str = "pca",
        max_svd_samples: int = 0,
    ) -> Tuple[list, list]:
        if len(ball_nodes) <= 1:
            return ball_nodes, []

        idx_arr = np.asarray(ball_nodes, dtype=np.int64)
        X = x_norm[idx_arr]
        direction = None

        if (
            split_mode == "centroid"
            and labels_list is not None
            and train_mask_list is not None
        ):
            train_in_ball = [n for n in ball_nodes if train_mask_list[n]]
            counts = Counter(labels_list[n] for n in train_in_ball)
            if len(counts) >= 2:
                (c1, _), (c2, _) = counts.most_common(2)
                idx1 = [n for n in train_in_ball if labels_list[n] == c1]
                idx2 = [n for n in train_in_ball if labels_list[n] == c2]
                m1 = x_norm[np.asarray(idx1, dtype=np.int64)].mean(axis=0)
                m2 = x_norm[np.asarray(idx2, dtype=np.int64)].mean(axis=0)
                direction = m1 - m2
                dn = float(np.linalg.norm(direction))
                if dn < 1e-9:
                    direction = None
                else:
                    direction = direction / dn

        if direction is None:
            if max_svd_samples > 0 and len(ball_nodes) > max_svd_samples:
                sub_idx = np.random.choice(
                    len(ball_nodes), size=max_svd_samples, replace=False
                )
                sample_rows = x_norm[idx_arr[sub_idx]]
            else:
                sample_rows = X
            Xc_s = sample_rows - sample_rows.mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(Xc_s, full_matrices=False)
                direction = vh[0].astype(np.float64, copy=False)
            except Exception:
                direction = np.ones((Xc_s.shape[1],), dtype=np.float64)
                direction /= (np.linalg.norm(direction) + 1e-12)

        X_mean = X.mean(axis=0, keepdims=True)
        proj = ((X - X_mean) @ direction).ravel()
        med = float(np.median(proj))
        left_mask = proj <= med
        right_mask = ~left_mask

        left = [ball_nodes[i] for i in np.where(left_mask)[0].tolist()]
        right = [ball_nodes[i] for i in np.where(right_mask)[0].tolist()]

        if len(left) == 0 or len(right) == 0:
            order = np.argsort(proj)
            mid = max(1, int(len(ball_nodes) // 2))
            left = [ball_nodes[i] for i in order[:mid].tolist()]
            right = [ball_nodes[i] for i in order[mid:].tolist()]

        if len(left) == 0 or len(right) == 0:
            perm = np.random.permutation(len(ball_nodes))
            mid = max(1, int(len(ball_nodes) // 2))
            left = [ball_nodes[i] for i in perm[:mid].tolist()]
            right = [ball_nodes[i] for i in perm[mid:].tolist()]

        return left, right

    def _build_granular_balls_topdown(
        self,
        x_norm: np.ndarray,
        labels_list: list,
        *,
        min_leaf_size: int,
        purity_stop: float,
        entropy_stop: float,
        max_depth: int,
        max_total_balls: int,
        root_nodes: list = None,
        train_mask_list: list = None,
        split_mode: str = "pca",
        max_svd_samples: int = 0,
    ) -> list:
        n_all = int(x_norm.shape[0])
        root = (
            list(root_nodes)
            if root_nodes is not None
            else list(range(n_all))
        )
        stack = [(root, 0)]
        leaves = []

        while stack:
            nodes, depth = stack.pop()
            if depth >= int(max_depth) or not self._should_split(
                nodes,
                labels_list,
                min_leaf_size=min_leaf_size,
                purity_stop=purity_stop,
                entropy_stop=entropy_stop,
                train_mask_list=train_mask_list,
            ):
                leaves.append(nodes)
                continue

            if len(leaves) + len(stack) + 1 >= int(max_total_balls):
                leaves.append(nodes)
                continue

            left, right = self._split_ball(
                nodes,
                x_norm,
                labels_list=labels_list,
                train_mask_list=train_mask_list,
                split_mode=split_mode,
                max_svd_samples=max_svd_samples,
            )
            if len(left) < int(min_leaf_size) or len(right) < int(min_leaf_size):
                leaves.append(nodes)
                continue

            stack.append((right, depth + 1))
            stack.append((left, depth + 1))

        return leaves

    def _refine_balls_to_budget(
        self,
        balls: Dict[int, list],
        ball_majority: Dict[int, int],
        ball_purity: Dict[int, float],
        ball_ig: Dict[int, float],
        ball_size: Dict[int, int],
        *,
        x_norm: np.ndarray,
        labels_list: list,
        train_mask_list: list,
        unique_classes: list,
        budget_map: Dict[int, int],
        default_budget: int,
        train_class_counts: Dict[int, int],
        min_leaf_size: int,
        split_mode: str,
        max_svd_samples: int,
        global_entropy: float,
    ) -> Tuple[Dict[int, list], Dict[int, int], Dict[int, float], Dict[int, float], Dict[int, int]]:
        """Budget-aware granular-ball refinement (Stage 2).

        When a class has fewer candidate balls than its allocated budget,
        its most dispersed ball (largest radius rho = mean distance to the
        ball center) is split further, until the class reaches budget or no
        candidate ball can be split validly.

        Phase A (strict): a split is accepted only if both children
        (1) reach min_leaf_size nodes, (2) contain training evidence, and
        (3) keep the target class as majority - so every stored prototype
        stays a granular-ball of its class. Phase B (relaxed): classes
        still short of budget split the highest-radius ball containing
        their train nodes and accept children of any majority, which keeps
        candidate counts monotone and guarantees the budget whenever the
        data admits it. Classes without any candidate ball first receive a
        class-root ball (all their train nodes), which is refined like any
        other ball.
        """
        next_bid = max(balls.keys(), default=-1) + 1

        def _cls_min_leaf(cls_int: int) -> int:
            # Long-tailed classes: allow smaller balls so a class can form
            # its budget of balls (otherwise tiny classes, e.g. ogbn-products
            # tail, store far fewer prototypes than the data allows while
            # the split uses all available nodes).
            b = int(budget_map.get(cls_int, default_budget))
            n_c = int(train_class_counts.get(cls_int, 0))
            return max(1, min(int(min_leaf_size), n_c // max(1, b)))
        # Safety bound on total split attempts per phase.
        max_attempts = 8 * max(1, int(max(budget_map.values(), default=default_budget))) * max(1, len(unique_classes))

        ball_train_counts: Dict[int, Counter] = {}

        def _ball_stats(nodes: list) -> Tuple[int, float, float, int, Counter]:
            cnt = Counter(labels_list[n] for n in nodes if train_mask_list[n])
            if not cnt:
                return -1, 0.0, 0.0, 0, cnt
            total = float(sum(cnt.values()))
            maj_label = int(cnt.most_common(1)[0][0])
            purity = float(cnt.most_common(1)[0][1]) / total
            ball_entropy = 0.0
            for c in cnt.values():
                p = float(c) / total
                if p > 0.0:
                    ball_entropy -= p * math.log2(p)
            ig = float(global_entropy - ball_entropy)
            return maj_label, purity, ig, int(sum(cnt.values())), cnt

        def _radius(nodes: list) -> float:
            arr = x_norm[np.asarray(nodes, dtype=np.int64)]
            mu = arr.mean(axis=0, keepdims=True)
            return float(np.mean(np.linalg.norm(arr - mu, axis=1)))

        def _add_ball(nodes: list):
            nonlocal next_bid
            maj, pur, ig, n_train, cnt = _ball_stats(nodes)
            if n_train <= 0:
                return
            balls[next_bid] = [int(n) for n in nodes]
            ball_majority[next_bid] = maj
            ball_purity[next_bid] = pur
            ball_ig[next_bid] = ig
            ball_size[next_bid] = int(len(nodes))
            ball_train_counts[next_bid] = cnt
            next_bid += 1

        # Class-root fallback: every class starts with at least one candidate.
        for cls in unique_classes:
            cls_int = int(cls)
            if any(lab == cls_int for lab in ball_majority.values()):
                continue
            cls_nodes = [
                i for i, t in enumerate(train_mask_list)
                if t and labels_list[i] == cls_int
            ]
            if cls_nodes:
                _add_ball(cls_nodes)

        def _split_and_replace(bid: int, refine_cls: int, require_majority) -> bool:
            left, right = self._split_ball(
                balls[bid],
                x_norm,
                labels_list=labels_list,
                train_mask_list=train_mask_list,
                split_mode=split_mode,
                max_svd_samples=max_svd_samples,
            )
            # Condition 1: both children reach the class's min_leaf.
            mls = _cls_min_leaf(refine_cls)
            if len(left) < mls or len(right) < mls:
                return False
            maj_l, pur_l, ig_l, n_l, _ = _ball_stats(left)
            maj_r, pur_r, ig_r, n_r, _ = _ball_stats(right)
            # Condition 2: training evidence.
            if n_l <= 0 or n_r <= 0:
                return False
            # Condition 3 (Phase A only): class consistency.
            if (
                require_majority is not None
                and (maj_l != require_majority or maj_r != require_majority)
            ):
                return False
            # Replace the parent ball with the two accepted children.
            balls.pop(bid)
            ball_majority.pop(bid)
            ball_purity.pop(bid)
            ball_ig.pop(bid)
            ball_size.pop(bid)
            ball_train_counts.pop(bid, None)
            _add_ball(left)
            _add_ball(right)
            return True

        # Phase A (strict): split the highest-radius ball of each
        # under-budget class, keeping the class-consistency condition.
        failed: set = set()
        n_attempts = 0
        while n_attempts < max_attempts:
            progress = False
            for cls in unique_classes:
                cls_int = int(cls)
                cls_budget = int(budget_map.get(cls_int, default_budget))
                cand_bids = [
                    bid for bid, lab in ball_majority.items()
                    if lab == cls_int
                ]
                if len(cand_bids) >= cls_budget:
                    continue
                bid = max(
                    (
                        b for b in cand_bids
                        if b not in failed
                        and len(balls[b]) >= 2 * _cls_min_leaf(cls_int)
                    ),
                    key=lambda b: _radius(balls[b]),
                    default=None,
                )
                if bid is None:
                    continue
                n_attempts += 1
                if _split_and_replace(bid, cls_int, require_majority=cls_int):
                    progress = True
                else:
                    failed.add(bid)
            if not progress:
                break

        # Phase B (relaxed): classes still short split the highest-radius
        # ball containing their train nodes and accept children of any
        # majority. A child keeps its own majority's candidate count, so
        # per-class counts never decrease and the budget is filled whenever
        # the data admits it.
        failed: set = set()
        n_attempts = 0
        while n_attempts < max_attempts:
            progress = False
            for cls in unique_classes:
                cls_int = int(cls)
                cls_budget = int(budget_map.get(cls_int, default_budget))
                if (
                    sum(1 for lab in ball_majority.values() if lab == cls_int)
                    >= cls_budget
                ):
                    continue
                bid = max(
                    (
                        b for b, cnt in ball_train_counts.items()
                        if b not in failed
                        and len(balls[b]) >= 2 * _cls_min_leaf(cls_int)
                        and cnt.get(cls_int, 0) > 0
                    ),
                    key=lambda b: _radius(balls[b]),
                    default=None,
                )
                if bid is None:
                    continue
                n_attempts += 1
                if _split_and_replace(bid, cls_int, None):
                    progress = True
                else:
                    failed.add(bid)
            if not progress:
                break

        return balls, ball_majority, ball_purity, ball_ig, ball_size

    def _generate_granular_balls(self, task, labels, budget_map=None):
        default_budget = int(self.budget)
        if budget_map is None:
            budget_map = {int(c): default_budget for c in torch.unique(labels).tolist()}
        device = task.x.device
        labels = labels.to(device)
        num_nodes = task.x.size(0)
        train_mask = task.train_mask.to(device)

        granular_balls = []
        feat_cond_list, labels_cond_list = [], []

        labels_list = labels.detach().to("cpu").tolist()
        train_mask_list = train_mask.detach().to("cpu").tolist()
        train_labels_global = [labels_list[i] for i, is_train in enumerate(train_mask_list) if is_train]
        if len(train_labels_global) == 0:
            feat_cond = torch.empty((0, task.x.size(1)), device=device)
            labels_cond = torch.empty((0,), dtype=torch.long, device=device)
            stats = {"avg_purity": 0.0, "selected_balls": 0}
            return feat_cond, labels_cond, granular_balls, stats
        x_cpu = task.x.detach().to("cpu", dtype=torch.float32).numpy()
        x_norm = x_cpu / (np.linalg.norm(x_cpu, axis=1, keepdims=True) + 1e-12)

        def _entropy_from_counts(counts) -> float:
            total = float(sum(counts))
            if total <= 0:
                return 0.0
            h = 0.0
            for cnt in counts:
                p = float(cnt) / total
                if p > 0.0:
                    h -= p * math.log2(p)
            return h

        global_counts = Counter(train_labels_global)
        global_entropy = _entropy_from_counts(global_counts.values())

        def _ball_ig_and_purity(ball_nodes) -> Tuple[float, float]:
            node_list = list(ball_nodes)
            if not node_list:
                return 0.0, 0.0
            ball_counts = Counter(labels_list[n] for n in node_list if train_mask_list[n])
            if not ball_counts:
                return 0.0, 0.0
            ball_entropy = _entropy_from_counts(ball_counts.values())
            ig = global_entropy - ball_entropy
            majority_count = ball_counts.most_common(1)[0][1]
            purity = float(majority_count) / float(sum(ball_counts.values()))
            return float(ig), float(purity)

        unique_classes = sorted(set(train_labels_global))
        num_classes = max(1, len(unique_classes))

        ball_cfg = self._resolve_ball_hparams(num_nodes, num_classes)
        min_leaf_size = int(ball_cfg["min_leaf_size"])
        purity_stop = float(ball_cfg["purity_stop"])
        entropy_stop = float(ball_cfg["entropy_stop"])
        max_depth = int(ball_cfg["max_depth"])
        max_total_balls = int(ball_cfg["max_total_balls"])
        split_mode = str(ball_cfg["split_mode"])
        max_svd_samples = int(ball_cfg["max_svd_samples"])
        train_root_only = bool(ball_cfg["train_root_only"])
        root_nodes = None
        if train_root_only:
            root_nodes = [i for i in range(num_nodes) if train_mask_list[i]]

        raw_balls = self._build_granular_balls_topdown(
            x_norm,
            labels_list,
            min_leaf_size=min_leaf_size,
            purity_stop=purity_stop,
            entropy_stop=entropy_stop,
            max_depth=max_depth,
            max_total_balls=max_total_balls,
            root_nodes=root_nodes,
            train_mask_list=train_mask_list,
            split_mode=split_mode,
            max_svd_samples=max_svd_samples,
        )

        ball_majority: Dict[int, int] = {}
        ball_purity: Dict[int, float] = {}
        ball_ig: Dict[int, float] = {}
        ball_size: Dict[int, int] = {}
        balls: Dict[int, list] = {}
        for bid, nodes in enumerate(raw_balls):
            if len(nodes) < min_leaf_size:
                continue
            ig, pur = _ball_ig_and_purity(nodes)
            counts = Counter(labels_list[n] for n in nodes if train_mask_list[n])
            if not counts:
                continue
            maj_label = int(counts.most_common(1)[0][0])
            balls[bid] = [int(n) for n in nodes]
            ball_majority[bid] = maj_label
            ball_purity[bid] = float(pur)
            ball_ig[bid] = float(ig)
            ball_size[bid] = int(len(nodes))

        # Budget-aware refinement: split the most dispersed (largest-radius)
        # ball of each under-budget class until every class has enough
        # candidate balls (or no candidate ball can be split validly).
        (
            balls,
            ball_majority,
            ball_purity,
            ball_ig,
            ball_size,
        ) = self._refine_balls_to_budget(
            balls,
            ball_majority,
            ball_purity,
            ball_ig,
            ball_size,
            x_norm=x_norm,
            labels_list=labels_list,
            train_mask_list=train_mask_list,
            unique_classes=unique_classes,
            budget_map=budget_map,
            default_budget=default_budget,
            train_class_counts=global_counts,
            min_leaf_size=min_leaf_size,
            split_mode=split_mode,
            max_svd_samples=max_svd_samples,
            global_entropy=global_entropy,
        )

        def _minmax(v: float, vmin: float, vmax: float) -> float:
            eps = 1e-12
            if vmax - vmin <= eps:
                return 0.0
            return (v - vmin) / (vmax - vmin + eps)

        def _append_ball_prototype(cand_nodes: list, cls_int: int):
            cand_tensor = torch.tensor(cand_nodes, dtype=torch.long, device=device)
            x_cand = task.x[cand_tensor]
            center_feat = x_cand.mean(dim=0, keepdim=True)
            x_cand_norm = F.normalize(x_cand, p=2, dim=1)
            center_norm = F.normalize(center_feat, p=2, dim=1)
            cos_sim = (x_cand_norm @ center_norm.t()).squeeze(-1)
            cos_shift = cos_sim - cos_sim.min() + 1e-6
            weights = cos_shift / cos_shift.sum()
            mean_feat = torch.sum(
                weights.unsqueeze(-1) * x_cand, dim=0, keepdim=True
            )
            feat_cond_list.append(mean_feat)
            labels_cond_list.append(torch.tensor([cls_int], dtype=torch.long, device=device))

        selected_purities = []
        selected_cnt_per_class: Dict[int, int] = {}
        for cls in unique_classes:
            cls_int = int(cls)
            selected_cnt_per_class[cls_int] = 0
            cand_bids = [bid for bid, lab in ball_majority.items() if lab == int(cls)]
            if not cand_bids:
                continue

            sizes = [float(ball_size[bid]) for bid in cand_bids]
            igs = [float(ball_ig[bid]) for bid in cand_bids]
            size_min, size_max = min(sizes), max(sizes)
            ig_min, ig_max = min(igs), max(igs)

            w_sz = float(self.args.get("ball_score_w_sz", 0.4))
            w_ig = float(self.args.get("ball_score_w_ig", 0.4))
            w_p = float(self.args.get("ball_score_w_p", 0.2))
            w_sum = w_sz + w_ig + w_p
            if w_sum <= 0:
                w_sz, w_ig, w_p = 0.4, 0.4, 0.2
                w_sum = 1.0
            w_sz, w_ig, w_p = w_sz / w_sum, w_ig / w_sum, w_p / w_sum

            scored = []
            for bid in cand_bids:
                size_norm = _minmax(float(ball_size[bid]), size_min, size_max)
                ig_norm = _minmax(float(ball_ig[bid]), ig_min, ig_max)
                score = (
                    w_sz * size_norm
                    + w_ig * ig_norm
                    + w_p * float(ball_purity[bid])
                )
                scored.append((score, bid))
            scored.sort(reverse=True)

            cls_budget = int(budget_map.get(cls_int, default_budget))
            for _, bid in scored[:cls_budget]:
                cand = balls[bid]
                granular_balls.append(cand)
                selected_purities.append(float(ball_purity[bid]))
                _append_ball_prototype(cand, cls_int)
                selected_cnt_per_class[cls_int] += 1

        # Every class contributes at least one prototype if refinement stalls.
        for cls in unique_classes:
            cls_int = int(cls)
            if int(selected_cnt_per_class.get(cls_int, 0)) > 0:
                continue
            cls_train_nodes = [
                i for i in range(num_nodes)
                if train_mask_list[i] and labels_list[i] == cls_int
            ]
            if not cls_train_nodes:
                continue
            granular_balls.append(cls_train_nodes)
            _append_ball_prototype(cls_train_nodes, cls_int)
            selected_purities.append(1.0)
            selected_cnt_per_class[cls_int] = 1

        if feat_cond_list:
            feat_cond = torch.cat(feat_cond_list, dim=0)
            labels_cond = torch.cat(labels_cond_list, dim=0)
        else:
            feat_cond = torch.empty((0, task.x.size(1)), device=device)
            labels_cond = torch.empty((0,), dtype=torch.long, device=device)

        stats = {
            "avg_purity": float(sum(selected_purities) / len(selected_purities)) if selected_purities else 0.0,
            "selected_balls": int(len(selected_purities)),
        }
        return feat_cond, labels_cond, granular_balls, stats

    def _embedding(self, task, feat_cond, labels_cond):
        device = self.device

        if feat_cond.numel() == 0:
            return Data(
                x=torch.empty((0, task.num_features), device=device),
                y=torch.empty((0,), dtype=torch.long, device=device),
                adj_t=SparseTensor.eye(0).to(device)
            )

        feat_cond_param = torch.nn.Parameter(feat_cond.clone().to(device))
        opt_feat = torch.optim.Adam([feat_cond_param], lr=float(self.feat_lr))

        encoder = Encoder(task.num_features, self.hid_dim, self.emb_dim,
                          self.n_layers, self.hop, self.activation).to(device)

        classes = torch.unique(task.y[task.train_mask]).to(device)
        cls_train_masks = [(task.y.to(device) == cls).logical_and(task.train_mask.to(device))
                           for cls in classes]

        for epoch in range(self.n_encoders):
            encoder.initialize()

            with torch.no_grad():
                emb_real = encoder.encode(task.x.to(device), task.adj_t.to(device))
                emb_real = F.normalize(emb_real, p=2, dim=1)

            emb_cond = encoder.encode_without_e(feat_cond_param)
            emb_cond = F.normalize(emb_cond, p=2, dim=1)

            class_in_loss = torch.tensor(0.0, device=device)

            for i, cls in enumerate(classes):
                real_mask = cls_train_masks[i]
                if real_mask.sum() == 0:
                    continue
                real_emb_at_class = emb_real[real_mask]
                cond_emb_at_class = emb_cond[labels_cond == cls]
                if cond_emb_at_class.size(0) == 0:
                    continue
                real_center = real_emb_at_class.mean(dim=0)
                cond_center = cond_emb_at_class.mean(dim=0)
                cos_sim = F.cosine_similarity(real_center.unsqueeze(0),
                                              cond_center.unsqueeze(0), dim=1)
                class_in_loss += (1.0 - cos_sim).squeeze()

            loss = class_in_loss
            opt_feat.zero_grad()
            loss.backward()
            opt_feat.step()

        replayed_graph = Data(
            x=feat_cond_param.detach().to(device),
            y=labels_cond.detach().to(device),
            adj_t=self._make_replay_adj_t(feat_cond_param.detach().to(device))
        )
        replayed_graph.train_mask = torch.ones(len(labels_cond), dtype=torch.bool, device=device)

        return replayed_graph
