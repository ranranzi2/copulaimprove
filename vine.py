"""VineCopula module."""
from tqdm import tqdm
import logging
import sys
import warnings

import numpy as np
import pandas as pd

from copulas.bivariate.base import Bivariate, CopulaTypes
from copulas.multivariate.base import Multivariate
from copulas.multivariate.tree import Tree, get_tree
from copulas.univariate.gaussian_kde import GaussianKDE
from copulas.utils import (
    EPSILON,
    check_valid_values,
    get_qualified_name,
    random_state,
    store_args,
    validate_random_state,
)

LOGGER = logging.getLogger(__name__)


class VineCopula(Multivariate):
    """Vine copula model.

    A :math:`vine` is a graphical representation of one factorization of the n-variate probability
    distribution in terms of :math:`n(n − 1)/2` bivariate copulas by means of the chain rule.

    It consists of a sequence of levels and as many levels as variables. Each level consists of
    a tree (no isolated nodes and no loops) satisfying that if it has :math:`n` nodes there must
    be :math:`n − 1` edges.

    Each node in tree :math:`T_1` is a variable and edges are couplings of variables constructed
    with bivariate copulas.

    Each node in tree :math:`T_{k+1}` is a coupling in :math:`T_{k}`, expressed by the copula
    of the variables; while edges are couplings between two vertices that must have one variable
    in common, becoming a conditioning variable in the bivariate copula. Thus, every level has
    one node less than the former. Once all the trees are drawn, the factorization is the product
    of all the nodes.

    Args:
        vine_type (str):
            type of the vine copula, could be 'center','direct','regular'
        random_state (int or np.random.RandomState):
            Random seed or RandomState to use.


    Attributes:
        model (copulas.univariate.Univariate):
            Distribution to compute univariates.
        u_matrix (numpy.array):
            Univariates.
        n_sample (int):
            Number of samples.
        n_var (int):
            Number of variables.
        columns (pandas.Series):
            Names of the variables.
        tau_mat (numpy.array):
            Kendall correlation parameters for data.
        truncated (int):
            Max level used to build the vine.
        depth (int):
            Vine depth.
        trees (list[Tree]):
            List of trees used by this vine.
        ppfs (list[callable]):
            percent point functions from the univariates used by this vine.
    """

    @store_args
    def __init__(self, vine_type, random_state=None):
        if sys.version_info > (3, 8):
            warnings.warn(
                'Vines have not been fully tested on Python >= 3.8 and might produce wrong results.'
            )

        self.random_state = validate_random_state(random_state)
        self.vine_type = vine_type
        self.u_matrix = None

        self.model = GaussianKDE
        self.original_sum_var = None 

    @classmethod
    def _deserialize_trees(cls, tree_list):
        previous = Tree.from_dict(tree_list[0])
        trees = [previous]

        for tree_dict in tree_list[1:]:
            tree = Tree.from_dict(tree_dict, previous)
            trees.append(tree)
            previous = tree

        return trees

    def to_dict(self):
        """Return a `dict` with the parameters to replicate this Vine.

        Returns:
            dict:
                Parameters of this Vine.
        """
        result = {
            'type': get_qualified_name(self),
            'vine_type': self.vine_type,
            'fitted': self.fitted,
        }

        if not self.fitted:
            return result

        result.update({
            'n_sample': self.n_sample,
            'n_var': self.n_var,
            'depth': self.depth,
            'truncated': self.truncated,
            'trees': [tree.to_dict() for tree in self.trees],
            'tau_mat': self.tau_mat.tolist(),
            'u_matrix': self.u_matrix.tolist(),
            'unis': [distribution.to_dict() for distribution in self.unis],
            'columns': self.columns,
        })
        return result

    @classmethod
    def from_dict(cls, vine_dict):
        """Create a new instance from a parameters dictionary.

        Args:
            params (dict):
                Parameters of the Vine, in the same format as the one
                returned by the ``to_dict`` method.

        Returns:
            Vine:
                Instance of the Vine defined on the parameters.
        """
        instance = cls(vine_dict['vine_type'])
        fitted = vine_dict['fitted']
        if fitted:
            instance.fitted = fitted
            instance.n_sample = vine_dict['n_sample']
            instance.n_var = vine_dict['n_var']
            instance.truncated = vine_dict['truncated']
            instance.depth = vine_dict['depth']
            instance.trees = cls._deserialize_trees(vine_dict['trees'])
            instance.unis = [GaussianKDE.from_dict(uni) for uni in vine_dict['unis']]
            instance.ppfs = [uni.percent_point for uni in instance.unis]
            instance.columns = vine_dict['columns']
            instance.tau_mat = np.array(vine_dict['tau_mat'])
            instance.u_matrix = np.array(vine_dict['u_matrix'])

        return instance

    @check_valid_values
    def fit(self, X, truncated=3,weight=1):
        """Fit a vine model to the data.

        1. Transform all the variables by means of their marginals.
        In other words, compute

        .. math:: u_i = F_i(x_i), i = 1, ..., n

        and compose the matrix :math:`u = u_1, ..., u_n,` where :math:`u_i` are their columns.

        Args:
            X (numpy.ndarray):
                Data to be fitted to.
            truncated (int):
                Max level to build the vine.
        """
        LOGGER.info('Fitting VineCopula("%s")', self.vine_type)
        self.n_sample, self.n_var = X.shape
        self.columns = X.columns
        self.tau_mat = X.corr(method='kendall').to_numpy()

        # 增加x1的权重（改动部分）
        x1_index = -1  # 假设x1是第一个变量
        self.tau_mat[x1_index, :] *= weight
        self.tau_mat[:, x1_index] *= weight
        self.tau_mat = np.clip(self.tau_mat, -1, 1)

        self.u_matrix = np.empty([self.n_sample, self.n_var])

        self.truncated = truncated
        self.depth = self.n_var - 1
        self.trees = []

        self.unis, self.ppfs = [], []
        for i, col in enumerate(X):
            uni = self.model()
            uni.fit(X[col])
            self.u_matrix[:, i] = uni.cumulative_distribution(X[col])
            self.unis.append(uni)
            self.ppfs.append(uni.percent_point)

        self.train_vine(self.vine_type)
        self.fitted = True
        self.original_sum_var = X.sum(axis=1).var()
        # print("Initial parameters:")
        # for p in self.get_parameters():
        #     print(f"Tree {p['tree_level']} Edge {p['edge_idx']}: theta={p['theta']}")

    def _calculate_sum_variance(self, samples):
        """计算样本总和的方差"""
        return samples.sum(axis=1).var()
    
    def get_parameters(self):
        """获取所有copula参数"""
        params = []
        for tree_level, tree in enumerate(self.trees):
            for edge_idx, edge in enumerate(tree.edges):
                params.append({
                    'tree_level': tree_level,
                    'edge_idx': edge_idx,
                    'copula_type': edge.name,
                    'theta': edge.theta
                })
        return params

    def set_parameters(self, params):
        """设置所有copula参数"""
        for param in params:
            tree_level = param['tree_level']
            edge_idx = param['edge_idx']
            copula_type = param['copula_type']
            theta = param['theta']

            if tree_level >= len(self.trees):
                continue

            tree = self.trees[tree_level]
            if edge_idx < len(tree.edges):
                edge = tree.edges[edge_idx]
                edge.name = copula_type
                edge.theta = self._clip_param(copula_type, theta)

    def _clip_param(self, copula_type, theta):
        """确保参数在合法范围内"""
        if copula_type == CopulaTypes.CLAYTON.name:
            return max(theta, 1e-5)
        elif copula_type == CopulaTypes.GAUSSIAN.name:
            return np.clip(theta, -0.999, 0.999)
        elif copula_type == CopulaTypes.GUMBEL.name:
            return max(theta, 1.0 + 1e-5)
        else:
            return theta

    def random_perturbation(
        self, 
        num_iterations=100,
        num_samples=1000,
        delta_percent=0.01,
        delta_absolute=0.01,
        tolerance=1e-4,
        verbose=True
    ):
        """对模型参数进行随机微扰，优化样本总和方差接近原始方差。

        Args:
            num_iterations (int): 随机扰动迭代次数
            num_samples (int): 每次评估生成的样本量
            delta_percent (float): 参数扰动的百分比幅度（适用于非零参数）
            delta_absolute (float): 零参数的绝对扰动幅度
            tolerance (float): 方差差异容忍阈值，达到则提前终止
            verbose (bool): 是否显示进度条
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call 'fit' first.")
        if self.original_sum_var is None:
            raise ValueError("Original sum variance not calculated. Fit with data first.")

        current_params = self.get_parameters()
        best_loss = self._evaluate_current_loss(num_samples)

        with tqdm(total=num_iterations, disable=not verbose) as pbar:
            for _ in range(num_iterations):
                # 生成扰动参数
                trial_params = [param.copy() for param in current_params]
                for param in trial_params:
                    theta = param['theta']
                    copula_type = param['copula_type']
                    
                    # 计算扰动量
                    if theta == 0:
                        perturbation = delta_absolute * np.random.uniform(-1, 1)
                    else:
                        perturbation = theta * delta_percent * np.random.uniform(-1, 1)
                    
                    # 应用扰动并裁剪到合法范围
                    new_theta = theta + perturbation
                    param['theta'] = self._clip_param(copula_type, new_theta)

                # 评估扰动后参数
                self.set_parameters(trial_params)
                try:
                    current_loss = self._evaluate_current_loss(num_samples)
                except Exception as e:
                    current_loss = np.inf

                # 保留更优参数
                if current_loss < best_loss:
                    current_params = trial_params
                    best_loss = current_loss
                    self.set_parameters(current_params)  # 立即应用最佳参数

                pbar.update(1)
                pbar.set_postfix({'best_loss': best_loss})

                # 提前终止检查
                if best_loss <= tolerance:
                    break

        # 确保最终参数已应用
        self.set_parameters(current_params)

    def _evaluate_current_loss(self, num_samples):
        """计算当前参数配置的方差差异损失"""
        samples = self.sample(num_samples)
        simulated_var = samples.sum(axis=1).var()
        return abs(simulated_var - self.original_sum_var)

    def perturb_parameters(
        self, 
        num_samples=1000,
        delta_percent=0.01,
        num_steps=10,
        max_iters=5,
        tolerance=1e-4,
        random_state=None,
        verbose=True  # 新增verbose控制输出
    ):
        if self.original_sum_var is None:
            raise ValueError("Fit the model first.")

        param_positions = [
            (t_idx, e_idx)
            for t_idx, tree in enumerate(self.trees)
            for e_idx, edge in enumerate(tree.edges)
            if edge.theta is not None
        ]

        best_loss = np.inf
        progress = None
        
        try:
            # 外层迭代进度条
            if verbose:
                progress = tqdm(total=max_iters, desc="Global Iterations", position=0)

            for iter in range(max_iters):
                improved = False
                
                # 参数遍历进度条
                param_progress = tqdm(
                    param_positions, 
                    desc=f"Iter {iter+1} | Best Loss: {best_loss:.4e}",
                    leave=False,
                    disable=not verbose
                )

                for t_idx, e_idx in param_progress:
                    tree = self.trees[t_idx]
                    edge = tree.edges[e_idx]
                    original_theta = edge.theta
                    
                    # 生成候选值
                    current_delta = abs(original_theta)*delta_percent if original_theta !=0 else 0.01
                    candidates = np.linspace(
                        original_theta - current_delta,
                        original_theta + current_delta,
                        num_steps
                    )

                    # 候选值评估进度
                    candidate_progress = tqdm(
                        candidates,
                        desc=f"Tree {t_idx} Edge {e_idx}",
                        leave=False,
                        disable=not verbose
                    )

                    best_theta = original_theta
                    min_loss = np.inf
                    for candidate in candidate_progress:
                        edge.theta = self._clip_param(edge.name, candidate)
                        try:
                            samples = self.sample(num_samples)
                            simulated_var = self._calculate_sum_variance(samples)
                            loss = abs(simulated_var - self.original_sum_var)
                        except Exception as e:
                            loss = np.inf

                        candidate_progress.set_postfix_str(f"loss={loss:.4e}")

                        if loss < min_loss:
                            min_loss = loss
                            best_theta = candidate

                    edge.theta = self._clip_param(edge.name, best_theta)
                    param_progress.set_postfix_str(f"improved={min_loss < best_loss}")

                    if min_loss < best_loss:
                        best_loss = min_loss
                        improved = True

                # 更新全局进度条
                if progress:
                    progress.update(1)
                    progress.set_postfix_str(f"Best Loss={best_loss:.4e}")

                if best_loss <= tolerance or not improved:
                    break

        finally:
            # 确保关闭所有进度条
            if progress:
                progress.close()
        
        # 最终验证
        if verbose:
            final_loss = abs(
                self._calculate_sum_variance(self.sample(num_samples)) - self.original_sum_var
            )
            print(f"\nOptimization finished. Final loss: {final_loss:.4e}")

    def full_grid_search(
        self,
        num_samples=1000,
        delta_percent=0.01,
        num_steps=3,  # 注意：当参数多时，num_steps必须很小！
        tolerance=1e3,
        random_state=None,
        verbose=True
    ):
        """全参数组合网格搜索（慎用！计算量指数爆炸）"""
        if self.original_sum_var is None:
            raise ValueError("Fit the model first.")

        # 获取所有可调参数信息
        params_info = [
            {
                "tree_idx": t_idx,
                "edge_idx": e_idx,
                "original": edge.theta,
                "name": edge.name,
                "candidates": self._generate_candidates(
                    edge.theta, 
                    delta_percent, 
                    num_steps,
                    edge.name
                )
            }
            for t_idx, tree in enumerate(self.trees)
            for e_idx, edge in enumerate(tree.edges)
            if edge.theta is not None
        ]

        n_params = len(params_info)
        print(n_params)
        total_combinations = num_steps ** n_params
        
        # 安全警告
        if verbose:
            print(f"Warning: This will evaluate {total_combinations} combinations!")
            # if total_combinations > 1e6:
            #     raise ValueError("Too many combinations to evaluate!")

        # 生成参数网格
        param_grid = np.array(np.meshgrid(
            *[p['candidates'] for p in params_info]
        )).T.reshape(-1, n_params)

        best_loss = np.inf
        best_params = None
        original_params = self.get_parameters()

        try:
            with tqdm(total=total_combinations, 
                    desc="Grid Search", 
                    disable=not verbose) as pbar:
                
                for i, combination in enumerate(param_grid):
                    # 设置当前参数组合
                    for param_idx, value in enumerate(combination):
                        t_idx = params_info[param_idx]['tree_idx']
                        e_idx = params_info[param_idx]['edge_idx']
                        edge = self.trees[t_idx].edges[e_idx]
                        edge.theta = value

                    # 评估
                    try:
                        samples = self.sample(num_samples)
                        current_var = self._calculate_sum_variance(samples)
                        loss = abs(current_var - self.original_sum_var)
                    except:
                        loss = np.inf

                    # 更新最佳
                    if loss < best_loss:
                        best_loss = loss
                        best_params = combination.copy()
                        pbar.set_postfix_str(f"Best Loss: {best_loss:.4e}")

                    pbar.update(1)

                    # 提前终止
                    if best_loss <= tolerance:
                        break

        finally:
            # 恢复原始参数
            self.set_parameters(original_params)

        # 应用最佳参数
        if best_params is not None:
            for param_idx, value in enumerate(best_params):
                t_idx = params_info[param_idx]['tree_idx']
                e_idx = params_info[param_idx]['edge_idx']
                edge = self.trees[t_idx].edges[e_idx]
                edge.theta = value

        # 结果输出
        if verbose:
            print(f"\nBest loss achieved: {best_loss:.4e}")
            print(f"Total combinations evaluated: {i+1}/{total_combinations}")

        return best_loss

    def _generate_candidates(self, original, delta_percent, num_steps, copula_type):
        """生成合法候选值"""
        if original == 0:
            delta = 0.01
        else:
            delta = abs(original) * delta_percent
        
        raw = np.linspace(original - delta, original + delta, num_steps)
        return [self._clip_param(copula_type, x) for x in raw]
    
    def directional_perturbation(
        self,
        num_samples=1000,
        delta_percent=0.01,
        max_total_change=0.05,
        max_iter=100,
        tolerance=1e-4,
        random_state=None,
        verbose=True,
        step=2
    ):
        """定向扰动优化（逐个参数尝试正负扰动，选择全局最优改进）"""
        if self.original_sum_var is None:
            raise ValueError("Fit the model first.")

        # 获取参数初始状态
        params_info = []
        for t_idx, tree in enumerate(self.trees):
            for e_idx, edge in enumerate(tree.edges):
                if edge.theta is not None:
                    params_info.append({
                        'tree_idx': t_idx,
                        'edge_idx': e_idx,
                        'original': edge.theta,
                        'current': edge.theta,
                        'name': edge.name,
                        'total_change': 0.0,
                        'locked': False,
                        'is_zero': abs(edge.theta) < 1e-8
                    })

        best_loss = np.inf
        original_params = self.get_parameters()
        current_params = original_params.copy()
        best_params = original_params.copy()  # 初始化最佳参数
        progress = None
        termination_info = {
            'reason': None,
            'iter': None,
            'active_params': None,
            'loss_change': None
        }

        try:
            if verbose:
                progress = tqdm(total=max_iter, desc="Iterations", position=0)

            # 初始评估
            best_loss = self._evaluate_current_loss(num_samples)
            if verbose:
                print(f"Initial loss: {best_loss:.4e}")

            for iter_num in range(max_iter):
                improved = False
                active_params = [p for p in params_info if not p['locked']]

                if verbose:
                    print(f"\n=== Iteration {iter_num+1} ===")
                    print(f"Active parameters: {len(active_params)}")

                # 保存当前参数快照
                current_params = self.get_parameters()
                best_candidate = None
                best_param_info = None
                min_loss = np.inf

                # 生成所有候选扰动
                candidates = []
                for param in active_params:
                    t_idx = param['tree_idx']
                    e_idx = param['edge_idx']
                    current_val = param['current']
                    
                    # 计算扰动步长
                    if param['is_zero']:
                        delta = 1e-4  # 零参数使用固定步长
                    else:
                        delta = current_val * delta_percent
                    
                    # 生成两个候选值并限制范围
                    for delta_dir in np.linspace(-abs(delta),abs(delta),num=step):#[+delta, -delta]:
                        cand_value = self._clip_param(param['name'], current_val + delta_dir)
                        candidates.append({
                            'param': param,
                            'value': cand_value,
                            'tree_idx': t_idx,
                            'edge_idx': e_idx
                        })

                if verbose:
                    # 子进度条（候选评估级别）
                    candidate_progress = tqdm(
                        total=len(candidates),
                        desc=f"Iter {iter_num+1} Candidates",
                        position=1,
                        leave=False  # 迭代结束后自动清除
                    )

                # 评估所有候选
                for cand in candidates:
                    # 恢复初始参数状态
                    self.set_parameters(current_params)
                    
                    # 应用候选参数
                    edge = self.trees[cand['tree_idx']].edges[cand['edge_idx']]
                    edge.theta = cand['value']
                    
                    # 评估损失
                    try:
                        loss = self._evaluate_current_loss(num_samples)
                    except:
                        loss = np.inf
                    
                    # 更新最佳候选
                    if loss < min_loss:
                        min_loss = loss
                        best_candidate = cand
                        best_param_info = cand['param']
                    if verbose:
                        candidate_progress.update(1)
                        candidate_progress.set_postfix_str(
                            f"Current Loss: {loss:.2e} | Best: {min_loss:.2e}"
                        )

                if verbose:
                    candidate_progress.close()

                # 应用全局最佳候选
                if best_candidate and min_loss < best_loss:
                    # 恢复参数后应用最佳候选
                    self.set_parameters(current_params)
                    edge = self.trees[best_candidate['tree_idx']].edges[best_candidate['edge_idx']]
                    edge.theta = best_candidate['value']
                    
                    # 更新参数信息
                    param = best_param_info
                    original_val = param['original']
                    current_val = best_candidate['value']
                    param['current'] = current_val

                    # 计算累积变化
                    if param['is_zero']:
                        param['total_change'] = abs(current_val - original_val)
                        if param['total_change'] >= max_total_change * 1e-3:
                            param['locked'] = True
                    else:
                        param['total_change'] = abs(current_val - original_val) / abs(original_val)
                        if param['total_change'] >= max_total_change:
                            param['locked'] = True

                    best_loss = min_loss
                    best_params = self.get_parameters()
                    improved = True

                # 检查终止条件
                current_loss = self._evaluate_current_loss(num_samples)
                active_count = len([p for p in params_info if not p['locked']])

                if progress:
                    progress.update(1)
                    progress.set_postfix_str(
                        f"Active: {active_count} Loss: {current_loss:.2e}"
                    )

                if active_count == 0:
                    termination_info.update({
                        'reason': '所有参数已锁定',
                        'iter': iter_num+1,
                        'active_params': 0
                    })
                    break
                elif abs(current_loss - best_loss) < tolerance:
                    termination_info.update({
                        'reason': '损失变化小于阈值',
                        'iter': iter_num+1,
                        'loss_change': abs(current_loss - best_loss)
                    })
                    break
                elif not improved:
                    termination_info.update({
                        'reason': '无法进一步改进',
                        'iter': iter_num+1
                    })
                    break

        finally:
            self.set_parameters(best_params)
            if progress:
                progress.close()

        # 诊断报告（保持不变）
        if verbose:
            print("\n=== Optimization Report ===")
            print(f"Termination reason: {termination_info['reason']}")
            print(f"Iterations completed: {termination_info.get('iter', 0)}")
            print(f"Final loss: {best_loss:.4e}")
            print("\nParameter status:")
            for p in params_info:
                status = "Locked" if p['locked'] else "Active"
                change_type = "Absolute" if p['is_zero'] else "Relative"
                print(f"Tree{p['tree_idx']}-Edge{p['edge_idx']}: {status} | "
                    f"Change: {p['total_change']:.2e} ({change_type})")

        return best_loss

    def _evaluate_current_loss(self, num_samples):
        """评估当前参数配置的损失"""
        samples = self.sample(num_samples)
        current_var = self._calculate_sum_variance(samples)
        return abs(current_var - self.original_sum_var)

    def _set_single_param(self, param_info, value):
        """设置单个参数值"""
        tree = self.trees[param_info['tree_idx']]
        edge = tree.edges[param_info['edge_idx']]
        edge.theta = value


    def train_vine(self, tree_type):
        r"""Build the vine.

        1. For the construction of the first tree :math:`T_1`, assign one node to each variable
           and then couple them by maximizing the measure of association considered.
           Different vines impose different constraints on this construction. When those are
           applied different trees are achieved at this level.

        2. Select the copula that best fits to the pair of variables coupled by each edge in
           :math:`T_1`.

        3. Let :math:`C_{ij}(u_i , u_j )` be the copula for a given edge :math:`(u_i, u_j)`
           in :math:`T_1`. Then for every edge in :math:`T_1`, compute either

           .. math:: {v^1}_{j|i} = \\frac{\\partial C_{ij}(u_i, u_j)}{\\partial u_j}

           or similarly :math:`{v^1}_{i|j}`, which are conditional cdfs. When finished with
           all the edges, construct the new matrix with :math:`v^1` that has one less column u.

        4. Set k = 2.

        5. Assign one node of :math:`T_k` to each edge of :math:`T_ {k−1}`. The structure of
           :math:`T_{k−1}` imposes a set of constraints on which edges of :math:`T_k` are
           realizable. Hence the next step is to get a linked list of the accesible nodes for
           every node in :math:`T_k`.

        6. As in step 1, nodes of :math:`T_k` are coupled maximizing the measure of association
           considered and satisfying the constraints impose by the kind of vine employed plus the
           set of constraints imposed by tree :math:`T_{k−1}`.

        7. Select the copula that best fit to each edge created in :math:`T_k`.

        8. Recompute matrix :math:`v_k` as in step 4, but taking :math:`T_k` and :math:`vk−1`
           instead of :math:`T_1` and u.

        9. Set :math:`k = k + 1` and repeat from (5) until all the trees are constructed.

        Args:
            tree_type (str or TreeTypes):
                Type of trees to use.
        """
        LOGGER.debug('start building tree : 0')
        # 1
        tree_1 = get_tree(tree_type)
        tree_1.fit(0, self.n_var, self.tau_mat, self.u_matrix)
        self.trees.append(tree_1)
        LOGGER.debug('finish building tree : 0')

        for k in range(1, min(self.n_var - 1, self.truncated)):
            # get constraints from previous tree
            self.trees[k - 1]._get_constraints()
            tau = self.trees[k - 1].get_tau_matrix()
            LOGGER.debug(f'start building tree: {k}')
            tree_k = get_tree(tree_type)
            tree_k.fit(k, self.n_var - k, tau, self.trees[k - 1])
            self.trees.append(tree_k)
            LOGGER.debug(f'finish building tree: {k}')

    def get_likelihood(self, uni_matrix):
        """Compute likelihood of the vine."""
        num_tree = len(self.trees)
        values = np.empty([1, num_tree])

        for i in range(num_tree):
            value, new_uni_matrix = self.trees[i].get_likelihood(uni_matrix)
            uni_matrix = new_uni_matrix
            values[0, i] = value

        return np.sum(values)

    def _sample_row(self):
        """Generate a single sampled row from vine model.

        Returns:
            numpy.ndarray
        """
        unis = np.random.uniform(0, 1, self.n_var)
        # randomly select a node to start with
        first_ind = np.random.randint(0, self.n_var)
        adj = self.trees[0].get_adjacent_matrix()
        visited = []
        explore = [first_ind]

        sampled = np.zeros(self.n_var)
        itr = 0
        while explore:
            current = explore.pop(0)
            adj_is_one = adj[current, :] == 1
            neighbors = np.where(adj_is_one)[0].tolist()
            if itr == 0:
                new_x = self.ppfs[current](unis[current])

            else:
                for i in range(itr - 1, -1, -1):
                    current_ind = -1

                    if i >= self.truncated:
                        continue

                    current_tree = self.trees[i].edges
                    # get index of edge to retrieve
                    for edge in current_tree:
                        if i == 0:
                            if (edge.L == current and edge.R == visited[0]) or (
                                edge.R == current and edge.L == visited[0]
                            ):
                                current_ind = edge.index
                                break
                        else:
                            if edge.L == current or edge.R == current:
                                condition = set(edge.D)
                                condition.add(edge.L)  # noqa: PD005
                                condition.add(edge.R)  # noqa: PD005

                                visit_set = set(visited)
                                visit_set.add(current)  # noqa: PD005

                                if condition.issubset(visit_set):
                                    current_ind = edge.index
                                break

                    if current_ind != -1:
                        # the node is not indepedent contional on visited node
                        copula_type = current_tree[current_ind].name
                        copula = Bivariate(copula_type=CopulaTypes(copula_type))
                        copula.theta = current_tree[current_ind].theta

                        U = np.array([unis[visited[0]]])
                        if i == itr - 1:
                            tmp = copula.percent_point(np.array([unis[current]]), U)[0]
                        else:
                            tmp = copula.percent_point(np.array([tmp]), U)[0]

                        tmp = min(max(tmp, EPSILON), 0.99)

                new_x = self.ppfs[current](np.array([tmp]))

            sampled[current] = new_x

            for s in neighbors:
                if s not in visited:
                    explore.insert(0, s)

            itr += 1
            visited.insert(0, current)

        return sampled

    @random_state
    def sample(self, num_rows):
        """Sample new rows.

        Args:
            num_rows (int):
                Number of rows to sample

        Returns:
            pandas.DataFrame:
                sampled rows.
        """
        sampled_values = []
        for i in range(num_rows):
            sampled_values.append(self._sample_row())

        return pd.DataFrame(sampled_values, columns=self.columns)
