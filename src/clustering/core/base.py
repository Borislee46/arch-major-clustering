"""
聚类算法基类模块
定义聚类算法的通用接口和基础功能
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import ClusteringConfig
from ..utils.exceptions import ValidationError
from ..utils.helpers import validate_input_data, calculate_similarity_matrix
from ..utils.logging import default_logger as logger
from ..utils.cache import EmbeddingCache
from ..utils.metrics import evaluate_clustering

class BaseClusterer(ABC):
    """聚类算法基类
    
    定义了聚类算法的基本接口和通用功能
    
    Attributes:
        config: 聚类配置对象
        model: 文本编码模型
        embeddings: 文本嵌入向量
        similarities: 相似度矩阵
    """
    
    def __init__(
        self,
        config: ClusteringConfig,
        model: Optional[SentenceTransformer] = None,
        model_name: str = 'intfloat/multilingual-e5-large-instruct',
        use_cache: bool = True
    ):
        self.config = config
        self.model = model or SentenceTransformer(model_name)
        self.model_name = model_name
        self.embeddings = None
        self.similarities = None
        self.items = None
        self._item_to_index = None
        self.use_cache = use_cache
        self.cache = EmbeddingCache() if use_cache else None
        logger.info(f"初始化{self.__class__.__name__}聚类器")
    
    def preprocess(self, items: List[str]) -> None:
        """数据预处理
        
        验证输入数据并计算文本嵌入向量和相似度矩阵
        
        Args:
            items: 待聚类的文本列表
        """
        validate_input_data(items)
        logger.info(f"开始处理{len(items)}个项目")
        
        # 指令前缀（E5-Instruct等模型可提升效果）
        prefix = self.config.embedding_prefix or ''
        prefixed_items = [f"{prefix}{it}" if prefix else it for it in items]
        self.items = items
        self._item_to_index = {item: idx for idx, item in enumerate(items)}
        
        # 尝试从缓存加载嵌入向量
        if self.use_cache and self.cache:
            logger.debug("尝试从缓存加载嵌入向量")
            self.embeddings = self.cache.get(prefixed_items, self.model_name)
        
        # 如果缓存未命中，计算嵌入向量
        if self.embeddings is None:
            logger.debug("计算文本嵌入向量")
            self.embeddings = self.model.encode(prefixed_items, show_progress_bar=True)
            
            # 保存到缓存
            if self.use_cache and self.cache:
                self.cache.set(prefixed_items, self.model_name, self.embeddings)
        
        logger.debug("计算相似度矩阵")
        self.similarities = calculate_similarity_matrix(self.embeddings)

    def _get_indices_for_items(self, items: List[str]) -> List[int]:
        return [self._item_to_index[item] for item in items if item in self._item_to_index]

    def compute_cluster_centroids(self, clusters: Dict[int, List[str]]) -> Dict[int, np.ndarray]:
        """计算每个簇的簇心（平均嵌入）"""
        centroids = {}
        for label, items in clusters.items():
            if label == -1:
                continue
            indices = self._get_indices_for_items(items)
            if not indices:
                continue
            vectors = self.embeddings[indices]
            centroid = vectors.mean(axis=0)
            # 归一化以用于余弦相似度
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            centroids[label] = centroid
        return centroids

    def build_middle_clusters(self, clusters: Dict[int, List[str]]) -> Dict[int, int]:
        """构建中簇：返回 小簇label -> 中簇id 的映射
        支持三种方法：threshold / modularity / hdbscan
        """
        method = getattr(self.config, 'middle_method', 'threshold')
        if method == 'threshold':
            centroids = self.compute_cluster_centroids(clusters)
            labels = list(centroids.keys())
            if not labels:
                return {}
            matrix = np.stack([centroids[l] for l in labels], axis=0)
            sims = matrix @ matrix.T
            threshold = self.config.middle_connectivity_threshold
            n = len(labels)
            visited = [False] * n
            middle_id = 0
            cluster_to_middle: Dict[int, int] = {}
            for i in range(n):
                if visited[i]:
                    continue
                stack = [i]
                visited[i] = True
                group_indices = [i]
                while stack:
                    u = stack.pop()
                    for v in range(n):
                        if not visited[v] and sims[u, v] >= threshold and u != v:
                            visited[v] = True
                            stack.append(v)
                            group_indices.append(v)
                for gi in group_indices:
                    cluster_to_middle[labels[gi]] = middle_id
                middle_id += 1
            return cluster_to_middle
        elif method == 'modularity':
            try:
                import networkx as nx
            except ImportError:
                # networkx未安装，回退到阈值方法
                logger.warning("networkx未安装，回退到threshold方法")
                setattr(self.config, 'middle_method', 'threshold')
                return self.build_middle_clusters(clusters)
            centroids = self.compute_cluster_centroids(clusters)
            labels = list(centroids.keys())
            if not labels:
                return {}
            matrix = np.stack([centroids[l] for l in labels], axis=0)
            sims = matrix @ matrix.T
            threshold = self.config.middle_connectivity_threshold
            G = nx.Graph()
            for i, li in enumerate(labels):
                G.add_node(li)
            for i in range(len(labels)):
                for j in range(i+1, len(labels)):
                    if sims[i, j] >= threshold:
                        G.add_edge(labels[i], labels[j], weight=float(sims[i, j]))
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G, weight='weight'))
            except (ImportError, AttributeError) as e:
                # 社区检测模块不可用或失败，回退
                logger.warning(f"社区检测失败: {str(e)}，回退到threshold方法")
                setattr(self.config, 'middle_method', 'threshold')
                return self.build_middle_clusters(clusters)
            mapping: Dict[int, int] = {}
            for cid, comm in enumerate(communities):
                for label in comm:
                    mapping[int(label)] = cid
            # 孤立点处理
            used = set(mapping.keys())
            next_id = len(set(mapping.values()))
            for l in labels:
                if int(l) not in used:
                    mapping[int(l)] = next_id
                    next_id += 1
            return mapping
        elif method == 'hdbscan':
            # 基于簇心再做 HDBSCAN，适合不规则密度
            try:
                import hdbscan  # type: ignore
            except ImportError:
                # hdbscan未安装，回退
                logger.warning("hdbscan未安装，回退到threshold方法")
                setattr(self.config, 'middle_method', 'threshold')
                return self.build_middle_clusters(clusters)
            centroids = self.compute_cluster_centroids(clusters)
            labels = list(centroids.keys())
            if not labels:
                return {}
            X = np.stack([centroids[l] for l in labels], axis=0)
            # 余弦距离 = 1 - 余弦相似度；HDBSCAN 支持 precomputed，亦可直接在向量上跑 metric='euclidean'
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self.config.middle_hdbscan_min_cluster_size,
                min_samples=self.config.middle_hdbscan_min_samples,
                metric='euclidean'
            )
            h_labels = clusterer.fit_predict(X)
            mapping: Dict[int, int] = {}
            # 将噪声(-1)各自独立成中簇，避免全部变 0
            next_id = int(h_labels[h_labels >= 0].max() + 1) if (h_labels >= 0).any() else 0
            for idx, l in enumerate(labels):
                if h_labels[idx] >= 0:
                    mapping[int(l)] = int(h_labels[idx])
                else:
                    mapping[int(l)] = next_id
                    next_id += 1
            return mapping
        elif method == 'kmeans':
            # 使用簇心做 KMeans，支持自动选 K
            from sklearn.cluster import KMeans
            centroids = self.compute_cluster_centroids(clusters)
            labels = list(centroids.keys())
            if not labels:
                return {}
            X = np.stack([centroids[l] for l in labels], axis=0)
            k_min = getattr(self.config, 'middle_kmeans_k_min', 2)
            k_max = getattr(self.config, 'middle_kmeans_k_max', 6)
            auto = getattr(self.config, 'middle_kmeans_auto_select', True)
            chosen_k = k_min
            if auto and X.shape[0] > 1:
                # 简单的肘部法启发：选择使得inertia相对下降最显著的K
                inertias = []
                candidate_ks = [k for k in range(k_min, min(k_max, X.shape[0]) + 1) if k >= 1]
                for k in candidate_ks:
                    km = KMeans(n_clusters=k, n_init=10, random_state=42)
                    km.fit(X)
                    inertias.append(km.inertia_)
                if len(candidate_ks) >= 2:
                    # 计算相对下降比
                    drops = []
                    for i in range(1, len(inertias)):
                        prev, cur = inertias[i-1], inertias[i]
                        drops.append((candidate_ks[i], (prev - cur) / max(prev, 1e-8)))
                    # 选择下降最大的K
                    chosen_k = max(drops, key=lambda x: x[1])[0]
                else:
                    chosen_k = candidate_ks[0]
            else:
                chosen_k = min(max(k_min, 1), X.shape[0])
            km = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
            km_labels = km.fit_predict(X)
            mapping: Dict[int, int] = {}
            for idx, l in enumerate(labels):
                mapping[int(l)] = int(km_labels[idx])
            return mapping
        else:
            # 未知方法回退
            return {}
    
    @abstractmethod
    def fit(self, items: List[str]) -> Dict[int, List[str]]:
        """执行聚类
        
        Args:
            items: 待聚类的文本列表
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典，键为类别标签，值为该类别的文本列表
        """
        pass
    
    def get_cluster_items(
        self,
        items: List[str],
        labels: np.ndarray
    ) -> Dict[int, List[str]]:
        """将聚类标签转换为结果字典
        
        Args:
            items: 原始文本列表
            labels: 聚类标签数组
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典
        """
        clusters = {}
        for item, label in zip(items, labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(item)
        return clusters
    
    def validate_results(
        self,
        clusters: Dict[int, List[str]]
    ) -> None:
        """验证聚类结果
        
        Args:
            clusters: 聚类结果字典
            
        Raises:
            ValidationError: 当结果不满足要求时
        """
        if not clusters:
            raise ValidationError("聚类结果为空")
        
        for label, items in clusters.items():
            if not items:
                raise ValidationError(f"类别{label}为空")
            
            if (label != -1 and
                len(items) > self.config.max_cluster_size):
                raise ValidationError(
                    f"类别{label}的大小({len(items)})超过最大限制"
                    f"({self.config.max_cluster_size})"
                )
    
    def evaluate_results(self, clusters: Dict[int, List[str]]) -> Dict[str, float]:
        """评估聚类质量
        
        使用标准指标评估聚类结果
        
        Args:
            clusters: 聚类结果字典
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        if self.embeddings is None or self.items is None:
            logger.warning("嵌入向量或项目列表为空，无法评估")
            return {}
        
        return evaluate_clustering(self.embeddings, clusters, self.items) 