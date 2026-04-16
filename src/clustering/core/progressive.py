"""
渐进式聚类实现
通过逐步调整参数实现更精细的聚类控制
"""

from typing import List, Dict, Optional
import numpy as np
import networkx as nx

from .base import BaseClusterer
from ..utils.exceptions import OptimizationError
from ..utils.logging import default_logger as logger
from ..utils.helpers import get_cluster_statistics

class ProgressiveClusterer(BaseClusterer):
    """渐进式聚类实现
    
    通过以下策略实现渐进式聚类：
    1. 首先尝试基于连通性的预分类
    2. 对大类进行递归细分
    3. 动态调整参数以获得最优结果
    """
    
    def fit(self, items: List[str]) -> Dict[int, List[str]]:
        self.preprocess(items)
        
        logger.info("开始基于连通性的预分类")
        pre_clusters = self._connectivity_based_clustering(items)
        
        if pre_clusters and self._is_valid_clustering(pre_clusters):
            logger.info("连通性预分类结果有效，无需进一步处理")
            return pre_clusters
        
        logger.info("开始渐进式聚类")
        return self._progressive_clustering(items)
    
    def _connectivity_based_clustering(self, items: List[str]) -> Optional[Dict[int, List[str]]]:
        G = nx.Graph()
        n = len(self.similarities)
        
        for i in range(n):
            for j in range(i+1, n):
                if self.similarities[i][j] >= self.config.connectivity_threshold:
                    G.add_edge(i, j, weight=self.similarities[i][j])
        
        components = list(nx.connected_components(G))
        
        if len(components) == 1:
            if len(components[0]) <= self.config.max_cluster_size:
                return {0: [items[idx] for idx in components[0]]}
            return None
        
        for comp in components:
            if len(comp) > self.config.max_cluster_size:
                return None
        
        clusters = {}
        for ci, comp in enumerate(components):
            clusters[ci] = [items[idx] for idx in comp]
        
        return clusters
    
    def _progressive_clustering(self, items: List[str]) -> Dict[int, List[str]]:
        best_clusters = None
        best_score = -1
        
        original_threshold = self.config.connectivity_threshold
        lower = max(0.0, original_threshold - 0.25)
        upper = min(0.99, original_threshold + 0.3)
        thresholds = np.linspace(lower, upper, 9)
        
        for threshold in thresholds:
            logger.debug(f"尝试连通性阈值: {threshold:.2f}")
            
            clusters = self._recursive_clustering(items, threshold=threshold)
            
            if not self._is_valid_clustering(clusters):
                continue
            
            score = self._calculate_clustering_score(clusters)
            
            if score > best_score:
                best_score = score
                best_clusters = clusters
                logger.debug(f"找到更好的阈值，分数: {score:.4f}")
        
        if best_clusters is None:
            raise OptimizationError("无法找到合适的聚类结果")
        
        logger.info(f"聚类完成，最终分数: {best_score:.4f}")
        return best_clusters
    
    def _recursive_clustering(
        self,
        items: List[str],
        depth: int = 0,
        threshold: Optional[float] = None
    ) -> Dict[int, List[str]]:
        if threshold is None:
            threshold = self.config.connectivity_threshold

        if depth >= self.config.max_depth:
            return {0: items}
        
        if len(items) <= self.config.max_cluster_size:
            return {0: items}
        
        global_indices = self._get_indices_for_items(items)
        if not global_indices:
            return {0: items}
        sub_similarities = self.similarities[np.ix_(global_indices, global_indices)]
        
        G = nx.Graph()
        n_local = len(global_indices)
        for i in range(n_local):
            for j in range(i+1, n_local):
                if sub_similarities[i][j] >= threshold:
                    G.add_edge(i, j)
        
        components = list(nx.connected_components(G))
        
        if len(components) == 1:
            higher_threshold = min(threshold + 0.1, 0.9)
            G = nx.Graph()
            for i in range(n_local):
                for j in range(i+1, n_local):
                    if sub_similarities[i][j] >= higher_threshold:
                        G.add_edge(i, j)
            components = list(nx.connected_components(G))
        
        clusters = {}
        next_label = 0
        
        for component in components:
            comp_items = [items[i] for i in component]
            if len(comp_items) > self.config.max_cluster_size:
                sub_clusters = self._recursive_clustering(
                    comp_items,
                    depth + 1,
                    threshold=threshold
                )
                for label, cluster_items in sub_clusters.items():
                    if label != -1:
                        clusters[next_label] = cluster_items
                        next_label += 1
            else:
                clusters[next_label] = comp_items
                next_label += 1
        
        return clusters
    
    def _is_valid_clustering(self, clusters: Dict[int, List[str]]) -> bool:
        if not clusters:
            return False
        total_items = sum(len(items) for items in clusters.values())
        if total_items == 0:
            return False
        noise_items = len(clusters.get(-1, []))
        if noise_items / total_items > 0.3:
            return False
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}
        if len(valid_clusters) < self.config.min_clusters:
            return False
        for label, items in valid_clusters.items():
            if len(items) > self.config.max_cluster_size:
                return False
        return True
    
    def _calculate_clustering_score(self, clusters: Dict[int, List[str]]) -> float:
        """计算聚类质量分数
        
        权重分配：类内相似度 0.35 + 类间差异度 0.25 + 噪声比例 0.20 + 大小均匀性 0.20
        """
        if not clusters or (len(clusters) == 1 and -1 in clusters):
            return 0.0
        
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}
        if not valid_clusters:
            return 0.0
        
        intra_similarities = []
        for items in valid_clusters.values():
            if len(items) > 1:
                indices = self._get_indices_for_items(items)
                if len(indices) < 2:
                    continue
                cluster_similarities = self.similarities[np.ix_(indices, indices)]
                denominator = len(indices) * (len(indices) - 1)
                if denominator > 0:
                    avg_sim = (cluster_similarities.sum() - len(indices)) / denominator
                    intra_similarities.append(avg_sim)
        
        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0
        
        inter_similarities = []
        cluster_indices = []
        for items in valid_clusters.values():
            cluster_indices.append(self._get_indices_for_items(items))
        
        for i in range(len(cluster_indices)):
            for j in range(i+1, len(cluster_indices)):
                if cluster_indices[i] and cluster_indices[j]:
                    similarities = self.similarities[
                        np.ix_(cluster_indices[i], cluster_indices[j])
                    ]
                    inter_similarities.append(np.mean(similarities))
        
        avg_inter_similarity = np.mean(inter_similarities) if inter_similarities else 0
        
        total_items = sum(len(items) for items in clusters.values())
        noise_ratio = len(clusters.get(-1, [])) / total_items
        
        avg_size, std_size, _ = get_cluster_statistics(valid_clusters)
        size_uniformity = max(0.0, 1 - (std_size / avg_size)) if avg_size > 0 else 0
        
        score = (
            0.35 * avg_intra_similarity +
            0.25 * (1 - avg_inter_similarity) +
            0.20 * (1 - noise_ratio) +
            0.20 * size_uniformity
        )
        
        return score
