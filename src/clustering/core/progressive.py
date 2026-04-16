"""
渐进式聚类实现
通过逐步调整参数实现更精细的聚类控制
"""

from typing import List, Dict, Optional
import copy
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
        """执行渐进式聚类
        
        Args:
            items: 待聚类的文本列表
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典
        """
        self.preprocess(items)
        
        logger.info("开始基于连通性的预分类")
        pre_clusters = self._connectivity_based_clustering(items)
        
        if pre_clusters and self._is_valid_clustering(pre_clusters):
            logger.info("连通性预分类结果有效，无需进一步处理")
            return pre_clusters
        
        logger.info("开始渐进式聚类")
        return self._progressive_clustering(items)
    
    def _connectivity_based_clustering(self, items: List[str]) -> Optional[Dict[int, List[str]]]:
        """基于连通性的预分类
        
        Args:
            items: 待聚类的文本列表
            
        Returns:
            Optional[Dict[int, List[str]]]: 聚类结果字典，如果预分类失败则返回None
        """
        G = nx.Graph()
        n = len(self.similarities)
        
        for i in range(n):
            for j in range(i+1, n):
                if self.similarities[i][j] >= self.config.connectivity_threshold:
                    G.add_edge(i, j, weight=self.similarities[i][j])
        
        components = list(nx.connected_components(G))
        
        if len(components) == 1:
            if len(components[0]) <= self.config.max_cluster_size:
                return {0: [items[i] for i in components[0]]}
            return None
        
        for comp in components:
            if len(comp) > self.config.max_cluster_size:
                return None
        
        clusters = {}
        for i, comp in enumerate(components):
            clusters[i] = [items[i] for i in comp]
        
        return clusters
    
    def _progressive_clustering(self, items: List[str]) -> Dict[int, List[str]]:
        """执行渐进式聚类
        
        Args:
            items: 待聚类的文本列表
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典
            
        Raises:
            OptimizationError: 当无法找到合适的聚类结果时
        """
        best_clusters = None
        best_score = -1
        
        # 扩大阈值搜索范围：从当前阈值向下和向上各扩展，避免卡住
        original_threshold = self.config.connectivity_threshold
        lower = max(0.0, original_threshold - 0.25)
        upper = min(0.99, original_threshold + 0.3)
        thresholds = np.linspace(lower, upper, 9)
        
        try:
            for threshold in thresholds:
                logger.debug(f"尝试连通性阈值: {threshold:.2f}")
                
                # 临时修改阈值
                self.config.connectivity_threshold = threshold
                
                try:
                    clusters = self._recursive_clustering(items)
                    
                    if not self._is_valid_clustering(clusters):
                        continue
                    
                    score = self._calculate_clustering_score(clusters)
                    
                    if score > best_score:
                        best_score = score
                        best_clusters = clusters
                        logger.debug(f"找到更好的阈值，分数: {score:.4f}")
                
                except Exception as e:
                    logger.debug(f"阈值{threshold}无效: {str(e)}")
                    continue
        finally:
            # 恢复原始阈值
            self.config.connectivity_threshold = original_threshold
        
        if best_clusters is None:
            raise OptimizationError("无法找到合适的聚类结果")
        
        logger.info(f"聚类完成，最终分数: {best_score:.4f}")
        return best_clusters
    
    def _recursive_clustering(
        self,
        items: List[str],
        depth: int = 0
    ) -> Dict[int, List[str]]:
        """递归聚类
        
        Args:
            items: 待聚类的文本列表
            depth: 当前递归深度
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典
        """
        if depth >= self.config.max_depth:
            return {0: items}
        
        if len(items) <= self.config.max_cluster_size:
            return {0: items}
        
        all_items = items
        item_indices = {item: i for i, item in enumerate(all_items)}
        indices = [item_indices[item] for item in items]
        sub_similarities = self.similarities[np.ix_(indices, indices)]
        
        G = nx.Graph()
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if sub_similarities[i][j] >= self.config.connectivity_threshold:
                    G.add_edge(i, j)
        
        components = list(nx.connected_components(G))
        
        if len(components) == 1:
            higher_threshold = min(
                self.config.connectivity_threshold + 0.1,
                0.9
            )
            G = nx.Graph()
            for i in range(len(indices)):
                for j in range(i+1, len(indices)):
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
                    depth + 1
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
        """检查聚类结果是否有效
        
        Args:
            clusters: 聚类结果字典
            
        Returns:
            bool: 是否有效
        """
        try:
            self.validate_results(clusters)
            
            total_items = sum(len(items) for items in clusters.values())
            noise_items = len(clusters.get(-1, []))
            if noise_items / total_items > 0.3:
                return False
            
            valid_clusters = {k: v for k, v in clusters.items() if k != -1}
            if len(valid_clusters) < self.config.min_clusters:
                return False
            
            return True
        
        except Exception:
            return False
    
    def _calculate_clustering_score(self, clusters: Dict[int, List[str]]) -> float:
        """计算聚类质量分数
        
        考虑以下因素：
        1. 类内相似度
        2. 类间相似度
        3. 噪声点比例
        4. 类大小分布
        
        Args:
            clusters: 聚类结果字典
            
        Returns:
            float: 聚类质量分数，范围[0,1]
        """
        if not clusters or (len(clusters) == 1 and -1 in clusters):
            return 0.0
        
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}
        if not valid_clusters:
            return 0.0
        
        all_items = []
        for items in clusters.values():
            all_items.extend(items)
        item_indices = {item: i for i, item in enumerate(all_items)}
        
        intra_similarities = []
        for items in valid_clusters.values():
            if len(items) > 1:
                indices = [item_indices[item] for item in items if item in item_indices]
                if not indices or len(indices) < 2:
                    continue
                cluster_similarities = self.similarities[np.ix_(indices, indices)]
                # 添加边界检查，避免除零错误
                denominator = len(indices) * (len(indices) - 1)
                if denominator > 0:
                    avg_sim = (cluster_similarities.sum() - len(indices)) / denominator
                    intra_similarities.append(avg_sim)
        
        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0
        
        inter_similarities = []
        cluster_indices = []
        for items in valid_clusters.values():
            indices = [item_indices[item] for item in items]
            cluster_indices.append(indices)
        
        for i in range(len(cluster_indices)):
            for j in range(i+1, len(cluster_indices)):
                similarities = self.similarities[
                    np.ix_(cluster_indices[i], cluster_indices[j])
                ]
                inter_similarities.append(np.mean(similarities))
        
        avg_inter_similarity = np.mean(inter_similarities) if inter_similarities else 0
        
        total_items = sum(len(items) for items in clusters.values())
        noise_ratio = len(clusters.get(-1, [])) / total_items
        
        avg_size, std_size, _ = get_cluster_statistics(valid_clusters)
        size_uniformity = 1 - (std_size / avg_size) if avg_size > 0 else 0
        
        score = (
            0.35 * avg_intra_similarity +     # 类内相似度
            0.25 * (1 - avg_inter_similarity) + # 类间差异度
            0.20 * (1 - noise_ratio) +        # 噪声点比例
            0.20 * size_uniformity            # 类大小均匀性
        )
        
        return score 