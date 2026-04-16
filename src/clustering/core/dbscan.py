"""
基于DBSCAN的聚类实现
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from collections import defaultdict

from .base import BaseClusterer
from ..utils.exceptions import OptimizationError
from ..utils.logging import default_logger as logger
from ..utils.helpers import get_cluster_statistics

class DBSCANClusterer(BaseClusterer):
    """基于DBSCAN的聚类实现
    
    使用DBSCAN算法进行聚类，支持参数优化和结果验证
    """
    
    def fit(self, items: List[str]) -> Dict[int, List[str]]:
        """执行DBSCAN聚类
        
        Args:
            items: 待聚类的文本列表
            
        Returns:
            Dict[int, List[str]]: 聚类结果字典
            
        Raises:
            OptimizationError: 当无法找到合适的参数时
        """
        self.preprocess(items)
        
        best_clusters = None
        best_score = -1
        best_params = None
        
        # 计算余弦距离矩阵（使用嵌入向量而非相似度矩阵）
        # 余弦距离 = 1 - 余弦相似度，范围 [0, 2]
        distance_matrix = squareform(pdist(self.embeddings, metric='cosine'))
        
        eps_range, min_samples_range = self.config.get_eps_min_samples_range(len(items))
        
        for eps in eps_range:
            for min_samples in min_samples_range:
                logger.debug(f"尝试参数: eps={eps}, min_samples={min_samples}")
                
                labels = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    metric='precomputed'
                ).fit(distance_matrix).labels_
                
                clusters = self.get_cluster_items(items, labels)
                
                try:
                    self.validate_results(clusters)
                    
                    score = self._calculate_clustering_score(clusters)
                    
                    if score > best_score:
                        best_score = score
                        best_clusters = clusters
                        best_params = (eps, min_samples)
                        logger.debug(f"找到更好的参数组合，分数: {score:.4f}")
                except Exception as e:
                    logger.debug(f"参数组合无效: {str(e)}")
                    continue
        
        if best_clusters is None:
            raise OptimizationError("无法找到合适的聚类参数")
        
        logger.info(f"最优参数: eps={best_params[0]}, min_samples={best_params[1]}")
        logger.info(f"聚类分数: {best_score:.4f}")
        
        avg_size, std_size, cluster_sizes = get_cluster_statistics(best_clusters)
        logger.info(f"平均类大小: {avg_size:.2f} ± {std_size:.2f}")
        logger.info(f"类大小分布: {cluster_sizes}")
        
        return best_clusters
    
    def _calculate_clustering_score(self, clusters: Dict[int, List[str]]) -> float:
        """计算聚类质量分数
        
        考虑以下因素：
        1. 类内平均相似度
        2. 噪声点比例
        3. 类大小分布的均匀性
        
        Args:
            clusters: 聚类结果字典
            
        Returns:
            float: 聚类质量分数，范围[0,1]
        """
        if not clusters or (len(clusters) == 1 and -1 in clusters):
            return 0.0
        
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}
        total_items = sum(len(items) for items in clusters.values())
        noise_ratio = len(clusters.get(-1, [])) / total_items
        
        if not valid_clusters:
            return 0.0
        
        intra_similarities = []
        for items in valid_clusters.values():
            if len(items) > 1:
                # 使用基类提供的方法获取正确的索引
                indices = self._get_indices_for_items(items)
                if not indices:
                    continue
                cluster_similarities = self.similarities[np.ix_(indices, indices)]
                avg_sim = (cluster_similarities.sum() - len(indices)) / (len(indices) * (len(indices) - 1))
                intra_similarities.append(avg_sim)
        
        avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0
        
        avg_size, std_size, _ = get_cluster_statistics(valid_clusters)
        size_uniformity = 1 - (std_size / avg_size) if avg_size > 0 else 0

        score = (
            0.4 * avg_intra_similarity +
            0.3 * (1 - noise_ratio) +    
            0.3 * size_uniformity        
        )
        
        return score 