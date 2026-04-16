"""
聚类评估指标模块
提供标准的聚类质量评估方法
"""

import numpy as np
from typing import List, Dict
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .logging import default_logger as logger


def calculate_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """计算轮廓系数 (Silhouette Score)
    
    范围: [-1, 1]
    值越大表示聚类质量越好
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 聚类标签数组
        
    Returns:
        float: 轮廓系数
    """
    try:
        # 过滤噪声点
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            logger.warning("有效样本数少于2，无法计算轮廓系数")
            return 0.0
        
        n_clusters = len(np.unique(labels[valid_mask]))
        if n_clusters < 2:
            logger.warning("聚类数少于2，无法计算轮廓系数")
            return 0.0
        
        score = silhouette_score(
            embeddings[valid_mask],
            labels[valid_mask],
            metric='cosine'
        )
        return float(score)
    except Exception as e:
        logger.warning(f"计算轮廓系数失败: {str(e)}")
        return 0.0


def calculate_davies_bouldin_score(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """计算Davies-Bouldin指数
    
    范围: [0, +∞)
    值越小表示聚类质量越好
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 聚类标签数组
        
    Returns:
        float: Davies-Bouldin指数
    """
    try:
        # 过滤噪声点
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            logger.warning("有效样本数少于2，无法计算Davies-Bouldin指数")
            return float('inf')
        
        n_clusters = len(np.unique(labels[valid_mask]))
        if n_clusters < 2:
            logger.warning("聚类数少于2，无法计算Davies-Bouldin指数")
            return float('inf')
        
        score = davies_bouldin_score(
            embeddings[valid_mask],
            labels[valid_mask]
        )
        return float(score)
    except Exception as e:
        logger.warning(f"计算Davies-Bouldin指数失败: {str(e)}")
        return float('inf')


def calculate_calinski_harabasz_score(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """计算Calinski-Harabasz指数 (方差比准则)
    
    范围: [0, +∞)
    值越大表示聚类质量越好
    
    Args:
        embeddings: 嵌入向量矩阵
        labels: 聚类标签数组
        
    Returns:
        float: Calinski-Harabasz指数
    """
    try:
        # 过滤噪声点
        valid_mask = labels != -1
        if np.sum(valid_mask) < 2:
            logger.warning("有效样本数少于2，无法计算Calinski-Harabasz指数")
            return 0.0
        
        n_clusters = len(np.unique(labels[valid_mask]))
        if n_clusters < 2:
            logger.warning("聚类数少于2，无法计算Calinski-Harabasz指数")
            return 0.0
        
        score = calinski_harabasz_score(
            embeddings[valid_mask],
            labels[valid_mask]
        )
        return float(score)
    except Exception as e:
        logger.warning(f"计算Calinski-Harabasz指数失败: {str(e)}")
        return 0.0


def evaluate_clustering(
    embeddings: np.ndarray,
    clusters: Dict[int, List[str]],
    items: List[str]
) -> Dict[str, float]:
    """综合评估聚类质量
    
    Args:
        embeddings: 嵌入向量矩阵
        clusters: 聚类结果字典
        items: 原始文本列表
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
    # 构建标签数组
    labels = np.full(len(items), -1, dtype=int)
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    
    for label, cluster_items in clusters.items():
        for item in cluster_items:
            if item in item_to_idx:
                labels[item_to_idx[item]] = label
    
    metrics = {
        'silhouette_score': calculate_silhouette_score(embeddings, labels),
        'davies_bouldin_score': calculate_davies_bouldin_score(embeddings, labels),
        'calinski_harabasz_score': calculate_calinski_harabasz_score(embeddings, labels),
        'n_clusters': len([k for k in clusters.keys() if k != -1]),
        'n_noise': len(clusters.get(-1, [])),
        'noise_ratio': len(clusters.get(-1, [])) / len(items) if items else 0
    }
    
    logger.info("聚类评估指标:")
    logger.info(f"  - 轮廓系数: {metrics['silhouette_score']:.4f}")
    logger.info(f"  - Davies-Bouldin指数: {metrics['davies_bouldin_score']:.4f}")
    logger.info(f"  - Calinski-Harabasz指数: {metrics['calinski_harabasz_score']:.2f}")
    logger.info(f"  - 聚类数: {metrics['n_clusters']}")
    logger.info(f"  - 噪声点数: {metrics['n_noise']}")
    logger.info(f"  - 噪声比例: {metrics['noise_ratio']:.2%}")
    
    return metrics

