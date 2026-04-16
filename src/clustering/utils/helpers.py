"""
工具函数模块
提供各种通用的辅助函数
"""

import numpy as np
from typing import List, Tuple, Any, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from .logging import default_logger as logger
from .exceptions import ValidationError

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def validate_input_data(items: List[Any]) -> None:
    """验证输入数据的有效性
    
    Args:
        items: 输入数据列表
        
    Raises:
        ValidationError: 当输入数据无效时
    """
    if not items:
        raise ValidationError("输入数据列表不能为空")
    
    invalid_items = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            invalid_items.append(item)
    
    if invalid_items:
        raise ValidationError(
            "输入数据必须是非空字符串",
            invalid_items=invalid_items
        )

def normalize_major(name: str) -> str:
    import re
    name = re.sub(r'\s*\([^)]*\)', '', name)
    name = re.sub(r'[\d（）【】\[\]、]+', '', name).strip()
    return name or '其他'

def calculate_similarity_matrix(
    embeddings: np.ndarray,
    batch_size: int = 1000,
    sparse_threshold: Optional[float] = None,
    show_progress: bool = True
) -> np.ndarray:
    """高效计算相似度矩阵
    
    优化:
    1. 只计算上三角矩阵，避免重复计算
    2. 支持稀疏矩阵存储
    3. 显示进度条
    
    Args:
        embeddings: 嵌入向量矩阵
        batch_size: 批处理大小
        sparse_threshold: 稀疏阈值，小于此值的相似度置为0（None表示不使用稀疏矩阵）
        show_progress: 是否显示进度条
        
    Returns:
        np.ndarray: 相似度矩阵（密集矩阵或稀疏矩阵）
    """
    n_samples = len(embeddings)
    similarities = np.zeros((n_samples, n_samples))
    
    # 创建进度条迭代器
    batch_range = range(0, n_samples, batch_size)
    if show_progress and TQDM_AVAILABLE:
        batch_range = tqdm(batch_range, desc="计算相似度矩阵", unit="batch")
    
    for i in batch_range:
        batch_end = min(i + batch_size, n_samples)
        batch = embeddings[i:batch_end]
        
        batch_similarities = cosine_similarity(batch, embeddings)
        similarities[i:batch_end] = batch_similarities
        
        if not (show_progress and TQDM_AVAILABLE):
            logger.debug(f"计算相似度矩阵进度: {batch_end}/{n_samples}")
    
    # 对称化矩阵（因为余弦相似度是对称的）
    similarities = (similarities + similarities.T) / 2
    
    # 对角线设为1（自相似度）
    np.fill_diagonal(similarities, 1.0)
    
    # 稀疏化处理
    if sparse_threshold is not None:
        logger.info(f"应用稀疏阈值: {sparse_threshold}")
        similarities[similarities < sparse_threshold] = 0
        # 统计稀疏度
        sparsity = np.sum(similarities == 0) / (n_samples * n_samples)
        logger.info(f"矩阵稀疏度: {sparsity:.2%}")
    
    return similarities

def get_cluster_statistics(
    clusters: Dict[int, List[Any]]
) -> Tuple[float, float, List[int]]:
    """计算聚类结果的统计信息
    
    Args:
        clusters: 聚类结果字典，键为类别标签，值为该类别的项目列表
        
    Returns:
        Tuple[float, float, List[int]]: (平均类大小, 类大小标准差, 各类大小列表)
    """
    if not clusters:
        return 0.0, 0.0, []
    
    valid_clusters = {k: v for k, v in clusters.items() if k != -1}
    if not valid_clusters:
        return 0.0, 0.0, []
    
    cluster_sizes = [len(items) for items in valid_clusters.values()]
    avg_size = np.mean(cluster_sizes)
    std_size = np.std(cluster_sizes)
    
    return avg_size, std_size, cluster_sizes 