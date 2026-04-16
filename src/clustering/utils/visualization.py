"""
可视化工具模块
提供聚类结果的可视化功能
"""

import numpy as np
from typing import List, Dict, Optional
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path
from .logging import default_logger as logger


def plot_clusters_2d(
    embeddings: np.ndarray,
    clusters: Dict[int, List[str]],
    items: List[str],
    output_file: str = "clusters_visualization.png",
    method: str = "tsne",
    random_state: int = 42
) -> None:
    """将聚类结果降维到2D并可视化
    
    Args:
        embeddings: 嵌入向量矩阵
        clusters: 聚类结果字典
        items: 原始文本列表
        output_file: 输出图像文件路径
        method: 降维方法 ('tsne', 'umap', 'pca')
        random_state: 随机种子
    """
    try:
        # 构建标签数组
        labels = np.full(len(items), -1, dtype=int)
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        for label, cluster_items in clusters.items():
            for item in cluster_items:
                if item in item_to_idx:
                    labels[item_to_idx[item]] = label
        
        # 降维
        if method == "tsne":
            from sklearn.manifold import TSNE
            logger.info("使用t-SNE进行降维...")
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(items)-1))
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "umap":
            try:
                import umap
                logger.info("使用UMAP进行降维...")
                reducer = umap.UMAP(n_components=2, random_state=random_state)
                embeddings_2d = reducer.fit_transform(embeddings)
            except ImportError:
                logger.warning("UMAP未安装，回退到t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=random_state, perplexity=min(30, len(items)-1))
                embeddings_2d = reducer.fit_transform(embeddings)
        elif method == "pca":
            from sklearn.decomposition import PCA
            logger.info("使用PCA进行降维...")
            reducer = PCA(n_components=2, random_state=random_state)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            raise ValueError(f"未知的降维方法: {method}")
        
        # 绘图
        plt.figure(figsize=(12, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        colors = plt.cm.get_cmap('tab20', len(unique_labels))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if label == -1:
                # 噪声点用灰色x标记
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c='gray',
                    marker='x',
                    alpha=0.5,
                    label='噪声',
                    s=50
                )
            else:
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[colors(i)],
                    label=f'簇 {label}',
                    alpha=0.7,
                    s=100,
                    edgecolors='black',
                    linewidth=0.5
                )
        
        plt.title(f'聚类可视化 ({method.upper()})', fontsize=16, fontproperties='SimHei')
        plt.xlabel('维度 1', fontsize=12, fontproperties='SimHei')
        plt.ylabel('维度 2', fontsize=12, fontproperties='SimHei')
        
        # 添加图例（限制最多20个）
        if len(unique_labels) <= 20:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', prop={'family': 'SimHei'})
        
        plt.tight_layout()
        
        # 保存图像
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化图像已保存至: {output_file}")
        
    except Exception as e:
        logger.error(f"可视化失败: {str(e)}")


def plot_cluster_size_distribution(
    clusters: Dict[int, List[str]],
    output_file: str = "cluster_sizes.png"
) -> None:
    """绘制簇大小分布图
    
    Args:
        clusters: 聚类结果字典
        output_file: 输出图像文件路径
    """
    try:
        valid_clusters = {k: v for k, v in clusters.items() if k != -1}
        cluster_sizes = [len(items) for items in valid_clusters.values()]
        
        if not cluster_sizes:
            logger.warning("没有有效的聚类，无法绘制分布图")
            return
        
        plt.figure(figsize=(10, 6))
        
        # 绘制柱状图
        plt.bar(range(len(cluster_sizes)), sorted(cluster_sizes, reverse=True))
        plt.xlabel('簇索引', fontsize=12, fontproperties='SimHei')
        plt.ylabel('簇大小', fontsize=12, fontproperties='SimHei')
        plt.title('簇大小分布', fontsize=16, fontproperties='SimHei')
        plt.grid(axis='y', alpha=0.3)
        
        # 添加统计信息
        mean_size = np.mean(cluster_sizes)
        median_size = np.median(cluster_sizes)
        plt.axhline(y=mean_size, color='r', linestyle='--', label=f'平均值: {mean_size:.1f}')
        plt.axhline(y=median_size, color='g', linestyle='--', label=f'中位数: {median_size:.1f}')
        plt.legend(prop={'family': 'SimHei'})
        
        plt.tight_layout()
        
        # 保存图像
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"簇大小分布图已保存至: {output_file}")
        
    except Exception as e:
        logger.error(f"绘制簇大小分布图失败: {str(e)}")


def create_visualization_report(
    embeddings: np.ndarray,
    clusters: Dict[int, List[str]],
    items: List[str],
    output_dir: str = "visualizations"
) -> None:
    """创建完整的可视化报告
    
    Args:
        embeddings: 嵌入向量矩阵
        clusters: 聚类结果字典
        items: 原始文本列表
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("生成可视化报告...")
    
    # 2D可视化
    plot_clusters_2d(
        embeddings, clusters, items,
        output_file=str(output_path / "clusters_tsne.png"),
        method="tsne"
    )
    
    plot_clusters_2d(
        embeddings, clusters, items,
        output_file=str(output_path / "clusters_pca.png"),
        method="pca"
    )
    
    # 簇大小分布
    plot_cluster_size_distribution(
        clusters,
        output_file=str(output_path / "cluster_sizes.png")
    )
    
    logger.info(f"可视化报告已生成至: {output_dir}")

