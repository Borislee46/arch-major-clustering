"""
聚类算法工厂模块
用于创建和管理不同的聚类算法实例
"""

from typing import Dict, Type, Optional
from sentence_transformers import SentenceTransformer

from .base import BaseClusterer
from .dbscan import DBSCANClusterer
from .progressive import ProgressiveClusterer
from ..config import ClusteringConfig
from ..utils.exceptions import ConfigurationError
from ..utils.logging import default_logger as logger

class ClustererFactory:
    """聚类算法工厂类
    
    用于创建和管理不同的聚类算法实例
    支持注册新的聚类算法和根据配置创建算法实例
    """
    
    _clusterers: Dict[str, Type[BaseClusterer]] = {
        'dbscan': DBSCANClusterer,
        'progressive': ProgressiveClusterer
    }
    
    @classmethod
    def register_clusterer(
        cls,
        name: str,
        clusterer_class: Type[BaseClusterer]
    ) -> None:
        """注册新的聚类算法
        
        Args:
            name: 算法名称
            clusterer_class: 聚类算法类
            
        Raises:
            TypeError: 当clusterer_class不是BaseClusterer的子类时
        """
        if not issubclass(clusterer_class, BaseClusterer):
            raise TypeError(
                f"{clusterer_class.__name__}必须是BaseClusterer的子类"
            )
        
        cls._clusterers[name] = clusterer_class
        logger.info(f"注册聚类算法: {name}")
    
    @classmethod
    def create_clusterer(
        cls,
        algorithm: str,
        config: Optional[ClusteringConfig] = None,
        model: Optional[SentenceTransformer] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseClusterer:
        """创建聚类算法实例
        
        Args:
            algorithm: 算法名称
            config: 配置对象，如果为None则使用默认配置
            model: 预训练的文本编码模型
            model_name: 模型名称，当model为None时使用
            **kwargs: 传递给聚类算法的其他参数
            
        Returns:
            BaseClusterer: 聚类算法实例
            
        Raises:
            ConfigurationError: 当算法名称无效或配置无效时
        """
        if algorithm not in cls._clusterers:
            raise ConfigurationError(
                f"未知的聚类算法: {algorithm}，"
                f"可用算法: {list(cls._clusterers.keys())}"
            )
        
        config = config or ClusteringConfig()
        
        try:
            clusterer = cls._clusterers[algorithm](
                config=config,
                model=model,
                model_name=model_name,
                **kwargs
            )
            logger.info(f"创建{algorithm}聚类器实例")
            return clusterer
            
        except Exception as e:
            raise ConfigurationError(f"创建聚类器失败: {str(e)}")
    
    @classmethod
    def list_algorithms(cls) -> Dict[str, str]:
        """列出所有可用的聚类算法
        
        Returns:
            Dict[str, str]: 算法名称和描述的字典
        """
        return {
            name: clusterer.__doc__.split('\n')[0]
            for name, clusterer in cls._clusterers.items()
        } 