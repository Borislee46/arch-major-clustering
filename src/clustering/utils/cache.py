"""
缓存工具模块
提供嵌入向量的缓存功能
"""

import hashlib
import pickle
import os
from pathlib import Path
from typing import List, Optional, Any
import numpy as np
from .logging import default_logger as logger


class EmbeddingCache:
    """嵌入向量缓存类
    
    用于缓存文本的嵌入向量，避免重复计算
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        """初始化缓存
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"初始化嵌入缓存: {self.cache_dir}")
    
    def _get_cache_key(self, texts: List[str], model_name: str) -> str:
        """生成缓存键
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            
        Returns:
            str: 缓存键（MD5哈希）
        """
        # 将文本和模型名称组合后计算哈希
        content = f"{model_name}|||{'|||'.join(sorted(texts))}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            Path: 缓存文件路径
        """
        return self.cache_dir / f"{cache_key}.pkl"
    
    def get(self, texts: List[str], model_name: str) -> Optional[np.ndarray]:
        """从缓存中获取嵌入向量
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            
        Returns:
            Optional[np.ndarray]: 嵌入向量，如果缓存不存在则返回None
        """
        cache_key = self._get_cache_key(texts, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embeddings = pickle.load(f)
                logger.info(f"从缓存加载嵌入向量: {cache_key}")
                return embeddings
            except Exception as e:
                logger.warning(f"加载缓存失败: {str(e)}")
                return None
        
        return None
    
    def set(self, texts: List[str], model_name: str, embeddings: np.ndarray) -> None:
        """将嵌入向量保存到缓存
        
        Args:
            texts: 文本列表
            model_name: 模型名称
            embeddings: 嵌入向量
        """
        cache_key = self._get_cache_key(texts, model_name)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            logger.info(f"保存嵌入向量到缓存: {cache_key}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {str(e)}")
    
    def clear(self) -> None:
        """清空所有缓存"""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            logger.info("已清空所有缓存")
    
    def get_cache_info(self) -> dict:
        """获取缓存信息
        
        Returns:
            dict: 缓存统计信息
        """
        if not self.cache_dir.exists():
            return {"count": 0, "total_size": 0}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "count": len(cache_files),
            "total_size": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }

