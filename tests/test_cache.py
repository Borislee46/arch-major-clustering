"""
测试缓存模块
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path

from src.clustering.utils.cache import EmbeddingCache


class TestEmbeddingCache(unittest.TestCase):
    """测试嵌入缓存"""
    
    def setUp(self):
        """设置测试环境"""
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache = EmbeddingCache(self.tmpdir.name)
    
    def tearDown(self):
        """清理测试环境"""
        self.tmpdir.cleanup()
    
    def test_cache_miss(self):
        """测试缓存未命中"""
        texts = ["文本1", "文本2"]
        model_name = "test_model"
        
        result = self.cache.get(texts, model_name)
        self.assertIsNone(result)
    
    def test_cache_hit(self):
        """测试缓存命中"""
        texts = ["文本1", "文本2"]
        model_name = "test_model"
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # 保存到缓存
        self.cache.set(texts, model_name, embeddings)
        
        # 从缓存加载
        cached_embeddings = self.cache.get(texts, model_name)
        self.assertIsNotNone(cached_embeddings)
        np.testing.assert_array_equal(cached_embeddings, embeddings)
    
    def test_cache_different_texts(self):
        """测试不同文本的缓存"""
        texts1 = ["文本1", "文本2"]
        texts2 = ["文本3", "文本4"]
        model_name = "test_model"
        embeddings1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        embeddings2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        # 保存两组缓存
        self.cache.set(texts1, model_name, embeddings1)
        self.cache.set(texts2, model_name, embeddings2)
        
        # 验证独立性
        cached1 = self.cache.get(texts1, model_name)
        cached2 = self.cache.get(texts2, model_name)
        
        np.testing.assert_array_equal(cached1, embeddings1)
        np.testing.assert_array_equal(cached2, embeddings2)
    
    def test_cache_different_models(self):
        """测试不同模型的缓存"""
        texts = ["文本1", "文本2"]
        model1 = "model1"
        model2 = "model2"
        embeddings1 = np.array([[1.0, 2.0], [3.0, 4.0]])
        embeddings2 = np.array([[5.0, 6.0], [7.0, 8.0]])
        
        # 保存两个模型的缓存
        self.cache.set(texts, model1, embeddings1)
        self.cache.set(texts, model2, embeddings2)
        
        # 验证独立性
        cached1 = self.cache.get(texts, model1)
        cached2 = self.cache.get(texts, model2)
        
        np.testing.assert_array_equal(cached1, embeddings1)
        np.testing.assert_array_equal(cached2, embeddings2)
    
    def test_clear_cache(self):
        """测试清空缓存"""
        texts = ["文本1", "文本2"]
        model_name = "test_model"
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # 保存到缓存
        self.cache.set(texts, model_name, embeddings)
        
        # 清空缓存
        self.cache.clear()
        
        # 验证缓存已清空
        result = self.cache.get(texts, model_name)
        self.assertIsNone(result)
    
    def test_get_cache_info(self):
        """测试获取缓存信息"""
        # 初始状态
        info = self.cache.get_cache_info()
        self.assertEqual(info['count'], 0)
        
        # 添加一些缓存
        texts = ["文本1", "文本2"]
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]])
        self.cache.set(texts, "model1", embeddings)
        self.cache.set(texts, "model2", embeddings)
        
        # 检查信息
        info = self.cache.get_cache_info()
        self.assertEqual(info['count'], 2)
        self.assertGreater(info['total_size'], 0)


if __name__ == '__main__':
    unittest.main()

