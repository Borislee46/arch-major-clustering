"""
测试配置模块
"""

import unittest
import tempfile
from pathlib import Path
import yaml

from src.clustering.config import ClusteringConfig


class TestClusteringConfig(unittest.TestCase):
    """测试ClusteringConfig类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ClusteringConfig()
        
        self.assertEqual(config.connectivity_threshold, 0.58)
        self.assertEqual(config.min_similarity_threshold, 0.52)
        self.assertEqual(config.max_cluster_size, 45)
        self.assertEqual(config.max_depth, 5)
        self.assertIsNotNone(config.eps_range)
        self.assertIsNotNone(config.min_samples_range)
    
    def test_validation(self):
        """测试配置验证"""
        # 有效配置
        config = ClusteringConfig()
        config.validate()  # 不应抛出异常
        
        # 无效的connectivity_threshold
        with self.assertRaises(ValueError):
            config = ClusteringConfig(connectivity_threshold=-0.1)
        
        with self.assertRaises(ValueError):
            config = ClusteringConfig(connectivity_threshold=1.5)
        
        # 无效的max_cluster_size
        with self.assertRaises(ValueError):
            config = ClusteringConfig(max_cluster_size=1)
        
        # 无效的max_depth
        with self.assertRaises(ValueError):
            config = ClusteringConfig(max_depth=0)
    
    def test_to_dict_from_dict(self):
        """测试字典转换"""
        config1 = ClusteringConfig(
            connectivity_threshold=0.7,
            max_cluster_size=50
        )
        
        config_dict = config1.to_dict()
        config2 = ClusteringConfig.from_dict(config_dict)
        
        self.assertEqual(config1.connectivity_threshold, config2.connectivity_threshold)
        self.assertEqual(config1.max_cluster_size, config2.max_cluster_size)
    
    def test_yaml_io(self):
        """测试YAML文件读写"""
        config1 = ClusteringConfig(
            connectivity_threshold=0.7,
            max_cluster_size=50
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_config.yaml"
            
            # 保存
            config1.to_yaml(str(yaml_path))
            self.assertTrue(yaml_path.exists())
            
            # 加载
            config2 = ClusteringConfig.from_yaml(str(yaml_path))
            self.assertEqual(config1.connectivity_threshold, config2.connectivity_threshold)
            self.assertEqual(config1.max_cluster_size, config2.max_cluster_size)
    
    def test_get_eps_min_samples_range(self):
        """测试动态参数范围"""
        config = ClusteringConfig()
        
        # 小数据集
        eps_range, min_samples_range = config.get_eps_min_samples_range(5)
        self.assertTrue(len(eps_range) > 0)
        self.assertTrue(len(min_samples_range) > 0)
        
        # 中等数据集
        eps_range, min_samples_range = config.get_eps_min_samples_range(20)
        self.assertTrue(len(eps_range) > 0)
        self.assertTrue(len(min_samples_range) > 0)
        
        # 大数据集
        eps_range, min_samples_range = config.get_eps_min_samples_range(100)
        self.assertTrue(len(eps_range) > 0)
        self.assertTrue(len(min_samples_range) > 0)


if __name__ == '__main__':
    unittest.main()

