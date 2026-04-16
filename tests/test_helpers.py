"""
测试辅助函数模块
"""

import unittest
import numpy as np

from src.clustering.utils.helpers import (
    validate_input_data,
    calculate_similarity_matrix,
    get_cluster_statistics
)
from src.clustering.utils.exceptions import ValidationError


class TestHelpers(unittest.TestCase):
    """测试辅助函数"""
    
    def test_validate_input_data_valid(self):
        """测试有效输入数据"""
        items = ["文本1", "文本2", "文本3"]
        validate_input_data(items)  # 不应抛出异常
    
    def test_validate_input_data_empty(self):
        """测试空输入"""
        with self.assertRaises(ValidationError):
            validate_input_data([])
    
    def test_validate_input_data_invalid_type(self):
        """测试无效类型"""
        with self.assertRaises(ValidationError):
            validate_input_data([1, 2, 3])
    
    def test_validate_input_data_empty_string(self):
        """测试空字符串"""
        with self.assertRaises(ValidationError):
            validate_input_data(["", "  ", "有效文本"])
    
    def test_calculate_similarity_matrix(self):
        """测试相似度矩阵计算"""
        # 创建简单的嵌入向量
        embeddings = np.array([
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0]
        ])
        
        similarities = calculate_similarity_matrix(
            embeddings,
            show_progress=False
        )
        
        # 检查形状
        self.assertEqual(similarities.shape, (3, 3))
        
        # 检查对角线为1
        np.testing.assert_almost_equal(np.diag(similarities), [1.0, 1.0, 1.0])
        
        # 检查对称性
        np.testing.assert_array_almost_equal(similarities, similarities.T)
        
        # 检查相似度范围
        self.assertTrue(np.all(similarities >= -1))
        self.assertTrue(np.all(similarities <= 1))
    
    def test_calculate_similarity_matrix_sparse(self):
        """测试稀疏相似度矩阵"""
        embeddings = np.random.randn(10, 5)
        
        similarities = calculate_similarity_matrix(
            embeddings,
            sparse_threshold=0.5,
            show_progress=False
        )
        
        # 检查稀疏化
        self.assertTrue(np.any(similarities == 0))
    
    def test_get_cluster_statistics(self):
        """测试聚类统计"""
        clusters = {
            0: ["a", "b", "c"],
            1: ["d", "e"],
            2: ["f", "g", "h", "i"],
            -1: ["x", "y"]  # 噪声点
        }
        
        avg_size, std_size, sizes = get_cluster_statistics(clusters)
        
        # 检查统计值（不包括噪声点）
        self.assertEqual(len(sizes), 3)
        self.assertEqual(avg_size, 3.0)  # (3+2+4)/3
        self.assertGreater(std_size, 0)
    
    def test_get_cluster_statistics_empty(self):
        """测试空聚类"""
        avg_size, std_size, sizes = get_cluster_statistics({})
        
        self.assertEqual(avg_size, 0.0)
        self.assertEqual(std_size, 0.0)
        self.assertEqual(len(sizes), 0)


if __name__ == '__main__':
    unittest.main()

