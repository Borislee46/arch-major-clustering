"""
聚类算法的自定义异常类模块
"""

class ClusteringError(Exception):
    """聚类算法基础异常类"""
    pass

class ConfigurationError(ClusteringError):
    """配置相关错误"""
    pass

class ConnectivityError(ClusteringError):
    """连通性检查相关错误"""
    pass

class QualityError(ClusteringError):
    """聚类质量相关错误"""
    pass

class OptimizationError(ClusteringError):
    """聚类优化相关错误"""
    pass

class ValidationError(ClusteringError):
    """数据验证相关错误"""
    def __init__(self, message: str, invalid_items: list = None):
        super().__init__(message)
        self.invalid_items = invalid_items or [] 