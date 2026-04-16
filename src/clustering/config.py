# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import yaml
from pathlib import Path

@dataclass
class ClusteringConfig:
    connectivity_threshold: float = 0.58
    min_similarity_threshold: float = 0.52
    max_cluster_size: int = 45
    max_depth: int = 5
    min_clusters: int = 1
    middle_connectivity_threshold: float = 0.55
    middle_method: str = 'hdbscan'
    middle_hdbscan_min_cluster_size: int = 2
    middle_hdbscan_min_samples: Optional[int] = 1
    middle_kmeans_k_min: int = 2
    middle_kmeans_k_max: int = 6
    middle_kmeans_auto_select: bool = True
    embedding_prefix: str = 'passage: '
    eps_range: Optional[np.ndarray] = None
    min_samples_range: Optional[range] = None
    
    def __post_init__(self):
        self.validate()
        if self.eps_range is None:
            self.eps_range = np.arange(0.08, 0.28, 0.02)
        if self.min_samples_range is None:
            self.min_samples_range = range(2, 8)
    
    def validate(self) -> None:
        if not 0 <= self.connectivity_threshold <= 1:
            raise ValueError("connectivity_threshold must be between 0 and 1")
        if not 0 <= self.min_similarity_threshold <= 1:
            raise ValueError("min_similarity_threshold must be between 0 and 1")
        if not 0 <= self.middle_connectivity_threshold <= 1:
            raise ValueError("middle_connectivity_threshold must be between 0 and 1")
        if self.middle_method not in {'threshold', 'modularity', 'hdbscan', 'kmeans'}:
            raise ValueError("middle_method must be one of {'threshold','modularity','hdbscan','kmeans'}")
        if self.max_cluster_size < 2:
            raise ValueError("max_cluster_size must be at least 2")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if self.min_clusters < 1:
            raise ValueError("min_clusters must be at least 1")
        if self.middle_hdbscan_min_cluster_size < 2:
            raise ValueError("middle_hdbscan_min_cluster_size must be at least 2")
        if self.middle_kmeans_k_min < 1 or self.middle_kmeans_k_max < self.middle_kmeans_k_min:
            raise ValueError("invalid kmeans k range")
    
    def get_eps_min_samples_range(self, cluster_size: int) -> tuple[np.ndarray, range]:
        if cluster_size < 10:
            return np.arange(0.05, 0.2, 0.02), range(2, 4)
        elif cluster_size < 30:
            return np.arange(0.1, 0.25, 0.02), range(3, 6)
        else:
            return self.eps_range, self.min_samples_range
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ClusteringConfig':
        if 'eps_range' in config_dict:
            config_dict['eps_range'] = np.array(config_dict['eps_range'])
        if 'min_samples_range' in config_dict:
            start, stop = config_dict['min_samples_range']
            config_dict['min_samples_range'] = range(start, stop)
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ClusteringConfig':
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        config_dict = {
            'connectivity_threshold': self.connectivity_threshold,
            'min_similarity_threshold': self.min_similarity_threshold,
            'max_cluster_size': self.max_cluster_size,
            'max_depth': self.max_depth,
            'min_clusters': self.min_clusters,
            'middle_connectivity_threshold': self.middle_connectivity_threshold,
            'middle_method': self.middle_method,
            'middle_hdbscan_min_cluster_size': self.middle_hdbscan_min_cluster_size,
            'middle_hdbscan_min_samples': self.middle_hdbscan_min_samples,
            'middle_kmeans_k_min': self.middle_kmeans_k_min,
            'middle_kmeans_k_max': self.middle_kmeans_k_max,
            'middle_kmeans_auto_select': self.middle_kmeans_auto_select,
            'embedding_prefix': self.embedding_prefix,
        }
        if self.eps_range is not None:
            config_dict['eps_range'] = self.eps_range.tolist()
        if self.min_samples_range is not None:
            config_dict['min_samples_range'] = [
                self.min_samples_range.start,
                self.min_samples_range.stop
            ]
        return config_dict
    
    def to_yaml(self, yaml_path: str) -> None:
        config_dict = self.to_dict()
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_dict, f, allow_unicode=True)
