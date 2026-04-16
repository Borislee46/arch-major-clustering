# -*- coding: utf-8 -*-
"""
聚类分析主程序
提供命令行接口和主要功能入口
"""

import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

from src.clustering.config import ClusteringConfig
from src.clustering.core.factory import ClustererFactory
from src.clustering.utils.logging import setup_logger
from src.clustering.utils.exceptions import ClusteringError
from src.clustering.utils.helpers import normalize_major
from src.clustering.utils.visualization import create_visualization_report

def load_data(file_path: str) -> List[str]:
    df = pd.read_excel(file_path)
    majors = [m for m in df['major_1'].dropna().unique() if isinstance(m, str)]
    return [normalize_major(m) for m in majors]

def save_results(
    clusters: Dict[int, List[str]],
    output_file: str,
    middle_mapping: Optional[Dict[int, int]] = None
) -> None:
    rows = []
    for label, items in clusters.items():
        for item in items:
            row = {'cluster_id': label, 'text': item}
            if middle_mapping is not None and label in middle_mapping:
                row['middle_cluster_id'] = middle_mapping[label]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_excel(output_file, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--algorithm', choices=['dbscan', 'progressive'], default='progressive')
    parser.add_argument('--log_file', default='clustering.log')
    parser.add_argument('--model_name', default='intfloat/multilingual-e5-large-instruct')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--viz_dir', default='visualizations')
    args = parser.parse_args()
    
    logger = setup_logger('clustering', log_file=args.log_file)
    try:
        if Path(args.config).exists():
            config = ClusteringConfig.from_yaml(args.config)
        else:
            config = ClusteringConfig()
            config.to_yaml(args.config)
        
        items = load_data(args.input_file)
        
        clusterer = ClustererFactory.create_clusterer(
            algorithm=args.algorithm,
            config=config,
            model_name=args.model_name
        )
        
        clusters = clusterer.fit(items)
        metrics = clusterer.evaluate_results(clusters)
        
        try:
            middle_mapping = clusterer.build_middle_clusters(clusters)
        except (AttributeError, ValueError, TypeError):
            middle_mapping = None
        
        save_results(clusters, args.output_file, middle_mapping)
        
        if args.visualize:
            create_visualization_report(
                clusterer.embeddings,
                clusters,
                items,
                args.viz_dir
            )
    except ClusteringError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(str(e))
        raise

if __name__ == '__main__':
    main() 