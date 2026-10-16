"""Wrapper for backward compatibility with k_means.py"""
from k_means import train_kmeans_model, predict_cluster, run_kmeans_analysis

if __name__ == "__main__":
    run_kmeans_analysis()