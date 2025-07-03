import json
from pathlib import Path
import heapq
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict

# Пути к папкам
CITED_BY_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cited_by"
CITES_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cites"


def build_citation_network():
    """Строит полную сеть цитирований"""
    citation_network = defaultdict(set)

    # Обрабатываем все файлы в обеих папках
    for folder in [CITES_DIR, CITED_BY_DIR]:
        for file in Path(folder).glob("*.json"):
            source_id = file.stem.lstrip('W')
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Добавляем связи (значения из JSON)
                    for target_id in data.values():
                        citation_network[source_id].add(str(target_id))
                except json.JSONDecodeError:
                    continue
    return citation_network


def get_top_and_related_articles(citation_network, top_n=100):
    """Возвращает топ-N статей и все связанные статьи"""
    # Считаем общее количество цитирований
    citation_counts = defaultdict(int)
    for paper, cited in citation_network.items():
        citation_counts[paper] += len(cited)
        for cited_paper in cited:
            citation_counts[cited_paper] += 1

    # Выбираем топ-N статей
    top_articles = heapq.nlargest(top_n, citation_counts.items(), key=lambda x: x[1])
    top_ids = {paper_id for paper_id, _ in top_articles}

    # Находим все связанные статьи
    related_articles = set()
    for paper_id in top_ids:
        related_articles.update(citation_network.get(paper_id, set()))
        # Находим статьи, которые цитируют текущую топ-статью
        for citing_paper, cited in citation_network.items():
            if paper_id in cited:
                related_articles.add(citing_paper)

    return top_ids.union(related_articles), citation_counts


def get_articles_metadata(conn, article_ids):
    """Получает метаданные из БД"""
    with conn.cursor() as cursor:
        # Преобразуем ID в bigint
        numeric_ids = [int(pid) for pid in article_ids if pid.isdigit()]

        cursor.execute("""
            SELECT 
                p.id,
                p.name as title,
                p.date_year as year,
                array_agg(a.name) as authors,
                COUNT(a.id) as author_count
            FROM publication p
            LEFT JOIN publication_author pa ON p.id = pa.publication_id
            LEFT JOIN author a ON pa.author_id = a.id
            WHERE p.id = ANY(%s::bigint[])
            GROUP BY p.id
        """, (numeric_ids,))

        return {
            str(row[0]): {
                "title": row[1],
                "year": row[2] if row[2] is not None else 0,
                "authors": row[3],
                "author_count": row[4]
            }
            for row in cursor.fetchall()
        }


def visualize_clusters(df, centroids, n_clusters):
    """Создает профессиональные графики кластеризации"""
    plt.figure(figsize=(18, 6))
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", n_colors=n_clusters)

    # График 1: Кластеры с центроидами
    plt.subplot(1, 3, 1)
    sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='cluster',
        palette=palette,
        size='citation_count',
        sizes=(30, 300),
        alpha=0.7
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='X', s=200, c='red', label='Centroids'
    )
    plt.title(f'Тематические кластеры (k={n_clusters})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # График 2: Цитирования (логарифмическая шкала)
    plt.subplot(1, 3, 2)
    sc = plt.scatter(
        df['x'], df['y'],
        c=np.log1p(df['citation_count']),
        cmap='viridis',
        s=100,
        alpha=0.7
    )
    plt.colorbar(sc, label='log(Цитирования + 1)')
    plt.title('Интенсивность цитирования')

    # График 3: Годы публикации (ограниченный диапазон)
    plt.subplot(1, 3, 3)
    sc = plt.scatter(
        df['x'], df['y'],
        c=df['year'],
        cmap='plasma',
        s=100,
        alpha=0.7,
        vmin=1980,  # Минимальное значение шкалы
        vmax=2025  # Максимальное значение шкалы
    )
    cbar = plt.colorbar(sc, label='Год публикации')
    cbar.set_ticks(range(1980, 2026, 5))  # Метки каждые 5 лет
    plt.title('Распределение по годам (1980-2025)')

    plt.tight_layout()
    plt.savefig('cluster_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # 1. Построение сети цитирований
    print("Построение сети цитирований...")
    citation_network = build_citation_network()

    # 2. Получение топ и связанных статей
    print("Выбор топ-100 и связанных статей...")
    all_article_ids, citation_counts = get_top_and_related_articles(citation_network)
    print(f"Всего статей для анализа: {len(all_article_ids)}")

    # 3. Получение метаданных
    print("Загрузка метаданных...")
    conn = psycopg2.connect(
        dbname="scopus",
        user="postgres",
        password="123456789",
        host="localhost",
        port="5432"
    )
    articles_metadata = get_articles_metadata(conn, all_article_ids)
    conn.close()

    # 4. Обогащение данных
    for article_id, meta in articles_metadata.items():
        meta['citation_count'] = citation_counts.get(article_id, 0)

    # 5. Векторизация
    print("Создание эмбеддингов...")
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    texts = [f"{meta['title']} {' '.join(meta['authors'])}" for meta in articles_metadata.values()]
    embeddings = model.encode(texts)

    # 6. Кластеризация
    print("Кластеризация...")
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Автоматический подбор числа кластеров
    silhouette_scores = []
    cluster_range = range(2, 8)
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    optimal_clusters = cluster_range[np.argmax(silhouette_scores)]
    print(f"Оптимальное число кластеров: {optimal_clusters}")

    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # 7. Подготовка данных для визуализации
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': clusters,
        'title': [meta['title'] for meta in articles_metadata.values()],
        'citation_count': [meta['citation_count'] for meta in articles_metadata.values()],
        'year': [meta['year'] for meta in articles_metadata.values()],
        'author_count': [meta['author_count'] for meta in articles_metadata.values()]
    })

    # 8. Визуализация
    print("Визуализация результатов...")
    visualize_clusters(df, kmeans.cluster_centers_[:, :2], optimal_clusters)

    # 9. Дополнительная информация
    print("\nСтатистика по кластерам:")
    print(df.groupby('cluster').agg({
        'citation_count': ['mean', 'median', 'count'],
        'year': ['mean', 'median'],
        'author_count': ['mean', 'median']
    }).round(2))