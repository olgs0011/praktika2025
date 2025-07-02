import json
from pathlib import Path
import heapq
import psycopg2
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os

# Пути к папкам (замените на свои)
CITED_BY_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cited_by"
CITES_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cites"


# 1. Функция для сбора топ-100 статей
def get_top_articles(cited_by_dir=CITED_BY_DIR, cites_dir=CITES_DIR, top_n=100):
    """Находит топ-N статей по сумме цитирующих и цитируемых статей"""
    citation_scores = {}

    # Считаем общее количество связей (цитирующие + цитируемые)
    for folder in [cited_by_dir, cites_dir]:
        for file in Path(folder).glob("*.json"):
            with open(file, "r") as f:
                data = json.load(f)
            # ID статьи - это название файла (без .json)
            article_id = file.stem.lstrip('W')  # Удаляем 'W' если есть
            # Суммируем все связи
            citation_scores[article_id] = citation_scores.get(article_id, 0) + len(data)

    # Выбираем топ-N статей
    top_articles = heapq.nlargest(top_n, citation_scores.items(), key=lambda x: x[1])
    return [article_id for article_id, _ in top_articles]


# 2. Функция для получения метаданных из БД
def get_articles_metadata(conn, article_ids):
    """Получает названия и авторов статей из БД Scopus"""
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.id,
                p.name as title,
                array_agg(a.name) as authors
            FROM publication p
            LEFT JOIN publication_author pa ON p.id = pa.publication_id
            LEFT JOIN author a ON pa.author_id = a.id
            WHERE p.id = ANY(%s::bigint[])
            GROUP BY p.id
        """, (article_ids,))

        return {row[0]: {"title": row[1], "authors": row[2]} for row in cursor.fetchall()}


# 3. Основной пайплайн
if __name__ == "__main__":
    # 1. Загрузка модели
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    # 2. Получаем топ-100 статей
    top_100_ids = get_top_articles()
    print(f"Найдено {len(top_100_ids)} топ-статей")

    # 3. Получаем метаданные из БД
    conn = psycopg2.connect(
        dbname="scopus",
        user="postgres",
        password="123456789",
        host="localhost",
        port="5432"
    )
    articles_metadata = get_articles_metadata(conn, top_100_ids)
    conn.close()

    # 4. Векторизация
    texts = [
        f"{meta['title']} {' '.join(meta['authors'])}"
        for meta in articles_metadata.values()
    ]
    embeddings = model.encode(texts)

    # 5. Кластеризация
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(embeddings)

    # 6. Расширенная визуализация
    plt.figure(figsize=(18, 5))

    # График 1: Кластеризация статей (как ранее)
    plt.subplot(1, 3, 1)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', s=50, alpha=0.7)
    plt.colorbar(scatter, label='Кластер')
    plt.title('1. Кластеризация по тематикам')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')

    # График 2: Распределение цитирований
    plt.subplot(1, 3, 2)
    citation_counts = [len(meta.get('cited_by', [])) for meta in articles_metadata.values()]
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=citation_counts, cmap='plasma', s=50)
    plt.colorbar(label='Число цитирований')
    plt.title('2. Влияние статей (по цитированиям)')
    plt.xlabel('PCA Component 1')

    # График 3: Распределение авторов
    plt.subplot(1, 3, 3)
    author_counts = [len(meta['authors']) for meta in articles_metadata.values()]
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=author_counts, cmap='cool', s=50)
    plt.colorbar(label='Число авторов')
    plt.title('3. Коллаборации (по числу авторов)')
    plt.xlabel('PCA Component 1')

    plt.tight_layout()
    plt.show()