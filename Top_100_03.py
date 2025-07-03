import json
from pathlib import Path
import heapq
import psycopg2
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import ConvexHull
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
from kneed import KneeLocator

# Конфигурация
CITED_BY_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cited_by"
CITES_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cites"
TOP_N = 100


def build_citation_network():
    """Строит полную сеть цитирований"""
    citation_network = defaultdict(set)

    for folder in [CITES_DIR, CITED_BY_DIR]:
        for file in Path(folder).glob("*.json"):
            source_id = file.stem.lstrip('W')
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Добавляем целевые ID (значения из JSON)
                    for target_id in data.values():
                        citation_network[source_id].add(str(target_id))
                except json.JSONDecodeError:
                    continue
    return citation_network


def get_top_and_related_articles(citation_network, top_n=TOP_N):
    """Возвращает топ-N статей и все связанные статьи"""
    citation_counts = defaultdict(int)
    for paper, cited in citation_network.items():
        citation_counts[paper] += len(cited)
        for cited_paper in cited:
            citation_counts[cited_paper] += 1

    top_articles = heapq.nlargest(top_n, citation_counts.items(), key=lambda x: x[1])
    top_ids = {paper_id for paper_id, _ in top_articles}

    related_articles = set()
    for paper_id in top_ids:
        related_articles.update(citation_network.get(paper_id, set()))
        for citing_paper, cited in citation_network.items():
            if paper_id in cited:
                related_articles.add(citing_paper)

    return top_ids.union(related_articles), citation_counts


def get_articles_metadata(conn, article_ids):
    """Получает метаданные из БД"""
    numeric_ids = [int(pid) for pid in article_ids if pid.isdigit()]

    with conn.cursor() as cursor:
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
            WHERE p.id = ANY(%s)
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


def enhanced_cluster_analysis(embeddings, max_clusters=10):
    """Улучшенный анализ кластеров с тематическим разделением"""
    # 1. Метод силуэта для определения оптимального числа кластеров
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    optimal_k = np.argmax(silhouette_scores) + 2

    # 2. Уточненная кластеризация с увеличенным числом итераций
    kmeans = KMeans(
        n_clusters=optimal_k,
        init='k-means++',
        n_init=50,  # Увеличиваем количество инициализаций
        max_iter=500,  # Увеличиваем число итераций
        random_state=42
    )
    clusters = kmeans.fit_predict(embeddings)

    # 3. Проверка разделимости кластеров
    cluster_std = []
    for i in range(optimal_k):
        cluster_points = embeddings[clusters == i]
        cluster_std.append(np.mean(np.std(cluster_points, axis=0)))

    print("\nАнализ качества кластеризации:")
    print(f"Среднее стандартное отклонение по кластерам: {np.mean(cluster_std):.2f}")
    print(f"Силуэтный коэффициент: {silhouette_score(embeddings, clusters):.2f}")

    return kmeans, optimal_k

# Исправленная функция visualize_thematic_clusters
def visualize_thematic_clusters(df, centroids, n_clusters, articles_metadata):
    """Визуализация с четким тематическим разделением"""
    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")

    # 1. Основной график кластеров
    plt.subplot(1, 2, 1)
    palette = sns.color_palette("husl", n_colors=n_clusters)

    sc = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='cluster',
        palette=palette,
        s=100,
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )

    # Рисуем границы кластеров
    for i in range(n_clusters):
        cluster_points = df[df['cluster'] == i][['x', 'y']].values
        if len(cluster_points) > 2:  # Нужно минимум 3 точки для ConvexHull
            try:
                hull = ConvexHull(cluster_points)
                for simplex in hull.simplices:
                    plt.plot(cluster_points[simplex, 0], cluster_points[simplex, 1],
                             color=palette[i], linestyle='--', alpha=0.3)
            except:
                continue

    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='X', s=300, c='red', label='Центроиды'
    )

    # Добавляем подписи для самых характерных статей
    for i in range(n_clusters):
        cluster_df = df[df['cluster'] == i]
        if not cluster_df.empty:
            centroid = centroids[i]
            distances = np.linalg.norm(cluster_df[['x', 'y']].values - centroid, axis=1)
            idx = cluster_df.index[np.argmin(distances)]
            plt.text(
                df.loc[idx, 'x'] + 0.02,
                df.loc[idx, 'y'] + 0.02,
                f"Cluster {i}",
                fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
            )

    plt.title(f'Тематические кластеры (k={n_clusters})')
    plt.xlabel('Компонента PCA 1')
    plt.ylabel('Компонента PCA 2')
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. Тематический анализ (топ-слов по кластерам)
    plt.subplot(1, 2, 2)
    vectorizer = TfidfVectorizer(max_features=5, stop_words='english')

    cluster_texts = []
    for i in range(n_clusters):
        cluster_titles = [articles_metadata[aid]['title']
                          for aid in df[df['cluster'] == i]['article_id']]
        cluster_texts.append(" ".join(cluster_titles))

    X = vectorizer.fit_transform(cluster_texts)
    top_words = []
    for i in range(n_clusters):
        feature_array = np.array(vectorizer.get_feature_names_out())
        tfidf_sorting = np.argsort(X[i].toarray()).flatten()[::-1]
        top_words.append(", ".join(feature_array[tfidf_sorting][:3]))

    plt.barh(range(n_clusters), [len(df[df['cluster'] == i]) for i in range(n_clusters)],
             color=palette, alpha=0.6)

    for i, (count, words) in enumerate(zip(
            [len(df[df['cluster'] == i]) for i in range(n_clusters)],
            top_words
    )):
        plt.text(5, i, words, va='center', fontsize=10)

    plt.yticks(range(n_clusters), [f"Кластер {i}" for i in range(n_clusters)])
    plt.xlabel('Количество статей')
    plt.title('Тематические ключевые слова по кластерам')
    plt.tight_layout()

    plt.savefig('thematic_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_clusters(embeddings, max_clusters=10):
    """Анализ оптимального числа кластеров"""
    # Метод локтя
    distortions = []
    K_range = range(1, max_clusters + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    kneedle = KneeLocator(K_range, distortions, curve='convex', direction='decreasing')
    elbow_k = kneedle.elbow

    # Метод силуэта
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    optimal_k = np.argmax(silhouette_scores) + 2

    # Визуализация
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Инерция')
    plt.title('Метод локтя')
    if elbow_k:
        plt.axvline(x=elbow_k, linestyle='--', color='r', label=f'Локоть при k={elbow_k}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'bx-')
    plt.xlabel('Число кластеров (k)')
    plt.ylabel('Силуэтный коэффициент')
    plt.title('Метод силуэта')
    plt.axvline(x=optimal_k, linestyle='--', color='r', label=f'Оптимальное k={optimal_k}')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cluster_selection.png', dpi=300)
    plt.show()

    # Добавить проверку качества кластеризации
    if silhouette_scores[optimal_k - 2] < 0.5:
        print("Внимание: низкое качество кластеризации (силуэт < 0.5). Рекомендуется:")
        print("- Увеличить размер выборки")
        print("- Проверить качество эмбеддингов")
        print("- Использовать другую модель кластеризации (например, DBSCAN)")

    return optimal_k


def visualize_results(df, centroids, n_clusters):
    """Финальная визуализация"""
    plt.figure(figsize=(18, 6))

    # 1. Кластеры
    plt.subplot(1, 3, 1)
    sc = plt.scatter(
        df['x'], df['y'],
        c=df['cluster'],
        cmap='tab20',
        s=50,
        alpha=0.7
    )
    plt.scatter(
        centroids[:, 0], centroids[:, 1],
        marker='X', s=200, c='red', label='Центроиды'
    )
    plt.title(f'Кластеризация (k={n_clusters})')
    plt.colorbar(sc, label='Кластер')

    # 2. Цитирования
    plt.subplot(1, 3, 2)
    log_cites = np.log1p(df['citation_count'])
    sc = plt.scatter(
        df['x'], df['y'],
        c=log_cites,
        cmap='viridis',
        s=50,
        alpha=0.7
    )
    cbar = plt.colorbar(sc, label='log(Цитирования + 1)')
    original_values = [0, 1, 10, 100, 1000]
    cbar.set_ticks(np.log1p(original_values))
    cbar.set_ticklabels(original_values)
    plt.title('Интенсивность цитирований')

    # 3. Годы публикации (только если есть данные)
    if df['year'].notnull().any():
        plt.subplot(1, 3, 3)
        sc = plt.scatter(
            df['x'], df['y'],
            c=df['year'],
            cmap='plasma',
            s=50,
            alpha=0.7,
            vmin=1980,
            vmax=2025
        )
        cbar = plt.colorbar(sc, label='Год публикации')
        cbar.set_ticks(range(1980, 2026, 5))
        plt.title('Распределение по годам')
    else:
        print("Нет данных по годам для визуализации")

    plt.tight_layout()
    plt.savefig('final_analysis.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    print("1. Построение сети цитирований...")
    citation_network = build_citation_network()

    print("2. Выбор топ-100 и связанных статей...")
    all_article_ids, citation_counts = get_top_and_related_articles(citation_network)
    print(f"Всего статей для анализа: {len(all_article_ids)}")

    print("3. Загрузка метаданных из БД...")
    try:
        conn = psycopg2.connect(
            dbname="scopus",
            user="postgres",
            password="123456789",
            host="localhost",
            port="5432"
        )
        articles_metadata = get_articles_metadata(conn, all_article_ids)
    except Exception as e:
        print(f"Ошибка при работе с БД: {e}")
        raise
    finally:
        if conn:
            conn.close()

    print("4. Обогащение данных...")
    for article_id, meta in articles_metadata.items():
        meta['citation_count'] = citation_counts.get(article_id, 0)

    print("5. Создание эмбеддингов...")
    model = SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
    texts = [f"{meta['title']} {' '.join(meta['authors'])}" for meta in articles_metadata.values()]
    embeddings = model.encode(texts)

    print("6. Анализ распределения цитирований...")
    citation_counts = [meta['citation_count'] for meta in articles_metadata.values()]
    plt.figure(figsize=(10, 5))
    plt.hist(np.log1p(citation_counts), bins=50)
    plt.xlabel('log(Цитирования + 1)')
    plt.ylabel('Частота')
    plt.title('Распределение цитирований (логарифмическая шкала)')
    plt.savefig('citation_distribution.png', dpi=300)
    plt.show()

    print("7. Определение оптимального числа кластеров...")
    optimal_k = analyze_clusters(embeddings)
    print(f"Выбрано число кластеров: {optimal_k}")

    print("8. Улучшенная кластеризация...")
    from scipy.spatial import ConvexHull

    # 1. Сначала создаем PCA-компоненты
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)  # Создаем embeddings_2d здесь

    # 2. Затем проводим кластеризацию
    kmeans, optimal_k = enhanced_cluster_analysis(embeddings)
    clusters = kmeans.labels_

    print("9. Подготовка данных для визуализации...")
    # Теперь embeddings_2d доступен для использования
    df = pd.DataFrame({
        'article_id': list(articles_metadata.keys()),
        'x': embeddings_2d[:, 0],  # Теперь правильно
        'y': embeddings_2d[:, 1],
        'cluster': clusters,
        'title': [meta['title'] for meta in articles_metadata.values()],
        'citation_count': [meta['citation_count'] for meta in articles_metadata.values()],
        'year': [meta['year'] for meta in articles_metadata.values()]
    })

    print("10. Визуализация тематических кластеров...")
    visualize_thematic_clusters(df, kmeans.cluster_centers_[:, :2], optimal_k, articles_metadata)

    print("11. Сохранение результатов кластеризации...")
    df.to_csv('clustered_articles.csv', index=False)