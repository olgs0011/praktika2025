import json
from pathlib import Path
import heapq
import psycopg2
from sklearn.manifold import TSNE
from matplotlib.patches import Ellipse
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
from collections import defaultdict, Counter
from kneed import KneeLocator

# Конфигурация
CITED_BY_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cited_by"
CITES_DIR = r"C:\Users\User\OneDrive\Рабочий стол\metadata\metadata\f_cites"
TOP_N = 100


def build_citation_network():
    """Строит сеть цитирований"""
    citation_network = defaultdict(set)
    for folder in [CITES_DIR, CITED_BY_DIR]:
        for file in Path(folder).glob("*.json"):
            source_id = file.stem.lstrip('W')
            with open(file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for target_id in data.values():
                        citation_network[source_id].add(str(target_id))
                except json.JSONDecodeError:
                    continue
    return citation_network


def get_top_and_related_articles(citation_network, top_n=TOP_N):
    """Возвращает топ-N статей и связанные статьи"""
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


def load_asjc_categories(conn):
    """Загружает ASJC классификацию из БД"""
    with conn.cursor() as cursor:
        cursor.execute("SELECT code, field, subject_area FROM asjc")
        return {
            row[0]: {
                "field": row[1],
                "subject_area": row[2]
            }
            for row in cursor.fetchall()
        }


def get_articles_metadata(conn, article_ids, asjc_categories):
    """Получает метаданные с ASJC-классификацией"""
    numeric_ids = [int(pid) for pid in article_ids if pid.isdigit()]

    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT 
                p.id,
                p.name as title,
                p.date_year as year,
                p.citations_num,
                array_agg(a.code) as asjc_codes
            FROM publication p
            LEFT JOIN publication_asjc pa ON p.id = pa.publication_id
            LEFT JOIN asjc a ON pa.asjc_code = a.code
            WHERE p.id = ANY(%s)
            GROUP BY p.id
        """, (numeric_ids,))

        metadata = {}
        for row in cursor.fetchall():
            asjc_info = []
            for code in (row[4] if row[4] else []):
                if code in asjc_categories:
                    asjc_info.append({
                        "code": code,
                        "field": asjc_categories[code]["field"],
                        "subject_area": asjc_categories[code]["subject_area"]
                    })

            metadata[str(row[0])] = {
                "title": row[1],
                "year": row[2] if row[2] else None,
                "citation_count": row[3] if row[3] else 0,
                "asjc": asjc_info
            }
        return metadata


def enhanced_cluster_visualization(df, n_clusters, articles_metadata):
    plt.figure(figsize=(14, 10))

    # Создаем палитру цветов
    palette = sns.color_palette("husl", n_clusters)

    # Улучшенный scatter plot
    scatter = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='cluster',
        palette=palette,
        s=100,
        alpha=0.7,
        edgecolor='w',
        linewidth=0.5
    )

    # Добавляем эллипсы для каждого кластера
    for cluster_id in range(n_clusters):
        cluster_data = df[df['cluster'] == cluster_id]
        if len(cluster_data) > 1:
            # Рассчитываем ковариационную матрицу
            cov = np.cov(cluster_data[['x', 'y']].T)

            try:
                # Вычисляем собственные значения и векторы
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)

                # Получаем центр кластера как numpy array
                center = cluster_data[['x', 'y']].mean().values  # Исправление здесь - добавлен .values

                # Создаем эллипс
                ell = Ellipse(
                    xy=(center[0], center[1]),  # Явное преобразование в кортеж
                    width=lambda_[0] * 3,  # Увеличиваем размер для наглядности
                    height=lambda_[1] * 3,
                    angle=np.degrees(np.arctan2(v[1, 0], v[0, 0])),
                    fill=False,
                    linestyle='--',
                    linewidth=1.5,
                    color=palette[cluster_id]
                )
                plt.gca().add_patch(ell)
            except np.linalg.LinAlgError:
                continue

    # Добавляем подписи кластеров
    cluster_themes = get_cluster_themes(df, articles_metadata)
    for cluster_id, theme in cluster_themes.items():
        centroid = df[df['cluster'] == cluster_id][['x', 'y']].median().values  # Исправление здесь
        plt.text(
            centroid[0],  # Доступ по индексу к numpy array
            centroid[1],
            f"Cluster {cluster_id}\n{theme}",
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'),
            fontsize=9
        )

    plt.title("Улучшенная визуализация кластеров с эллипсами", pad=20)
    plt.xlabel("Компонента 1 (t-SNE)")
    plt.ylabel("Компонента 2 (t-SNE)")
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('improved_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_cluster_themes(df, articles_metadata):
    """Определяет тематики кластеров по ASJC"""
    cluster_themes = {}
    for cluster_id in df['cluster'].unique():
        counter = Counter()
        for aid in df[df['cluster'] == cluster_id]['article_id']:
            for category in articles_metadata[aid]['asjc']:
                counter[category['field']] += 1

        if counter:
            top_theme = counter.most_common(1)[0]
            cluster_themes[cluster_id] = f"{top_theme[0]}\n({top_theme[1]} ст.)"

    return cluster_themes


def visualize_year_distribution(articles_metadata):
    """Гистограмма распределения по годам"""
    years = [meta['year'] for meta in articles_metadata.values() if meta['year']]
    plt.figure(figsize=(12, 6))
    plt.hist(years, bins=range(min(years), max(years) + 1, 5), edgecolor='black')
    plt.xlabel('Год публикации')
    plt.ylabel('Количество статей')
    plt.title('Распределение публикаций по годам')
    plt.grid(True)
    plt.savefig('year_distribution.png', dpi=300)
    plt.show()


def visualize_citation_distribution(articles_metadata):
    """Гистограмма цитирований"""
    citations = [meta['citation_count'] for meta in articles_metadata.values()]
    plt.figure(figsize=(12, 6))
    plt.hist(np.log1p(citations), bins=50, edgecolor='black')
    plt.xlabel('log(Цитирования + 1)')
    plt.ylabel('Количество статей')
    plt.title('Распределение цитирований')
    plt.grid(True)
    plt.savefig('citation_distribution.png', dpi=300)
    plt.show()


def visualize_cluster_selection(embeddings):
    """Графики выбора числа кластеров"""
    # Метод локтя
    distortions = []
    K_range = range(1, 16)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    kneedle = KneeLocator(K_range, distortions, curve='convex', direction='decreasing')
    elbow_k = kneedle.elbow

    # Метод силуэта
    silhouette_scores = []
    for k in range(2, 16):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Инерция')
    plt.title('Метод локтя')
    if elbow_k:
        plt.axvline(x=elbow_k, linestyle='--', color='r', label=f'k={elbow_k}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(2, 16), silhouette_scores, 'bx-')
    plt.xlabel('Число кластеров')
    plt.ylabel('Силуэтный коэффициент')
    plt.title('Метод силуэта')
    plt.axvline(x=np.argmax(silhouette_scores) + 2, linestyle='--', color='r')
    plt.tight_layout()
    plt.savefig('cluster_selection.png', dpi=300)
    plt.show()


def visualize_thematic_clusters(df, centroids, n_clusters, articles_metadata):
    """Визуализация тематических кластеров"""
    plt.figure(figsize=(14, 8))
    palette = sns.color_palette("husl", n_clusters)
    cluster_themes = get_cluster_themes(df, articles_metadata)

    # Основной график кластеров
    sns.scatterplot(
        data=df, x='x', y='y',
        hue='cluster', palette=palette,
        s=70, alpha=0.8
    )

    # Подписи кластеров
    for cluster_id, theme in cluster_themes.items():
        plt.text(
            centroids[cluster_id, 0],
            centroids[cluster_id, 1],
            theme,
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
            fontsize=9
        )

    plt.title(f'Тематические кластеры (k={n_clusters})')
    plt.xlabel('Компонента PCA 1')
    plt.ylabel('Компонента PCA 2')
    plt.legend(title='Кластер', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.savefig('thematic_clusters.png', dpi=300)
    plt.show()


def analyze_clusters(embeddings, max_clusters=15):
    """Определяет оптимальное число кластеров"""
    # Метод локтя
    distortions = []
    K_range = range(1, max_clusters + 1)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(embeddings)
        distortions.append(kmeans.inertia_)

    # Метод силуэта
    silhouette_scores = []
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette_scores.append(silhouette_score(embeddings, labels))

    # Визуализация
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K_range, distortions, 'bx-')
    plt.title('Метод локтя')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, 'rx-')
    plt.title('Метод силуэта')
    plt.tight_layout()
    plt.savefig('cluster_selection.png', dpi=300)
    plt.show()

    return np.argmax(silhouette_scores) + 2  # Оптимальное k


if __name__ == "__main__":
    print("1. Построение сети цитирований...")
    citation_network = build_citation_network()

    print("2. Выбор топ-статей...")
    article_ids, citation_counts = get_top_and_related_articles(citation_network)
    print(f"Всего статей для анализа: {len(article_ids)}")

    print("3. Загрузка метаданных...")
    try:
        conn = psycopg2.connect(
            dbname="scopus",
            user="postgres",
            password="123456789",
            host="localhost",
            port="5432"
        )
        asjc_categories = load_asjc_categories(conn)
        articles_metadata = get_articles_metadata(conn, article_ids, asjc_categories)
    except Exception as e:
        print(f"Ошибка при работе с БД: {e}")
        raise
    finally:
        if conn:
            conn.close()

    print("4. Создание эмбеддингов...")
    model = SentenceTransformer(
        "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code=True
    )
    texts = [
        meta['title'] + (" " + meta['abstract'] if meta.get('abstract') else "")
        for meta in articles_metadata.values()
    ]
    embeddings = model.encode(texts)

    print("5. Анализ распределения...")
    visualize_year_distribution(articles_metadata)
    visualize_citation_distribution(articles_metadata)

    print("6. Определение числа кластеров...")
    optimal_k = analyze_clusters(embeddings)  # Определяем оптимальное k
    visualize_cluster_selection(embeddings)  # Визуализируем выбор k

    print("7. Кластеризация...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    print("8. Подготовка данных для визуализации...")
    # Заменяем PCA на TSNE (лучше разделяет кластеры)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        'article_id': list(articles_metadata.keys()),
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': clusters,
        'citation_count': [meta['citation_count'] for meta in articles_metadata.values()],
        'year': [meta['year'] for meta in articles_metadata.values()]
    })

    print("9. Визуализация кластеров...")
    enhanced_cluster_visualization(df, optimal_k, articles_metadata)


    print("10. Сохранение данных...")
    df.to_csv('clustered_articles.csv', index=False)
    print("Анализ завершен! Все графики сохранены.")
