import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage

# -------------------------------------------------------
# 🟣 App Config
# -------------------------------------------------------
st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")

st.title("🟣 News Topic Discovery Dashboard")
st.markdown("""
This system uses **Hierarchical Clustering** to automatically group similar news articles 
based on textual similarity.
""")

# -------------------------------------------------------
# 📂 Sidebar Controls
# -------------------------------------------------------

st.sidebar.header("📂 Dataset Upload")

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.sidebar.success("Dataset Loaded Successfully")

    # Auto-detect text column
    text_columns = df.select_dtypes(include=['object']).columns.tolist()
    text_column = st.sidebar.selectbox("Select Text Column", text_columns)

    texts = df[text_column].astype(str)

    # -------------------------------------------------------
    # 📝 Text Vectorization Controls
    # -------------------------------------------------------

    st.sidebar.header("📝 TF-IDF Settings")

    max_features = st.sidebar.slider(
        "Maximum TF-IDF Features",
        100, 2000, 1000
    )

    remove_stopwords = st.sidebar.checkbox("Use English Stopwords", value=True)

    ngram_option = st.sidebar.selectbox(
        "N-gram Range",
        ["Unigrams", "Bigrams", "Unigrams + Bigrams"]
    )

    if ngram_option == "Unigrams":
        ngram_range = (1, 1)
    elif ngram_option == "Bigrams":
        ngram_range = (2, 2)
    else:
        ngram_range = (1, 2)

    # -------------------------------------------------------
    # 🌳 Clustering Controls
    # -------------------------------------------------------

    st.sidebar.header("🌳 Hierarchical Clustering Settings")

    linkage_method = st.sidebar.selectbox(
        "Linkage Method",
        ["ward", "complete", "average", "single"]
    )

    distance_metric = "euclidean"

    dendro_articles = st.sidebar.slider(
        "Number of Articles for Dendrogram",
        20, 200, 50
    )

    # -------------------------------------------------------
    # Vectorization
    # -------------------------------------------------------

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english" if remove_stopwords else None,
        ngram_range=ngram_range
    )

    X = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # -------------------------------------------------------
    # 🟦 Generate Dendrogram Button
    # -------------------------------------------------------

    if st.button("🟦 Generate Dendrogram"):

        st.subheader("🌳 Dendrogram")

        subset_size = min(dendro_articles, X.shape[0])
        X_subset = X[:subset_size].toarray()

        Z = linkage(X_subset, method=linkage_method)

        fig, ax = plt.subplots(figsize=(10, 5))
        dendrogram(Z, ax=ax)
        ax.set_title("Hierarchical Clustering Dendrogram")
        ax.set_xlabel("Article Index")
        ax.set_ylabel("Distance")
        st.pyplot(fig)

        st.info("""
        Large vertical gaps indicate strong separation between article groups.
        Choose number of clusters based on natural breaks in the tree.
        """)

    # -------------------------------------------------------
    # 🟩 Apply Clustering
    # -------------------------------------------------------

    st.subheader("🟩 Apply Clustering")

    n_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

    if st.button("Apply Clustering"):

        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )

        labels = model.fit_predict(X.toarray())
        df["Cluster"] = labels

        # -------------------------------------------------------
        # 📊 PCA Visualization
        # -------------------------------------------------------

        st.subheader("📊 2D Cluster Visualization (PCA)")

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X.toarray())

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels
        )
        ax2.set_title("Cluster Projection (PCA)")
        st.pyplot(fig2)

        # -------------------------------------------------------
        # 📊 Silhouette Score
        # -------------------------------------------------------

        score = silhouette_score(X, labels)

        st.subheader("📊 Silhouette Score")
        st.write("Score:", round(score, 4))

        if score > 0.5:
            st.success("Well-separated clusters")
        elif score > 0.2:
            st.warning("Moderate structure detected")
        elif score >= 0:
            st.info("Clusters overlap somewhat")
        else:
            st.error("Poor clustering detected")

        # -------------------------------------------------------
        # 📋 Cluster Summary
        # -------------------------------------------------------

        st.subheader("📋 Cluster Summary")

        summary_data = []

        for cluster_id in range(n_clusters):

            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)

            cluster_tfidf = X[cluster_indices].mean(axis=0)
            top_indices = np.argsort(cluster_tfidf.A1)[-10:]
            top_keywords = [feature_names[i] for i in top_indices]

            sample_text = texts.iloc[cluster_indices[0]][:150]

            summary_data.append([
                cluster_id,
                cluster_size,
                ", ".join(top_keywords),
                sample_text
            ])

        summary_df = pd.DataFrame(
            summary_data,
            columns=[
                "Cluster ID",
                "Number of Articles",
                "Top Keywords",
                "Sample Article Snippet"
            ]
        )

        st.dataframe(summary_df)

        # -------------------------------------------------------
        # 🧠 Business Interpretation Section
        # -------------------------------------------------------

        st.subheader("🧠 Business Interpretation")

        for row in summary_data:
            st.markdown(f"""
            🔵 **Cluster {row[0]}**
            - Contains {row[1]} articles  
            - Common themes include: {row[2]}  
            - Representative content: "{row[3]}..."
            """)

        # -------------------------------------------------------
        # 💡 Insight Box
        # -------------------------------------------------------

        st.info("""
        Articles grouped in the same cluster share similar vocabulary and themes.
        These clusters can be used for automatic tagging, recommendations,
        and content organization.
        """)
