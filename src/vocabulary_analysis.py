"""
Omar - Vocabulary Analysis (Section 1.2 Requirement)
Builds variety vocabularies, computes Jaccard & TF-IDF cosine similarities,
and generates a heatmap of cross-variety similarity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import warnings
warnings.filterwarnings('ignore')


class VocabularyAnalysis:
    def __init__(self, df_all: pd.DataFrame, text_col: str = 'text',
                 variety_col: str = 'variety', save_path: str = './reports'):
        self.df = df_all
        self.text_col = text_col
        self.variety_col = variety_col
        self.save_path = save_path
        self.figures_path = os.path.join(save_path, 'figures')
        os.makedirs(self.figures_path, exist_ok=True)

        self.varieties = sorted(df_all[variety_col].unique())
        self.vocab_per_variety: dict[str, set] = {}

    # ------------------------------------------------------------------
    # 1. Build per-variety vocabularies
    # ------------------------------------------------------------------
    def build_vocabularies(self) -> dict[str, set]:
        """Build a set of unique lowercased tokens for each variety."""
        for variety in self.varieties:
            texts = self.df[self.df[self.variety_col] == variety][self.text_col]
            tokens = set(
                word.lower()
                for text in texts.dropna()
                for word in str(text).split()
            )
            self.vocab_per_variety[variety] = tokens

        print("Vocabulary sizes:")
        for v, vocab in self.vocab_per_variety.items():
            print(f"  {v}: {len(vocab):,} unique words")

        return self.vocab_per_variety

    # ------------------------------------------------------------------
    # 2. Jaccard similarity
    # ------------------------------------------------------------------
    @staticmethod
    def jaccard(set_a: set, set_b: set) -> float:
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    def compute_jaccard_matrix(self) -> pd.DataFrame:
        """Compute pairwise Jaccard similarity between all variety pairs."""
        if not self.vocab_per_variety:
            self.build_vocabularies()

        n = len(self.varieties)
        matrix = np.zeros((n, n))

        for i, v1 in enumerate(self.varieties):
            for j, v2 in enumerate(self.varieties):
                matrix[i, j] = self.jaccard(
                    self.vocab_per_variety[v1],
                    self.vocab_per_variety[v2]
                )

        df_jac = pd.DataFrame(matrix, index=self.varieties, columns=self.varieties)

        print("\nJaccard similarity matrix:")
        print(df_jac.round(4))

        # Pairwise readable output
        print("\nPairwise Jaccard scores:")
        pairs = [
            ('en-AU', 'en-IN'),
            ('en-AU', 'en-UK'),
            ('en-IN', 'en-UK'),
        ]
        for a, b in pairs:
            score = df_jac.loc[a, b]
            intersection = len(self.vocab_per_variety[a] & self.vocab_per_variety[b])
            print(f"  {a} ↔ {b}: {score:.4f}  (shared words: {intersection:,})")

        return df_jac

    # ------------------------------------------------------------------
    # 3. TF-IDF cosine similarity
    # ------------------------------------------------------------------
    def compute_tfidf_cosine_matrix(self) -> pd.DataFrame:
        """
        Concatenate all texts per variety into one document,
        fit TF-IDF, then compute cosine similarity between variety documents.
        """
        variety_docs = []
        for v in self.varieties:
            texts = self.df[self.df[self.variety_col] == v][self.text_col].dropna()
            variety_docs.append(' '.join(texts.astype(str).tolist()))

        vectorizer = TfidfVectorizer(
            max_features=10_000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        tfidf_matrix = vectorizer.fit_transform(variety_docs)
        cos_sim = cosine_similarity(tfidf_matrix)

        df_cos = pd.DataFrame(cos_sim, index=self.varieties, columns=self.varieties)

        print("\nTF-IDF cosine similarity matrix:")
        print(df_cos.round(4))

        print("\nPairwise cosine scores:")
        pairs = [('en-AU', 'en-IN'), ('en-AU', 'en-UK'), ('en-IN', 'en-UK')]
        for a, b in pairs:
            print(f"  {a} ↔ {b}: {df_cos.loc[a, b]:.4f}")

        return df_cos

    # ------------------------------------------------------------------
    # 4. Heatmap of both similarities
    # ------------------------------------------------------------------
    def plot_similarity_heatmap(self, df_jac: pd.DataFrame,
                                 df_cos: pd.DataFrame, save: bool = True):
        """Side-by-side heatmaps for Jaccard and TF-IDF cosine similarity."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        for ax, df_sim, title in zip(
            axes,
            [df_jac, df_cos],
            ['Jaccard similarity', 'TF-IDF cosine similarity']
        ):
            mask = np.eye(len(df_sim), dtype=bool)  # hide diagonal
            sns.heatmap(
                df_sim, annot=True, fmt='.3f', cmap='Blues',
                vmin=0, vmax=1, ax=ax,
                linewidths=0.5, linecolor='white',
                mask=mask, annot_kws={'size': 11}
            )
            # Show diagonal as 1.0 in grey
            for k in range(len(df_sim)):
                ax.text(k + 0.5, k + 0.5, '1.000',
                        ha='center', va='center', fontsize=11, color='grey')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Variety')
            ax.set_ylabel('Variety')

        plt.suptitle('Cross-variety vocabulary similarity', fontweight='bold', y=1.02)
        plt.tight_layout()

        if save:
            path = os.path.join(self.figures_path, 'vocabulary_similarity_heatmap.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Heatmap saved to: {path}")

        plt.show()

    # ------------------------------------------------------------------
    # 5. Run full analysis
    # ------------------------------------------------------------------
    def run(self, save: bool = True):
        print("=" * 55)
        print("VOCABULARY ANALYSIS")
        print("=" * 55)

        self.build_vocabularies()
        df_jac = self.compute_jaccard_matrix()
        df_cos = self.compute_tfidf_cosine_matrix()
        self.plot_similarity_heatmap(df_jac, df_cos, save=save)

        return df_jac, df_cos, self.vocab_per_variety
