"""
Omar - Linguistic Feature Analysis
Compares sarcastic vs non-sarcastic texts across:
  - Punctuation usage (!, ?, ...)
  - ALL CAPS word count
  - Emoji presence
  - Average sentence length
And extracts variety-specific terms (Australian slang, Indian English, British colloquialisms).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import warnings
warnings.filterwarnings('ignore')

# Emoji detection — uses regex range (no external emoji library needed)
EMOJI_PATTERN = re.compile(
    "[\U00010000-\U0010ffff"
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF\u2700-\u27BF]+",
    flags=re.UNICODE
)

# Variety-specific seed term lists for extraction guidance
VARIETY_SEEDS = {
    'en-AU': ['arvo', 'ute', 'mate', 'heaps', 'reckon', 'servo', 'brekkie',
              'arvo', 'footy', 'thongs', 'sunnies', 'bikkie', 'maccas',
              'chook', 'doco', 'sickie', 'tradie', 'rego', 'ambo'],
    'en-IN': ['yaar', 'bhai', 'ji', 'thoda', 'bahut', 'acha', 'bas',
              'lakh', 'crore', 'rupee', 'chai', 'desi', 'timepass',
              'jugaad', 'prepone', 'revert', 'doubts', 'discuss about'],
    'en-UK': ['cheers', 'mate', 'brilliant', 'lovely', 'rubbish', 'gutted',
              'chuffed', 'bloke', 'lass', 'quid', 'dodgy', 'knackered',
              'gobsmacked', 'miffed', 'reckon', 'fancy', 'fortnight', 'whilst'],
}


class LinguisticFeatureAnalysis:
    def __init__(self, df_all: pd.DataFrame, text_col: str = 'text',
                 sarcasm_col: str = 'Sarcasm', variety_col: str = 'variety',
                 save_path: str = './reports'):
        self.df = df_all.copy()
        self.text_col = text_col
        self.sarcasm_col = sarcasm_col
        self.variety_col = variety_col
        self.save_path = save_path
        self.figures_path = os.path.join(save_path, 'figures')
        os.makedirs(self.figures_path, exist_ok=True)

        self._extract_features()

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------
    def _extract_features(self):
        """Add linguistic feature columns to the dataframe."""
        txt = self.df[self.text_col].fillna('').astype(str)

        self.df['feat_exclamation']   = txt.str.count(r'!')
        self.df['feat_question']      = txt.str.count(r'\?')
        self.df['feat_ellipsis']      = txt.str.count(r'\.\.\.')
        self.df['feat_caps_words']    = txt.apply(
            lambda t: sum(1 for w in t.split() if w.isupper() and len(w) > 1)
        )
        self.df['feat_has_emoji']     = txt.apply(
            lambda t: int(bool(EMOJI_PATTERN.search(t)))
        )
        self.df['feat_emoji_count']   = txt.apply(
            lambda t: len(EMOJI_PATTERN.findall(t))
        )
        # Average sentence length (words per sentence)
        self.df['feat_avg_sent_len']  = txt.apply(self._avg_sentence_length)

    @staticmethod
    def _avg_sentence_length(text: str) -> float:
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return np.mean([len(s.split()) for s in sentences])

    # ------------------------------------------------------------------
    # 1. Sarcastic vs non-sarcastic comparison
    # ------------------------------------------------------------------
    def compare_sarcasm_features(self, save: bool = True) -> pd.DataFrame:
        """Compare mean linguistic features between sarcastic and non-sarcastic texts."""
        features = {
            'Exclamation marks (!)': 'feat_exclamation',
            'Question marks (?)':    'feat_question',
            'Ellipsis (...)':        'feat_ellipsis',
            'ALL CAPS words':        'feat_caps_words',
            'Emoji count':           'feat_emoji_count',
            'Avg sentence length':   'feat_avg_sent_len',
        }

        self.df[self.sarcasm_col] = self.df[self.sarcasm_col].astype(int)
        groups = self.df.groupby(self.sarcasm_col)

        rows = []
        for label, col in features.items():
            non_sarc = groups.get_group(0)[col].mean()
            sarc = groups.get_group(1)[col].mean()
            diff_pct = ((sarc - non_sarc) / (non_sarc + 1e-9)) * 100
            rows.append({
                'Feature': label,
                'Non-sarcastic (mean)': round(non_sarc, 3),
                'Sarcastic (mean)': round(sarc, 3),
                'Δ (%)': round(diff_pct, 1),
            })

        df_compare = pd.DataFrame(rows)
        print("\nSarcastic vs Non-sarcastic — linguistic features:")
        print(df_compare.to_string(index=False))

        self._plot_feature_comparison(df_compare, save)
        return df_compare

    def _plot_feature_comparison(self, df_compare: pd.DataFrame, save: bool):
        features = df_compare['Feature'].tolist()
        non_sarc_vals = df_compare['Non-sarcastic (mean)'].tolist()
        sarc_vals = df_compare['Sarcastic (mean)'].tolist()

        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 5))
        bars1 = ax.bar(x - width/2, non_sarc_vals, width,
                       label='Non-sarcastic', color='#66b3ff', edgecolor='black')
        bars2 = ax.bar(x + width/2, sarc_vals, width,
                       label='Sarcastic', color='#ff9999', edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel('Mean value per text', fontsize=11)
        ax.set_title('Linguistic features: sarcastic vs non-sarcastic',
                     fontweight='bold', fontsize=13)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        if save:
            path = os.path.join(self.figures_path, 'linguistic_features_sarcasm.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"\n✅ Figure saved to: {path}")
        plt.show()

    # ------------------------------------------------------------------
    # 2. Variety-specific term extraction
    # ------------------------------------------------------------------
    def extract_variety_terms(self, top_n: int = 30,
                               save: bool = True) -> dict[str, pd.DataFrame]:
        """
        Extract the top-N terms that are distinctive to each variety
        using a simple term-frequency ratio (TF in variety / TF elsewhere).
        Also checks for seed terms from VARIETY_SEEDS.
        """
        varieties = self.df[self.variety_col].unique()
        variety_term_dfs = {}

        all_words: dict[str, int] = {}
        for text in self.df[self.text_col].fillna('').astype(str):
            for w in re.findall(r"[a-z']+", text.lower()):
                all_words[w] = all_words.get(w, 0) + 1
        total_all = sum(all_words.values())

        print("\n" + "=" * 55)
        print("VARIETY-SPECIFIC TERM EXTRACTION")
        print("=" * 55)

        for variety in sorted(varieties):
            subset = self.df[self.df[self.variety_col] == variety][self.text_col]
            variety_words: dict[str, int] = {}
            for text in subset.fillna('').astype(str):
                for w in re.findall(r"[a-z']+", text.lower()):
                    variety_words[w] = variety_words.get(w, 0) + 1
            total_variety = sum(variety_words.values()) or 1

            rows = []
            for word, count in variety_words.items():
                if count < 3:
                    continue
                tf_variety = count / total_variety
                tf_all = all_words.get(word, 0) / (total_all or 1)
                ratio = tf_variety / (tf_all + 1e-9)
                is_seed = word in VARIETY_SEEDS.get(variety, [])
                rows.append({
                    'term': word,
                    'count_in_variety': count,
                    'tf_variety': round(tf_variety * 1000, 3),  # per-1000
                    'tf_overall': round(tf_all * 1000, 3),
                    'distinctiveness_ratio': round(ratio, 2),
                    'is_seed_term': is_seed,
                })

            df_terms = pd.DataFrame(rows).sort_values(
                'distinctiveness_ratio', ascending=False
            ).head(top_n)

            variety_term_dfs[variety] = df_terms

            print(f"\nTop {top_n} distinctive terms for {variety}:")
            print(df_terms[['term', 'count_in_variety', 'distinctiveness_ratio',
                             'is_seed_term']].to_string(index=False))

            # Check which seed terms appear
            seeds_found = [w for w in VARIETY_SEEDS.get(variety, [])
                           if w in variety_words]
            if seeds_found:
                print(f"  ✅ Seed terms found: {', '.join(seeds_found)}")
            else:
                print(f"  ⚠️  No seed terms found for {variety} "
                      f"(may be present in different form)")

            if save:
                csv_path = os.path.join(
                    self.save_path, f'variety_terms_{variety.replace("-","_")}.csv'
                )
                df_terms.to_csv(csv_path, index=False)
                print(f"  ✅ Saved to: {csv_path}")

        return variety_term_dfs

    # ------------------------------------------------------------------
    # 3. Error analysis prep — tricky cases per variety
    # ------------------------------------------------------------------
    def identify_tricky_cases(self, n_per_variety: int = 10,
                               save: bool = True) -> pd.DataFrame:
        """
        Identify tricky cases: sarcastic texts with positive sentiment
        OR non-sarcastic texts with very high punctuation density.
        These are the cases most likely to fool a classifier.
        """
        self.df['feat_punct_density'] = (
            self.df['feat_exclamation'] + self.df['feat_question'] + self.df['feat_ellipsis']
        )

        tricky_rows = []
        for variety in sorted(self.df[self.variety_col].unique()):
            sub = self.df[self.df[self.variety_col] == variety].copy()

            # Case 1: sarcastic + positive sentiment (confusing label combo)
            sarc_positive = sub[
                (sub[self.sarcasm_col] == 1) &
                (sub.get('Sentiment', sub.get('sentiment', pd.Series(dtype=int))).astype(float) == 1.0)
            ].head(n_per_variety // 2)

            # Case 2: high punctuation density but NOT sarcastic (could confuse model)
            punct_threshold = sub['feat_punct_density'].quantile(0.9)
            high_punct_not_sarc = sub[
                (sub[self.sarcasm_col] == 0) &
                (sub['feat_punct_density'] >= punct_threshold)
            ].head(n_per_variety // 2)

            for df_sub, reason in [(sarc_positive, 'sarcastic+positive'),
                                   (high_punct_not_sarc, 'high-punct+non-sarcastic')]:
                for _, row in df_sub.iterrows():
                    tricky_rows.append({
                        'variety': variety,
                        'reason': reason,
                        'text': str(row[self.text_col])[:300],
                        'Sarcasm': int(row[self.sarcasm_col]),
                        'punct_density': int(row['feat_punct_density']),
                        'caps_words': int(row['feat_caps_words']),
                    })

        df_tricky = pd.DataFrame(tricky_rows)

        print(f"\n{'='*55}")
        print("ERROR ANALYSIS PREP — tricky cases")
        print(f"{'='*55}")
        for variety in sorted(df_tricky['variety'].unique()):
            sub = df_tricky[df_tricky['variety'] == variety]
            print(f"\n{variety} ({len(sub)} cases):")
            for _, row in sub.iterrows():
                print(f"  [{row['reason']}] {row['text'][:120]}...")

        if save:
            csv_path = os.path.join(self.save_path, 'tricky_cases_error_analysis.csv')
            df_tricky.to_csv(csv_path, index=False)
            print(f"\n✅ Tricky cases saved to: {csv_path}")

        return df_tricky

    # ------------------------------------------------------------------
    # 4. Run full analysis
    # ------------------------------------------------------------------
    def run(self, save: bool = True):
        df_compare = self.compare_sarcasm_features(save=save)
        variety_terms = self.extract_variety_terms(save=save)
        df_tricky = self.identify_tricky_cases(save=save)
        return df_compare, variety_terms, df_tricky
