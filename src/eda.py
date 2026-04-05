import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import warnings
import spacy

nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, df_all, df_train, df_val, df_test):
        self.df_all = df_all
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test

        os.makedirs("./reports/figures", exist_ok=True)
        os.makedirs("./reports", exist_ok=True)

    def save_figure(self, save_path="./reports/figures", filename="plot.png"):
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure saved to: {path}")

    def plot_distribution(self, df, column, groupby=None, title=None, xlabel=None, ylabel=None,
                          plot_type='countplot', save=False, save_path="./reports/figures",
                          filename=None, **kwargs):

        plt.figure(figsize=(8, 5))

        # stacked bar plot
        if plot_type == 'stacked_bar':
            if groupby is None:
                raise ValueError("groupby parameter is required for stacked_bar")
            crosstab = pd.crosstab(df[column], df[groupby])
            ax = crosstab.plot(kind='bar', stacked=True,
                              color=['#4285F4', '#EA4335'],
                              edgecolor='black')

            for i, idx in enumerate(crosstab.index):
                total = crosstab.loc[idx].sum()
                ax.text(i, total + 5, f'Total: {total}',
                       ha='center', va='bottom', fontsize=9)

        #grouped bar plot
        elif plot_type == 'grouped_bar':
            if groupby is None:
                raise ValueError("groupby parameter is required for grouped_bar")
            crosstab = pd.crosstab(df[column], df[groupby])
            percentages = crosstab.div(crosstab.sum(axis=1), axis=0) * 100

            num_groups = len(percentages.columns)
            colors = ['#66b3ff', '#ff9999', '#99ff99'][:num_groups]

            ax = percentages.plot(kind='bar', color=colors, edgecolor='black', width=0.7)
            plt.ylabel(ylabel or "Percentage (%)")
            plt.xlabel(xlabel or column)
            plt.xticks(rotation=0)
            plt.ylim(0, 100)
            plt.grid(axis='y', alpha=0.3)

            for i, idx in enumerate(percentages.index):
                for j, col in enumerate(percentages.columns):
                    value = percentages.loc[idx, col]
                    if value > 0:
                        ax.text(i + (j-0.5)*0.3, value + 1, f'{value:.1f}%',
                               ha='center', va='bottom', fontsize=9)
            plt.legend(title=groupby)

        #heatmap
        elif plot_type == 'heatmap':
            if groupby is None:
                raise ValueError("groupby parameter is required for heatmap")
            crosstab = pd.crosstab(df[column], df[groupby])
            ax = sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues',
                            cbar_kws={'label': 'Count'})

            if 'highlight' in kwargs:
                highlight_row, highlight_col = kwargs['highlight']
                rect = plt.Rectangle((highlight_col, highlight_row), 1, 1,
                                    fill=False, edgecolor='red', linewidth=5)
                ax.add_patch(rect)
            plt.ylabel(ylabel or column)
            plt.xlabel(xlabel or groupby)

        elif plot_type == 'countplot':
            ax = sns.countplot(x=column, data=df, palette='Set2', edgecolor='black')
            plt.ylabel(ylabel or "Count")

            for bar in ax.patches:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.xlabel(xlabel or column)

        elif plot_type == 'bar':
            counts = df[column].value_counts()
            bars = plt.bar(counts.index.astype(str), counts.values, color='skyblue', edgecolor='black')
            plt.ylabel(ylabel or "Count")

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=10)
            plt.xlabel(xlabel or column)

        elif plot_type == 'pie':
            counts = df[column].value_counts()
            plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
            plt.ylabel('')

        plt.title(title or f"{column} Distribution", fontweight='bold')

        if plot_type not in ['heatmap', 'pie']:
            plt.grid(axis='y', alpha=0.3)

        if filename is None:
            filename = f"{column.lower()}_distribution.png"

        plt.tight_layout()

        if save:
            self.save_figure(save_path, filename)

        plt.show()
        return plt.gcf()

    def create_source_variety_table(self, crosstab, save_path="./reports"):
        os.makedirs(save_path, exist_ok=True)

        table_data = []
        for variety in ['en-AU', 'en-IN', 'en-UK']:
            google_count = crosstab.loc[variety, 'Google']
            reddit_count = crosstab.loc[variety, 'Reddit']
            total = google_count + reddit_count
            google_pct = (google_count / total) * 100
            reddit_pct = (reddit_count / total) * 100

            table_data.append([
                variety,
                f"{google_count} ({google_pct:.1f}%)",
                f"{reddit_count} ({reddit_pct:.1f}%)",
                total
            ])

        table_df = pd.DataFrame(table_data, columns=['Variety', 'Google', 'Reddit', 'Total'])
        print(table_df.to_string(index=False))

        filepath = os.path.join(save_path, "source_by_variety.csv")
        table_df.to_csv(filepath, index=False)
        print(f"\n✅ Table saved to: {filepath}")
        return table_df

    def create_sarcasm_source_table(self, crosstab, save_path="./reports"):
        os.makedirs(save_path, exist_ok=True)

        table_data = []
        for source in ['Google', 'Reddit']:
            sarcastic_count = crosstab.loc[source, 1]
            non_sarcastic_count = crosstab.loc[source, 0]
            total = sarcastic_count + non_sarcastic_count
            sarcastic_pct = (sarcastic_count / total) * 100
            non_sarcastic_pct = (non_sarcastic_count / total) * 100

            table_data.append([
                source,
                f"{sarcastic_count} ({sarcastic_pct:.1f}%)",
                f"{non_sarcastic_count} ({non_sarcastic_pct:.1f}%)",
                total
            ])

        table_df = pd.DataFrame(table_data, columns=['Source', 'Sarcastic', 'Non-sarcastic', 'Total'])
        print(table_df.to_string(index=False))

        filepath = os.path.join(save_path, "sarcasm_by_source.csv")
        table_df.to_csv(filepath, index=False)
        print(f"\n✅ Table saved to: {filepath}")
        return table_df

    def create_sentiment_source_table(self, crosstab, save_path="./reports"):
        os.makedirs(save_path, exist_ok=True)

        table_data = []
        for source in ['Google', 'Reddit']:
            positive = crosstab.loc[source, 1]
            negative = crosstab.loc[source, 0]
            total = positive + negative
            positive_pct = (positive / total) * 100
            negative_pct = (negative / total) * 100

            table_data.append([
                source,
                f"{positive} ({positive_pct:.1f}%)",
                f"{negative} ({negative_pct:.1f}%)",
                total
            ])

        table_df = pd.DataFrame(table_data, columns=['Source', 'Positive', 'Negative', 'Total'])
        print(table_df.to_string(index=False))

        filepath = os.path.join(save_path, "sentiment_by_source.csv")
        table_df.to_csv(filepath, index=False)
        print(f"\n✅ Table saved to: {filepath}")
        return table_df

    # DATASET ANALYSIS METHODS
    def sarcasm_dist(self):
        self.df_train["Sarcasm"] = self.df_train["Sarcasm"].astype(int)
        self.plot_distribution(
            df=self.df_train,
            column='Sarcasm',
            title="Sarcasm Distribution",
            xlabel="Sarcasm (0=No, 1=Yes)",
            ylabel="Count",
            plot_type='countplot',
            save=True,
            filename="sarcasm_distribution.png"
        )

    def sentiment_dist(self):
        self.df_train["Sentiment"] = self.df_train["Sentiment"].astype(int)
        self.plot_distribution(
            df=self.df_train,
            column='Sentiment',
            title="Sentiment Distribution",
            xlabel="Sentiment (0=Negative, 1=Positive)",
            ylabel="Count",
            plot_type='countplot',
            save=True,
            filename="sentiment_distribution.png"
        )

    def variety_source_dist(self, df):
        self.plot_distribution(
            df=df,
            column='variety',
            title="Distribution Across Varieties",
            xlabel="Variety",
            ylabel="Count",
            plot_type='countplot',
            save=True,
            filename="variety_distribution.png"
        )

        self.plot_distribution(
            df=df,
            column='source',
            title="Source Distribution",
            xlabel="Source",
            ylabel="Count",
            plot_type='countplot',
            save=True,
            filename="source_distribution.png"
        )

        return df["variety"].value_counts(), df["source"].value_counts()

    def split_distribution_per_variety(self):
        split_series = pd.Series(
            ["train"] * len(self.df_train) +
            ["validation"] * len(self.df_val) +
            ["test"] * len(self.df_test),
            name="split"
        )
        return pd.crosstab(self.df_all["variety"], split_series)

    def source_per_variety(self):
        crosstab = pd.crosstab(self.df_all["variety"], self.df_all["source"])
        self.create_source_variety_table(crosstab, save_path="./reports")

        self.plot_distribution(
            df=self.df_all,
            column='variety',
            groupby='source',
            title="Source Distribution by Variety",
            xlabel="Variety",
            ylabel="Count",
            plot_type='stacked_bar',
            save=True,
            filename="source_by_variety.png"
        )

        return crosstab

    def sarcasm_by_source(self):
        crosstab = pd.crosstab(self.df_train["source"], self.df_train["Sarcasm"])
        self.create_sarcasm_source_table(crosstab, save_path="./reports")

        self.plot_distribution(
            df=self.df_train,
            column='source',
            groupby='Sarcasm',
            title="Sarcasm Distribution by Source",
            xlabel="Source",
            ylabel="Count",
            plot_type='stacked_bar',
            save=True,
            filename="sarcasm_by_source.png"
        )

        return crosstab

    def sentiment_by_source(self):
        crosstab = pd.crosstab(self.df_train["source"], self.df_train["Sentiment"])
        self.create_sentiment_source_table(crosstab, save_path="./reports")

        self.plot_distribution(
            df=self.df_train,
            column='source',
            groupby='Sentiment',
            title="Sentiment Distribution by Source",
            xlabel="Source",
            ylabel="Count",
            plot_type='stacked_bar',
            save=True,
            filename="sentiment_by_source.png"
        )

        return crosstab

    def sarcasm_sentiment_correlation(self):
        crosstab = pd.crosstab(self.df_train["Sarcasm"], self.df_train["Sentiment"])

        # Print insight
        if 1 in crosstab.index and 0 in crosstab.columns:
            sarcasm_neg = crosstab.loc[1, 0]
            total_sarcasm = crosstab.loc[1].sum()
            perc = (sarcasm_neg / total_sarcasm) * 100
            print(f"\n🔍 {perc:.2f}% of sarcastic texts have negative sentiment")
            print(f"   ({sarcasm_neg} out of {total_sarcasm} sarcastic instances)")

        self.plot_distribution(
            df=self.df_train,
            column='Sarcasm',
            groupby='Sentiment',
            title="Sarcasm vs Sentiment Correlation\n(Red box shows sarcastic-negative texts)",
            xlabel="Sentiment",
            ylabel="Sarcasm",
            plot_type='heatmap',
            save=True,
            highlight=(1, 0) if 1 in crosstab.index and 0 in crosstab.columns else None,
            filename="sarcasm_sentiment_correlation.png"
        )

        return crosstab

    def sarcasm_imbalance(self):
        """Sarcasm imbalance analysis with plots"""
        self.df_all["Sarcasm"] = self.df_all["Sarcasm"].astype(int)

        # Overall imbalance
        overall = self.df_all["Sarcasm"].value_counts(normalize=True) * 100
        print("\n" + "="*50)
        print("OVERALL SARcASM IMBALANCE")
        print("="*50)
        print(f"Sarcastic (1): {overall[1]:.2f}%")
        print(f"Non-sarcastic (0): {overall[0]:.2f}%")

        self.plot_distribution(
            df=self.df_all,
            column='Sarcasm',
            title="Sarcasm Distribution (Overall)",
            plot_type='pie',
            save=True,
            filename="sarcasm_overall.png"
        )

        # Per variety imbalance
        per_variety = pd.crosstab(self.df_all["variety"], self.df_all["Sarcasm"], normalize="index") * 100
        print("\n" + "="*50)
        print("SARcASM IMBALANCE PER VARIETY")
        print("="*50)
        print(per_variety.round(2))

        self.plot_distribution(
            df=self.df_all,
            column='variety',
            groupby='Sarcasm',
            title="Sarcasm Distribution by Variety",
            xlabel="Variety",
            ylabel="Percentage (%)",
            plot_type='grouped_bar',
            save=True,
            filename="sarcasm_by_variety.png"
        )

        # Per variety per split imbalance
        per_split = pd.crosstab([self.df_all["variety"], self.df_all["split"]],
                                self.df_all["Sarcasm"],
                                normalize="index") * 100
        print("\n" + "="*50)
        print("SARcASM IMBALANCE PER VARIETY & SPLIT")
        print("="*50)
        print(per_split.round(2))

        # Create combined column for plotting
        df_temp = self.df_all.copy()
        df_temp['variety_split'] = df_temp['variety'] + "\n(" + df_temp['split'] + ")"

        self.plot_distribution(
            df=df_temp,
            column='variety_split',
            groupby='Sarcasm',
            title="Sarcasm Distribution by Variety and Split",
            xlabel="Variety (Split)",
            ylabel="Percentage (%)",
            plot_type='grouped_bar',
            save=True,
            filename="sarcasm_by_variety_split.png"
        )

        return overall, per_variety, per_split
        
    # 8. Sentiment imbalance in whole dataset
    def sentiment_imbalance(self):
         self.df_all["Sarcasm"] = self.df_all["Sentiment"].astype(int)
        overall = self.df_all["Sentiment"].value_counts(normalize=True) * 100

        self.plot_distribution(
            df=self.df_all,
            column='Sentiment',
            title="Sentiment Distribution (Overall)",
            plot_type='pie',
            save=True,
            filename="sentiment_overall.png"
        )

        # Per variety imbalance
        per_variety = pd.crosstab(self.df_all["variety"], self.df_all["Sentiment"], normalize="index") * 100

        self.plot_distribution(
            df=self.df_all,
            column='variety',
            groupby='Sentiment',
            title="Sentiment Distribution by Variety",
            xlabel="Variety",
            ylabel="Percentage (%)",
            plot_type='grouped_bar',
            save=True,
            filename="sentiment_by_variety.png"
        )

        # Per variety per split imbalance
        per_split = pd.crosstab([self.df_all["variety"], self.df_all["split"]],
                                self.df_all["Sentiment"],
                                normalize="index") * 100

        # Create combined column for plotting
        df_temp = self.df_all.copy()
        df_temp['variety_split'] = df_temp['variety'] + "\n(" + df_temp['split'] + ")"

        self.plot_distribution(
            df=df_temp,
            column='variety_split',
            groupby='Sentiment',
            title="Sentiment Distribution by Variety and Split",
            xlabel="Variety (Split)",
            ylabel="Percentage (%)",
            plot_type='grouped_bar',
            save=True,
            filename="sentiment_by_variety_split.png"
        )
        return overall, per_variety, per_split

    def text_length(self, save=False):
        # Add length columns
        self.df_all['text_length'] = self.df_all['text'].str.len()
        self.df_all['word_count'] = self.df_all['text'].str.split().str.len()

        # Avg length per variety
        variety_length = self.df_all.groupby('variety')['text_length'].mean().round(1)

        # Avg length per sentiment class
        sentiment_length = self.df_all.groupby('Sentiment')['text_length'].mean().round(1)

        # Avg length per sarcasm class
        sarcasm_length = self.df_all.groupby('Sarcasm')['text_length'].mean().round(1)

        # Identify outliers
        threshold = self.df_all['text_length'].quantile(0.99)
        outliers = self.df_all[self.df_all['text_length'] > threshold]

        # Call plotting function
        self.plot_text_length_analysis(variety_length, sentiment_length, sarcasm_length, threshold, save)

        return variety_length, sentiment_length, sarcasm_length

    def plot_text_length_analysis(self, variety_length, sentiment_length, sarcasm_length, threshold, save=False):
        save_path = "./reports/figures"
        os.makedirs(save_path, exist_ok=True)

        #Histogram
        plt.figure(figsize=(10, 5))
        plt.hist(self.df_all['text_length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(self.df_all['text_length'].mean(), color='red', linestyle='--',
                    label=f'Mean: {self.df_all["text_length"].mean():.0f}')
        plt.axvline(threshold, color='orange', linestyle='--',
                    label=f'Outlier Threshold: {threshold:.0f}')
        plt.title("Distribution of Text Length", fontsize=14, fontweight='bold')
        plt.xlabel("Text Length (characters)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "text_length_histogram.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")


        #Box plot by variety
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='variety', y='text_length', data=self.df_all, palette='Set2')
        plt.title("Text Length Distribution by Variety", fontsize=14, fontweight='bold')
        plt.xlabel("Variety", fontsize=12)
        plt.ylabel("Text Length (characters)", fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "text_length_by_variety_box.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")


        # PLOT 3: Box plot by sentiment
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Sentiment', y='text_length', data=self.df_all, palette='Set2')
        plt.title("Text Length Distribution by Sentiment", fontsize=14, fontweight='bold')
        plt.xlabel("Sentiment (0=Negative, 1=Positive)", fontsize=12)
        plt.ylabel("Text Length (characters)", fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "text_length_by_sentiment_box.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")


        # PLOT 4: Box plot by sarcasm
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='Sarcasm', y='text_length', data=self.df_all, palette='Set2')
        plt.title("Text Length Distribution by Sarcasm", fontsize=14, fontweight='bold')
        plt.xlabel("Sarcasm (0=No, 1=Yes)", fontsize=12)
        plt.ylabel("Text Length (characters)", fontsize=12)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "text_length_by_sarcasm_box.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")


        # PLOT 5: Bar chart of average length per variety
        plt.figure(figsize=(8, 5))
        variety_length.plot(kind='bar', color=['#ff9999', '#66b3ff', '#99ff99'], edgecolor='black')
        plt.title("Average Text Length by Variety", fontsize=14, fontweight='bold')
        plt.xlabel("Variety", fontsize=12)
        plt.ylabel("Average Text Length (characters)", fontsize=12)
        plt.xticks(rotation=0)
        plt.grid(axis='y', alpha=0.3)
        for i, v in enumerate(variety_length.values):
            plt.text(i, v + 2, f'{v:.1f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "avg_text_length_by_variety.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")

        # PLOT 6: Comparison of average length (sentiment vs sarcasm)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sentiment_length.plot(kind='bar', ax=axes[0], color=['#ff9999', '#66b3ff'], edgecolor='black')
        axes[0].set_title("Avg Text Length by Sentiment", fontweight='bold')
        axes[0].set_xlabel("Sentiment (0=Negative, 1=Positive)")
        axes[0].set_ylabel("Avg Length (characters)")
        axes[0].set_xticklabels(['Negative', 'Positive'], rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(sentiment_length.values):
            axes[0].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom')

        sarcasm_length.plot(kind='bar', ax=axes[1], color=['#ff9999', '#66b3ff'], edgecolor='black')
        axes[1].set_title("Avg Text Length by Sarcasm", fontweight='bold')
        axes[1].set_xlabel("Sarcasm (0=No, 1=Yes)")
        axes[1].set_ylabel("Avg Length (characters)")
        axes[1].set_xticklabels(['Non-sarcastic', 'Sarcastic'], rotation=0)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(sarcasm_length.values):
            axes[1].text(i, v + 2, f'{v:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(save_path, "avg_text_length_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()
        print("\n")


    def compare_domains(self, save=False):
        # Add text_length if not already present
        if 'text_length' not in self.df_all.columns:
            self.df_all['text_length'] = self.df_all['text'].str.len()

        # Average length per domain
        domain_length = self.df_all.groupby('source')['text_length'].mean().round(1)

        # Basic vocabulary differences
        google_texts = self.df_all[self.df_all['source'] == 'Google']['text'].str.cat(sep=' ').lower().split()
        reddit_texts = self.df_all[self.df_all['source'] == 'Reddit']['text'].str.cat(sep=' ').lower().split()

        google_vocab = set(google_texts)
        reddit_vocab = set(reddit_texts)
        overlap = len(google_vocab & reddit_vocab)

        print("VOCABULARY COMPARISON")
        print(f"Google unique words: {len(google_vocab):,}")
        print(f"Reddit unique words: {len(reddit_vocab):,}")
        print(f"Overlap: {overlap:,}")


        self.plot_domain_comparison(domain_length, google_vocab, reddit_vocab, overlap, save)

        return domain_length, google_vocab, reddit_vocab, overlap

    def plot_domain_comparison(self, domain_length, google_vocab, reddit_vocab, overlap, save=False):
        save_path = "./reports/figures"
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 5))

        bars = plt.bar(domain_length.index, domain_length.values,
                      color=['#4285F4', '#EA4335'], edgecolor='black')

        plt.title("Average Text Length by Domain", fontsize=14, fontweight='bold')
        plt.xlabel("Source", fontsize=12)
        plt.ylabel("Average Text Length (characters)", fontsize=12)
        plt.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        vocab_info = f"Google Vocab: {len(google_vocab):,}\nReddit Vocab: {len(reddit_vocab):,}\nOverlap: {overlap:,}"
        plt.text(0.98, 0.95, vocab_info, transform=plt.gca().transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        if save:
            plt.savefig(os.path.join(save_path, "domain_comparison.png"), dpi=150, bbox_inches='tight')
        plt.show()

    def pos_for_sarcasm(self, n_samples=500):
        sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 1]['text'].sample(
            min(n_samples, len(self.df_all[self.df_all['Sarcasm']==1]))
        ).tolist()
        
        non_sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 0]['text'].sample(
            min(n_samples, len(self.df_all[self.df_all['Sarcasm']==0]))
        ).tolist()
        
        sarcastic_pos = {}
        non_sarcastic_pos = {}
    
        for text in sarcastic_texts:
            doc = nlp(text)
            for token in doc:
                sarcastic_pos[token.pos_] = sarcastic_pos.get(token.pos_, 0) + 1

        for text in non_sarcastic_texts:
            doc = nlp(text)
            for token in doc:
                non_sarcastic_pos[token.pos_] = non_sarcastic_pos.get(token.pos_, 0) + 1
        
        total_sarc = sum(sarcastic_pos.values())
        total_non = sum(non_sarcastic_pos.values())

        pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ', 'PRON', 'ADP']
        sarc_pcts = [(sarcastic_pos.get(pos, 0) / total_sarc) * 100 for pos in pos_tags]
        non_pcts = [(non_sarcastic_pos.get(pos, 0) / total_non) * 100 for pos in pos_tags]
    
        return {
            'pos_tags': pos_tags,
            'sarcastic_pcts': sarc_pcts,
            'non_sarcastic_pcts': non_pcts
        }, sarcastic_pos, non_sarcastic_pos
    
    def sarcastic_phrases_analysis(self):
        sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 1]['text']

        patterns = [
            'yeah right', 'oh great', 'wonderful', 'brilliant', 'thanks a lot', 
            'as if', 'sure', 'of course', 'how nice', 'how lovely', 'well done',
            'good job', 'nice one', 'really?', 'seriously?', 'obviously',
            'tell me about it', 'big surprise', 'what a surprise', 'fantastic'
        ]
        
        found_patterns = []
        pattern_counts = []
        
        for pattern in patterns:
            count = sarcastic_texts.str.lower().str.contains(pattern).sum()
            if count > 0:
                print(f"   '{pattern}': found in {count} sarcastic texts")
                found_patterns.append(pattern)
                pattern_counts.append(count)
        
        examples_by_variety = {}
        for variety in ['en-AU', 'en-IN', 'en-UK']:
            examples = self.df_all[(self.df_all['variety'] == variety) & (self.df_all['Sarcasm'] == 1)]['text'].head(3).tolist()
            examples_by_variety[variety] = [ex[:120] + "..." if len(ex) > 120 else ex for ex in examples]

        return {
            'found_patterns': found_patterns,
            'pattern_counts': pattern_counts
        }, examples_by_variety



def get_sarcasm_extremes(per_variety):
    most_sarcastic = per_variety[1].idxmax()
    least_sarcastic = per_variety[1].idxmin()
    
    most_sarcastic_pct = per_variety.loc[most_sarcastic, 1]
    least_sarcastic_pct = per_variety.loc[least_sarcastic, 1]
    
    return {
        'most_sarcastic': most_sarcastic,
        'most_sarcastic_pct': most_sarcastic_pct,
        'least_sarcastic': least_sarcastic,
        'least_sarcastic_pct': least_sarcastic_pct
    }

#VARIETY-SPECIFIC SLANG
def variety_slang(df_all):
    slang_dictionary = {
        'en-AU': [
            'arvo', 'brekkie', 'servo', 'maccas', 'bottle-o', 'esky', 'straya',
            'ute', 'mate', 'bogan', 'thongs', 'sunnies', 'trackies', 'ambo',
            'pollie', 'tradie', 'garbo', 'sparky', 'chippy',
            'footy', 'crikey', 'fair dinkum', 'true blue', 'no worries',
            'she\'ll be right', 'stoked', 'heaps', 'rack off', 'dunny',
            'tucker', 'bush tucker', 'yakka', 'snag', 'tinnie', 'bathers',
            'cossies', 'togs', 'barbie'
        ],

        'en-IN': [
            'yaar', 'na', 're', 'bhai', 'acha', 'accha', 'chai', 'jugaad',
            'arre', 'kya', 'machaa', 'bahut', 'thoda', 'theek', 'hai',
            'nahi', 'waah', 'abey', 'bhaiya', 'didi',
            'tension', 'matlab', 'actually', 'basically', 'seriously', 'generally',
            'only', 'itself', 'too much', 'very much', 'kindly', 'timepass',
            'prepone', 'passing out', 'cousin brother', 'cousin sister',
            'batchmate', 'rest is fine', 'do one thing', 'what to do',
            'chalta hai', 'thoda adjust', 'mind it', 'just now'
        ],

        'en-UK': [

            'bloody', 'brilliant', 'cheers', 'lorry', 'boot', 'flat', 'mate',
            'bob', 'chuffed', 'gobsmacked', 'knackered', 'gutted', 'peckish',
            'bloke', 'bird', 'geezer', 'lad', 'lass', 'chap', 'missus',
            'innit', 'proper', 'sorted', 'taking the piss', 'fancy', 'quite',
            'queue', 'telly', 'loo', 'bog', 'cuppa', 'pub',
            'nowt', 'owt', 'canny', 'bairn', 'wee', 'aye', 'nae', 'ken'
        ]
    }

    results = {}

    for en_variety, slang_list in slang_dictionary.items():
        en_variety_texts = df_all[df_all['variety'] == en_variety]['text'].str.lower()

        slang_found = []
        for slang in slang_list:
            matches = en_variety_texts[en_variety_texts.str.contains(slang, na=False)]
            if len(matches) > 0:
                example = matches.iloc[0]
                slang_found.append((slang, example))

        results[en_variety] = slang_found
        if slang_found:
            for slang, example in slang_found[:5]:
                example_short = example[:100] + "..." if len(example) > 100 else example
        else:
            print("   No slang examples found")

    return results
