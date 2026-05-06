import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import os
from collections import Counter
import warnings
import spacy

nlp = spacy.load('en_core_web_sm')
warnings.filterwarnings('ignore')

class EDA:
    def __init__(self, df_train, df_validation, df_test):
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.df_all = pd.concat([df_train, df_validation, df_test], ignore_index=True)
        os.makedirs("./reports/figures", exist_ok=True)
        os.makedirs("./reports", exist_ok=True)


    def plot_counts(self, df, column, title, xlabel, ylabel,save=False, filename=None):
        fig, ax =plt.subplots(figsize=(7.2, 4.5))
        sns.countplot(x=column, data= df, palette='viridis', ax=ax, edgecolor ='black')

        for c in ax.containers:
            ax.bar_label(c, padding=3, fontsize=9)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        plt.title(title, pad=10)
        if save:
            self.save_figure(filename=filename or f"{column}_counts.png")
        plt.show()

    def plot_grouped_bar(self, df, column, groupby, title, xlabel, ylabel, save=False,filename=None):
        table= pd.crosstab(df[column], df[groupby], normalize='index').mul(100)

        ax = table.plot(kind='bar', figsize=(8.2, 5), width= 0.7, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, 115)
        for c in ax.containers:
            ax.bar_label(c, fmt='%.1f%%', padding  =2, fontsize=8)

        plt.title(title, fontweight='semibold')
        if save:
            self.save_figure(filename=f"{column}_vs_{groupby}_pct.png")
        plt.show()

    def plot_stacked_bar(self, df, column, groupby, title, xlabel, ylabel, save=False,filename=None):
        table = pd.crosstab(df[column], df[groupby])
        custom_colors = ['#34495e', '#e67e22', '#27ae60']
        ax = table.plot(kind='bar', stacked=True, color=custom_colors,
                        figsize=(8.4, 4.6), edgecolor='white', linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        for c in ax.containers:
            ax.bar_label(c, label_type='center', color='white', weight='bold')

        plt.title(title)
        if save:
            path = f"stacked_{column}_by_{groupby}.png"
            self.save_figure(filename=path)
        plt.show()

    def plot_heatmap(self, df, column, groupby, title, xlabel, ylabel,highlight=None, save=False,filename=None):
        counts_table = pd.crosstab(df[column], df[groupby])
        fig, ax = plt.subplots(figsize=(7.8, 5.2))

        sns.heatmap(counts_table, annot=True, fmt='d', cmap='magma', cbar=False, ax=ax)
        ax.set_xlabel(xlabel, fontsize=10,fontweight='medium')
        ax.set_ylabel(ylabel,fontsize=10,  fontweight='medium')

        if highlight:
            r, c = highlight
            rectangle=plt.Rectangle((c, r), 1, 1, fill=False, edgecolor='red', lw=5, ls='-')
            ax.add_patch(rectangle)
        plt.title(title,  pad=15)
        plt.tight_layout()
        if save:
            self.save_figure(filename=f"heatmap_{column}_{groupby}.png")
        plt.show()
  
    def variety_source_dist(self, df):
        self.plot_counts(
            df=df,
            column='variety',
            title="Distribution Across Varieties",
            xlabel="Variety Name",
            ylabel="Number of samples",
            save=True,
            filename="variety_distribution.png"
        )

        self.plot_counts(
            df=df,
            column='source',
            title="Source Distribution",
            xlabel="Source Name",
            ylabel="Number of samples",
            save=True,
            filename="source_distribution.png"
        )

        return df["variety"].value_counts(), df["source"].value_counts()

    def split_distribution_per_variety(self, save=False):
        split_series = pd.Series(
            ["train"] * len(self.df_train) +
            ["validation"] * len(self.df_validation) +
            ["test"] * len(self.df_test),
            name="split"
        )

        table = pd.crosstab(self.df_all["variety"], split_series.values)
        english_varieties = ['en-AU', 'en-IN', 'en-UK']
        filtered_table = table.loc[english_varieties]
        x = np.arange(len(english_varieties))
        width = 0.2
        fig, ax = plt.subplots(figsize=(9, 5))

        splits = ['train', 'validation', 'test']
        colors = ['#4e79a7', '#f28e2b', '#e15759']

        for i, split in enumerate(splits):
            counts = filtered_table[split].values
            bars = ax.bar(x + i * width, counts, width, label=split.title(), color=colors[i])
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                        f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(english_varieties)
        ax.set_ylabel('Total Count')
        ax.set_title('Dataset Split by English Variety', fontweight='bold')
        ax.legend(frameon=False)

        plt.tight_layout()
        if save:
            self.save_figure(filename="split_dist_plot.png")
        plt.show()
        return filtered_table

    def source_per_variety(self):
        table = pd.crosstab(self.df_all["variety"], self.df_all["source"])
        self.plot_stacked_bar(
            df=self.df_all,
            column='variety',
            groupby='source',
            title="Source Distribution by Variety",
            xlabel="Variety Name",
            ylabel="Number of samples",
            save=True,
            filename="source_by_variety.png"
        )

        return table

    def sarcasm_sentiment_correlation(self):
        table = pd.crosstab(self.df_train["Sarcasm"], self.df_train["Sentiment"])
        self.plot_heatmap(
            df=self.df_train,
            column='Sarcasm',
            groupby='Sentiment',
            title="Sarcasm vs Sentiment Correlation\n(Red box shows sarcastic-negative texts)",
            xlabel="Sentiment",
            ylabel="Sarcasm",
            save=True,
            highlight=(1, 0) if 1 in table.index and 0 in table.columns else None,
            filename="sarcasm_sentiment_correlation.png"
        )
        return table


    def sentiment_imbalance(self):
        self.df_all["Sentiment"] = self.df_all["Sentiment"].astype(int)
        overall=self.df_all["Sentiment"].value_counts(normalize=True) * 100
        per_variety = pd.crosstab(self.df_all["variety"], self.df_all["Sentiment"], normalize="index") * 100

        self.plot_grouped_bar(
            df=self.df_all,
            column='variety',
            groupby='Sentiment',
            title="Sentiment Distribution by Variety",
            xlabel="Varieties",
            ylabel= "Percentages (%)",
            save=True,
            filename="sentiment_by_variety.png"
        )

        per_split =  pd.crosstab([self.df_all["variety"], self.df_all["split"]],self.df_all["Sentiment"],  normalize="index") * 100
        return overall, per_variety, per_split

    def pos_for_sarcasm(self, n_samples=500):

          s_df = self.df_all[self.df_all['Sarcasm'] == 1]
          n_df = self.df_all[self.df_all['Sarcasm'] == 0]
          sarc_texts = s_df['text'].sample(min(n_samples, len(s_df))).tolist()
          non_sarc_texts = n_df['text'].sample(min(n_samples, len(n_df))).tolist()
          
          sarc_counts = {}
          non_sarc_tags = {}
          for text in sarc_texts:
              doc = nlp(text)
              for token in doc:
                  sarc_counts[token.pos_] = sarc_counts.get(token.pos_, 0) + 1

          # Same for non-sarcastic texts
          for text in non_sarc_texts:

              doc = nlp(text)

              for token in doc:
              
                  non_sarc_tags[token.pos_] = non_sarc_tags.get(token.pos_, 0) + 1
          
          total_sarc_counts = sum(sarc_counts.values())
          total_non_sarc_counts = sum(non_sarc_tags.values())
          tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ', 'PRON', 'ADP']
          sarcastic_pcts = [(sarc_counts.get(pos, 0) / total_sarc_counts) * 100 for pos in tags]
          non_sarcastic_pcts = [100*(non_sarc_tags.get(pos, 0) / total_non_sarc_counts) for pos in tags]
      
          return {
              'pos_tags': tags, 'sarcastic_pcts': sarcastic_pcts,'non_sarcastic_pcts': non_sarcastic_pcts
          }, sarc_counts, non_sarc_tags


    def sarcasm_imbalance(self):
        self.df_all["Sarcasm"] = self.df_all["Sarcasm"].astype(int)
        overall = self.df_all["Sarcasm"].value_counts(normalize=True) * 100
        per_variety = pd.crosstab(self.df_all["variety"], self.df_all["Sarcasm"], normalize="index") * 100

        self.plot_grouped_bar(
            df=self.df_all,
            column='variety',
            groupby='Sarcasm',
            title="Sarcasm Distribution by Variety",
            xlabel="Varieties",
            ylabel="Percentages (%)",
            save=True,
            filename="sarcasm_by_variety.png"
        )
        per_split = pd.crosstab([self.df_all["variety"], self.df_all["split"]],
                                self.df_all["Sarcasm"],
                                normalize="index") * 100

        return overall, per_variety, per_split

    def sarcastic_phrases_analysis(self):
        sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 1]['text']
        patterns = [
            'yeah right', 'oh great', 'wonderful', 'brilliant', 'thanks a lot',
            'as if', 'sure', 'of course', 'how nice', 'how lovely', 'well done',
            'good job', 'nice one', 'really?', 'seriously?', 'obviously',
            'tell me about it', 'big surprise', 'what a surprise', 'fantastic'
        ]
        matched = []
        pattern_counts = []

        for pattern in patterns:
            count = sarcastic_texts.str.lower().str.contains(pattern).sum()
            if count > 0:
                print(f"   '{pattern}': found in {count} sarcastic texts")
                matched.append(pattern)
                pattern_counts.append(count)

        examples_by_variety = {}
        for variety in ['en-AU', 'en-IN', 'en-UK']:
            examples = self.df_all[(self.df_all['variety'] == variety) & (self.df_all['Sarcasm'] == 1)]['text'].head(3).tolist()
            examples_by_variety[variety] = [ex[:120] + "..." if len(ex) > 120 else ex for ex in examples]

        return {
            'found_patterns': matched,
            'pattern_counts': pattern_counts
        }, examples_by_variety
        
    def sarcasm_by_source(self):
        tab = pd.crosstab(self.df_all["source"], self.df_all["Sarcasm"])
        return tab

    def sentiment_by_source(self):
        cb_table = pd.crosstab(
            self.df_all["source"], 
            self.df_all["Sentiment"])
        return cb_table

    def save_figure(self, save_path="./reports/figures", filename="plot.png"):
        folder = Path(save_path)
        if not folder.exists():
            folder.mkdir(parents=True)
        full_path = folder / filename
        plt.savefig(str(full_path), dpi=200, facecolor='white')
        print(f"Saved: {full_path}")

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
