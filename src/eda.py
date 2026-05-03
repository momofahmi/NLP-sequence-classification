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
    def __init__(self, df_all, df_train, df_validation, df_test):
        self.df_all_ = df_all
        self.df_train = df_train
        self.df_validation = df_validation
        self.df_test = df_test
        self.df_all = pd.concat([df_train, df_validation, df_test])
        os.makedirs("./reports/figures", exist_ok=True)
        os.makedirs("./reports", exist_ok=True)

    def save_figure(self, save_path="./reports/figures", filename="plot.png"):
        os.makedirs(save_path, exist_ok=True)
        path = os.path.join(save_path, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Figure saved to: {path}")

    #Description: draws plots of multiple types whenever needed.
    def plot_distribution(self, df, column, groupby=None, title=None, xlabel=None, ylabel=None,
                          plot_type='countplot', save=False, save_path="./reports/figures",
                          filename=None, **kwargs):

        plt.figure(figsize=(8, 5))

        if plot_type == 'stacked_bar':
            if groupby is None:
                raise ValueError("groupby parameter required.")
            table = pd.crosstab(df[column], df[groupby])
            ax = table.plot(kind='bar', stacked=True,
                              color=['#4285F4', '#EA4335'],
                              edgecolor='black')

            for i, idx in enumerate(table.index):
                total = table.loc[idx].sum()
                ax.text(i, total + 5, f'Total: {total}',
                       ha='center', va='bottom', fontsize=9)

        elif plot_type == 'grouped_bar':
            if groupby is None:
                raise ValueError("groupby parameter is required for grouped_bar")
            table = pd.crosstab(df[column], df[groupby])
            percentages = table.div(table.sum(axis=1), axis=0) * 100

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
            
        elif plot_type == 'heatmap':
            if groupby is None:
                raise ValueError("groupby parameter is required for heatmap")
            table = pd.crosstab(df[column], df[groupby])
            ax = sns.heatmap(table, annot=True, fmt='d', cmap='Blues',
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

        if filename is None:
            filename = f"{column.lower()}_distribution.png"

        plt.tight_layout()

        if save:
            self.save_figure(save_path, filename)

        plt.show()
        return plt.gcf()

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

    def split_distribution_per_variety(self, save=False, save_path="./reports/figures"):
        split_series = pd.Series(
            ["train"] * len(self.df_train) +
            ["validation"] * len(self.df_validation) +
            ["test"] * len(self.df_test),
            name="split"
        )
        table = pd.crosstab(self.df_all["variety"], split_series)
        data = {
            'en-AU': {
                'train': table.loc['en-AU', 'train'],
                'validation': table.loc['en-AU', 'validation'],
                'test': table.loc['en-AU', 'test']
            },
            'en-IN': {
                'train': table.loc['en-IN', 'train'],
                'validation': table.loc['en-IN', 'validation'],
                'test': table.loc['en-IN', 'test']
            },
            'en-UK': {
                'train': table.loc['en-UK', 'train'],
                'validation': table.loc['en-UK', 'validation'],
                'test': table.loc['en-UK', 'test']
            }
        }
        english_varieties = list(data.keys())
        data_splits = ['train', 'validation', 'test']
        
        x = np.arange(len(english_varieties))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, split in enumerate(data_splits):
            values = [data[v][split] for v in english_varieties]
            bars = ax.bar(x + i * width, values, width, label=split.capitalize())
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Variety', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Split Distribution by Variety', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(english_varieties)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        if save:
            self.save_figure(save_path, "split_distribution.png")
        plt.show()
        return table

    def source_per_variety(self):
        table = pd.crosstab(self.df_all["variety"], self.df_all["source"])
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

        return table
        
    def sarcasm_sentiment_correlation(self):
        table = pd.crosstab(self.df_train["Sarcasm"], self.df_train["Sentiment"])
        self.plot_distribution(
            df=self.df_train,
            column='Sarcasm',
            groupby='Sentiment',
            title="Sarcasm vs Sentiment Correlation\n(Red box shows sarcastic-negative texts)",
            xlabel="Sentiment",
            ylabel="Sarcasm",
            plot_type='heatmap',
            save=True,
            highlight=(1, 0) if 1 in table.index and 0 in table.columns else None,
            filename="sarcasm_sentiment_correlation.png"
        )
        return table

    def sarcasm_imbalance(self):
        self.df_all["Sarcasm"] = self.df_all["Sarcasm"].astype(int)
        overall = self.df_all["Sarcasm"].value_counts(normalize=True) * 100
        per_variety = pd.crosstab(self.df_all["variety"], self.df_all["Sarcasm"], normalize="index") * 100

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
        per_split = pd.crosstab([self.df_all["variety"], self.df_all["split"]],
                                self.df_all["Sarcasm"],
                                normalize="index") * 100

        return overall, per_variety, per_split
        
    def sentiment_imbalance(self):
        self.df_all["Sarcasm"] = self.df_all["Sentiment"].astype(int)
        overall = self.df_all["Sentiment"].value_counts(normalize=True) * 100
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

        per_split = pd.crosstab([self.df_all["variety"], self.df_all["split"]],
                                self.df_all["Sentiment"],
                                normalize="index") * 100
        return overall, per_variety, per_split

    def pos_for_sarcasm(self, n_samples=500):
        sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 1]['text'].sample(
            min(n_samples, len(self.df_all[self.df_all['Sarcasm']==1]))
        ).tolist()
        
        non_sarcastic_texts = self.df_all[self.df_all['Sarcasm'] == 0]['text'].sample(
            min(n_samples, len(self.df_all[self.df_all['Sarcasm']==0]))
        ).tolist()
        
        sarcastic_pos_tags = {}
        non_sarcastic_pos_tags = {}
        for text in sarcastic_texts:
            doc = nlp(text)
            for token in doc:
                sarcastic_pos_tags[token.pos_] = sarcastic_pos_tags.get(token.pos_, 0) + 1

        for text in non_sarcastic_texts:
            doc = nlp(text)
            for token in doc:
                non_sarcastic_pos_tags[token.pos_] = non_sarcastic_pos_tags.get(token.pos_, 0) + 1
        
        total_sarcastic_pos_tags = sum(sarcastic_pos_tags.values())
        total_non_sarcastic_pos_tags = sum(non_sarcastic_pos_tags.values())
        pos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ', 'PRON', 'ADP']
        sarcastic_pcts = [(sarcastic_pos_tags.get(pos, 0) / total_sarcastic_pos_tags) * 100 for pos in pos_tags]
        non_sarcastic_pcts = [(non_sarcastic_pos_tags.get(pos, 0) / total_non_sarcastic_pos_tags) * 100 for pos in pos_tags]
    
        return {
            'pos_tags': pos_tags,
            'sarcastic_pcts': sarcastic_pcts,
            'non_sarcastic_pcts': non_sarcastic_pcts
        }, sarcastic_pos_tags, non_sarcastic_pos_tags
    
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
