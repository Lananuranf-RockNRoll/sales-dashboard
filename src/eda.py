import matplotlib.pyplot as plt
import seaborn as sns

def get_basic_metrics(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "revenue_rate": round(df["Revenue"].mean() * 100, 2)
    }

def plot_revenue_distribution(df):
    fig, ax = plt.subplots()
    df["Revenue"].value_counts().plot(kind="bar", ax=ax)
    return fig

def plot_numeric_distribution(df, column):
    fig, ax = plt.subplots()
    sns.histplot(df[column], kde=True, ax=ax)
    return fig

def plot_correlation(df):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", ax=ax)
    return fig

def plot_categorical_distribution(df, column):
    fig, ax = plt.subplots()
    df[column].value_counts().plot(kind="bar", ax=ax)
    return fig