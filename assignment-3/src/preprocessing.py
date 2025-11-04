import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



def clean_data(df):
    df = df.dropna(subset=['ID', 'class'])
    df['ID'] = df['ID'].str.replace('id_', '').astype(int)
    df['class'] = df['class'].map({'H': 0, 'P': 1})
    return df


def standardize_data(df):
    scaler = StandardScaler()
    feature_cols = [col for col in df.columns if col not in ['ID', 'class']]
    scaled = scaler.fit_transform(df[feature_cols])
    df_scaled = pd.DataFrame(scaled, columns=feature_cols, index=df.index)
    df_scaled['ID'] = df['ID']
    df_scaled['class'] = df['class']
    df_scaled = df_scaled[['ID'] + feature_cols + ['class']]

    return df_scaled


def pca_reduction(df):
    pca = PCA(n_components=0.95)
    df_PCAed = pca.fit_transform(df.drop(columns=['class']))
    df_PCAed = pd.DataFrame(df_PCAed)
    df_PCAed['class'] = df['class'].values

    return df_PCAed


if __name__ == "__main__":
    dp = pd.read_csv("..\\data\\data.csv")
    if not dp.empty:
        rows = len(dp)
        print("------------Data before preprocessing------------")
        print(dp.dtypes)
        dp = clean_data(dp)

        dp = standardize_data(dp)
        pca = pca_reduction(dp)
        print("------------Data after preprocessing------------")
        print(dp.dtypes)
        print("Percentage of data remaining after preprocessing = ", len(dp) / rows * 100, "%")

        dp.to_csv("..\\data\\dataProcessed.csv", index=False)
        pca.to_csv("..\\data\\dataPCAed.csv", index=False)
