# Import the modules
import pandas as pd
import numpy as np
import statsmodels.api as sm

from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional


class ModelPreprocessor:
    def __init__(self, df: pd.DataFrame, target_col: str, random_state=1):
        self.df = df
        self.index_col: str = df.index.name
        self.target_col: str = target_col
        self.random_state: int = random_state

    def remove_cols_with_mostly_same_values(self, freq_value_threshold=0.9) -> pd.DataFrame:
        self.df = self.df.loc[:, self.df.apply(
            lambda col: col.value_counts(normalize=True).max() < freq_value_threshold
        )]
        return self.df

    def get_x_y(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df.copy().drop(self.target_col, axis=1), df[self.target_col]

    def get_x_y_reshaped(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        return df.copy().drop(self.target_col, axis=1), df[self.target_col].reshape(-1, 1)

    def train_test_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X, y = self.get_x_y(df)
        return train_test_split(X, y, random_state=self.random_state)

    def get_numeric_columns(self, df: Optional[pd.DataFrame] = None) -> list:
        if df is None:
            df = self.df
        return list(df.select_dtypes(include='number').columns)

    def get_categorical_columns(self, df: Optional[pd.DataFrame] = None) -> list:
        if df is None:
            df = self.df
        return list(df.select_dtypes(include=['object', 'category']).columns)

    def scale_numeric_columns(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        numeric_cols = self.get_numeric_columns(df.drop(self.target_col, axis=1))
        scaled_vals = StandardScaler().fit_transform(df[numeric_cols])
        # Create a DataFrame from the scaled data
        return pd.DataFrame(scaled_vals, columns=numeric_cols)

    def scale_numeric_columns_with_idx(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        index_col: str = df.index.name
        numeric_cols = self.get_numeric_columns(df.drop(self.target_col, axis=1))
        scaled_vals = StandardScaler().fit_transform(df[numeric_cols])
        # Create a DataFrame from the scaled data
        scaled_df = pd.DataFrame(scaled_vals, columns=numeric_cols)
        # Recreate the index if needed
        if index_col is not None:
            # Recreate the index
            scaled_df[index_col] = df.index
            scaled_df = scaled_df.set_index(index_col)
        return scaled_df

    def encode_categorical_columns_ohe(self, df: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        if df is None:
            df = self.df
        encode_cols = self.get_categorical_columns(df.drop(self.target_col, axis=1))
        if encode_cols is None or len(encode_cols) == 0:
            return None
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.set_output(transform="pandas")
        return ohe.fit_transform(self.df[encode_cols])

    def get_scaled_and_encoded_df(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        scaled_df = self.scale_numeric_columns(df)
        encoded_df = self.encode_categorical_columns_ohe(df)
        if encoded_df is None:
            return scaled_df
        return pd.concat([scaled_df, encoded_df, df[self.target_col]], axis=1)

    def get_encoded_and_scaled_df(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if df is None:
            df = self.df
        sae_df = self.get_scaled_and_encoded_df(df)
        return self.scale_numeric_columns(sae_df)

    def get_pd_corr_abs_numeric_features_df(self,
                                            scaled_df: pd.DataFrame) -> pd.DataFrame:
        X, y = self.get_x_y(scaled_df.dropna())
        corr_data_abs = X.corrwith(other=y).abs()
        corr_coeff_abs_df = pd.DataFrame(
            corr_data_abs,
            columns=['correlation']
        )
        return corr_coeff_abs_df

    def get_ridge_coeff_abs_numeric_features_df(self,
                                                scaled_df: pd.DataFrame) -> pd.DataFrame:
        X, y = self.get_x_y(scaled_df.dropna())
        ridge = RidgeCV(alphas=np.logspace(-6, 6, num=7)).fit(X, y)
        print(f"Best alpha: {ridge.alpha_} (R^2 score: {ridge.score(X, y): .2f})")
        importance: np.ndarray = np.abs(ridge.coef_)
        importance /= importance.max()
        coefs_df = pd.DataFrame(
            importance,
            columns=['coefficient'],
            index=X.columns,
        )
        return coefs_df

    def get_lr_pvals_numeric_features_df(self,
                                         scaled_df: pd.DataFrame) -> pd.DataFrame:
        # Looking forp-values < 0.05 which are statistically signicant so subtract from 1 to get values to add to total
        X_train, X_test, y_train, y_test = self.train_test_split(scaled_df.dropna())
        lr = sm.OLS(y_train, X_train).fit()
        pvals = round(lr.pvalues, 6).values
        pvals_inv = 1. - pvals
        pvals_df = pd.DataFrame(
            pvals_inv,
            columns=['pval_inv'],
            index=lr.pvalues.index,
        )
        return pvals_df

    def get_vif_numeric_features_df(self, scaled_df: pd.DataFrame) -> pd.DataFrame:
        X, y = self.get_x_y(scaled_df.dropna())
        vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_inv = [1 / v for v in vif]
        vif_df = pd.DataFrame()
        vif_df["variables"] = X.columns
        vif_df["vif"] = vif
        vif_df["vif_inv"] = vif_inv
        return vif_df.set_index("variables")

    def get_combined_important_numeric_features_df(self,
                                                   scaled_df: pd.DataFrame) -> pd.DataFrame:
        corr_df = self.get_pd_corr_abs_numeric_features_df(scaled_df)
        coef_df = self.get_ridge_coeff_abs_numeric_features_df(scaled_df)
        pvals_df = self.get_lr_pvals_numeric_features_df(scaled_df)
        vif_df = self.get_vif_numeric_features_df(scaled_df)
        combo_df = pd.concat([corr_df, coef_df, pvals_df, vif_df], axis=1)
        
        # combo_df = corr_df.merge(coef_df, left_index=True, right_index=True)
        # combo_df = combo_df.merge(pvals_df, left_index=True, right_index=True)
        # combo_df = self.scale_numeric_columns(combo_df)
        combo_df['total'] = combo_df['correlation'] + combo_df['coefficient'] + combo_df['pval_inv']
        return combo_df.sort_values(by='total', ascending=False)

    def get_top_total_numeric_features_list(self,
                                            scaled_df: pd.DataFrame,
                                            total_min: float) -> list:
        combo_df = self.get_combined_important_numeric_features_df(scaled_df)
        return list(combo_df[combo_df['total'] > total_min].index)

    def plot_feature_selection(self, df: pd.DataFrame) -> None:
        df.plot(
            kind='bar',
            title='Feature selection',
            legend=True,
            xlabel='Feature',
            ylabel='Values')

    def get_pd_corr_abs_encoded_features_df(self, encoded_df: pd.DataFrame) -> pd.DataFrame:
        abs_correlation_matrix = encoded_df.corrwith(
            encoded_df[self.target_col]
        ).abs().sort_values(ascending=False)
        abs_correlation_matrix_df = pd.DataFrame(
            abs_correlation_matrix,
            columns=['correlation']
        ).drop(self.target_col)
        return abs_correlation_matrix_df

    def get_top_corr_abs_encoded_features_list(self, encoded_df: pd.DataFrame, corr_min: float) -> list:
        correlation_matrix_abs_df = self.get_pd_corr_abs_encoded_features_df(encoded_df)
        top_corr_encoded_features = list(
            correlation_matrix_abs_df[correlation_matrix_abs_df['correlation'] > corr_min].index
        )
        return list(set([item.split('_')[0] for item in top_corr_encoded_features]))

    def get_top_feature_cols(self,
                             scaled_df: pd.DataFrame,
                             total_min: float,
                             encoded_df: pd.DataFrame,
                             corr_min: float) -> list:
        top_feature_cols: list[str] = self.get_top_total_numeric_features_list(scaled_df, total_min)
        top_feature_cols.extend(self.get_top_corr_abs_encoded_features_list(encoded_df, corr_min))
        return top_feature_cols

    def get_random_forest_important_features_df(self, scaled_df: pd.DataFrame, n_estimators: int) -> pd.DataFrame:
        # Train and evaluate the Random Forest model
        X, y = self.get_x_y(scaled_df)
        clf = RandomForestClassifier(random_state=self.random_state, n_estimators=n_estimators)
        clf.fit(X, y)
        score = clf.score(X, y)
        acc_score = accuracy_score(y, clf.predict(self.X))
        print(f'RF: training Score: {score}, accuracy score: {acc_score}')
        rf_imp_df = pd.DataFrame(
            clf.feature_importances_,
            index=X.columns, columns=['rf_importance']
        ).sort_values('rf_importance', ascending=False)
        return rf_imp_df

    def get_elbow_df(self, k_start: int, k_end: int) -> pd.DataFrame:
        # Create a dictionary that holds the list of values for k and inertia
        elbow_data = {
            'k': list(range(k_start, k_end)),
            'inertia': []
        }

        # Create a for-loop where the interia for each value of k is evaluated using the K-Means model
        for k in elbow_data['k']:
            k_model = KMeans(n_clusters=k, n_init='auto',
                             random_state=self.random_state)
            k_model.fit(self.df)
            elbow_data['inertia'].append(k_model.inertia_)

        # Convert the dictionary to a DataFrame
        return pd.DataFrame(elbow_data)

    def plot_elbow_curve(self, elbow_df: pd.DataFrame) -> None:
        # Plot the elbow curve
        axs: Axes = elbow_df.plot.line(x='k',
                                       y='inertia',
                                       title='Elbow Curve',
                                       xticks=elbow_df['k'].values)
        # Add the best k value
        x = self.get_best_k(elbow_df)
        y = elbow_df['inertia'][x-1]
        axs.plot(x, y, 'ro')
        axs.annotate('Best k', xy=(x, y), xytext=(x+0.15, y+0.15))

    def get_best_k(self, elbow_df: pd.DataFrame, print_details=False) -> int:
        """
        Calculates the optimal value of k for K-Means Clustering Model based on the Elbow Curve.
        Note: This is a best guess based on the first k value where the inertia decreases less than 6%
        or hits 90% or greater of the total inertia range.

        Args:
            elbow_df (pd.DataFrame): The Elbow Curve DataFrame to be evaluated.

        Returns:
            int: The optimal value of k.
        """
        best_k = 0
        inertia_range = elbow_df['inertia'].max() - elbow_df['inertia'].min()
        pct_decrease_total = 0
        for i in range(1, len(elbow_df['k'])):
            pct_decrease = (elbow_df['inertia'][i-1] -
                            elbow_df['inertia'][i]) / inertia_range * 100
            pct_decrease_total += pct_decrease
            if best_k == 0:
                if pct_decrease_total > 90:
                    best_k = elbow_df['k'][i]
                if pct_decrease < 6:
                    best_k = elbow_df['k'][i-1]
            if print_details:
                print(f"""Pct decrease from k={elbow_df['k'][i-1]} to k={elbow_df['k'][i]}: {pct_decrease: .2f} %, \
                      Total decrease: {pct_decrease_total: .2f} %""")
        if print_details:
            print(f"Best k: {best_k}")
        return best_k

    def print_best_k_details(self, elbow_df: pd.DataFrame) -> None:
        self.get_best_k(elbow_df, print_details=True)


class JgKMeans(ModelPreprocessor):

    def __init__(self, df: pd.DataFrame, random_state=1):
        super().__init__(df, random_state)
        self.model = None

    def get_model(self) -> KMeans:
        return self.model

    def fit(self, k: int) -> None:
        self.model = KMeans(n_clusters=k, n_init='auto',
                            random_state=self.random_state)
        self.model.fit(self.df)

    def predict(self, predict_col_name: str) -> pd.DataFrame:
        # Create a copy of the DataFrame and add the new predictions column
        predicted_df = self.df.copy()
        predicted_df[predict_col_name] = self.model.predict(self.df)
        return predicted_df

    def fit_and_predict(self, k: int, predict_col_name: str) -> pd.DataFrame:
        self.fit(k)
        return self.predict(predict_col_name)

    def get_centroids(self) -> pd.DataFrame:
        return self.model.cluster_centers_


class JgPCA(JgKMeans):

    def __init__(self, n_components: int, original_df: pd.DataFrame, random_state=1):
        self.n_components = n_components
        self.original_df = original_df
        self.index_col = original_df.index.name
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.pca_cols = []
        super().__init__(self._get_pca_df(), random_state)

    def get_pca(self) -> PCA:
        return self.pca

    def get_pca_component_weights_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.pca.components_.T, columns=self.pca_cols, index=self.original_df.columns)

    def _get_pca_df(self) -> pd.DataFrame:
        self.pca_cols = []
        for n in range(1, self.n_components + 1):
            self.pca_cols.append(f"PCA{n}")
        pca_data = self.pca.fit_transform(self.original_df)
        pca_df = pd.DataFrame(
            pca_data,
            columns=self.pca_cols
        )
        pca_df[self.index_col] = self.original_df.index
        pca_df = pca_df.set_index(self.index_col)
        return pca_df
