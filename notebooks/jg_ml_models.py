# Import the modules
import pandas as pd
from matplotlib.axes import Axes
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ModelPreprocessor:
    def __init__(self, df: pd.DataFrame, random_state=1):
        self.df = df
        self.index_col = df.index.name
        self.random_state = random_state

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_random_state(self) -> int:
        return self.random_state

    def set_random_state(self, random_state: int) -> None:
        self.random_state = random_state

    def get_numeric_columns(self) -> list:
        return list(self.df.select_dtypes(include='number').columns)

    def get_correlation_matrix_abs_numeric_features_df(self,
                                                       numeric_cols: list,
                                                       target_col: str) -> pd.DataFrame:
        scaled_df = self.scale_numeric_columns(numeric_cols)
        correlation_matrix_abs = scaled_df.corrwith(scaled_df[target_col]).abs().sort_values(ascending=False)
        correlation_matrix_abs_df = pd.DataFrame(correlation_matrix_abs, columns=['correlation']).drop(target_col)
        return correlation_matrix_abs_df

    def get_top_correlated_abs_numeric_features_list(self,
                                                     numeric_cols: list,
                                                     target_col: str,
                                                     corr_min: float) -> list:
        correlation_matrix_abs_df = self.get_correlation_matrix_abs_numeric_features_df(numeric_cols, target_col)
        return list(correlation_matrix_abs_df[correlation_matrix_abs_df['correlation'] > corr_min].index)

    def get_correlation_matrix_abs_encoded_features_df(self, encode_cols: list, target_col: str) -> pd.DataFrame:
        encoded_df = self.encode_string_columns_ohe(encode_cols, target_col)
        abs_correlation_matrix = encoded_df.corrwith(encoded_df[target_col]).abs().sort_values(ascending=False)
        abs_correlation_matrix_df = pd.DataFrame(abs_correlation_matrix, columns=['correlation']).drop(target_col)
        return abs_correlation_matrix_df

    def get_top_correlated_abs_encoded_features_list(self, encode_cols: list, target_col: str, corr_min: float) -> list:
        correlation_matrix_abs_df = self.get_correlation_matrix_abs_encoded_features_df(encode_cols, target_col)
        top_corr_encoded_features = list(
            correlation_matrix_abs_df[correlation_matrix_abs_df['correlation'] > corr_min].index
        )
        return list(set([item.split('_')[0] for item in top_corr_encoded_features]))

    def get_top_feature_cols(self,
                             numeric_cols: list,
                             corr_min_1: float,
                             encode_cols: list,
                             corr_min_2: float,
                             target_col: str) -> list:
        top_feature_cols: list[str] = self.get_top_correlated_abs_numeric_features_list(
            numeric_cols, target_col, corr_min_1
        )
        top_feature_cols.extend(self.get_top_correlated_abs_encoded_features_list(
            encode_cols=encode_cols, target_col=target_col, corr_min=corr_min_2)
        )
        return top_feature_cols

    def get_non_numeric_columns(self) -> list:
        return list(self.df.select_dtypes(exclude='number').columns)

    def get_columns_to_encode(self) -> list:
        return list(self.df.select_dtypes(include='object').columns)

    def scale_numeric_columns(self, numeric_cols: list) -> pd.DataFrame:
        # Standardize the data
        scaled_vals = StandardScaler().fit_transform(self.df[numeric_cols])
        # Create a DataFrame from the scaled data
        scaled_df = pd.DataFrame(scaled_vals, columns=numeric_cols)
        # Recreate the index if needed
        if self.index_col is not None:
            # Recreate the index
            scaled_df[self.index_col] = self.df.index
            scaled_df = scaled_df.set_index(self.index_col)
        return scaled_df

    def encode_string_columns(self, non_numeric_cols: list) -> pd.DataFrame:
        # Return a DataFrame with the encoded data as new columns
        return pd.get_dummies(self.df[non_numeric_cols])

    def encode_string_columns_ohe(self, encode_cols: list, target_col: str) -> pd.DataFrame:
        # Return a DataFrame with the encoded data as new columns
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        ohe.set_output(transform="pandas")
        encoded_df = ohe.fit_transform(self.df[encode_cols])
        encoded_df[target_col] = self.df[target_col]
        return encoded_df

    def get_scaled_and_encoded_df(self, scale_cols: list, encode_cols: list) -> pd.DataFrame:
        scaled_df = self.scale_numeric_columns(scale_cols)
        if encode_cols is None or len(encode_cols) == 0:
            return scaled_df
        encoded_df = self.encode_string_columns(encode_cols)
        return pd.concat([scaled_df, encoded_df], axis=1)

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
            pct_decrease = (elbow_df['inertia'][i-1] - elbow_df['inertia'][i]) / inertia_range * 100
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
        self.model = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state)
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
