from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process._gpr import _check_optimize_result
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from typing import Optional, Union
import pickle
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.model_selection import train_test_split, KFold
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances, pairwise_distances_argmin_min
from tqdm import tqdm 
import scipy.optimize


class MyGPR(GaussianProcessRegressor):
    '''
    Gaussian Process Regressor model

    :params: See GaussianProcessActiveLearner
    '''
    def __init__(self, alpha=1e-10, kernel=None, n_restarts_optimizer=0, random_state=None, copy_X_train=True, _max_iter=2e05):
        super().__init__(alpha=alpha, kernel=kernel, n_restarts_optimizer=n_restarts_optimizer, random_state=random_state, copy_X_train=copy_X_train)
        self._max_iter = _max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        if self.optimizer == "fmin_l_bfgs_b":
            opt_res = scipy.optimize.minimize(obj_func, initial_theta, method="L-BFGS-B", jac=True, bounds=bounds, options={'maxiter':self._max_iter})
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
        elif callable(self.optimizer):
            theta_opt, func_min = self.optimizer(obj_func, initial_theta, bounds=bounds)
        else:
            raise ValueError("Unknown optimizer %s." % self.optimizer)
        return theta_opt, func_min
        
        
class GaussianProcessActiveLearner:
    def __init__(self, n_initial=100, n_iter=10, max_samples=5000, max_samples_per_query=100, alpha=1e-10, kernel=None, n_restarts_optimizer=0, max_queries=100, strategy='std', random_state=42, max_iter=2000000):
        '''
        Gaussian Process Regressor model with Active Learning

        :param n_initial: Number of initial samples to start active learning
        :param n_iter: Number of times to repeat the AL training. If you want use all training subsets, n_iter=len(X_train)/max_samples
        :param max_samples: Maximum number of samples to consider
        :param max_samples_per_query: Maximum number of samples to query in each iteration
        :param alpha: Value added to the diagonal of the kernel matrix during fitting.
        :param kernel: Kernel specifying the covariance function of the GP.
        :param n_restarts_optimizer: The number of restarts of the optimizer for finding the kernel's parameters.
        :param max_queries: max queries that can be done in training, One query could return multiple new samples (depends on sampling strategy). 
        :param strategy: querying strategy among uncertainty 'std', entropy 'entropy',
        euclidian based diversity 'ebd', angle based diversity 'abd', cluster based diversity 'cbd',
        :param random_state: 
        :param _max_iter: max iter in GPR optimizer
        '''
        if kernel is None:
            # Use a default RBF kernel if not provided
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

        self.n_initial = n_initial
        self.n_iter = n_iter
        self.max_samples = max_samples
        self.max_samples_per_query = max_samples_per_query
        self.alpha = alpha
        self.kernel = kernel
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_queries = max_queries
        self.strategy = strategy
        self.random_state = random_state
        self.max_iter = max_iter
        
        # Create an active learner instance
        self.active_learner = None

    
    def get_sampling_strategy(self):
        ''' 
        Return right sampling startgey function
        '''
        if self.strategy=='std':
            return self.GP_regression_std
        if self.strategy=='ebd':
            return self.ebd
        if self.strategy=='abd':
            return self.abd
        if self.strategy=='cbd':
            return self.cbd
        if self.strategy=='entropy':
            return self.entropy
        else:
            raise Exception('Sampling strategy not valid')
        return


    def ebd(self, regressor, X_remaining: np.array) -> Union[int, np.array]:
        ''' 
        Euclidean based distance sampling strategy.
        Selects the samples in the candidate set that are distant 
        from the current training set using their squared Euclidean distance.

        :param X_remaining: Data points remaining to sample from
        '''
        # Get the current training set
        X_train = self.active_learner.X_training
        # Compute squared Euclidean distances between each training point and the query points
        distances = pairwise_distances(X_remaining, X_train, metric='l2')
        # Find the index of the maximum value in the entire distances array
        flat_query_idx = np.argmax(distances)
        # Convert the flat index to row and column indices
        query_idx_row, query_idx_col = np.unravel_index(flat_query_idx, distances.shape)
        return query_idx_row, X_remaining[query_idx_row]

    
    def abd(self, regressor, X_remaining: np.array) -> Union[int, np.array]:
        ''' 
        Cosine similarity based distance sampling strategy.
        Selects the samples in the candidate set that are distant 
        from the current training set using cosine similarity.

        :param X_remaining: Data points remaining to sample from
        '''
        # Get the current training set
        X_train = self.active_learner.X_training
        # Compute cosine similarity between each training point and the query points
        similarities = 1 - pairwise_distances(X_remaining, X_train, metric='cosine')
        # Find the index of the minimum similarity in the entire similarities array
        flat_query_idx = np.argmin(similarities)
        # Convert the flat index to row and column indices
        query_idx_row, query_idx_col = np.unravel_index(flat_query_idx, similarities.shape)
        return query_idx_row, X_remaining[query_idx_row]

    
    def cbd(self, regressor, X):
        '''
        Sampling strategy based on sampling a point from each cluster, 
        where the point is the nearest to the cluster centroid.
        
        :param X: Data points remaining to sample from. n_clusters points will be added
        :return: Tuple containing the index and the sample from each cluster
        '''
        # Apply k-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Initialize arrays to store the indices and samples
        query_indices = []
        query_samples = []

        # For each cluster, find the nearest sample to the cluster centroid
        for cluster_label in range(n_clusters):
            cluster_mask = (cluster_labels == cluster_label)
            cluster_points = X[cluster_mask]
            
            # Find the index and distance of the nearest sample to the cluster centroid
            nearest_index, _ = pairwise_distances_argmin_min([kmeans.cluster_centers_[cluster_label]], cluster_points)
            
            # Append the index and the corresponding sample to the arrays
            query_indices.append(nearest_index[0])
            query_samples.append(cluster_points[nearest_index[0]])

        # Return the tuple of indices and samples
        return np.array(query_indices), np.array(query_samples)

    
    def entropy(self, regressor, X):
        ''' 
        Sampling strategy based on sampling a point with the highest entropy
        '''
        # Get the predicted mean and standard deviation for all training points
        mean, std = regressor.predict(X, return_std=True)
        # Calculate the entropy of the predictive distribution
        entropies = entropy(np.vstack((mean, std)), axis=0)
        # Find the index with the highest entropy
        query_idx = np.argmax(entropies)

        return query_idx, X[query_idx]

    
    def GP_regression_std(self, regressor, X: np.array) -> Union[int, np.array]:
        ''' 
        Sampling strategy based on sampling point for which 
        regressor had biggest uncertainty (std)
        '''
        _, std = regressor.predict(X, return_std=True)
        query_idx = np.argmax(std)
        return query_idx, X[query_idx]


    def initialize_active_learner(self, X_initial: np.array, y_initial: np.array) -> None:
        ''' 
        Setup up the active learner

        :param X_initial: initial training data set
        :param y_initial: initial training labels
        '''

        # Initialize the Gaussian Process Regressor
        regressor = MyGPR(
            alpha=self.alpha,
            kernel=self.kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            copy_X_train=False,
            _max_iter=self.max_iter
        )

        # Initialize the active learner
        self.active_learner = ActiveLearner(
            estimator=regressor,
            X_training=X_initial.reshape(-1, X_initial.shape[1]),
            y_training=y_initial.reshape(-1, 1),
            query_strategy=self.get_sampling_strategy()
        )

        return



    
    def fit(self, X_train: np.array, y_train: np.array, X_test: np.array, y_test: np.array) -> None:
        '''
        Fit the model with active learning.
        The data will be split in len(X_train)/max_samples subsets.
        Ideally n_iter*max_samples = len(X_train). This means that all data subsets will be used once. If not, only part of the subsets are used for training.
        The GPR will have access to an initial set of size self.n_initial and can query sel.max_samples_per_query until reaching self.max_samples in total.
        The RMSE is computed on the remaining points.
        This is repeated self.n_iter times and the avegrage score is kept.

        :param X_train: Training data
        :param y_train: Training labels
        :param X_test: Tests data
        :param y_test: Test labels
        '''
        # Seperate train data in self.n_iter subsets -> random subset
        kf = KFold(n_splits=len(X_train)//self.max_samples, shuffle=True, random_state=self.random_state)

        rmse_scores = []
        avg_std = []
        for _ in tqdm(range(self.n_iter), desc='Iteration'):
            _, subset_index = next(kf.split(X_train))

            # Get the current subset to work on
            X_train_sub = X_train[subset_index][:self.max_samples] # trim if necessary
            y_train_sub = y_train[subset_index][:self.max_samples]

            # Split the training subset into an initial pool and remaining samples
            X_initial, X_remaining, y_initial, y_remaining = train_test_split(
                X_train_sub, y_train_sub, train_size=self.n_initial, random_state=self.random_state
            )

            self.initialize_active_learner(X_initial=X_initial, y_initial=y_initial)
            # Perform active learning until reaching self.max_samples in total
            samples_queried = self.n_initial
            sampling_iter = 0
            self.best_model = None
            best_rmse = np.inf

            while samples_queried < self.max_samples and sampling_iter < self.max_queries: #or stop when rmse good enough
                # Query new samples based on uncertainty
                query_idx, query_inst = self.active_learner.query(X_remaining)
                sampling_iter += 1

                # Limit the number of samples queried in each iteration if more than 1 idx
                if not np.isscalar(query_idx):
                    query_idx = query_idx[:self.max_samples_per_query]
                    query_inst = query_inst[:self.max_samples_per_query]

                # Label the queried samples
                query_labels = y_remaining[query_idx]
                if np.isscalar(query_labels):
                    query_labels = np.array(query_labels).reshape(-1,1) # num samples x 1
                    query_inst = query_inst.reshape(len(query_labels), query_inst.shape[0]) # num samples x num features
                else:
                    query_labels = query_labels.reshape(-1,1)

                # Update the active learner with the new samples
                self.active_learner.teach(query_inst, query_labels)

                # Move the newly labeled samples from the remaining pool to the training pool
                X_remaining = np.delete(X_remaining, query_idx, axis=0)
                y_remaining = np.delete(y_remaining, query_idx, axis=0)

                # Update the count of samples queried
                samples_queried += len(query_idx) if not np.isscalar(query_idx) else 1

            # Print the final model performance for each fold
            y_pred, y_std = self.active_learner.predict(X_test, return_std=True) 
            iter_rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_scores.append(iter_rmse)
            avg_std.append(np.mean(y_std))

            # Update the best model
            if iter_rmse < best_rmse:
                self.best_model = self.active_learner
                best_rmse = iter_rmse

        # Print the average performance over all folds
        avg_rmse = np.mean(rmse_scores)
        print(f'Avg Test RMSE: {avg_rmse} \nBest test RMSE: {best_rmse}')
        print(f'Avg prediction uncertainty (std): {np.mean(avg_std)}')
        
        return


    def save(self, model, model_filename: str) -> None:
        ''' 
        Save trained model

        :param model: trained model
        :param model_filename: path to save model as .pkl file
        '''
        # Save the trained model to a file using pickle
        with open(model_filename, 'wb') as file:
            pickle.dump(model, file)
        print(f'Trained model saved to {model_filename}')