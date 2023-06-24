import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator
from sklearn.metrics import silhouette_score
from tqdm.notebook import tqdm

class SparseJumpModel(BaseEstimator):
    """
    SparseJumpModel class implements the Sparse Jump method for unsupervised learning. 
    It extends the BaseEstimator class from sklearn.base module to leverage scikit-learn's 
    estimator interface for compatibility with its tools.
    """
    
    def __init__(self, n_states=2, max_features=10, jump_penalty=1e-5,
                 max_iter=10, tol=1e-4, n_init=10, verbose=False):
        """
        Initializes an instance of the SparseJumpModel class.
        
        Parameters:
        - n_states (int): The number of states.
        - max_features (int): The maximum number of features to consider.
        - jump_penalty (float): The penalty for jumping between states.
        - max_iter (int): The maximum number of iterations to run.
        - tol (float): The tolerance for stopping criteria.
        - n_init (int): The number of initializations to consider.
        - verbose (bool): Whether to print verbose output.
        """
        
        self.n_states = n_states
        self.max_features = max_features
        self.jump_penalty = jump_penalty
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.mu = None
        self.states_ = None
        self.feat_w_ = None

    def sparse_jump(self, Y, n_states, max_features, jump_penalty=1e-5,
                max_iter=100, tol=1e-4, n_init=10, verbose=False):
        '''
        Implementation of sparse jump model
        '''
        
        n_obs, n_features = Y.shape
        max_features = np.clip(max_features, a_min=1, a_max=np.sqrt(n_features))
        feat_w = np.repeat(1 / np.sqrt(n_features), n_features)
        states = None

        for it in range(max_iter):
            states = self.jump(Y * np.sqrt(feat_w + 1e-10),
                          n_states,
                          initial_states=states,
                          jump_penalty=jump_penalty,
                          n_init=n_init)
            if len(np.unique(states)) == 1:
                break
            else:
                new_w = self.get_weights(Y, states, max_features, n_states)
            if abs(new_w - feat_w).sum() / abs(feat_w).sum() < tol:
                break
            elif verbose:
                print('Iteration {}, w diff {:.6e}'.format(it, abs(new_w - feat_w).sum()))
            feat_w = new_w

        return states, feat_w
    
    def jump(self, Y, n_states, jump_penalty=1e-5, initial_states=None,
         max_iter=10, n_init=10, tol=None, verbose=False):
        '''
        # Fit jump model using framework of Bemporad et al. (2018)
        '''
        
        if initial_states is not None:
            initial_states = np.array(initial_states, dtype=np.int64)
            if len(np.unique(initial_states)) == n_states:
                s = initial_states.copy()
            else:
                s = self.init_states(Y, n_states)
        else:
            s = self.init_states(Y, n_states)

        n_obs, n_features = Y.shape
        Gamma = jump_penalty * (1 - np.eye(n_states)) 
        best_loss = None
        best_s = None

        for init in range(n_init):
            mu = np.zeros((n_states, n_features))
            loss_old = 1e10
            for it in range(max_iter):
                # Fit model by updating mean of observed states
                for i in np.unique(s):
                    mu[i] = np.mean(Y[s==i], axis=0)
                # Fit state sequence
                s_old = s.copy()
                loss_by_state = cdist(mu, Y, 'euclidean').T**2
                V = loss_by_state.copy()
                for t in range(n_obs-1, 0, -1):
                    V[t-1] = loss_by_state[t-1] + (V[t] + Gamma).min(axis=1)
                s[0] = V[0].argmin()
                for t in range(1, n_obs):
                    s[t] = (Gamma[s[t-1]] + V[t]).argmin()
                # Monitor convergence
                if len(np.unique(s)) == 1:
                    break
                loss = min(V[0])
                if verbose:
                    print('Iteration {}: {:.6e}'.format(it, loss))
                if tol:
                    epsilon = loss_old - loss 
                    if epsilon < tol:
                        break
                elif np.array_equal(s, s_old):
                    break
                loss_old = loss

            if (best_s is None) or (loss_old < best_loss):
                best_loss = loss_old
                best_s = s.copy()
            s = self.init_states(Y, n_states)

        return best_s

    def init_states(self, Y, n_states):
        '''
        # Generate initial states using K-means++ (Arthur and Vassilvitskii, 2007)
        '''
        
        n_obs, n_features = Y.shape
        centers = np.zeros((n_states, n_features))
        center_idx = np.random.randint(n_obs)
        centers[0] = Y.iloc[center_idx]
        n_local_trials = 2 + int(np.log(n_states))
        closest_dist_sq = cdist(centers[0, None], Y, 'euclidean')**2
        current_pot = closest_dist_sq.sum()

        for i in range(1, n_states):
            rand_vals = np.random.sample(n_local_trials) * current_pot
            candidate_ids = np.searchsorted(np.cumsum(closest_dist_sq), 
                                            rand_vals)
            distance_to_candidates = cdist(Y.iloc[candidate_ids], Y, 'euclidean')**2
            # Decide which candidate is the best
            best_candidate = None
            best_pot = None
            best_dist_sq = None
            for trial in range(n_local_trials):
                # Compute potential when including center candidate
                new_dist_sq = np.minimum(closest_dist_sq,
                                         distance_to_candidates[trial])
                new_pot = new_dist_sq.sum()

                # Store result if it is the best local trial so far
                if (best_candidate is None) or (new_pot < best_pot):
                    best_candidate = candidate_ids[trial]
                    best_pot = new_pot
                    best_dist_sq = new_dist_sq

            centers[i] = Y.iloc[best_candidate]
            current_pot = best_pot
            closest_dist_sq = best_dist_sq

        # Compute the state assignment
        states = cdist(centers, Y, 'euclidean').argmin(axis=0)

        return states

    def get_weights(self, Y, states, max_features, n_states):
        '''
        Find weights given a state sequence by maximizing the interstate distance
        '''
        
        BCSS = self.get_BCSS(Y, states)
        delta = self.binary_search(BCSS, max_features)
        w = self.calc_new_feature_weights(BCSS, delta)

        return w


    def get_BCSS(self, Y, states):
        '''
        Find BCSS given a state sequence
        '''
        
        WCSS = np.zeros(Y.shape[1])
        for i in np.unique(states):
            mask = (states == i)
            if mask.sum() > 1:
                WCSS += np.square(Y[mask] - np.mean(Y[mask], axis=0)).sum(axis=0)
        TSS = np.square(Y - np.mean(Y, axis=0)).sum(axis=0)

        return (TSS - WCSS) + 1e-10



    def binary_search(self, objective, norm_constraint, max_iter=15):
        """
        Performs binary search for finding the best trade-off between sparsity and performance.
        """
        
        l2n_arg = np.linalg.norm(objective)
        if l2n_arg == 0 or abs(objective / l2n_arg).sum() <= norm_constraint:
            return 0
        lam1 = 0
        lam2 = abs(objective).max() - 1e-5
        for iter in range(max_iter):
            su = self.soft_threshold(objective, (lam1 + lam2) / 2)
            if abs(su / np.linalg.norm(su)).sum() < norm_constraint:
                lam2 = (lam1 + lam2) / 2
            else:
                lam1 = (lam1 + lam2) / 2
            if (lam2 - lam1) < 1e-4:
                break

        return (lam1 + lam2) / 2


    def calc_new_feature_weights(self, objective, delta):
        '''
        Calculate feature weights using soft thresholding
        '''
        
        soft = self.soft_threshold(objective, delta)
        w = soft / np.linalg.norm(soft)

        return w


    def soft_threshold(self, x, delta):
        """
        Applies soft thresholding to the input.
        """

        return np.sign(x) * np.maximum(0, np.abs(x) - delta)

    def fit(self, X, y=None):
        """
        Fits the Sparse Jump Model to the data. It first runs the sparse_jump method 
        to initialize and optimize the states and feature weights. Then, it calculates
        the centroids of the clusters for prediction.
        """
        self.states_, self.feat_w_ = self.sparse_jump(X, self.n_states, self.max_features, self.jump_penalty,
                                                 self.max_iter, self.tol, self.n_init, self.verbose)
        # Compute the centroids after fitting
        self.mu = np.zeros((self.n_states, X.shape[1]))
        for i in np.unique(self.states_):
            self.mu[i] = np.mean(X[self.states_==i], axis=0)
        return self


    def predict(self, X):
        """
        Predicts the states for the input X based on the fitted model.
        """
        distances = cdist(self.mu, X, 'euclidean')  # calculate distances from centroids to each row in X
        states = np.argmin(distances, axis=0)  # assign state of closest centroid
        return states
    
    def score(self, X, y=None):
        """
        Uses silhouette score as the scorer.  Robust to clusters of 1 in random search.
        """
        labels = self.predict(X)
        if len(set(labels)) > 1:
            # If more than one label is predicted, calculate the silhouette score
            return silhouette_score(X, labels)
        else:
            # If only one label is predicted, return a very low score
            return -1  # Or some other low value
        
if __name__ == '__main__':
    model = SparseJumpModel()