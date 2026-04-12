import numpy as np
from ..core.base import BaseEstimator
class Node:
    def __init__(self, left = None, right = None, feature_index = None,feature_type = None,threshold= None, predicted_value = None, info_gain = None):
        # Decision Node
        self.left = left
        self.right = right
        self.feature_index = feature_index
        self.threshold = threshold
        self.info_gain = info_gain
        #Leaf Node
        self.predicted_value = predicted_value
        
class DecisionTreeClassifier(BaseEstimator):        
    def __init__(self, min_samples_split = 2, max_depth = 3, verbose = False):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.verbose = verbose
        self.depth = 0
        self.root = None
        
    def build_tree(self,X, y, current_depth = 0):
        n_samples,n_features = X.shape
        if(self.verbose and n_samples >= self.min_samples_split):
             print(f"Building tree at depth {current_depth} with {n_samples} samples")  # Debug statement to trace tree building
        if(n_samples >= self.min_samples_split and current_depth <= self.max_depth):
              best_split = self.get_best_split(X,y)
              print(X,y)
              if(self.verbose):
                print(f"Best split at depth {current_depth}: feature_index={best_split['feature_index']}, threshold={best_split['threshold']}, info_gain={best_split['info_gain']}")  # Debug statement to trace best split
              if best_split["info_gain"] > 0:
                  left_node = self.build_tree(best_split["left_X"], best_split["left_Y"], current_depth + 1)
                  right_node = self.build_tree(best_split["right_X"], best_split["right_Y"], current_depth + 1)
                  
                  return Node(left = left_node, right = right_node, feature_index= best_split["feature_index"], threshold=best_split["threshold"], info_gain=best_split["info_gain"])
        
        predicted_value = self.compute_predicted_value(y)
        return Node(predicted_value=predicted_value)      
                  
        
    def fit(self, X,y):
        self.root = self.build_tree(X,y, current_depth = 0)
        if(self.verbose):
            print(f"Tree building completed. Root node info_gain: {self.root.info_gain}, feature_index: {self.root.feature_index}, threshold: {self.root.threshold}")  # Debug statement to confirm tree building completion
    
    def compute_predicted_value(self, y):
        unique_classes = np.unique(y)
        predicted_value  = unique_classes[0] if len(unique_classes) > 0 else None
        for cls in unique_classes:
            if np.sum(y == cls) > np.sum(y == predicted_value):
                predicted_value = cls
        return predicted_value
    
    def print_tree(self, node=None, level=0, direction = "Root"):
        if node is None:
            node = self.root
        indent = "  " * level
        if node.predicted_value is not None:
            print(f"{indent} {direction} Leaf: Predict {node.predicted_value}")
        else:
            print(f"{indent} {direction} Node: feature_index={node.feature_index}, threshold={node.threshold}, info_gain={node.info_gain}")
            self.print_tree(node.left, level + 1, direction = "Left")
            self.print_tree(node.right, level + 1, direction = "Right")
        
    def predict(self, X):
        predictions = []
        for x in X:
            if(self.verbose):
                print("Predicting for input:", x)  # Debug statement to trace input
            node = self.root
            while node.predicted_value is None:
                if(self.verbose):
                    print(f"At node: feature_index={node.feature_index}, threshold={node.threshold}, info_gain={node.info_gain}")  # Debug statement to trace node details
                if(node.threshold and x[node.feature_index] <= node.threshold):
                    node = node.left
                else:
                    node = node.right    
            predictions.append(node.predicted_value)
            if(self.verbose):
                print(predictions)
            
        return np.array(predictions)            
    
    def compute_entropy(self, y):
        if(y.size == 0):
            return 0
        p1 = np.mean(y)
        return 0.0 if (p1 == 0 or p1 == 1) else - (p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
    

    def compute_information_gain(self, X, y, threshold, feature): 
        # Always return a dictionary with consistent keys so callers can rely on structure
        if X is None or len(X) == 0:
            empty = np.empty((0, X.shape[1])) if (X is not None and X.ndim == 2) else np.empty((0,))
            return {"left_X": empty, "left_Y": np.array([]), "right_X": empty, "right_Y": np.array([]), "info_gain": 0.0, "threshold": threshold}
        mask = X[:,feature] <= threshold
        X_left, X_right = X[mask],X[~mask]
        y_left, y_right = y[mask],y[~mask]
        parent_entropy = self.compute_entropy(y)
        left_weight = len(X_left) / len(X)
        left_entropy = self.compute_entropy(y_left)
        right_weight = len(X_right) / len(X)
        right_entropy = self.compute_entropy(y_right)
        weighted_entropy = (left_weight * left_entropy + right_weight * right_entropy)
        information_gain = parent_entropy - weighted_entropy
        return {"left_X":X_left,"left_Y":y_left,"right_X":X_right,"right_Y":y_right,"info_gain":information_gain, "threshold": threshold}
        
    
    def get_best_split(self,X, y):   
        num_features = X.shape[1]
        best_feature = -1
        best_information_gain = 0
        best_information_gain_dictionary = None
        for feature_idx in range(num_features):
            unique_values = np.unique(X[:,feature_idx])
            if len(unique_values) <= 1:
                continue
            thresholds = [(unique_values[i] + unique_values[i+1]) / 2 for i in range(self.min_samples_split, len(unique_values) - self.min_samples_split + 1)]
            for threshold in thresholds:
                gain_dictionary = self.compute_information_gain(X, y, threshold, feature_idx)
                if(gain_dictionary["info_gain"] > best_information_gain):
                    best_information_gain_dictionary = gain_dictionary
                    best_feature = feature_idx
                    best_information_gain = gain_dictionary["info_gain"]
                    
        # If no valid split found, return a safe default with info_gain 0
        if best_information_gain_dictionary is None:
            empty_X = np.empty((0, X.shape[1]))
            return {
                "left_X": empty_X,
                "left_Y": np.array([]),
                "right_X": empty_X,
                "right_Y": np.array([]),
                "info_gain": 0.0,
                "threshold": None,
                "feature_index": -1,
            }

        # ensure returned dict contains feature_index
        return {**best_information_gain_dictionary, "feature_index": best_feature}

        