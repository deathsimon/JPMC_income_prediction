import pandas as pd
from sklearn.model_selection import train_test_split                # For splitting dataset
from sklearn.ensemble import RandomForestClassifier                 # For Random Forest model
from sklearn.preprocessing import OneHotEncoder, StandardScaler     # For data preprocessing
from sklearn.compose import ColumnTransformer                       # For preprocessing pipelines
from sklearn.pipeline import Pipeline                               # For creating ML pipelines
from sklearn.metrics import accuracy_score, classification_report   # For evaluation metrics
from sklearn.tree import DecisionTreeClassifier                     # For decision tree
from sklearn.tree import _tree                                      # For accessing tree structure
from pathlib import Path
import argparse


def output_segmentation_model(decision_tree, classifier, selected_features, extended_features, numerical_cols, output_file=None):
    # Prepare to capture the output (creates a file if not exist)
    if output_file != None:
        file = open(output_file, 'w')
    
    # Output selected features
    print("[Selected Features with High Importance]:")
    if file != None:
        file.write("[Selected Features with High Importance]:\n")
    for feature in selected_features:
        print("  " + feature)
        if file != None:
            file.write(f"\t{feature}\n")
    
    # Retrieve means and standard deviation (stds) from the scaler used in preprocessing
    scaler = classifier.named_steps['preprocessor'].named_transformers_['numerical']
    means = scaler.mean_
    stds = scaler.scale_
    
    tree_ = decision_tree.tree_
    feature_name = [
        extended_features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    # Define the things needed for mapping back to original scale for each node
    def recurse_traverse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            # If feature is numerical, map threshold back to original scale
            if name in numerical_cols:
                idx = numerical_cols.index(name)
                threshold_original = threshold * stds[idx] + means[idx]
                threshold_display = round(threshold_original, 3)    # display with 3 decimal places
            else:
                threshold_display = round(threshold, 3)  # categorical or binary features
            
            print(f"{indent}if {name} <= {threshold_display}:")
            if file != None:
                file.write(f"{indent}if {name} <= {threshold_display}:\n")
            recurse_traverse(tree_.children_left[node], depth + 1)
            print(f"{indent}else:  # if {name} > {threshold_display}")
            if file != None:
                file.write(f"{indent}else:  # if {name} > {threshold_display}\n")
            recurse_traverse(tree_.children_right[node], depth + 1)
        else:
            value = tree_.value[node]
            class_idx = value.argmax()
            if class_idx == 1:
                print(f"{indent}predicts income >50K")
                if file != None:
                    file.write(f"{indent}predicts income >50K\n")
            else:
                print(f"{indent}predicts income <=50K")
                if file != None:
                    file.write(f"{indent}predicts income <=50K\n")
            
    
    # Recursively traverse and output the tree structure
    print("[Segmentation Model Decision Tree Structure]:")
    if file != None:
        file.write("[Segmentation Model Decision Tree Structure]:\n")
    recurse_traverse(0, 0)
    
    if file != None:
        file.close()
    
def prune_tree_same_class(tree, node=0):
    # Check if current node is internal
    if tree.children_left[node] == _tree.TREE_LEAF or tree.children_right[node] == _tree.TREE_LEAF:
        return

    left = tree.children_left[node]
    right = tree.children_right[node]

    # Recursively prune children first
    prune_tree_same_class(tree, left)
    prune_tree_same_class(tree, right)

    # If both children are leaves
    left_is_leaf = (tree.children_left[left] == _tree.TREE_LEAF and tree.children_right[left] == _tree.TREE_LEAF)
    right_is_leaf = (tree.children_left[right] == _tree.TREE_LEAF and tree.children_right[right] == _tree.TREE_LEAF)

    if left_is_leaf and right_is_leaf:
        left_class = tree.value[left].argmax()
        right_class = tree.value[right].argmax()
        if left_class == right_class:
            # Merge this node: make it a leaf predicting the same class
            tree.children_left[node] = _tree.TREE_LEAF
            tree.children_right[node] = _tree.TREE_LEAF
            tree.feature[node] = _tree.TREE_UNDEFINED
            tree.threshold[node] = -2
            tree.value[node] = tree.value[left]
 
def post_process_selected_features(selected_features, categorical_cols, numerical_cols):
    original_selected_features = []
    for feature in selected_features:
        if feature in numerical_cols:
            original_selected_features.append(feature)
        else:   # in categorical
            # Add the original categorical feature name (before one-hot encoding)
            prefix = feature.split('_')[0]
            if prefix not in original_selected_features:
                original_selected_features.append(prefix)
    
    return original_selected_features
    
def post_process_segmentation_model(decision_tree):
    # Prune the decision tree to merge nodes predicting the same class
    prune_tree_same_class(decision_tree.tree_)    
 
def build_segmentation_model(classifier, features, labels, selected_features, extended_features, max_depth=None, verbose=False):
    # Select the columns from the transformed matrix corresponding to selected_features
    feature_preprocessed = classifier.named_steps['preprocessor'].transform(features)
    feature_index = [extended_features.index(f) for f in selected_features]
    feature_segment = feature_preprocessed[:, feature_index]

    # Separate into training and testing sets
    features_train, features_test, labels_train, labels_test = train_test_split(feature_segment, labels, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier
    start_time = pd.Timestamp.now()
    decision_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    decision_tree.fit(features_train, labels_train)
    training_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    # Evaluate the segmentation model
    labels_pred = decision_tree.predict(features_test)
    print("Segmentation Model Accuracy: %.2f" % accuracy_score(labels_test, labels_pred))
    if verbose:
        print("Segmentation Model Classification Report:\n", classification_report(labels_test, labels_pred))
        print("Segmentation Model Training Time (seconds): %.2f\n" % training_time)
        
    return decision_tree
            
def extract_feature_importance(classifier, categorical_cols, numerical_cols, verbose=False):
    # retrieve the preprocessor from the pipeline
    preprocessor = classifier.named_steps['preprocessor']
    # Get categorical feature names after OneHotEncoding
    categorical_features = preprocessor.transformers_[1][1].get_feature_names_out()    
    # Combine numerical and categorical feature names
    extended_features = list(numerical_cols) + list(categorical_features)

    # Get feature importance scores from the Random Forest classifier
    importances = classifier.named_steps['classifier'].feature_importances_

    # Create DataFrame of features and their importance
    feat_imp_df = pd.DataFrame({
        'Feature': extended_features,
        'Importance': importances
    })

    # Sort by importance descending and select top 10
    top_10_features = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)['Feature'].tolist()
    if verbose:
        print("[Top 10 Important Features]:")
        for feature in top_10_features:
            print("  " + feature + " (Importance: %.4f)" % feat_imp_df[feat_imp_df['Feature'] == feature]['Importance'].values[0])
        print("\n")

    # reduced feature set based on top features
    selected_features = []
    for feature in top_10_features:
        if feature in numerical_cols:
            selected_features.append(feature)
        else:   # in categorical
            # Add all one-hot encoded columns that start with that categorical feature's prefix
            prefix = feature.split('_')[0]
            selected_features.extend([col for col in extended_features if col.startswith(prefix + '_')])
    # Remove duplicates        
    selected_features = list(dict.fromkeys(selected_features))

    return selected_features, extended_features   

def train_classifier_random_forest(features_train, labels_train, features_test, labels_test, categorical_cols, numerical_cols, n_estimators=100, max_depth=None, verbose=False):
    # Define preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', StandardScaler(), numerical_cols),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_cols),        
        ])
    
    # Create pipeline with preprocessing and Random Forest classifier
    classifier = Pipeline([
        ('preprocessor', preprocessor),
        # ('classifier', RandomForestClassifier(random_state=42)) // default number of trees: 100, max_depth=None
        ('classifier', RandomForestClassifier(n_estimators, max_depth=max_depth, random_state=42)) # lower the number of trees and limit depth for faster training
    ])

    # Train the model
    start_time = pd.Timestamp.now()
    classifier.fit(features_train, labels_train)
    training_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    # Predict and evaluate
    labels_pred = classifier.predict(features_test)
    
    print("Classification Model Accuracy: %.2f" % accuracy_score(labels_test, labels_pred))
    if verbose:        
        print("Classification Model Report:\n", classification_report(labels_test, labels_pred)) 
        print("Training Time (seconds): %.2f\n" % training_time)
        
    return classifier
    
def preprocess_data(data, verbose=False):    
    # Check for and handling missing values    
    # First, remove rows with missing labels
    data = data.dropna(subset=['label'])
    
    # Separate features and labels
    labels = data['label']
    features = data.drop('label', axis=1)
    
    # For the Label column, convert the labels to binary format: 1 for '50000+.', 0 for '- 50000.'
    labels = labels.apply(lambda x: 1 if x == '50000+.' else 0)
    
    # For the Features, identify categorical and numerical columns
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    total_rows = len(features)
    # For categorical column, replace the missing/meaningless values with 'Unknown'
    unknown_percent_categorical = {}    # for reporting
    for col in categorical_cols:
        features[col].fillna('Unknown', inplace=True)
        features[col] = features[col].replace(['Not in universe', 'Not in universe or children','?'], 'Unknown')
        unknown_percent_categorical[col] = (features[col] == 'Unknown').sum()* 100 / total_rows
    # For numerical columns, replace missing/meaningless values with -1
    zero_percent_numerical = {}
    for col in numerical_cols:
        features[col].fillna(-1, inplace=True)
        features[col] = features[col].replace(['Not in universe', '?'], -1)
        zero_percent_numerical[col] = (features[col] <= 0.0).sum()* 100 / total_rows
        
    if verbose:
        # Combine results into a DataFrame for easy viewing
        report_df = pd.DataFrame({
            'Column': list(unknown_percent_categorical.keys()) + list(zero_percent_numerical.keys()),
            'Percent Unknown or Zero': list(unknown_percent_categorical.values()) + list(zero_percent_numerical.values())
        })

        # Sort by percentage descending for clarity
        report_df = report_df.sort_values(by='Percent Unknown or Zero', ascending=False).reset_index(drop=True)
        # output the report regarding missing values in the dataset in command line
        print(report_df)
        
    return features, labels, categorical_cols, numerical_cols        

def load_data(data_path, columns_path):
    
    # Load column headers    
    if not Path(columns_path).is_file():
        raise FileNotFoundError(f"Column headers file not found: {columns_path}")    
    with open(columns_path, 'r') as f:
        headers = [line.strip() for line in f.readlines()]
        
    # Load dataset
    if not Path(data_path).is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = pd.read_csv(data_path, header=None)
    data.columns = headers
    
    return data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Income Prediction and Segmentation Model')
    parser.add_argument('--verbose', type=bool, default=False, help='Enable verbose output', const=False, nargs='?')
    parser.add_argument('--input_data', type=str, default='./Data/census-bureau.data', help='Input file name of the data set', const='./Data/census-bureau.data', nargs='?')
    parser.add_argument('--input_data_header', type=str, default='./Data/census-bureau.columns', help='Input file name of the data header', const='./Data/census-bureau.columns', nargs='?')
    parser.add_argument('--output_file_name', type=str, default='Segmentation_Model.txt', help='Output file name for segmentation model', const='', nargs='?')
    
    args = parser.parse_args()
    verbose = args.verbose
    output_file = args.output_file_name
    input_data = args.input_data    
    input_data_header = args.input_data_header
    
    # Load column headers and dataset
    data = load_data(input_data, input_data_header)
    
    features, labels, categorical_cols, numerical_cols = preprocess_data(data, verbose=verbose)
    
    # Split into training and testing sets (80% train, 20% test)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Train and evaluate Random Forest classifier
    classifier = train_classifier_random_forest(features_train, labels_train, features_test, labels_test, categorical_cols, numerical_cols, n_estimators=50, max_depth=10, verbose=verbose)
       
    # Extract high-importance features from the trained classifier, as well as the full list of features
    selected_features, extended_features = extract_feature_importance(classifier, categorical_cols, numerical_cols, verbose=verbose)
    
    # Build the segmentation model using Decision Tree on the selected features
    segmentation_model = build_segmentation_model(classifier, features, labels, selected_features, extended_features, max_depth=5, verbose=verbose)
    
    # Post-process the segmentation model to prune and display with original thresholds
    post_process_segmentation_model(segmentation_model)
    
    # Post-process selected features to map back to original feature names
    selected_features = post_process_selected_features(selected_features, categorical_cols, numerical_cols)
    
    # output the segmentation model structure
    output_dir = Path('./Output')
    output_dir.mkdir(parents=True, exist_ok=True)   # Create directory if not exists
    output_file = output_dir / output_file
    output_segmentation_model(segmentation_model, classifier, selected_features, extended_features, numerical_cols, output_file)

    
if __name__ == "__main__":
    main()
