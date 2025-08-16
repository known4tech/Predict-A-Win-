#!/usr/bin/env python3
"""
IPL Win Probability Predictor - Model Retraining Script
This script retrains the IPL win probability prediction model with scikit-learn==1.3.2
for compatibility with the Streamlit deployment.
Author: Updated for scikit-learn 1.3.2 compatibility
Date: August 17, 2025
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data():
    """
    Load and process the IPL match and delivery data.
    """
    print("Loading data...")
    
    # Load the datasets
    matches = pd.read_csv('matches.csv')
    deliveries = pd.read_csv('deliveries.csv')
    
    print(f"Matches shape: {matches.shape}")
    print(f"Deliveries shape: {deliveries.shape}")
    
    # Group by match_id and inning to get total runs per innings
    totalrun_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
    
    # Get first innings runs only (target for second innings)
    totalrun_df = totalrun_df[totalrun_df['inning'] == 1]
    totalrun_df['total_runs'] = totalrun_df['total_runs'].apply(lambda x: x + 1)  # Add 1 as target
    
    # Merge with matches data
    match_df = matches.merge(totalrun_df[['match_id', 'total_runs']], 
                            left_on='id', right_on='match_id')
    
    return match_df, deliveries

def clean_team_names(match_df):
    """
    Clean and standardize team names.
    """
    print("Cleaning team names...")
    
    # Replace team names for consistency
    match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    
    # Define main teams to keep
    teams = [
        'Sunrisers Hyderabad',
        'Mumbai Indians',
        'Royal Challengers Bangalore',
        'Kolkata Knight Riders',
        'Kings XI Punjab',
        'Chennai Super Kings',
        'Rajasthan Royals',
        'Delhi Capitals'
    ]
    
    # Filter to keep only main teams
    match_df = match_df[match_df['team1'].isin(teams)]
    match_df = match_df[match_df['team2'].isin(teams)]
    
    print(f"After filtering teams, matches shape: {match_df.shape}")
    
    return match_df

def create_ball_by_ball_features(match_df, deliveries):
    """
    Create ball-by-ball features for model training.
    """
    print("Creating ball-by-ball features...")
    
    # Remove DL method matches
    match_df = match_df[match_df['dl_applied'] == 0]
    
    # Keep only required columns
    match_df = match_df[['match_id', 'city', 'winner', 'total_runs']]
    
    # Merge with deliveries
    delivery_df = match_df.merge(deliveries, on='match_id')
    
    # Keep only second innings (prediction target)
    delivery_df = delivery_df[delivery_df['inning'] == 2]
    
    print(f"Ball-by-ball data shape: {delivery_df.shape}")
    
    # Convert to numeric type before cumsum operations
    delivery_df['total_runs_y'] = pd.to_numeric(delivery_df['total_runs_y'], errors='coerce').fillna(0)
    
    # Calculate current score (cumulative sum of runs)
    delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']
    
    # Calculate runs left
    delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']
    
    # Calculate balls left
    delivery_df['balls_left'] = 126 - (delivery_df['over'] * 6 + delivery_df['ball'])
    
    # Process player dismissed column
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
    delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(
        lambda x: x if x == "0" else "1"
    )
    
    # Convert to numeric type before cumsum operations
    delivery_df['player_dismissed'] = pd.to_numeric(delivery_df['player_dismissed'], errors='coerce').fillna(0)
    
    # Calculate wickets left
    wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
    delivery_df['wickets_left'] = 10 - wickets
    
    # Calculate current run rate
    delivery_df['cur_run_rate'] = (delivery_df['current_score'] * 6) / (
        120 - delivery_df['balls_left']
    )
    
    # Calculate required run rate
    delivery_df['req_run_rate'] = (delivery_df['runs_left'] * 6) / delivery_df['balls_left']
    
    # Create result column (1 if batting team wins, 0 otherwise)
    def result_func(row):
        return 1 if row['batting_team'] == row['winner'] else 0
    
    delivery_df['result'] = delivery_df.apply(result_func, axis=1)
    
    return delivery_df

def prepare_final_dataset(delivery_df):
    """
    Prepare the final dataset for training.
    """
    print("Preparing final dataset...")
    
    # Select final features
    final_df = delivery_df[[
        'batting_team', 'bowling_team', 'city', 'runs_left',
        'balls_left', 'wickets_left', 'total_runs_x', 'cur_run_rate',
        'req_run_rate', 'result'
    ]]
    
    print(f"Final dataset shape before cleaning: {final_df.shape}")
    
    # Remove null values
    final_df = final_df.dropna()
    
    # Remove rows where balls_left is 0 (to avoid division by zero)
    final_df = final_df[final_df['balls_left'] != 0]
    
    print(f"Final dataset shape after cleaning: {final_df.shape}")
    print(f"Null values: {final_df.isnull().sum().sum()}")
    
    return final_df

def train_model(final_df):
    """
    Train the logistic regression model.
    """
    print("Training model...")
    
    # Prepare features and target
    X = final_df.drop(['result'], axis=1)
    y = final_df['result']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Create column transformer for categorical variables
    column_transformer = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(sparse_output=False, drop='first'), 
             ['batting_team', 'bowling_team', 'city'])
        ],
        remainder='passthrough'
    )
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', column_transformer),
        ('classifier', LogisticRegression(solver='liblinear', random_state=42))
    ])
    
    # Train the model
    print("Fitting the pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Make predictions and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Show sample predictions
    sample_probs = pipeline.predict_proba(X_test)[:5]
    print("\nSample predictions (first 5):")
    for i, prob in enumerate(sample_probs):
        print(f"Sample {i+1}: Loss prob: {prob[0]:.3f}, Win prob: {prob[1]:.3f}")
    
    return pipeline

def save_model(pipeline, filename='pipe.pkl'):
    """
    Save the trained model to a pickle file.
    """
    print(f"Saving model to {filename}...")
    
    with open(filename, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Model saved successfully to {filename}")
    
    # Verify the saved model can be loaded
    try:
        with open(filename, 'rb') as f:
            loaded_model = pickle.load(f)
        print("Model loading verification: SUCCESS")
        return True
    except Exception as e:
        print(f"Model loading verification: FAILED - {e}")
        return False

def main():
    """
    Main execution function.
    """
    print("=" * 60)
    print("IPL Win Probability Predictor - Model Retraining")
    print("Using scikit-learn==1.3.2")
    print("=" * 60)
    
    try:
        # Step 1: Load and process data
        match_df, deliveries = load_and_process_data()
        
        # Step 2: Clean team names
        match_df = clean_team_names(match_df)
        
        # Step 3: Create ball-by-ball features
        delivery_df = create_ball_by_ball_features(match_df, deliveries)
        
        # Step 4: Prepare final dataset
        final_df = prepare_final_dataset(delivery_df)
        
        # Step 5: Train model
        pipeline = train_model(final_df)
        
        # Step 6: Save model
        success = save_model(pipeline, 'pipe.pkl')
        
        if success:
            print("\n" + "=" * 60)
            print("MODEL RETRAINING COMPLETED SUCCESSFULLY!")
            print("The new 'pipe.pkl' file is ready for the Streamlit app.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("MODEL RETRAINING FAILED!")
            print("Please check the error messages above.")
            print("=" * 60)
    
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Model retraining failed. Please check the error and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
