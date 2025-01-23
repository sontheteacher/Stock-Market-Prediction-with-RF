import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import the data
def load_and_prepare_data():
    # Load your BrightFuture data (adjust the path as needed)
    # df = pd.read_csv('QFC_fin_modelling_template/brightfuture_data.csv')
    df = pd.read_csv('brightfuture_data.csv')
    
    # Calculate price direction for BrightFuture
    df['price_direction'] = df['Direction14_BrightFuture']

    # Print all columns in the dataset
    print("Available columns in the dataset:")
    print(df.columns.tolist())
    
    # Create features (modify these based on your available data)
    feature_columns = [
        'Day', 'BrightFuture_Price', 'RSI_BrightFuture', 'Stochastic_BrightFuture', 'Williams_%R_BrightFuture', 'MACD_BrightFuture', 'PROC_BrightFuture'
        # Add more relevant features here
    ]
    
    # Remove last row since we can't calculate its direction
    df = df.dropna()
    
    X = df[feature_columns]
    y = df['price_direction']
    
    return X, y

def create_stock_prediction_model():
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Sort by date
    X = X.sort_values('Day')
    y = y[X.index]

    # Calculate split point
    split_idx = int(len(X) * 0.75)

    # Split based on date
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    # Remove Date column before scaling
    X_train = X_train.drop('Day', axis=1)
    X_test = X_test.drop('Day', axis=1)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def predict_next_direction(model, latest_data):
    # Make prediction
    prediction = model.predict(latest_data)
    return prediction

if __name__ == "__main__":
    # Train the model
    model = create_stock_prediction_model()
    # Save the trained model to the models folder
    joblib.dump(model, 'models/bright_model.pkl')
