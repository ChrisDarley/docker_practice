from joblib import load

# run the inference script
def run_inference():
        
    # print to check if file runs
    print("inference.py running")

    # loading test dataset and model
    X_test = load('X_test.csv')
    y_test = load('y_test.csv')
    model = load('model.csv')

    # scoring model
    score = model.score(X_test, y_test)
    print(score)