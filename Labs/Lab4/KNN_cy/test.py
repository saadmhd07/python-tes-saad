import knn  # Import the compiled module
import numpy as np 

def test_knn():
    # Example test data
    x_train = np.random.rand(100, 5).astype(np.float64)
    class_train = np.random.randint(0, 2, size=100).astype(np.int32)
    x_test = np.random.rand(10, 5).astype(np.float64)
    k = 5
    
    # Call the knn function
    y_pred = knn.knn(x_train, class_train, x_test, k)
    print(y_pred)

if __name__ == "__main__":
    test_knn()