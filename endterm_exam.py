'''
-------------------------ENDTERM EXAM-----------------
DO NOT DELETE THE FOLLOWING CODE
'''
import sys
try:
    input1 = sys.argv[1]
except:
    pass
'''
In the following file, do not delete anything (comments, code, ...). Just add you code in every part (one per exercise).
Use my variable for input (if there is any), use my printing for output (if there is any).
You can upload your code to codepost.io to check the tests. A sucess in one test doesn't always mean than your exercise is correct,
a fail doesn't always mean that your exercise is wrong. I will check all codes.
At the end of exam, you should upload the last version of your code to codepost.io or to the online folder on Teams.
The only authorized packages are:
- pandas
- pyarrow
- fastparquet
- numpy
- sklearn
- matplotlib
- datetime

'''
if input1 == '4':
# ----------------------EXERCISE 4 - Machine Learning II--------------------------------------
# Instructions:
# You have a list of feature named feature and a list of label named label.
# Make a linear regression with the features and the label and display the equation and the accuracy of your model.
# Do not use scaling or train/test sets. Choose the best printing instructions.
    import sklearn.linear_model as skmod
    import numpy as np

    features = [-8, -1, -6, -8, -1, -9, -1, -7, -4, 3, -8, 2, -8, -5, -2, 4, 1, 3, -3, -7]
    label = [34.0, -5.0, 27.0, 29.0, -5.0, 51.0, -5.0, 28.0, 21.0, -25.0, 59.0, -17.0, 34.0, 24.0, 5.0, -36.0, -19.0, -32.0, 10.0, 40.0]
    import sklearn.linear_model as skmod
    import numpy as np

    arr_x = np.array(features).reshape(-1, 1)
    arr_y = np.array(label).reshape(-1, 1)
    modelSK = skmod.LinearRegression()
    model = modelSK.fit(arr_x, arr_y)


    w1 = model.coef_[0][0]
    b = model.intercept_[0]
    R2 = model.score(arr_x, arr_y)
    print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(w1, b, R2))






# Here is several printings, choose the most appropriate one.	
    #print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(w1, b, R2))	
    #print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(b, w1, R2))	
    #print("The most accurate linear regression has the following equation: y' = {:0.2f}*x + {:0.2f} and its accuracy is: {:0.3f}".format(R2, w1, b))
# ----------------------End of EXERCISE 4 --------------------------------------

elif input1 == '5':
# ----------------------EXERCISE 5 - Machine Learning III--------------------------------------
# Instructions:
# You have two features : feature1 and feature2. You objective is to make a two columns matrix and
# to separate them into a train and a test set. The size of train set is 80 % of the original matrix.
# Remember to use shuffle = False in the train_test_split function of the scikit-learn package.
# At the end print only the test arrays. 
# You can find the desired matrix in the document Ex_5_x_tests_matrix_V17.txt
    import numpy as np
    import sklearn.model_selection as sksel


    feature1 = [-61, -296, -418, -922, 383, -964, -731, -63, 109, 360, 427, -206, 222, 78, 273, 459, -969, -674, 229, 171, 295, -515, 269, -435, -335, -44, -544, -944, -188, -255]
    feature2 = [-10, -3, -4, 2, -3, -3, -7, -2, 3, 0, -7, 2, -5, -8, -1, 0, 1, 0, -1, -1, 3, 4, 2, 4, -2, -1, -6, -2, -7, -4]
    feature1 = np.array(feature1).reshape(-1, 1)
    feature2 = np.array(feature2).reshape(-1, 1)

    x = np.hstack([feature1, feature2])
    x_train, x_test = sksel.train_test_split(x, train_size = 0.8, shuffle = False)



    # Here is the print instructions to print test arrays.
    print(np.float64(x_test))

# ----------------------End of EXERCISE 5 --------------------------------------