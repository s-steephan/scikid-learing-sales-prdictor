from sklearn.linear_model import LinearRegression

x = [[2,4],[3,6],[4,5],[6,7],[3,3],[2,5],[5,2]]
y = [14,21,22,32,15,16,19]

genius_regression_model = LinearRegression()
genius_regression_model.fit(x,y)

print genius_regression_model.predict([[8,4]])