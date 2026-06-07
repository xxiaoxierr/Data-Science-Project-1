import statsmodels.api as sm

def fit_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

