from sklearn.ensemble import GradientBoostingRegressor

def trained_gbm_model(x_train, y_train):
     # set best params from earlier testing
     best_n_estimators_value = 100
     best_max_depth_value = 3
     best_learning_rate_value = 0.1

     # fit rf
     best_gb_regressor = GradientBoostingRegressor(
          n_estimators=best_n_estimators_value,
          max_depth=best_max_depth_value,
          learning_rate=best_learning_rate_value
     )

     best_gb_regressor.fit(x_train, y_train)

     return best_gb_regressor