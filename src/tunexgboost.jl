
using DataFrames, CSVFiles, Pipe, MLJ, LossFunctions
@load XGBoostClassifier()
# reload the saved data
fullX = CSVFiles.load("data/fullX.csv") |> DataFrame
fullY = CSVFiles.load("data/fullY.csv") |> DataFrame

# create array of training and testing rows
train, test = partition(1:2230, 0.7, shuffle=true)
validate = [2230:size(fullY, 1)...]

#################################################
xgb = XGBoostClassifier()
xgb.num_round = 6
xgb.max_depth = 3
xgb.min_child_weight = 4.2105263157894735
xgb.gamma = 11
xgb.eta = .35
xgb.subsample = 0.6142857142857143
xgb.colsample_bytree = 1.0

y = @pipe categorical(fullY.y) |> recode(_, 0=>"lose",1=>"win");
fullX_co = coerce(fullX, Count=>Continuous)

r_eta = range(xgb, :eta, lower=.3, upper=.4)

tm = TunedModel(model=xgb, tuning=Grid(resolution=2),
            resampling=CV(rng=234), ranges=r_eta,
            measure=cross_entropy)

mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)
yhat = predict(mtm, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(mtm, rows=test), y[test])


# measuring the number of rounds
r_numround = range(xgb, :num_round, lower=1, upper=100)
r_maxdepth = range(xgb, :max_depth, lower = 3, upper = 10)
r_minchildweight = range(xgb, :min_child_weight, lower=0, upper=5)
r_gamma = r = range(xgb, :gamma, lower=6, upper=20)
r_subsample = range(xgb, :subsample, lower=0.1, upper=1.0)
r_colsample = range(xgb, :colsample_bytree, lower=0.1, upper=1.0)
r_eta = range(xgb, :eta, lower=.01, upper=.4)
r_lambda = range(xgb, :lambda, lower=1.0, upper = 100.0)

all_ranges = [r_numround, r_maxdepth, r_minchildweight, r_gamma, r_subsample, r_colsample, r_eta, r_lambda]

#########################################
# Tuning lam and colsample
xgbm = machine(xgb, fullX_co, y)
global xgb = XGBoostClassifier()
# iterates over models and finds optimal values
for ranger in all_ranges
    tm = TunedModel(model=xgb, tuning=Grid(resolution=1),
                resampling=CV(rng=234), ranges=ranger,
                measure=cross_entropy)
    mtm = machine(tm, fullX_co, y)
    fit!(mtm, rows=train)
    global xgb = fitted_params(mtm).best_model
    best = getfield(report(mtm).best_model, ranger.field)
    println(ranger.field, ":", best)
end

mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)
yhat = predict(mtm, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(mtm, rows=test), y[test])
