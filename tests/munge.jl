## MM2020 Train/Test
# Create full submission dataset
submission_features = hcat(seed_submission, eff_submission, elo_submission, momentum_submission, ranef_submission)

##########################################################################

# TRAINING

# Join the two feature sets
featurecols = [names(seed_submission), names(eff_submission), names(elo_submission), names(momentum_submission), names(ranef_submission)]
featurecols = collect(Iterators.flatten(featurecols))
fullX = [fdata[featurecols]; submission_features[featurecols]]
fullY = [seeds_features_min.Result; repeat([0], size(submission_features, 1))]

####################################0
using DataFrames, CSVFiles, Pipe, MLJ, LossFunctions
# reload the saved data
#save("fullX.csv", fullX)
#save("fullY.csv", DataFrame(y=fullY))

fullX = CSVFiles.load("fullX.csv") |> DataFrame
fullY = CSVFiles.load("fullY.csv") |> DataFrame

# create array of training and testing rows
train, test = partition(1:2230, 0.7, shuffle=true)
validate = [2231:size(fullY, 1)...]

# Recode result to win/ loss
y = @pipe categorical(fullY.y) |> recode(_, 0=>"lose",1=>"win");


#################################################
@load XGBoostClassifier()
xgb = XGBoostClassifier()
fullX_co = coerce(fullX, Count=>Continuous)
#--- Setting the rounds of the xgb, then tuning depth and children
xgb.num_round = 4
xgb.max_depth = 3
xgb.min_child_weight = 4.2105263157894735
xgb.gamma = 11
xgb.eta = .35
xgb.subsample = 0.6142857142857143
xgb.colsample_bytree = 1.0

xgb_forest = EnsembleModel(atom=xgb, n=1000);
#xgb_forest.bagging_fraction = .8
xg_model = machine(xgb_forest, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])
# This is a working single model for XGBOOST Classifier
xgb_forest = EnsembleModel(atom=xgb, n=100);
xgb_forest.bagging_fraction = .72
N_range = range(xgb_forest, :n,
                lower=1, upper=200)
tm = TunedModel(model=xgb_forest,
                tuning=Grid(resolution=200), # 10x10 grid
                resampling=Holdout(fraction_train=0.8, rng=42),
                ranges=N_range)
tuned_ensemble = machine(tm, fullX_co, y)
fit!(tuned_ensemble, rows=train);
yhat = predict(tuned_ensemble, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(tuned_ensemble, rows=test), y[test])

params, measures = report(tuned_ensemble).plotting.parameter_values, report(tuned_ensemble).plotting.measurements
plot(params[:, 1], measures, seriestype=:scatter)


####################################################
# Predict onto the submission_sample
sub_sample = predict(xg_model, rows = validate)
submission_sample[:Pred] = pdf.(sub_sample, "win")
save("submission_xgb.csv", submission_sample)
############################### TUNING THE SUBMISSION
using LossFunctions
loss = LogitDistLoss()

logloss(yhat, y) = -1/length(y) * sum(y .* log.(yhat.+1e-15) .+ (1 .- y) .* log.(1 .- yhat .+1e-15) )
threshold = [.95, .92, .90, .88, .85, .80, .75, .70, .65]
mce_out = []
ytest = Float32(1.0) .* (y[test] .== "win")
for thresh in threshold
        recode_pred = Float64[ifelse(x >= thresh, 1.0, x) for x in pdf.(yhat, "win")]
        recode_pred = [ifelse(x <= (1.0-thresh), 0.0, x) for x in recode_pred]
        push!(mce_out, loss(ytest, recode_pred) |> sum)
end

###########################
# measuring the number of rounds
xgbm = machine(xgb, fullX_co, y)
r = range(xgb, :num_round, lower=1, upper=50)
curve = learning_curve!(xgbm, resampling=CV(nfolds=3),
                        range=r, resolution=20,
                        measure=cross_entropy)

plot(curve.parameter_values, curve.measurements)

r1 = range(xgb, :max_depth, lower = 3, upper = 10)
r2 = range(xgb, :min_child_weight, lower=0, upper=5)
tm = TunedModel(model = xgb, tuning = Grid(resolution = 20),
        resampling = CV(rng=11), ranges=[r1, r2],
        measure = cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows = train)

yhat = predict(mtm, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(mtm, rows=test), y[test])

###################################################
# TUNING GAMMA
xgbm = machine(xgb, fullX_co, y)
r = range(xgb, :gamma, lower=6, upper=20)
curve = learning_curve!(xgbm, resampling=CV(),
                        range=r, resolution=50,
                        measure=cross_entropy);
plot(curve.parameter_values, curve.measurements)
####################################################
# TUNING ETA
r = range(xgb, :eta, lower=.01, upper=.4)
tm = TunedModel(model = xgb, tuning = Grid(resolution = 20),
        resampling = CV(rng=11), ranges=r,
        measure = cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows = train)
######################################
# Tuning subsample and colsample
r1 = range(xgb, :subsample, lower=0.1, upper=1.0)
r2 = range(xgb, :colsample_bytree, lower=0.1, upper=1.0)
tm = TunedModel(model=xgb, tuning=Grid(resolution=8),
                resampling=CV(rng=234), ranges=[r1,r2],
                measure=cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)

#########################################
# Tuning lam and colsample
r1 = range(xgb, :subsample, lower=0.1, upper=1.0)
r2 = range(xgb, :colsample_bytree, lower=0.1, upper=1.0)
tm = TunedModel(model=xgb, tuning=Grid(resolution=8),
                resampling=CV(rng=234), ranges=[r1,r2],
                measure=cross_entropy)
mtm = machine(tm, fullX_co, y)
fit!(mtm, rows=train)


#######################################
rf = @load RandomForestClassifier pkg="ScikitLearn"

rf_forest = EnsembleModel(atom=rf, n=1);
rf_model = machine(rf_forest, fullX, y)
fit!(rf_model, rows = train)
yhat = predict(rf_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(rf_model, rows=test), y[test])


####################################################

ada = @load AdaBoostStumpClassifier pkg="DecisionTree"
ada_model = machine(ada, fullX, y)
fit!(ada_model, rows = train)
yhat = predict(ada_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(ada_model, rows=test), y[test])

# Train the model!

# make the submission prediction
final_prediction = predict_mode(tree, rows=validate)


xg = @load GradientBoostingClassifier pkg = "ScikitLearn"
fullX_co = coerce(fullX, Count=>Continuous)
xg_model = machine(xg, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])

@load XGBoostClassifier()
xgb = XGBoostClassifier()
xg_model = machine(xgb, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])

#########################################################
# This is a working single model for XGBOOST Classifier
xgb_forest = EnsembleModel(atom=xgb, n=1000);
xg_model = machine(xgb_forest, fullX_co, y)
fit!(xg_model, rows = train)
yhat = predict(xg_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(xg_model, rows=test), y[test])
#######################################################3

# get parameter values and misclassification scores
miss_rates = ens_model.report.plotting.measurements[:, 1]
alphas = ens_model.report.plotting.parameter_values[:, 1]
################################


#####################################3

yhat = predict(ens_model, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict(ens_model, rows=test), y[test])

atom = @load RidgeRegressor pkg=MultivariateStats
mach = machine(ensemble, X, y)
########################################
################ WORKING EXAMPLE
tree_model = @load DecisionTreeClassifier verbosity=1
tree = machine(tree_model, fullX[:, [:SeedDiff]], y)
fit!(tree, rows = train)
yhat = predict(tree, rows=test)
mce = cross_entropy(yhat, y[test]) |> mean
accuracy(predict_mode(tree, rows=test), y[test])

# Just checking on a GLM
df = hcat(fullY, fullX)
myform = @formula(y ~ SeedDiff + Diff_Pts_mean_mean)
mod = glm(myform, df[train, :], Binomial(), ProbitLink())


#### BOOSTING ALGOS

# HOW TO DO WEIGHTED BOOSTING

# INITIALIZE WEIGHTS
W = 1/length(y) .* fill(1, length(y)) # weights
y = [0,1,1,0,0]

# For M in M_ALL - CREATE PREDICTION
pred_g = [1, 0, 1, 0, 0]
# Generate the Error of the model Weights times indicator
err_m = sum(W .* (y .!= pred_g)) / sum(W)
# Alpha, transformation of the error (log(0)=1)
α_m = log( (1 - err_m) / err_m)
# update the weights
W .= W .* exp.(α_m .* (y .!= pred_g))

# FINAL OUTPUT - take the sum of all models =  alpha_m * prediction_m
