using MLJ, Test
using mm2020, CSVFiles, DataFrames

# Notes:
# winning model in 2019 used an xgboost model with a glmer measure of quality (RE's)
#   avg win rate in the last 14 days of the tournament
# Interesting features from second place:
#   Difference in the variance of game to game free throw percentage.
#   Difference in the variance of turnovers in the game to game free throw percentage.

# Get the submission sample
submission_sample = CSVFiles.load("data/MSampleSubmissionStage1_2020.csv") |> DataFrame

# Get the source seeds:
df_seeds = CSVFiles.load("data/MDataFiles_Stage1/MNCAATourneySeeds.csv") |> DataFrame
season_df = CSVFiles.load("data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv") |> DataFrame
season_df_detail = CSVFiles.load("data/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv") |> DataFrame
tourney_df  = CSVFiles.load("data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv") |> DataFrame
ranefs = CSVFiles.load("data/raneffects.csv") |> DataFrame
##############################################################
# Create training features for valid historical data
# SEEDS
seeds_features = make_seeds(copy(df_seeds), copy(tourney_df))
# EFFICIENCY
Wfdat, Lfdat, effdat = eff_stat_seasonal_means(copy(season_df_detail))
eff_features = get_eff_tourney_diffs(Wfdat, Lfdat, effdat, copy(tourney_df))
# ELO
season_elos = elo_ranks(Elo())
elo_features = get_elo_tourney_diffs(season_elos, copy(tourney_df))
# Momentum
momentum_features, momentum_df = make_momentum(copy(tourney_df), copy(season_df))
# Team Effects
ranef_features = make_ranef_features(copy(tourney_df), ranefs)

### Full feature dataset
seeds_features_min = filter(row -> row[:Season] >= 2003, seeds_features)
eff_features_min = filter(row -> row[:Season] >= 2003, eff_features)
elo_features_min = filter(row -> row[:Season] >= 2003, elo_features)
momentum_features_min = filter(row -> row[:Season] >= 2003, momentum_features)
ranef_features_min = filter(row -> row[:Season] >= 2003, ranef_features)

# create full stub

stub = join(seeds_features_min, eff_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);
fdata = join(stub, elo_features, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);
fdata = join(fdata, momentum_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);
fdata = join(fdata, ranef_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

exclude = [:Result, :Season, :LTeamID, :WTeamID]
deletecols!(fdata, exclude)

# Create features required to make submission predictions
seed_submission = get_seed_submission_diffs(copy(submission_sample), df_seeds)
eff_submission = get_eff_submission_diffs(copy(submission_sample), effdat) #see above
elo_submission = get_elo_submission_diffs(copy(submission_sample), season_elos)
momentum_submission = make_momentum_sub(copy(submission_sample), momentum_df)
ranef_submission = make_ranef_sub(copy(submission_sample), ranefs)
@test size(seed_submission, 1) == size(eff_submission, 1) == size(elo_submission, 1) == size(momentum_submission, 1)
