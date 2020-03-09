 # March Madness 2020 Men's Forecast

This repository generates features and a baseline model for the Men's 2020 NCAA [Kaggle](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament) prediction competition.

This script creates features and a ML modeling for predicting the outcome of the Men's NCAA March Madness Tournament.

First, load some dependencies. The primary machine learning methods are done using the [MLJ] (https://github.com/alan-turing-institute/MLJ.jl)

```julia
 using MLJ, Test, Pipe
 using mm2020, CSVFiles, DataFrames
```

 Get the submission sample which will be used to create the features for the final submission.

```julia
 submission_sample = CSVFiles.load("data/MSampleSubmissionStage1_2020.csv") |> DataFrame
```

Load all the source data required to generate basic features: the seeds, the compact season data, the detailed season data, and the seasonal tournament data.

```julia
 df_seeds = CSVFiles.load("data/MDataFiles_Stage1/MNCAATourneySeeds.csv") |> DataFrame
 season_df = CSVFiles.load("data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv") |> DataFrame
 season_df_detail = CSVFiles.load("data/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv") |> DataFrame
 tourney_df  = CSVFiles.load("data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv") |> DataFrame
```

Finally, due to some compatibility issues with MixedModels.jl, load team effects, which had to be generated and saved in a separate environment. I'll describe how these were created a bit later.

 ```julia
 ranefs = CSVFiles.load("data/raneffects.csv") |> DataFrame
```

### Load Seed Training Features

The differences in seed ranking for every team in the tournament.

```julia
seeds_features = make_seeds(copy(df_seeds), copy(tourney_df))
```

### Load Efficiency Training Features

Team efficiency scores, an advanced feature engineering exercise, ported from [this](https://www.kaggle.com/lnatml/feature-engineering-with-advanced-stats) Python notebook. Basically they measure how efficiently a team capitalizes on possessions.

```julia
Wfdat, Lfdat, effdat = eff_stat_seasonal_means(copy(season_df_detail))
eff_features = get_eff_tourney_diffs(Wfdat, Lfdat, effdat, copy(tourney_df))
```

### Load ELO Training Features

Generate ELO ranks from the data. This required creating a mutable struct that possesses all the features required to calculate the ELO rank, then running an ELO function on the struct.

```julia
season_elos = elo_ranks(Elo())
elo_features = get_elo_tourney_diffs(season_elos, copy(tourney_df))
```

### Momentum

Generate scores capturing momentum late in the season. The code below uses the last two weeks of the season data to get the median team score differences in this set of games.

```julia
momentum_features, momentum_df = make_momentum(copy(tourney_df), copy(season_df))
```

Finally, this code **would** generate team effects by season - team differences from the baseline probability of winning using a Mixed Effects model from the [MixedModels.jl](https://github.com/JuliaStats/MixedModels.jl) package. This package is essentially a port of the 'lmer' package in R.

### Team Effects
```julia
ranef_features = make_ranef_features(copy(tourney_df), ranefs)
```

### Filtering and joining

Most features don't go back very far, so 2003 is the earliest possible complete dataset available. The code below uses the 'filter' functionality in [DataFrames.jl](https://github.com/JuliaData/DataFrames.jl) to reduce the data to what is available.

```julia
seeds_features_min = filter(row -> row[:Season] >= 2003, seeds_features)
eff_features_min = filter(row -> row[:Season] >= 2003, eff_features)
elo_features_min = filter(row -> row[:Season] >= 2003, elo_features)
momentum_features_min = filter(row -> row[:Season] >= 2003, momentum_features)
ranef_features_min = filter(row -> row[:Season] >= 2003, ranef_features)
```

Join all the data to one dataframe for training.

```julia
stub = join(seeds_features_min, eff_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

fdata = join(stub, elo_features, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

fdata = join(fdata, momentum_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

fdata = join(fdata, ranef_features_min, on = [:WTeamID, :LTeamID, :Season, :Result], kind = :left);

exclude = [:Result, :Season, :LTeamID, :WTeamID]
deletecols!(fdata, exclude)
```
