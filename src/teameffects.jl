using CSVFiles, MixedModels, DataFrames

#season_df = CSVFiles.load("/home/swojcik/mm2020/data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv") |> DataFrame
#tourney_df  = CSVFiles.load("/home/swojcik/mm2020/data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv") |> DataFrame

function make_team_effects(season_df)

	season_df.LTeamID = CategoricalArray(season_df.LTeamID)
	season_df.WTeamID = CategoricalArray(season_df.WTeamID)
	season_df.Season = CategoricalArray(season_df.Season)

	wins = select(season_df, [:WTeamID, :Season])
	rename!(wins, :WTeamID => :TeamID)
	wins.Result = 1

	losses = select(season_df, [:LTeamID, :Season])
	rename!(losses, :LTeamID => :TeamID)
	losses.Result = 0

	dat = [wins;losses]
	sort!(dat, ( :Season, :TeamID))
	# Create an ID for random effects
	dat.ID = string.(dat.Season) .*"-".* string.(dat.TeamID)

	verbaggform = @formula(Result ~ 1+(1|ID));

	gm2 = fit(MixedModel, verbaggform, dat, Bernoulli())

	stub = DataFrame(ID = unique(dat.ID), ranef = Array(transpose(gm2.b[1]))[:, 1])
	out = join(dat, stub, on = :ID, kind = :left)
	out = unique(select(out, Not(:Result)))
	println("Saving Random Effects to raneffects.csv")
	save("raneffects.csv", select(out, Not(:ID)))
end

function make_ranef_features(tourney_df, ranefs)
	#tourney_df.ranef = 0.0
	wins  = rename(copy(ranefs), :TeamID => :WTeamID, :ranef => :wranefs)
	losses  = rename(copy(ranefs), :TeamID => :LTeamID, :ranef => :lranefs)
	stub = join(tourney_df, wins, on = [:WTeamID, :Season], kind = :left)
	full = join(stub, losses, on = [:LTeamID, :Season], kind = :left)
	full.RanefDiff = full.wranefs - full.lranefs

	df_wins = copy(full[[:Season, :WTeamID, :LTeamID, :RanefDiff]])
	df_wins.Result = 1

	df_losses = copy(full[[:Season, :WTeamID, :LTeamID]])
	df_losses.RanefDiff = full.RanefDiff*-1
	df_losses.Result = 0

	println("done")
	df_predictions = [df_wins; df_losses]
end


function make_ranef_sub(submission_sample, ranefs)
	submission_sample.RanefDiff = -99.0
	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		# get seeds for team1 and team
		row1 = filter(row -> row[:Season] == season && row[:TeamID] == team1, ranefs);
		row2 = filter(row -> row[:Season] == season && row[:TeamID] == team2, ranefs);
		submission_sample.RanefDiff[getfield(row, :row)] = (row1.ranef - row2.ranef)[1]
	end
	return submission_sample[:, [:RanefDiff]]
end
