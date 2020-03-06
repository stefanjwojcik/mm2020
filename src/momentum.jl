## create momentum scores for the last TK days of the season

#season_df = load("/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv") |> DataFrame

# Get the final days

function make_momentum(tourney_df, season_df)

    end_of_season = filter(row -> row[:DayNum] > 120, season_df)

    end_of_season.ScoreDiff = end_of_season.WScore - end_of_season.LScore
    deletecols!(season_df, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])

    wins = end_of_season[:, [:ScoreDiff, :WTeamID, :Season]]
    losses = end_of_season[:, [:ScoreDiff, :LTeamID, :Season]]

    rename!(wins, :WTeamID => :TeamID)
    rename!(losses, :LTeamID => :TeamID)

    losses.ScoreDiff .= 1*-losses.ScoreDiff

    fulldf = [wins; losses]

	# aggregate team diffs from end of season
    scores = aggregate(groupby(fulldf, [:TeamID, :Season]), median)
	rename!(scores, :ScoreDiff_median => :ScoreDiff)
	scores_out = copy(scores)
	# rename the colums and merge to winning
	rename!(scores, :ScoreDiff => :WScores, :TeamID => :WTeamID)
	dummy = join(tourney_df, scores, on = [:WTeamID, :Season], kind = :left)
	# rename the columns and merge to losing
	rename!(scores, :WTeamID => :LTeamID, :WScores => :LScores)
	dummy = join(dummy, scores, on = [:LTeamID, :Season], kind = :left)
	dummy.ScoreDiff = dummy.WScores - dummy.LScores

	# Make the tournament feature
	df_wins = copy(dummy[[:Season, :WTeamID, :LTeamID, :ScoreDiff]])
	df_wins.Result = 1

	df_losses = copy(dummy[[:Season, :WTeamID, :LTeamID, :ScoreDiff]])
	df_losses.ScoreDiff = df_losses.ScoreDiff*-1
	df_losses.Result = 0

	println("done")
	momentum_features = [df_wins; df_losses]
	return momentum_features, scores_out

end

function make_momentum_sub(submission_sample, momentum_df)
	submission_sample.ScoreDiff = 0.0
	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		row1 = filter(row -> row[:Season] == season && row[:TeamID] == team1, momentum_df);
		size(row1)[1]==0 && continue
		row2 = filter(row -> row[:Season] == season && row[:TeamID] == team2, momentum_df);
		size(row2)[1]==0 && continue
		submission_sample.ScoreDiff[getfield(row, :row)] = (row1.ScoreDiff - row2.ScoreDiff)[1]
	end
	return submission_sample[:, [:ScoreDiff]]
end
