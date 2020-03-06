"""
This file is responsible for creating basic ncaa seeds and for creating the outcome 'result'
"""
using DataFrames, CSVFiles

# need to create a utility function to map team combos to WTeamID, LTeamID
function create_ID_crosswalk(df)
	df_min = select(df, [:WTeamID, :LTeamID])
	matchup_code = String[]
	global count = 0
	for row in eachrow(df_min)
		push!(matchup_code, string(row[argmin(row)]) *"-"* string(row[argmax(row)]))
		global count += 1
	end
	return matchup_code
end

# The goal is to create historical matchup scores

#data_dir = '../input/'
function historic_tourney_matchups(df_path = "/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
	println("loading data..")

	# get the basic tour data
	df_tour = load(df_path) |> DataFrame;

	# generate a modelable version of the tourney data for later merging
	df_tour_norm = copy(df_tour)
	deletecols!(df_tour_norm, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])
	df_tour_norm.Result = 1
	df_tour_inverse = copy(df_tour_norm)
	df_tour_inverse.Result = 0
	df_tour_double = vcat(df_tour_norm, df_tour_inverse)
	df_tour_double.teamIDcross = create_ID_crosswalk(df_tour_double)

	# Create historical difference
	df_hist_norm = copy(df_tour)
	df_hist_norm.teamIDcross = create_ID_crosswalk(df_hist_norm)
	df_hist_norm.Diff_TScore = df_hist_norm.WScore - df_hist_norm.LScore
	deletecols!(df_hist_norm, [:DayNum, :WScore, :LScore, :WLoc, :NumOT, :WTeamID, :LTeamID])
	df_hist_norm = aggregate(df_hist_norm, [:teamIDcross, :Season], mean)
	df_hist_norm.Result = 1
	df_inverse = copy(df_hist_norm)
	df_inverse.Diff_TScore_mean = -1 .* df_inverse.Diff_TScore_mean
	df_inverse.Result = 0

	df_hist = vcat(df_hist_norm, df_inverse)
	# update the year so that you have priors
	df_hist.Season = df_hist.Season .+ 1

	# now merge the historical matchups with the current tourney data
	df_out = join(df_tour_double, df_hist, on = [:teamIDcross, :Season, :Result], kind = :left)
	for row in eachrow(df_out)
		!ismissing(row.Diff_TScore_mean) && continue
		season, teamIDcross, Result = row.Season, row.teamIDcross, row.Result
		dprior = filter(row -> row[:Season] < season && row[:teamIDcross]==teamIDcross && row.Result==Result, df_out)
		size(dropmissing(dprior))[1]==0 && continue
		val = by(dropmissing(dprior), [:teamIDcross],  :Diff_TScore_mean => x -> mean(x))
		row.Diff_TScore_mean = val.Diff_TScore_mean_function[1]
	end
	df_out.Diff_TScore_mean[ismissing.(df_out.Diff_TScore_mean)] .= 0
	return df_out
end

function historic_tourney_sub(submission_sample, hist_tourney_df)
	submission_sample.Diff_TScore_mean = 0.0
	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		teamIDcross = string(team1)*"-"*string(team2)
		# get values for team1 and team
		val = filter(row -> row[:Season] == season && row[:teamIDcross] == teamIDcross, hist_tourney_df);
		size(val)[1]==0 && continue
		submission_sample.Diff_TScore_mean[getfield(row, :row)] = val.Diff_TScore_mean[1]
	end
	#submission_sample.Diff_TScore_mean[ismissing.(submission_sample.Diff_TScore_mean)] .= 0.0
	return submission_sample[:, [:Diff_TScore_mean]]
end

function historic_season_matchups(
	df_path = "/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv",
	df_tour_path = "/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
	println("loading data..")

	# get the basic season data
	df_season = load(df_path) |> DataFrame;
	# generate a modelable version of the tourney data for later merging
	df_season_norm = copy(df_season)
	deletecols!(df_season_norm, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])
	df_season_norm.Result = 1
	df_season_inverse = copy(df_season_norm)
	df_season_inverse.Result = 0
	df_season_double = vcat(df_season_norm, df_season_inverse)
	df_season_double.teamIDcross = create_ID_crosswalk(df_season_double)

	# Create historical difference
	df_hist_norm = copy(df_season)
	df_hist_norm.teamIDcross = create_ID_crosswalk(df_hist_norm)
	df_hist_norm.Diff_TScore = df_hist_norm.WScore - df_hist_norm.LScore
	deletecols!(df_hist_norm, [:DayNum, :WScore, :LScore, :WLoc, :NumOT, :WTeamID, :LTeamID])
	df_hist_norm = aggregate(df_hist_norm, [:teamIDcross, :Season], mean)
	df_hist_norm.Result = 1
	df_inverse = copy(df_hist_norm)
	df_inverse.Diff_TScore_mean = -1 .* df_inverse.Diff_TScore_mean
	df_inverse.Result = 0

	df_hist = vcat(df_hist_norm, df_inverse)

	# Load the tour data
	df_tour_norm = load(df_tour_path) |> DataFrame
	deletecols!(df_tour_norm, [:DayNum, :WScore, :LScore, :WLoc, :NumOT])
	df_tour_norm.Result = 1
	df_tour_inverse = copy(df_tour_norm)
	df_tour_inverse.Result = 0
	df_tour_double = vcat(df_tour_norm, df_tour_inverse)
	df_tour_double.teamIDcross = create_ID_crosswalk(df_tour_double)


	# now merge the historical matchups with the current tourney data
	df_out = join(df_tour_double, df_hist, on = [:teamIDcross, :Season, :Result], kind = :left)
	# IF the value of history is missing take the mean over prior contests
	for row in eachrow(df_out)
		!ismissing(row.Diff_TScore_mean) && continue
		season, teamIDcross, Result = row.Season, row.teamIDcross, row.Result
		dprior = filter(row -> row[:Season] < season && row[:teamIDcross]==teamIDcross && row.Result==Result, df_out)
		size(dropmissing(dprior))[1]==0 && continue
		val = by(dropmissing(dprior), [:teamIDcross],  :Diff_TScore_mean => x -> mean(x))
		row.Diff_TScore_mean = val.Diff_TScore_mean_function[1]
	end
	df_out.Diff_TScore_mean[ismissing.(df_out.Diff_TScore_mean)] .= 0
	return df_out
end


function historic_season_sub(submission_sample, hist_season_df)
	submission_sample.Diff_TScore_mean = 0.0
	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		teamIDcross = string(team1)*"-"*string(team2)
		# get values for team1 and team
		val = filter(row -> row[:Season] == season && row[:teamIDcross] == teamIDcross, hist_season_df);
		size(val)[1]==0 && continue
		submission_sample.Diff_TScore_mean[getfield(row, :row)] = val.Diff_TScore_mean[1]
	end
	#submission_sample.Diff_TScore_mean[ismissing.(submission_sample.Diff_TScore_mean)] .= 0.0
	return submission_sample[:, [:Diff_TScore_mean]]
end
