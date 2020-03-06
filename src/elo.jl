# calculate season elo rankings from here: https://www.kaggle.com/lpkirwin/fivethirtyeight-elo-ratings
"""
This file is responsible for creating seasonal elo rankings as a predictive feature
"""

#def elo_pred(elo1, elo2):
#	return(1. / (10. ** (-(elo1 - elo2) / 400.) + 1.))

function elo_pred(elo1::Float64, elo2::Float64)
	1.0 / (10.0 ^ (-(elo1 - elo2) / 400.0) + 1.0)
end

#def expected_margin(elo_diff):
#	return((7.5 + 0.006 * elo_diff))

function expected_margin(elo_diff)
	(7.5 + 0.006 * elo_diff)
end

#def elo_update(w_elo, l_elo, margin):
#	K = 20.
#	elo_diff = w_elo - l_elo
#	pred = elo_pred(w_elo, l_elo)
#	mult = ((margin + 3.) ** 0.8) / expected_margin(elo_diff)
#	update = K * mult * (1 - pred)
#	return(pred, update)

function elo_update(w_elo, l_elo, margin)
	K = 20.0
	elo_diff = w_elo - l_elo
	pred = elo_pred(w_elo, l_elo)
	mult = ((margin + 3.0)^0.8) / expected_margin(elo_diff)
	update = K * mult * (1 - pred)
	pred, update
end

# FINAL ELO FOR THE SEASON: NEED TO LOOK IN WINNER OR LOSER POSITION

function final_elo_per_season(df::DataFrame, team_id::Int64)
	d = copy(df)
	d = d[ (d.WTeamID .== team_id) .| (d.LTeamID .== team_id ), : ]
	sort!(d, (:Season, :DayNum), rev=true)
	unique!(d, :Season)
	w_mask = d.WTeamID .== team_id
	l_mask = d.LTeamID .== team_id
	d.season_elo = 0.0
	d.season_elo[w_mask] .= d.w_elo[w_mask]
	d.season_elo[l_mask] .= d.l_elo[l_mask]
	out = DataFrame(team_id = team_id, season = d.Season, season_elo = d.season_elo)
end

mutable struct Elo
	data_path::String
	rs::DataFrame
	HOME_ADVANTAGE::Float64
	team_ids::Array{Int64, 1}
	elo_dict::Dict
	function Elo(data_path::String="",
		rs::DataFrame = DataFrame(),
		HOME_ADVANTAGE::Float64 = 100.0,
		team_ids::Array{Int64, 1} = Int64[],
		elo_dict::Dict = Dict())
        new(data_path, rs, HOME_ADVANTAGE, team_ids, elo_dict)
    end
end
	# I'm going to iterate over the games dataframe using
	# index numbers, so want to check that nothing is out
	# of order before I do that.

function iterate_games(elo_obj::Elo)

	# update the basic info
	elo_obj.data_path = "/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv"
	elo_obj.rs = load(elo_obj.data_path) |> DataFrame
	elo_obj.rs.margin = elo_obj.rs.WScore - elo_obj.rs.LScore
	elo_obj.HOME_ADVANTAGE = 100.0
	elo_obj.team_ids = unique([ elo_obj.rs.WTeamID ; elo_obj.rs.LTeamID ]) #all team ids
	elo_obj.elo_dict = Dict(); [elo_obj.elo_dict[x] = 1500 for x in elo_obj.team_ids]

	preds = []
	w_elo = []
	l_elo = []

	print("looping over every game in the dataframe")
	# Loop over all rows of the games dataframe
	#for row in self.rs.itertuples():

	for row in eachrow(elo_obj.rs)
		w = row.WTeamID
		l = row.LTeamID
		margin = row.margin
		wloc = row.WLoc

		# Does either team get a home-court advantage?
		w_ad, l_ad, = 0., 0.
		if wloc == "H"
			w_ad += elo_obj.HOME_ADVANTAGE
		elseif wloc == "A"
			l_ad += elo_obj.HOME_ADVANTAGE
		end
		# Get elo updates as a result of the game
		pred, update = elo_update(elo_obj.elo_dict[w] + w_ad,
								  elo_obj.elo_dict[l] + l_ad,
								  margin)
		elo_obj.elo_dict[w] += update
		elo_obj.elo_dict[l] -= update

		# Save prediction and new Elos for each round
		push!(preds, pred)
		push!(w_elo, elo_obj.elo_dict[w])
		push!(l_elo, elo_obj.elo_dict[l])
	end

	elo_obj.rs.w_elo = w_elo
	elo_obj.rs.l_elo = l_elo
	print("done")
	return elo_obj
end

# this is a superior function
function elo_ranks(elo_obj::Elo)
	# load data
	elo_obj = iterate_games(elo_obj)
	print("computing elo for each team..")
	df_list = [final_elo_per_season(elo_obj.rs, id) for id in elo_obj.team_ids]
	season_elos = df_list[1] # create a stub dataset
	[append!(season_elos, df_list[x]) for x in 2:length(df_list)] # append everything
	season_elos
end

function get_elo_tourney_diffs(season_elos, df_tour)
	# create difference scores
	df_winelo, df_losselo = copy(season_elos), copy(season_elos)
	rename!(df_winelo, :team_id => :WTeamID, :season => :Season, :season_elo => :W_elo)
	rename!(df_losselo, :team_id => :LTeamID, :season => :Season, :season_elo => :L_elo)
	# Merge in the compact results
	df_dummy = join(df_tour, df_winelo, on = [:Season, :WTeamID], kind = :left)
	df_concat = join(df_dummy, df_losselo, on = [:Season, :LTeamID])
	df_concat.Elo_diff = df_concat.W_elo - df_concat.L_elo
	deletecols!(df_concat, [:W_elo, :L_elo])

	df_wins = DataFrame()
	df_wins = copy(df_concat[ :, [:Season, :WTeamID, :LTeamID, :Elo_diff]])
	df_wins.Result = 1

	df_losses = DataFrame()
	df_losses = copy(df_concat[ :, [:Season, :WTeamID, :LTeamID]])
	df_losses.Elo_diff = -1*df_concat.Elo_diff
	df_losses.Result = 0

	df_out = [df_wins; df_losses]

	println("done")
	df_out
end

function get_elo_submission_diffs(submission_sample, season_elos)
	submission_sample.Elo_diff = -99.0
	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		row1 = filter(row -> row[:season] == season && row[:team_id] == team1, season_elos);
		row1 = mapcols(x -> mean(x), copy(select(row1, :season_elo))) #lambda fn for cols
		row2 = filter(row -> row[:season] == season && row[:team_id] == team2, season_elos);
		row2 = mapcols(x -> mean(x), copy(select(row2, :season_elo))) #lambda fn for cols
		submission_sample.Elo_diff[getfield(row, :row)] = (row1.season_elo - row2.season_elo)[1]
	end
	return submission_sample[:, [:Elo_diff]]
end
