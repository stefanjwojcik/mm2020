# efficiency scores - need to convert from PYTHON

## FROM : https://www.kaggle.com/lnatml/feature-engineering-with-advanced-stats
"""
eff_stats()
This file is responsible for creating 'advanced' features related to team efficiencies
"""

function eff_stat_seasonal_means(df) # this is season_df_detail
	#Points Winning/Losing Team
	df.WPts = 2*df.WFGM + df.WFGM3 + df.WFTM
	df.LPts = 2*df.LFGM + df.LFGM3 + df.LFTM
	#
	#Calculate Winning/losing Team Possesion Feature
	wPos = .96*(df.WFGA + df.WTO + 0.44*df.WFTA - df.WOR)
	lPos = .96*(df.LFGA + df.LTO + 0.44*df.LFTA - df.LOR)
	#two teams use almost the same number of possessions in a game
	#(plus/minus one or two - depending on how quarters end)
	#so let's just take the average
	df.Pos = (wPos+lPos)/2
	#
	print("computing offensive/defensive rating...")
	#Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
	df.WOffRtg = 100 * (df.WPts ./ df.Pos)
	df.LOffRtg = 100 * (df.LPts ./ df.Pos)
	#Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
	df.WDefRtg = df.LOffRtg
	df.LDefRtg = df.WOffRtg
	#Net Rating = Off.Rtg - Def.Rtg
	df.WNetRtg = df.WOffRtg - df.WDefRtg
	df.LNetRtg = df.LOffRtg - df.LDefRtg
	#Assist Ratio : Percentage of team possessions that end in assists
	df.WAstR = df.WAst ./ (df.WFGA + .44*df.WFTA + df.WAst + df.WTO)
	df.LAstR = df.LAst ./ (df.LFGA + .44*df.LFTA + df.LAst + df.LTO)

	print("computing turnovers...")
	#Turnover Ratio: Number of turnovers of a team per 100 possessions used.
	#(TO * 100) / (FGA + (FTA * 0.44) + AST + TO)
	df.WTOR = 100*df.WTO ./ (df.WFGA + .44*df.WFTA + df.WAst + df.WTO)
	df.LTOR = 100*df.LTO ./ (df.LFGA + .44*df.LFTA + df.LAst + df.LTO)
	#The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
	df.WTSP = df.WPts ./ (2 * (df.WFGA + .44 * df.WFTA))
	df.LTSP = df.LPts ./ (2 * (df.LFGA + .44 * df.LFTA))
	#eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
	df.WeFGP = (df.WFGM + 0.5 * df.WFGM3) ./ df.WFGA
	df.LeFGP = (df.LFGM + 0.5 * df.LFGM3) ./ df.LFGA
	#FTA Rate : How good a team is at drawing fouls.
	df.WFTAR = df.WFTA ./ df.WFGA
	df.LFTAR = df.LFTA ./ df.LFGA
	#OREB% : Percentage of team offensive rebounds
	df.WORP = df.WOR ./ (df.WOR + df.LDR)
	df.LORP = df.LOR ./ (df.LOR + df.WDR)
	#DREB% : Percentage of team defensive rebounds
	df.WDRP = df.WDR ./ (df.WDR + df.LOR)
	df.LDRP = df.LDR ./ (df.LDR + df.WOR)
	#REB% : Percentage of team total rebounds
	df.WRP = (df.WDR + df.WOR) ./ (df.WDR + df.WDR + df.LDR + df.LOR)
	df.LRP = (df.LDR + df.LOR) ./ (df.WDR + df.WDR + df.LDR + df.LOR)
	# Drop original measures
	#deletecols!(df, [:WFGM, :WFGA, :WFGM3, :WFGA3, :WFTM,
	#:WFTA, :WOR, :WDR, :WAst, :WTO, :WStl, :WBlk, :WPF, :WLoc])
	#deletecols!(df, [:LFGM, :LFGA, :LFGM3, :LFGA3, :LFTM,
	#:LFTA, :LOR, :LDR, :LAst, :LTO, :LStl, :LBlk, :LPF])

	#instead just dropping a few
	deletecols!(df, [:WLoc])

	# take mean, min, max of each of the advanced measures
	W_cols = [!in(x, [ "WScore"]) & occursin(r"W|Season", x)  for x in String.(names(df))]	# make win and loss average datasets
	L_cols = [!in(x, [ "LScore"]) & occursin(r"L|Season", x)  for x in String.(names(df))]	# make win and loss average datasets

	Wmean = aggregate(df[W_cols], [:WTeamID, :Season], mean)
	alt_names = [Symbol(replace(String(x), "W" => "")) for x in names(Wmean)]
	# And actually alter the names in place
	names!(Wmean, alt_names)
	# losing team
	Lmean = aggregate(df[L_cols], [:LTeamID, :Season], mean)

	alt_names = [Symbol(replace(String(x), "L" => "")) for x in names(Lmean)]
	# And actually alter the names in place
	names!(Lmean, alt_names)
	# concatenate both and take average over team and season
	fdat = [Wmean;Lmean] # this is how you concatenate in JULIA
	# get the mean when winning/losing
	fdat_mean = aggregate(fdat, [:TeamID, :Season], mean)
	alt_names = [Symbol(replace(String(x), "_mean" => "")) for x in names(fdat_mean)]
	names!(fdat_mean, alt_names)
	# create two functions - for when team wins/loses for merging
	Wfdat = copy(fdat_mean)
	Wfdat_names = Symbol.([x == "Season" ? x : "W"*x for x in String.(names(Wfdat))])
	names!(Wfdat, Wfdat_names)
	Lfdat = copy(fdat_mean)
	Lfdat_names = Symbol.([x == "Season" ? x : "L"*x for x in String.(names(Lfdat))])
	names!(Lfdat, Lfdat_names)
	return Wfdat, Lfdat, fdat_mean
end

function get_eff_tourney_diffs(Wfdat, Lfdat, fdat, df_tour)
	# NEED TO MAKE THIS COMPATIBLE WITH THE REST OF THE DATA: TAKE DIFFS AND CONCATENATE
	df_tour = load("/home/swojcik/github/mm2020.jl/data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv") |> DataFrame

	deletecols!(df_tour, [ :WScore, :LScore, :WLoc, :NumOT])
	df = join(df_tour, Wfdat, on = [:Season, :WTeamID], kind = :left)
	df = join(df, Lfdat, on = [:Season, :LTeamID], kind = :left)

		# Option to create two functions - one to create
	df_concat = DataFrame()
	vars_to_add = [String(x) for x in names(fdat) if !in(x, [:Season, :TeamID])]
	for var in vars_to_add
		df_concat[Symbol("Diff_"*var)] = df[Symbol("W"*var)]-df[Symbol("L"*var)]
	end

	pred_vars = names(df_concat)
	df_concat.WTeamID = df.WTeamID
	df_concat.LTeamID = df.LTeamID
	df_concat.Season = df.Season

	df_wins = copy(df_concat)
	df_wins.Result = 1

	df_losses = copy(df_concat[:, [:Season, :WTeamID, :LTeamID]])
	df_losses[:, pred_vars] = mapcols(x -> x*-1, copy(select(df_concat, pred_vars))) #lambda fn for cols
	df_losses.Result = 0

	df_out = [df_wins; df_losses]
	return dropmissing(df_out)
end

# drop missing obs in the data
#dropmissing!(eff_stats())
################# JULIA ############

function get_eff_submission_diffs(submission_sample, fdat)
	final_out = DataFrame()
	# The variables to take diffs
	vars_to_add = [String(x) for x in names(fdat) if !in(x, [:Season, :TeamID])]

	for row in eachrow(submission_sample)
		season, team1, team2 = parse.(Int, split(row.ID, "_"))
		# filter each dataframe to get each team avg when winning/losing - team 1
		row1 = filter(row -> row[:Season] == season && row[:TeamID] == team1, fdat);
		row1 = mapcols(x -> mean(x), copy(select(row1, Symbol.(vars_to_add)))) #lambda fn for cols
		row2 = filter(row -> row[:Season] == season && row[:TeamID] == team2, fdat);
		row2 = mapcols(x -> mean(x), copy(select(row2, Symbol.(vars_to_add)))) #lambda fn for cols
		out = DataFrame()
		# small loop to take diffs
		for var in vars_to_add
			out[Symbol("Diff_"*var)] = row1[Symbol(var)]-row2[Symbol(var)]
		end
		# append the final data
		append!(final_out, out)
	end
	return final_out
end
