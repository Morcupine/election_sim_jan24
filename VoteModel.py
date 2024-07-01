import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import json
#from goto import goto, comefrom, label
from append_residual import append_residual
from get_npv_ev import get_npv_ev
from get_state_breakdown import get_state_breakdown
from ideology import scale_candidate_ideology
from normalize import normalize

#predict the 2-party vote each state
output_fundamentals = False
output_regression = True#overwrite the regression dataframe containing the election prediction variables for each state and election
create_new_regression_str = False#Set this to False, unless you want to overwrite model_data.json with new smf.ols regression string

DIME_DATA_RANGE = range(1980,2022,2)
STATE_IDEOLOGY_SCALE = 'nom'#choose measurement of state ideology (e.g. nom, berry_cit, rstone, statelib, dime)
data_range = range(1948,2028,4)

f = open('ideology_coeffs.json')
ideology_coeffs = json.load(f)
state_votes = pd.read_excel('base_data/DemVoteShare28-20.xlsx', index_col = 'year')
state_lean = pd.read_excel('base_data/2PartyLean28-20.xlsx', index_col = 'Year')
delegations = pd.read_csv('essential_data/state_delegations_NOMINATE.csv')
rdi = pd.read_excel('base_data/RDI.xlsx')
fundamentals = pd.read_excel('essential_data/fundamentals.xlsx', index_col = 'Year')
state_liberalism = pd.read_csv('essential_data/state_policy_idealpoints-all.csv')#from dynamics of state liberalism paper
state_liberalism = state_liberalism.set_index(['year', 'abb'])
rstone_state_ideology = pd.read_csv('essential_data/rstone_state_ideology.csv')#rosenstone ratings of state liberalism on ndsw (1948, 1972) and race (1948, 1968)
rstone_state_ideology = rstone_state_ideology.set_index('state')
house = pd.read_csv('house_returns/state_house_returns.csv')
house = house.set_index(['year','state'])
berry = pd.read_excel('base_data/stateideology_v2018.xlsx')
berry = berry.set_index(['year','state'])
dime = pd.read_csv('DIME/dime_state_ideology.csv')
dime = dime.set_index(['year','state'])
nominate_citi = pd.read_csv('Berry_et_al/nominate_citi.csv')
nominate_citi = nominate_citi.set_index(['year','state'])
state_pop = pd.read_excel('essential_data/StatePopTranspose.xlsx', index_col = 'Year')
state_evs = pd.read_excel('essential_data/StateEVsTranspose.xlsx', index_col = 'Year')

#notes: .loc can also be used on an empty dataframe 
#setting an empty dataframe's index to a column and then adding values via .loc[index, 'col'] works
if output_fundamentals:
    fundamentals = pd.DataFrame()
    rdi.index = rdi['date']

    for YEAR in range(1948,2024,4):
        rdi.loc[YEAR, 'RDI']

        #change over one year, from october in the election year to october the year before
        fundamentals.loc[YEAR, 'yr_to_election_change'] = rdi.loc[YEAR + 0.75, 'RDI'] / rdi.loc[YEAR - 0.25, 'RDI']
        fundamentals.loc[YEAR, 'election_year_change'] = rdi.loc[YEAR + 1, 'RDI'] / rdi.loc[YEAR, 'RDI']

        if YEAR >= 1952:
            for i in range(16):
                qrtr = f'y{(i+1)/4}'
                date = YEAR - 4 + 0.25*i + 1.25
                fundamentals.loc[YEAR, f'{qrtr}'] = rdi.loc[date, 'RDI'] / rdi.loc[date - 0.25, 'RDI']

    fundamentals.to_csv('fundamentals_output2.csv')


#create csv with overall ideology scores and dim 1 and 2 -- state 
#df = pd.read_csv('/Users/marcetter/ElectionSimulation/SimulationData/Transposed/ElectionDFDefault.csv')
df = pd.read_csv('base_data/state_codes.csv')
state_codes = list(df['state'])#get list of us state codes (e.g AL, NJ, NY)

if output_regression:
    regression_df = pd.DataFrame()
    for YEAR in data_range: 
        print(f'Tabulating fundamentals for {YEAR}')
        #the congressional delegation ideology measure for a given election year is for the INCUMBENT congress DURING the election
        # Hence, YEAR - 2 is used, and denotes that the measure of congressional ideology in regression_data is from two years prior 
        congress = delegations[delegations['year'] == YEAR - 2]
        congress.index = congress['state']
        for state in state_codes:
            state_row = dict()
            state_row['year'] = YEAR
            state_row['state'] = state
            
            if YEAR != 2024:
                state_row['lean_current'] = state_lean.loc[YEAR][state]
                #for qrtr in range(1, 16, 1):
                #    state_row[f'rdi_q{qrtr}'] = fundamentals.loc[YEAR, f'y{qrtr/4}']#the state row entry will throw an error in regression if it includes '.'
                state_row['dem_share'] = state_votes.loc[YEAR][state]

            #Note: this is lean, not lean towards the incumbent
            state_row['lean_prev'] = state_lean.loc[YEAR-4][state]#state lean, where negative numbers denote democratic and positive republican
            #state_row['lean_avg8'] = np.mean([state_lean.loc[YEAR-4][state], state_lean.loc[YEAR-8][state]])
            #state_row['lean_avg12'] = np.mean([state_lean.loc[YEAR-4][state], state_lean.loc[YEAR-8][state], state_lean.loc[YEAR-12][state]])
            lean_prev2 = state_lean.loc[YEAR-8][state]
            lean_prev3 = state_lean.loc[YEAR-12][state]
            lean_prev4 = state_lean.loc[YEAR-16][state]
            
            state_row['rdi_yr_to_election'] = fundamentals.loc[YEAR, 'yr_to_election_change']
            state_row['rdi_election_year_change'] = fundamentals.loc[YEAR, 'election_yr_change']
            #state_row['inc_approval'] = int(fundamentals.loc[YEAR, 'IncumbentApproval'])
            #state_row['inc_net_approval'] = int(fundamentals.loc[YEAR, 'IncumbentNetApproval'])

            state_row['inc_midterm_change'] = fundamentals.loc[YEAR, 'MidtermChange']
            state_row['inc_has_house'] = fundamentals.loc[YEAR, 'inc_has_house']
            state_row['inc_tenure'] = fundamentals.loc[YEAR, 'IncumbentTenure']
            
            #other economic indicators
            state_row['gdp_yoy'] = fundamentals.loc[YEAR, 'GDPYOY']
            state_row['inflation_yoy'] = fundamentals.loc[YEAR, 'InflationYOY']
            state_row['unemployment'] = fundamentals.loc[YEAR, 'Unemployment']

            state_row['inc_pres'] = fundamentals.loc[YEAR, 'NomineeIsIncumbent']
            if not (state == 'USA') and not (YEAR == 2024):
                state_row['state_pop'] = state_pop.loc[YEAR, state]
            if not (state == 'USA'):
                state_row['state_evs'] = state_evs.loc[YEAR, state]

            #inc_party_cand_approval approval and inc_pres_approval distinguish between 
            #incumbent president and incumbent party candidate when the incumbent president is not running
            #this way, we are not forced to predict incumbent party candidate vote share using incumbent pres approval
            if fundamentals.loc[YEAR, 'NomineeIsIncumbent']:
                state_row['inc_party_cand_approval'] = fundamentals.loc[YEAR, 'IncumbentApproval']
            state_row['inc_pres_approval'] = fundamentals.loc[YEAR, 'IncumbentApproval']

            #fundamentals.index = fundamentals['Year']
            if fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                
                state_row['inc_party'] = 0
                state_row['inc_candidate'] = f'{fundamentals.loc[YEAR, "Democrat"]}_{YEAR}'
                state_row['noninc_candidate'] = f'{fundamentals.loc[YEAR, "Republican"]}_{YEAR}'

                #CANDIDATE IDEOLOGY
                state_row['rosenstone_inc_new_deal'] = fundamentals.loc[YEAR, 'norm_dem_new_deal']
                state_row['rosenstone_inc_race'] = fundamentals.loc[YEAR, 'norm_dem_race']
                state_row['rosenstone_noninc_new_deal'] = fundamentals.loc[YEAR, 'norm_gop_new_deal']
                state_row['rosenstone_noninc_race'] = fundamentals.loc[YEAR, 'norm_gop_race']
                #candidate extremism paper ideology (BPHI denotes the candidate ideology ratings from hi-info ANES respondents 1972-)
                state_row['spliced_BPHI_inc'] = fundamentals.loc[YEAR, 'spliced.BPHI.dem']
                state_row['spliced_BPHI_noninc'] = fundamentals.loc[YEAR, 'spliced.BPHI.rep']
                #CANDIDATE_NOMINATE_SCORES
                #state_row['inc_nom_dim1'] = fundamentals.loc[YEAR, 'dem_nom_dim1_interpolate']
                #state_row['inc_nom_dim2'] = fundamentals.loc[YEAR, 'dem_nom_dim2_interpolate']
                #state_row['noninc_nom_dim1'] = fundamentals.loc[YEAR, 'gop_nom_dim1_interpolate']                
                #state_row['noninc_nom_dim2'] = fundamentals.loc[YEAR, 'gop_nom_dim2_interpolate']
                state_row['inc_nom_dim1'] = fundamentals.loc[YEAR, 'dem_nom_dim1']
                state_row['inc_nom_dim2'] = fundamentals.loc[YEAR, 'dem_nom_dim2']
                state_row['noninc_nom_dim1'] = fundamentals.loc[YEAR, 'gop_nom_dim1']                
                state_row['noninc_nom_dim2'] = fundamentals.loc[YEAR, 'gop_nom_dim2']
                #CANDIDATE_CF_SCORES
                if YEAR in DIME_DATA_RANGE:
                    state_row['inc_cfscore'] = fundamentals.loc[YEAR, 'dem_cfscore']
                    state_row['noninc_cfscore'] = fundamentals.loc[YEAR, 'gop_cfscore']

                #lean towards the incumbent party, where positive values denote higher levels of support for the incumbent
                if not YEAR == 2024:
                    state_row['inc_share'] = state_row['dem_share']
                    state_row['inc_lean_current'] = -1*state_row['lean_current']

                try:
                    state_row['inc_hlean_prev'] = -1*house.loc[(YEAR - 2, state), 'hlean']
                    state_row['inc_hlean_prev2'] = -1*house.loc[(YEAR - 4, state), 'hlean']
                    state_row['inc_hlean_prev3'] = -1*house.loc[(YEAR - 6, state), 'hlean']
                    state_row['inc_hlean_prev4'] = -1*house.loc[(YEAR - 8, state), 'hlean']

                    state_row['inc_hshare_prev'] = house.loc[(YEAR-2,state), 'dem_hshare']
                    state_row['inc_hshare_prev2'] = house.loc[(YEAR-2,state), 'dem_hshare']
                    state_row['inc_hshare_prev3'] = house.loc[(YEAR-2,state), 'dem_hshare']
                    state_row['inc_hshare_prev4'] = house.loc[(YEAR-2,state), 'dem_hshare']
                except:
                    pass

                state_row['inc_lean_prev'] = -1*state_row['lean_prev']
                #state_row['inc_lean_avg8'] = -1*state_row['lean_avg8']
                #state_row['inc_lean_avg12'] = -1*state_row['lean_avg12']
                state_row['inc_lean_prev2'] = -1*lean_prev2
                state_row['inc_lean_prev3'] = -1*lean_prev3
                state_row['inc_lean_prev4'] = -1*lean_prev4

                state_row['inc_lean_trend'] = state_row['inc_lean_prev'] - state_row['inc_lean_prev3']

            elif fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':

                state_row['inc_party'] = 1
                state_row['inc_candidate'] = f'{fundamentals.loc[YEAR, "Republican"]}_{YEAR}'
                state_row['noninc_candidate'] = f'{fundamentals.loc[YEAR, "Democrat"]}_{YEAR}'

                #CANDIDATE IDEOLOGY
                state_row['rosenstone_inc_new_deal'] = fundamentals.loc[YEAR, 'norm_gop_new_deal']
                state_row['rosenstone_inc_race'] = fundamentals.loc[YEAR, 'norm_gop_race']
                state_row['rosenstone_noninc_new_deal'] = fundamentals.loc[YEAR, 'norm_dem_new_deal']
                state_row['rosenstone_noninc_race'] = fundamentals.loc[YEAR, 'norm_dem_race']
                #candidate extremism paper ideology (BPHI denotes the candidate ideology ratings from hi-info ANES respondents 1972-)
                state_row['spliced_BPHI_inc'] = fundamentals.loc[YEAR, 'spliced.BPHI.rep']
                state_row['spliced_BPHI_noninc'] = fundamentals.loc[YEAR, 'spliced.BPHI.dem']
                #CANDIDATE_NOMINATE_SCORES
                #state_row['inc_nom_dim1'] = fundamentals.loc[YEAR, 'gop_nom_dim1_interpolate']
                #state_row['inc_nom_dim2'] = fundamentals.loc[YEAR, 'gop_nom_dim2_interpolate']
                #state_row['noninc_nom_dim1'] = fundamentals.loc[YEAR, 'dem_nom_dim1_interpolate']                
                #state_row['noninc_nom_dim2'] = fundamentals.loc[YEAR, 'dem_nom_dim2_interpolate']
                state_row['inc_nom_dim1'] = fundamentals.loc[YEAR, 'gop_nom_dim1']
                state_row['inc_nom_dim2'] = fundamentals.loc[YEAR, 'gop_nom_dim2']
                state_row['noninc_nom_dim1'] = fundamentals.loc[YEAR, 'dem_nom_dim1']                
                state_row['noninc_nom_dim2'] = fundamentals.loc[YEAR, 'dem_nom_dim2']
                #CANDIDATE_CF_SCORES
                if YEAR in DIME_DATA_RANGE:
                    state_row['inc_cfscore'] = fundamentals.loc[YEAR, 'gop_cfscore']
                    state_row['noninc_cfscore'] = fundamentals.loc[YEAR, 'dem_cfscore']

                #lean towards the incumbent party, where positive values denote higher levels of support for the incumbent
                if not YEAR == 2024:
                    state_row['inc_share'] = 1 - state_row['dem_share']
                    state_row['inc_lean_current'] = state_row['lean_current']

                try:
                    state_row['inc_hlean_prev'] = house.loc[(YEAR - 2, state), 'hlean']
                    state_row['inc_hlean_prev2'] = house.loc[(YEAR - 4, state), 'hlean']
                    state_row['inc_hlean_prev3'] = house.loc[(YEAR - 6, state), 'hlean']
                    state_row['inc_hlean_prev4'] = house.loc[(YEAR - 8, state), 'hlean']

                    state_row['inc_hshare_prev'] = house.loc[(YEAR-2,state), 'gop_hshare']
                    state_row['inc_hshare_prev2'] = house.loc[(YEAR-2,state), 'gop_hshare']
                    state_row['inc_hshare_prev3'] = house.loc[(YEAR-2,state), 'gop_hshare']
                    state_row['inc_hshare_prev4'] = house.loc[(YEAR-2,state), 'gop_hshare']
                except:
                    pass

                state_row['inc_lean_prev'] = state_row['lean_prev']
                #state_row['inc_lean_avg8'] = state_row['lean_avg8']
                #state_row['inc_lean_avg12'] = state_row['lean_avg12']
                state_row['inc_lean_prev2'] = lean_prev2
                state_row['inc_lean_prev3'] = lean_prev3
                state_row['inc_lean_prev4'] = lean_prev4



            if not state == 'USA' and not YEAR == 2024:
                if state_row['inc_share'] > 0.5:
                    state_row['inc_evs'] = state_evs.loc[YEAR, state]
                else:
                    state_row['inc_evs'] = 0

            #state_row['inc_party'] = 0: gop, 1: dem

            #simple difference between candidates, not with respect to states
            
            state_row['spliced_diff_natl'] = np.abs(state_row['spliced_BPHI_inc']) - np.abs(state_row['spliced_BPHI_noninc'])
            """
            state_row['nom_dim1_diff_natl'] = np.abs(state_row['inc_nom_dim1']) - np.abs(state_row['noninc_nom_dim1'])
            state_row['nom_dim2_diff_natl'] = np.abs(state_row['inc_nom_dim2']) - np.abs(state_row['noninc_nom_dim2'])

            state_row['rstone_diff_ndsw'] = np.abs(state_row['rosenstone_inc_new_deal']) - np.abs(state_row['rosenstone_noninc_new_deal'])
            state_row['rstone_diff_race'] = np.abs(state_row['rosenstone_inc_race']) - np.abs(state_row['rosenstone_noninc_race'])
            """
            #previously, values to the below variables were assigned in one if else blocks for each variable
            state_row['inc_home_state'] = 0
            state_row['noninc_home_state'] = 0
            state_row['inc_vp_home_state'] = 0
            state_row['noninc_vp_home_state'] = 0

            confederacy = ['FL', 'TX', 'SC', 'NC', 'AL', 'MS', 'GA', 'TN', 'LA', 'AR', 'VA']
            
            state_row['inc_south_bonus'] = 0
            state_row['noninc_south_bonus'] = 0

            if state in confederacy and fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                if fundamentals.loc[YEAR, 'dem_home_state'] in confederacy:
                    state_row['inc_south_bonus'] = 1
                elif fundamentals.loc[YEAR, 'gop_home_state'] in confederacy:
                    state_row['noninc_south_bonus'] = 1
            elif state in confederacy and fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':
                if fundamentals.loc[YEAR, 'dem_home_state'] in confederacy:
                    state_row['noninc_south_bonus'] = 1
                elif fundamentals.loc[YEAR, 'gop_home_state'] in confederacy:                    
                    state_row['inc_south_bonus'] = 1

            if state == fundamentals.loc[YEAR, 'dem_home_state']:
                if fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                    state_row['inc_home_state'] = 1
                elif fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':
                    state_row['noninc_home_state'] = 1

            if state == fundamentals.loc[YEAR, 'gop_home_state']:
                if fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':
                    state_row['inc_home_state'] = 1
                elif fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                    state_row['noninc_home_state'] = 1

            if state == fundamentals.loc[YEAR, 'demvp_home_state']:
                if fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                    state_row['inc_vp_home_state'] = 1
                elif fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':
                    state_row['noninc_vp_home_state'] = 1

            if state == fundamentals.loc[YEAR, 'gopvp_home_state']:
                if fundamentals.loc[YEAR, 'IncumbentParty'] == 'GOP':
                    state_row['inc_vp_home_state'] = 1
                elif fundamentals.loc[YEAR, 'IncumbentParty'] == 'DEM':
                    state_row['noninc_vp_home_state'] = 1

            #state_row['overall_nom_dim1'] = congress.loc[state, 'overall_nom_dim1']
            #state_row['overall_nom_dim2'] = congress.loc[state, 'overall_nom_dim2']
            #state_row['avg4_overall_nom_dim1'] = congress.loc[state, 'avg4_overall_nom_dim1']
            #state_row['avg4_overall_nom_dim2'] = congress.loc[state, 'avg4_overall_nom_dim2']
                    
            ###PREVIOUS NOMINATE MEASURE###
            #state_row['state_nom_dim1'] = congress.loc[state, 'avg8_overall_nom_dim1']
            #state_row['state_nom_dim2'] = congress.loc[state, 'avg8_overall_nom_dim2']

            ###CONVERTING FROM BERRY ET AL TO NOMINATE VIA REGRESSION###
            try:
                state_row['state_nom_dim1'] = berry.loc[(YEAR,state),'citi6016'] * -0.00802486 + 0.32631745
                state_row['state_nom_dim2'] = congress.loc[state, 'avg8_overall_nom_dim2'] #included for legacy purposes

                ###WEIGHTED MEASURE OF STATE ELECTORATE IDEOLOGY BASED ON VOTE SHARES AND CONGRESSIONAL NOMINATE SCORES###
                state_row['state_citizen_nom'] = nominate_citi.loc[(YEAR-1,state),'ideology']
            except:
                pass
            
            #DYNAMICS OF STATE LIBERALISM SCORES
            if not (state == 'DC' or state == 'USA') and not (YEAR > 2012):
                state_row['state_liberalism'] = state_liberalism.loc[(YEAR,state),'median']

            #berry IDEOLOGY SCORES
            if not (state == 'DC' or state == 'USA') and (YEAR >= 1960) and (YEAR < 2020):
                try:
                    #.loc returns series, perhaps due to unordered double index?
                    state_row['berry_citizen'] = berry.loc[(YEAR,state),'citi6016'].values[0]
                except:
                    state_row['berry_citizen'] = berry.loc[(YEAR,state),'citi6016']
                
                try:
                    state_row['berry_gov'] = berry.loc[(YEAR,state),'inst6017_nom'].values[0]
                except:
                    state_row['berry_gov'] = berry.loc[(YEAR,state),'inst6017_nom']
            
            #ROSENSTONE STATE IDEOLOGY MEASURES, INTERPOLATED 
            if (YEAR >= 1948 and YEAR <= 1972 and not (state == 'DC') and not (state == 'USA')):
                ndsw_1948 = rstone_state_ideology.loc[state, 'rstone_ndsw_1948']
                ndsw_1972 = rstone_state_ideology.loc[state, 'rstone_ndsw_1972']
                race_1948 = rstone_state_ideology.loc[state, 'rstone_race_1948']
                race_1968 = rstone_state_ideology.loc[state, 'rstone_race_1968']

                try:
                    state_row['rstone_state_ndsw'] = ndsw_1972 - ndsw_1948
                    state_row['rstone_state_ndsw'] = state_row['rstone_state_ndsw'] * (YEAR - 1948) / (1972 - 1948) + ndsw_1948
                except:
                    if not np.isnan(rstone_state_ideology.loc[state, 'rstone_ndsw_1948']):
                        state_row['rstone_state_ndsw'] = ndsw_1948
                    if not np.isnan(rstone_state_ideology.loc[state, 'rstone_ndsw_1972']):
                        state_row['rstone_state_ndsw'] = ndsw_1972

                try:
                    state_row['rstone_state_race'] = race_1968 - race_1948
                    state_row['rstone_state_race'] = state_row['rstone_state_race'] * (YEAR - 1948) / (1968 - 1948) + race_1948
                except:
                    if not np.isnan(race_1948):
                        state_row['rstone_state_race'] = race_1948
                    if not np.isnan(race_1968):
                        state_row['rstone_state_race'] = race_1968

            #DIME CFSCORES, STATE AVERAGES
            if YEAR in DIME_DATA_RANGE and not state == 'DC':
                state_row['mean_cfscore_adj'] = dime.loc[(YEAR,state),'mean_cfscore_adj'] 
                state_row['mean_cfscore_unweighted'] = dime.loc[(YEAR,state),'mean_cfscore_unweighted']
                if YEAR <= 1980:
                    state_row['mean_cfscore_adj_smooth'] = dime.loc[(1980,state),'mean_cfscore_adj']
                    state_row['mean_cfscore_smooth'] = dime.loc[(1980,state),'mean_cfscore']
                else:
                    state_row['mean_cfscore_adj_smooth'] = (dime.loc[(YEAR,state),'mean_cfscore_adj'] + dime.loc[(YEAR-2,state),'mean_cfscore_adj']) / 2
                    state_row['mean_cfscore_smooth'] = (dime.loc[(YEAR,state),'mean_cfscore'] + dime.loc[(YEAR-2,state),'mean_cfscore']) / 2
                state_row['mean_cfscore'] = dime.loc[(YEAR,state),'mean_cfscore']
                
            #note: creating a df from a dict and transposing the df causes each element of the dictionary 
            #to become [index, value] tuples
            state_df = pd.DataFrame.from_dict(state_row, orient = 'index')
            state_df = state_df.transpose()
            regression_df = pd.concat([regression_df, state_df])
            regression_df = regression_df.set_index(['year', 'state'], drop = False)

            #scale_candidate_ideology(state, YEAR, STATE_IDEOLOGY_SCALE, ideology_coeffs, regression_df)


    #normalize the survey ratings so that the most extreme Democrat and Republican are at 100 and 0 resp.
    """
    regression_df['spliced_BPHI_noninc'] *= -1
    regression_df['spliced_BPHI_inc'] *= -1
    concat = pd.concat([regression_df['spliced_BPHI_inc'],regression_df['spliced_BPHI_noninc']]) 
    spliced_min = np.min(concat)
    spliced_max = np.max(concat)

    regression_df['spliced_BPHI_noninc'] = (regression_df['spliced_BPHI_noninc'] - spliced_min) / (spliced_max - spliced_min) * 100
    regression_df['spliced_BPHI_inc'] = (regression_df['spliced_BPHI_inc'] - spliced_min) / (spliced_max - spliced_min) * 100
    """
    #NORMALIZE CANDIDATE SPLICE IDEOLOGY TO berry CITIZEN IDEOLOGY
    berry_cit_mean = np.mean(regression_df['berry_citizen'])
    berry_cit_std = np.std(regression_df['berry_citizen'])

    
    ###NORMALIZE COHEN SURVEY RATINGS SO THAT STD AND MEAN ARE SAME AS BERRY STATE CITIZEN IDEOLOGY
    concat = pd.concat([regression_df['spliced_BPHI_inc'],regression_df['spliced_BPHI_noninc']])
    spliced_mean = np.mean(concat)
    spliced_std = np.std(concat)

    regression_df['spliced_BPHI_inc'] = regression_df['spliced_BPHI_inc'] - spliced_mean
    regression_df['spliced_BPHI_inc'] = regression_df['spliced_BPHI_inc'] / spliced_std * berry_cit_std * -1
    regression_df['spliced_BPHI_noninc'] = regression_df['spliced_BPHI_noninc'] - spliced_mean
    regression_df['spliced_BPHI_noninc'] = regression_df['spliced_BPHI_noninc'] / spliced_std * berry_cit_std * -1

    regression_df['spliced_BPHI_inc'] = regression_df['spliced_BPHI_inc'] + berry_cit_mean
    regression_df['spliced_BPHI_noninc'] = regression_df['spliced_BPHI_noninc'] + berry_cit_mean
    
    #NORMALIZE ROSENSTONE #rstone candidate is lib-cons; rstone state is cons-lib; multiply by -1
    regression_df['rstone_state_ndsw'] = (regression_df['rstone_state_ndsw'] - np.mean(regression_df['rstone_state_ndsw'])) * -1
    regression_df['rstone_state_ndsw'] = regression_df['rstone_state_ndsw'] / np.std(regression_df['rstone_state_ndsw']) 

    regression_df['rstone_state_race'] = (regression_df['rstone_state_race'] - np.mean(regression_df['rstone_state_race'])) * -1
    regression_df['rstone_state_race'] = regression_df['rstone_state_race'] / np.std(regression_df['rstone_state_race']) 
    
    concat = pd.concat([regression_df['rosenstone_inc_race'], regression_df['rosenstone_noninc_race']])
    race_mean = np.mean(concat)
    race_std = np.std(concat)
    
    regression_df['rosenstone_inc_race'] = regression_df['rosenstone_inc_race'] - race_mean
    regression_df['rosenstone_inc_race'] = regression_df['rosenstone_inc_race'] / race_std
    regression_df['rosenstone_noninc_race'] = regression_df['rosenstone_noninc_race'] - race_mean
    regression_df['rosenstone_noninc_race'] = regression_df['rosenstone_noninc_race'] / race_std

    #NORMALIZE RSTONE STATE IDEOLOGY SCALES TO MEAN ZERO, STDEV -1
    regression_df['rstone_state_race'] = normalize(regression_df['rstone_state_race'], 0, 1)
    regression_df['rstone_state_ndsw'] = normalize(regression_df['rstone_state_ndsw'], 0, 1)

    #NORMALIZE STATE LIBERALISM TO BERRY SCALE
    regression_df['state_liberalism'] = normalize(regression_df['state_liberalism'], berry_cit_mean, berry_cit_std)
    
    #NORMALIZE NOMINATE SCORES TO BERRY IDEOLOGY SCORES
    #regression_df['inc_nom_dim1'] = normalize(regression_df['inc_nom_dim1'], berry_cit_mean, -1*berry_cit_std)
    #regression_df['noninc_nom_dim1'] = normalize(regression_df['noninc_nom_dim1'], berry_cit_mean, -1*berry_cit_std)

    for YEAR in data_range:
        print(f'Computing relative extremism for {YEAR}')
        for state in state_codes:
            #valid arguments for scale_candidate_ideology: 'rstone', 'statelib', 'berry_cit', 'berry_gov' 
            #notice, some measures may not be normalized
            scale_candidate_ideology(state, YEAR, STATE_IDEOLOGY_SCALE, ideology_coeffs, regression_df)

    regression_df.to_csv('essential_data/regression_data.csv')

def make_regression_str(regress_vars):
        ln = len(regress_vars)
        str = 'inc_share ~ '
        for var in regress_vars:
            if regress_vars[var] == 1:
                str += var
                if not (var == list(regress_vars.keys())[ln-1]):
                    str += ' + '

        if str[len(str)-3: len(str)] == ' + ':#if last character is plus, drop to avoid error 
            str = str[:len(str) - 2]

        return str

if create_new_regression_str:
    #'inc_share ~ berry_inc_minus_noninc_citizen + 
    # inc_hlean_prev + 
    # inc_hlean_prev2 + 
    # inc_hshare_prev + 
    # inc_lean_prev + 
    # inc_lean_prev2 + 
    # inc_pres + 
    # rdi_yr_to_election + 
    # inflation_yoy + 
    # inc_tenure + 
    # inc_home_state + 
    # noninc_home_state'

    regress_vars = {
        #'state': 1,
        'berry_inc_minus_noninc_citizen': 1,
        'inc_pres': 1,
        #'noninc_candidate': 0,
        #'inc_party_cand_approval':0,
        #'inc_pres_approval': 0,
        'inc_lean_prev': 1,
        'inc_lean_prev2': 1,
        'inc_hshare_prev': 1,
        'inc_hlean_prev': 1,
        'inc_hlean_prev2': 1,
        'rdi_yr_to_election': 1,
        'inflation_yoy': 1,
        'inc_tenure': 1,
        'inc_home_state': 1,
        'noninc_home_state': 1,
        'lean_avg8': 0#lean where neg denotes dem, pos denotes gop
    }

    regression_str = make_regression_str(regress_vars)
    model_data = {'regression_str': regression_str}
    with open(f"model_data.json", "w") as outfile:
        json.dump(model_data, outfile)

regression_df = pd.read_csv('essential_data/regression_data.csv')
regression_df = regression_df.set_index(['year', 'state'], drop = False)

f = open('model_data.json')
ols_data = json.load(f)

results = smf.ols(ols_data['regression_str'], data = regression_df).fit()
residual_df = append_residual(results, regression_df)
npv_ev = get_npv_ev(residual_df)
state_breakdown = get_state_breakdown(residual_df)

#above line throws an error when output_regression = True and the program reads regression_df created in process of running program
#when the same dataframe is read through pd.read_csv, the error disappears for some reason

print(results.summary())
