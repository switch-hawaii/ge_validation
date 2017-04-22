#!/usr/bin/env python

import time, sys, os, textwrap, datetime, re, collections, itertools

try:
    import pandas as pd
except ImportError:
    print "This script requires the pandas module to manipulate data."
    print "Please use pip or another package manager to install pandas."
    raise

import numpy as np  # should succeed if pandas did

try:
    import openpyxl
except ImportError:
    print "This script requires the openpyxl module to access the data in Microsoft Excel files."
    print "Please use pip or another package manager to install openpyxl."
    raise

oahu_only = True
one_week = True

scenario_numbers = [2, 16] # range(2, 10) #[2, 16]
scenarios = pd.DataFrame.from_records(
    [
        (
            s, 
            "inputs_" + str(s).zfill(2) 
                + ('_forecasted' if f else '') 
                + ('_week' if one_week else '_year'), 
            f
        )
            for s in scenario_numbers for f in [False] # [True, False]
    ],
    columns=['scenario', 'scenario_dir', 'forecasted']
)
# this is used often, so we define it once here
scenario_dirs = list(scenarios[['scenario', 'scenario_dir']].itertuples(index=False))

# TODO: store all of this in the scenarios dataframe using simpler code
ge_scenario_sheets = {s: (
    os.path.join('GE RPS Study Input Data', 'Scenarios1-9.xlsx' if s <= 9 else 'Scenarios10-18.xlsx'),
    (
        'Oahu-s1s2' 
        if (1 <= s <= 2)
        else (( 
            'Oahu-s' 
            if 3 <= s <= 9
            else 's'
        ) + str(s))
    )
) for (s, _) in scenario_dirs}

# study_timepoints = [
#     ((2020 * 100 + month) * 100 + 15) * 100 + hour
#         for month in range(1, 13) for hour in range(24)
# ]
# first two weeks of june
# study_timepoints = [
#     ((2020 * 100 + 6) * 100 + day) * 100 + hour
#         for day in range(1, 15) for hour in range(24)
# ]

if one_week:
    # dates shown in Figures 8 & 9
    study_dates = [datetime.date(2020, 6, 22) + datetime.timedelta(days=d) for d in range(7)]
else:
    # all year
    study_dates = [datetime.date(2020, 1, 1) + datetime.timedelta(days=d) for d in range(366)]
study_timepoints = [
    ((date.year * 100 + date.month) * 100 + date.day) * 100 + hour
        for date in study_dates for hour in range(24)
]

# number of hours per timepoint
timepoint_duration = 1
# all projects should have an energy source in the list of fuels or in this list
non_fuel_energy_sources = ["Solar", "Wind", "Waste", "Biomass"]

def main():
    # run all the data creation functions (or at least the ones we want now)
    make_dirs()
    scenario_list()
    variable_capacity_factors()
    loads()
    load_zones()
    timescales()
    financials()
    fuels()
    gen_techs()
    project_info()
    proj_commit_bounds()

open_workbooks = dict()

def source_data(filename):
    """Convert the filename to a path in the source data directory."""
    return os.path.join('source_data', filename)
    
def get_workbook(xlsx_file):
    if xlsx_file not in open_workbooks:
        # load the file, ignore formula text
        open_workbooks[xlsx_file] = openpyxl.load_workbook(source_data(xlsx_file), data_only=True, read_only=True)
    return open_workbooks[xlsx_file]

def get_region(xlsx_file, range_name):
    # get a single rectangular region from the specified file in the source data directory
    wb = get_workbook(xlsx_file)
    if '!' in range_name:
        # passed a worksheet!cell reference
        ws_name, reg = range_name.split('!')
        if ws_name.startswith("'") and ws_name.endswith("'"):
            # optionally strip single quotes around sheet name
            ws_name = ws_name[1:-1]
        region = wb[ws_name][reg]
    else:
        # passed a named range
        full_range = wb.get_named_range(range_name)
        if full_range is None:
            raise ValueError(
                'Range "{}" not found in workbook "{}".'.format(range_name, xlsx_file))
        if len(full_range.destinations) > 1:
            raise ValueError(
                'Range "{}" in workbook "{}" contains more than one region.'.format(range_name, xlsx_file))
        ws, reg = full_range.destinations[0]
        region = ws[reg]
    return region

def data_frame_from_xlsx(xlsx_file, range_name):
    region = get_region(xlsx_file, range_name)
    return pd.DataFrame([cell.value for cell in row] for row in region)

def value_from_xlsx(xlsx_file, range_name):
    region = get_region(xlsx_file, range_name)
    if isinstance(region, collections.Iterable):
        raise ValueError(
            'Range "{}" in workbook "{}" does not refer to an individual cell.'.format(
                range_name, xlsx_file))
    return region.value

def dedent(text):
    return textwrap.dedent(text).lstrip()

def make_dirs():
    for scenario, scenario_dir in scenario_dirs:
        try:
            os.makedirs(scenario_dir)
        except OSError:
            # probably the directory already exists
            pass

def get_rps_project_info(columns=None):
    """ Retrieve the specified columns from Hawaii RPS Study Generator Table OCR.xlsx,
    optionally renaming them """
    
    proj_info = data_frame_from_xlsx(
        "Hawaii RPS Study Generator Table OCR.xlsx", "generator_data",
    )
    proj_info.columns = [c.replace("\n", " ") for c in proj_info.loc[0]]
    proj_info.columns.names = [None]
    proj_info = proj_info.loc[2:]     # drop the header and units rows
    
    # convert spaces to underscores in project names, so that doesn't have to be done elsewhere
    # (Pyomo treats space as a column delimiter in .tab files, even if quoted)
    proj_info["Plant"].replace(' ', '_', inplace=True, regex=True) 

    if columns:
        # convert single string to a list
        if isinstance(columns, str):
            columns = [columns]
        # if they just supply a string, assume they want to use the name as-is
        cols = [[r, r] if isinstance(r, str) else r for r in columns]
        orig_cols, new_cols = map(list, zip(*cols))

        # select and rename columns
        proj_info = proj_info[orig_cols]
        proj_info.columns = new_cols

    return proj_info

def scenario_list():
    with open('scenarios.txt', 'w') as f:
        for (scenario, scenario_dir, forecasted) in scenarios[['scenario', 'scenario_dir', 'forecasted']].itertuples(index=False):
            f.write(
                '--scenario-name scen{n:0>2}{fc} --inputs-dir {d}\n'.format(
                    n=scenario, fc='_forecast' if forecasted else '', d=scenario_dir
                )
            )

def waste_plant_capacity_factor():
    return value_from_xlsx(
         "RPS Study Resources.xlsx", "waste_plant_capacity_factor"
    )

def variable_capacity_factors():
    """ Create tables of capacity factors and reserve requirements for each scenario. """

    # get names and column-finding regexes for wind and solar projects
    proj_info = data_frame_from_xlsx("RPS Study Resources.xlsx", "renewable_project_info")
    proj_info = proj_info.T.set_index(0).T    # set column headers
    # if oahu_only:
    #     proj_info = proj_info[proj_info['Load Zone']=='Oahu']
    project_names = proj_info["Project Name"]
    project_regex = {
        p: re.compile(r)
            for p, r in proj_info[["Project Name", "Column Expression"]].itertuples(index=False)
    }

    # Get capacity of wind and solar built in each scenario
    capacities = data_frame_from_xlsx("RPS Study Resources.xlsx", "scenario_capacities")
    capacities.set_index(0, inplace=True)    # use first column as row index
    capacities.index.name = "Scenario"
    capacities.columns = capacities.iloc[0] # use top row as col indices
    capacities.columns.name = None
    capacities = capacities.iloc[1:] # drop the index rows (could use df.drop, but row labels are tricky here)
    capacities = capacities.loc[:, project_names]    # choose only the relevant columns

    fuels = get_rps_project_info([
        ["Plant", "Project Name"],
        ["Primary Fuel (Scen 1)", "Fuel"],
        ["Load Zone", "Load Zone"]
    ])
    if oahu_only:
        fuels = fuels[fuels["Load Zone"] == "Oahu"]
    # find the waste plants
    waste_plants = fuels.loc[fuels["Fuel"]=="Waste", "Project Name"].values

    for (scenario, scenario_dir, forecasted) in scenarios[['scenario', 'scenario_dir', 'forecasted']].itertuples(index=False):
        print "creating {d}variable_capacity_factors.tab and {d}reserve_requirements.tab...".format(d=os.path.join(scenario_dir, ''))
        # create tables showing capacity and production for each project in this scenario
        wb, ws = ge_scenario_sheets[scenario]
        # check whether the data start at row 2 or 3
        rng = 'A2:BZ8786' if value_from_xlsx(wb, ws + '!A2') else 'A3:BZ8787'
        hourly = data_frame_from_xlsx(wb, ws + '!' + rng).T.set_index(0).T
        
        if scenario <= 9 and not oahu_only:
            # still need to write code to merge the production and reserves data from the separate Maui sheet with the Oahu data
            raise NotImplementedError(
                "Scenario {} requires data from 'Maui' sheet of {} but that code hasn't been written yet.".format(scenario, wb)
            )

        # setup timestamp index
        # 'Date' column is actually a datetime value;
        # it has some microsecond errors, so we round it to the nearest minute (which should give a clean hour)
        # (based on http://stackoverflow.com/questions/13785932/how-to-round-a-pandas-datetimeindex)
        # Note: there may be some way to convert the Date column directly to an epoch, but I didn't find it.
        timestamps = pd.DatetimeIndex(hourly['Date']).astype(np.int64)    # counts in nanoseconds
        one_minute = 60 * 1e9
        timestamps = one_minute * np.round(timestamps / one_minute, 0)
        hourly['Date'] = timestamps.astype('datetime64[ns]')
        dt = hourly['Date'].dt
        hourly['TIMEPOINT'] = ((2020 * 100 + dt.month) * 100 + dt.day) * 100 + dt.hour
        hourly.set_index('TIMEPOINT', inplace=True)

        # break columns into groups as follows (each group is separated by one or more blank columns):
        # timestamps, production, forecast, reserves, day/night reserves, possibly reserve lookup table
        # note: groupby() technique is from http://stackoverflow.com/questions/38277182/splitting-numpy-array-based-on-value/38277460#38277460
        # note: we have to use column numbers instead of names, because some names are duplicated between groups
        def is_blank(i):
            # check whether column i has a blank header (nan, None or '')
            v = hourly.columns[i]
            return v in {None, ''} or isinstance(v, float) and np.isnan(v)  # could use v!=v from http://stackoverflow.com/questions/944700/how-to-check-for-nan-in-python
        group_cols = [
            list(g) 
                for k, g in itertools.groupby(range(len(hourly.columns)), is_blank) 
                if not k
        ]
        production_cols = hourly[group_cols[2 if forecasted else 1]]

        # We use the reserves specified in GE's spreadsheet. It's possible they actually
        # calculate reserves endogenously in GE MAPS, but we have no way to know.
        # Note: the RPS Study seems to have committed enough plants for contingency+operating 
        # reserves in day-ahead and real time, but then it dispatches plants to provide
        # all contingency reserves (but apparently not operating reserves) from baseload
        # and cycling plants. If they get reserve numbers only from these spreadsheets,
        # it's not clear how they know the contingency reserves (alone) for scenarios 1&2,
        # which only specify total reserves. (Maybe they use total from these spreadsheets
        # to do commitment, then do N-1 in real time? But how would that handle HVDC, which
        # is not a normal contingency?)
        
        # also note: there is a bug in GE's spreadsheet, so they always switch to daytime
        # reserve levels for the midnight hour; we just pass that through here.

        reserve_col_names = tuple(hourly.columns[group_cols[3]])
        if reserve_col_names == ('RESERVES',):
            # Oahu-only, bundled contingency and operating reserves
            reserves = pd.DataFrame({
                'LOAD_ZONE': 'Oahu',
                'regulating_reserve_requirement_mw': hourly['RESERVES']-180.0,
                'contingency_reserve_requirement_mw': 180.0,
            })
        elif reserve_col_names == ('OpeRES', 'Cont RES'):
            # Oahu-only, separate contingency and operating reserves
            reserves = pd.DataFrame({
                'LOAD_ZONE': 'Oahu',
                'regulating_reserve_requirement_mw': hourly['OpeRES'],
                'contingency_reserve_requirement_mw': hourly['Cont RES'],
            })
        elif reserve_col_names == ('OpeRES - Oahu', 'ConRES - Oahu', 'OpeRES - Maui', 'ConRES - Maui'):
            # Oahu and Maui, separate contingency and operating reserves
            reserves = pd.concat([
                pd.DataFrame({
                    'LOAD_ZONE': 'Oahu',
                    'regulating_reserve_requirement_mw': hourly['OpeRES - Oahu'],
                    'contingency_reserve_requirement_mw': hourly['ConRES - Oahu'],
                }),
                pd.DataFrame({
                    'LOAD_ZONE': 'Maui',
                    'regulating_reserve_requirement_mw': hourly['OpeRES - Maui'],
                    'contingency_reserve_requirement_mw': hourly['ConRES - Maui'],
                }),
            ])
        else:
            raise ValueError(
                "Unexpected names for reserve columns in sheet '{}' of '{}': {}"
                .format(ws, wb, reserve_col_names)
            )
        # sometimes GE used "-" for 0.0
        if reserves['regulating_reserve_requirement_mw'].dtype==object:
            reserves.loc[reserves['regulating_reserve_requirement_mw']=='-', 'regulating_reserve_requirement_mw'] = 0.0
        if reserves['contingency_reserve_requirement_mw'].dtype==object:
            reserves.loc[reserves['contingency_reserve_requirement_mw']=='-', 'contingency_reserve_requirement_mw'] = 0.0
        reserves = reserves.set_index('LOAD_ZONE', append=True).astype(float)

        if scenario <= 9:
            forecast_cols = group_cols[2][-1:]
        else:
            forecast_cols = group_cols[2][-3:-1]
            
        forecast_col_names = tuple(hourly.columns[forecast_cols])

        if forecast_col_names == ('Total',):
            # Oahu-only
            forecast = pd.DataFrame({
                'LOAD_ZONE': 'Oahu',
                'renewable_forecast_mw': hourly.iloc[:, forecast_cols[0]]
            })
        elif forecast_col_names == ('Total Oahu', 'Total Maui'):
            # Oahu and Maui
            forecast = pd.concat([
                pd.DataFrame({
                    'LOAD_ZONE': 'Oahu',
                    'renewable_forecast_mw': hourly.iloc[:, forecast_cols[0]]
                }),
                pd.DataFrame({
                    'LOAD_ZONE': 'Maui',
                    'renewable_forecast_mw': hourly.iloc[:, forecast_cols[1]]
                }),
            ])
        else:
            raise ValueError(
                "Unexpected names for forecast columns in sheet '{}' of '{}': {}"
                .format(ws, wb, forecast_col_names)
            )

        forecast = forecast.set_index('LOAD_ZONE', append=True).astype(float)

        # merge forecast and reserves tables
        reserves = forecast.join(reserves)
        
        # drop load_zone index to make filtering easier
        reserves = reserves.reset_index('LOAD_ZONE')
        
        # filter timepoints and load zones
        reserves = reserves.loc[study_timepoints]
        if oahu_only:
            reserves = reserves.loc[reserves['LOAD_ZONE']=='Oahu', :]

        # place load zone at start of index
        reserves = reserves.set_index('LOAD_ZONE', append=True)
        reserves.index = reserves.index.reorder_levels(['LOAD_ZONE', 'TIMEPOINT'])
        

        # make empty data frame to hold total production for each project
        production = pd.DataFrame(0.0, index=production_cols.index, columns=project_names)
        # assign output from each production column to a project
        for c in production_cols.columns:
            # see which project(s) this column belongs in
            match_proj = [proj for proj in project_names if project_regex[proj].match(c)]
            if c.lower().startswith('total'):
                # ignore 'Total' columns
                pass
            elif len(match_proj) < 1:
                raise ValueError("Column {} in sheet {} of {} did not match any known project.".format(c, ws, wb))
            elif len(match_proj) > 1:
                raise ValueError("Column {} in sheet {} of {} matched multiple projects: {}".format(c, ws, wb, match_proj))
            else:
                # add the output to the total for this project
                production[match_proj[0]] = production[match_proj[0]] + production_cols[c]
        
        # filter timepoints
        production = production.loc[study_timepoints]

        # create tables showing capacity and production for each project in this scenario
        scen_capacity = capacities.loc[scenario, :]

        # remove projects that don't have any capacity in this scenario or aren't in Oahu
        scen_capacity = scen_capacity[scen_capacity > 0]
        if oahu_only:
            scen_capacity = scen_capacity[[p for p in scen_capacity.index if p.startswith('Oahu')]]
        # apply same filtering to production table
        production = production.loc[:, scen_capacity.index]

        # calculate capacity factors
        scen_cap_factor = production / scen_capacity
        # double-check that the values are plausible (i.e., the capacity we're expecting for each
        # project at least roughly matches the capacity recorded in the hourly production worksheet)
        if any((scen_cap_factor.max(axis=0) < .7) | (scen_cap_factor.max(axis=0) > 1.1)):
            print "WARNING: One or more projects have invalid maximum capacity factors in {}".format(scenario_dir)
            print scen_cap_factor.max(axis=0)
        if any((scen_cap_factor.mean(axis=0) < .15) | (scen_cap_factor.mean(axis=0) > 0.50)):
            print "WARNING: One or more projects have invalid average capacity factors in {} (expected 0.15-0.50)".format(scenario_dir)
            print scen_cap_factor.mean(axis=0)
        # add waste plants
        for p in waste_plants:
            scen_cap_factor[p] = waste_plant_capacity_factor()
        # switch from table format to serial format (one row for each project/time)
        scen_cap_factor = scen_cap_factor.unstack()
        # write the .tab files
        scen_cap_factor.to_csv(
            os.path.join(scenario_dir, "variable_capacity_factors.tab"), 
            sep='\t',
            index_label=['PROJECT', 'timepoint'],
            header=['proj_max_capacity_factor']
        )
        reserves.to_csv(
            os.path.join(scenario_dir, "reserve_requirements.tab"), 
            sep='\t',
            # index_label=['load_zone', 'timepoint'],
        )


def loads():
    # load hourly loads, then assign timepoints and save in loads.tab files
    loads = data_frame_from_xlsx(
        os.path.join('GE RPS Study Input Data', 'Hawaii RPS Hourly Loads 2020.xlsx'),
        'Sheet1!A1:F8785'
    ).T.set_index(0).T
    # setup timestamp index
    # 'Date' column is actually a datetime value;
    # it has some microsecond errors, so we round it to the nearest minute (which should give a clean hour)
    timestamps = pd.DatetimeIndex(loads['Date']).astype(np.int64)    # counts in nanoseconds
    one_minute = 60 * 1e9
    timestamps = one_minute * np.round(timestamps / one_minute, 0)
    loads['Date'] = timestamps.astype('datetime64[ns]')
    dt = loads['Date'].dt
    loads['TIMEPOINT'] = ((2020 * 100 + dt.month) * 100 + dt.day) * 100 + dt.hour
    loads.set_index('TIMEPOINT', inplace=True)

    # drop unneeded columns and filter timepoints
    loads = loads[['Oahu Load', 'Maui Load']].loc[study_timepoints]
    # convert column names into load zone identifiers and unstack
    loads.columns = ['Oahu', 'Maui']
    loads.columns.name = 'LOAD_ZONE'
    loads = pd.DataFrame({'lz_demand_mw': loads.unstack()}) # turns the index into load_zone, timepoint

    if oahu_only:
        loads = loads.loc[['Oahu']]

    # store values in .tab files
    for scenario, scenario_dir in scenario_dirs:
        loads.to_csv(os.path.join(scenario_dir, "loads.tab"), sep='\t')

def load_zones():
    for scenario, scenario_dir in scenario_dirs:
        with open(os.path.join(scenario_dir, "load_zones.tab"), "w") as f:
            f.write("LOAD_ZONE\n")
            f.write("Oahu\n")
            if not oahu_only:
                f.write("Maui\n")

def timepoint_to_date(tp):
    s = str(tp)
    return datetime.date(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8]))

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

def timescales():
    for scenario, scenario_dir in scenario_dirs:
        with open(os.path.join(scenario_dir, "periods.tab"), "w") as f:
            f.write(dedent(
                """
                INVESTMENT_PERIOD\tperiod_start\tperiod_end
                2020\t{start}\t{end}
                """.format(
                    start=year_fraction(timepoint_to_date(study_timepoints[0])), 
                    end=year_fraction(timepoint_to_date(study_timepoints[-1])+datetime.timedelta(days=1))
                )
            ))
        with open(os.path.join(scenario_dir, "timeseries.tab"), "w") as f:
            f.write(dedent("""
                TIMESERIES\tts_period\tts_duration_of_tp\tts_num_tps\tts_scale_to_period
                2020\t2020\t{dur}\t{num}\t{scale}
            """.format(
                dur=timepoint_duration, num=len(study_timepoints), scale=1.0)
            ))
        with open(os.path.join(scenario_dir, "timepoints.tab"), "w") as f:
            f.write('timepoint_id\ttimestamp\ttimeseries\n')
            for tp in study_timepoints:
                stamp = str(tp)
                stamp = stamp[0:4] + '-' + stamp[4:6] + '-' + stamp[6:8] + '-' + stamp[8:10] + ':00'
                f.write('{tp}\t{stamp}\t2020\n'.format(tp=tp, stamp=stamp))
            
def financials():
    cc = value_from_xlsx("RPS Study Resources.xlsx", "cost_of_capital")
    for scenario, scenario_dir in scenario_dirs:
        with open(os.path.join(scenario_dir, "financials.dat"), "w") as f:
            f.write(dedent("""
                param base_financial_year := 2015;
                param interest_rate := {};
                param discount_rate := 0.0;
            """.format(cc)))

def fuels():
    fuel_costs = data_frame_from_xlsx("RPS Study Resources.xlsx", "fuel_prices")
    fuel_costs.iloc[0, 0] = "fuel"
    fuel_costs = fuel_costs.T.set_index(0).T    # set column headers
    fuel_costs.columns.name = 'load_zone'
    fuel_costs['period'] = 2020
    fuel_costs = pd.DataFrame({'fuel_cost': fuel_costs.set_index(['fuel', 'period']).stack()})
    fuel_costs = fuel_costs.reorder_levels(['load_zone', 'period', 'fuel']).sort_index()
    
    if oahu_only:
        fuel_costs = fuel_costs.loc[['Oahu']]
    
    fuels = fuel_costs.index.levels[2]

    # store data in .tab files
    for scenario, scenario_dir in scenario_dirs:
        fuel_costs.to_csv(os.path.join(scenario_dir, "fuel_cost.tab"), sep='\t')    
        # write list of available fuels
        with open(os.path.join(scenario_dir, "fuels.tab"), "w") as f:
            f.write("fuel\tco2_intensity\n")
            f.writelines('{f}\t0\n'.format(f=f) for f in fuels)
        # write list of available fuels
        with open(os.path.join(scenario_dir, "non_fuel_energy_sources.tab"), "w") as f:
            f.write("NON_FUEL_ENERGY_SOURCES\n")
            f.write("\n".join(non_fuel_energy_sources))    

def gen_techs():
    """Create generator_info.tab and gen_inc_heat_rates.tab"""
    # note: this could be rewritten to select specific columns and rename them here
    gen_info = get_rps_project_info()

    gen_info.set_index('Plant ID', drop=False, inplace=True)
    
    # convert "N/A" and "PPA" values to NaN (if any of these are output, to_csv() will 
    # convert them to ".", and then SWITCH will provide default values or raise an error)
    gen_info = gen_info.where((gen_info != "N/A") & (gen_info != "PPA"))

    # convert missing variable O&M values to 0.0
    gen_info.loc[gen_info['VOM'].isnull(), 'VOM'] = 0.0
    
    # del gen_info['Plant']
    if oahu_only:
        gen_info = gen_info[gen_info['Load Zone']=='Oahu']

    # define renewable energy technologies
    renewable_project_info = data_frame_from_xlsx("RPS Study Resources.xlsx", "renewable_project_info")
    renewable_project_info = renewable_project_info.T.set_index(0).T    # set column headers
    renewable_techs = renewable_project_info[["Generation Technology", "Energy Source"]].drop_duplicates()

    gen_info_renewable = pd.DataFrame(dict(
        generation_technology=renewable_techs["Generation Technology"],
        g_dbid='.',
        g_max_age=30,   # use 30 years as max age for all; matches cost of capital calcs
        g_scheduled_outage_rate=0.0,
        g_forced_outage_rate=0.0,   # assume no forced outages, to get 100% dispatch
        # treat both waste-to-energy plants as variable; we'll set their 
        # hourly capacity factor elsewhere
        g_is_variable=1,
        g_variable_o_m=0.0,
        g_energy_source=renewable_techs["Energy Source"],
        g_unit_size=".", # no unit size for discrete commitment
        g_min_load_fraction=".", # no minimum load
        g_startup_fuel="."
    ))
    gen_info_renewable.set_index("generation_technology", inplace=True)
    
    for scenario, scenario_dir in scenario_dirs:
        # min_load_col = "P-Min (Scen 1)"
        # kludge/bug: Figures 8 & 9 seem to use Scen 1 min load for Kahe 3-6 & Waiau 7-8
        # (actually, maybe not)
        min_load_col = "P-Min (Scen 1)" if scenario == 1 else "P-Min (Scen 2-18)"
        energy_source_col = 'Primary Fuel (Scen 1)' if scenario==1 else 'Primary Fuel (Scen 2-18)'
        
        gen_info_tab = pd.DataFrame(dict(
            g_dbid='.',
            g_max_age=30,   # use 30 years as max age for all; matches cost-of-capital calcs
            g_scheduled_outage_rate=0.0,
            g_forced_outage_rate=0.0,   # assume no forced outages, to get 100% dispatch
            # treat both waste-to-energy plants as variable; we'll set their 
            # hourly capacity factor elsewhere
            g_is_variable=(gen_info['Primary Fuel (Scen 1)']=="Waste").astype(int),
            g_variable_o_m=gen_info['VOM'],
            g_energy_source=gen_info[energy_source_col],
            g_unit_size=gen_info['Max Capacity'],
            g_min_load_fraction=gen_info[min_load_col]/gen_info['Max Capacity'],
            g_startup_fuel=gen_info['Start Energy (Hot)']/gen_info['Max Capacity'],
            g_min_uptime=gen_info['Min Up Time'],
            g_min_downtime=gen_info['Min Down Time'],
        ), index=gen_info.index)
        # note: we don't use baseload flags; we use proj_commit_bounds instead
        # we also omit a few other flags that are not relevant for this model
        # g_is_baseload=0,
        # g_is_flexible_baseload=0,
        # g_is_cogen=0,   # doesn't actually have any effect on the model
        # g_competes_for_space=gen_info[''],
        # omit full load heat rate, since it's specified with curves for each plant
        # g_full_load_heat_rate=gen_info['Full Load Heat Rate']/1000, # not actually used

        # set min load for waste plants, so they run at full power
        # (see John Cole e-mail, 9/28/16)
        gen_info_tab.loc[gen_info_tab['g_energy_source']=='Waste', "g_min_load_fraction"] \
            = waste_plant_capacity_factor()

        # select only the projects active in this scenario
        if scenario == 1:
            gen_info_tab = gen_info_tab[gen_info['Active in Scenario 1']=="Y"]
        else:
            gen_info_tab = gen_info_tab[gen_info['Active in Scenario 2-18']=="Y"]
        
        # append renewable energy technologies
        all_gen_info = pd.concat([gen_info_tab, gen_info_renewable])

        # set dummy values for a bunch of mandatory columns we don't use
        for c in ["g_is_baseload", "g_is_flexible_baseload", "g_is_cogen", "g_competes_for_space"]:
            all_gen_info[c] = 0
        all_gen_info["g_full_load_heat_rate"] = np.nan
        
        # store data in tab file
        all_gen_info.to_csv(
            os.path.join(scenario_dir, "generator_info.tab"), 
            sep='\t',
            index_label=['generation_technology'],
            na_rep="."
        )
        
        # calculate incremental heat rates
        heat_rate_cols = map(list, zip(
            [min_load_col, "min_load"],
            ["Max Capacity", "max_load"],
            ["Heat Rate Coef (A)", "mmbtu"],
            ["Heat Rate Coef (B)", "mmbtu_per_kwh"],
            ["Heat Rate Coef (C)", "mmbtu_per_kwh2"],
        ))
        heat_rates = gen_info[heat_rate_cols[0]]
        heat_rates.columns = heat_rate_cols[1]  # give deterministic and easier-to-use names
        # filter projects to match gen_info_tab and drop waste-to-energy projects
        heat_rates = heat_rates.loc[gen_info_tab.index]
        heat_rates = heat_rates.loc[heat_rates["mmbtu"].notnull()]
        # convert all remaining values (ints and floats) to float
        heat_rates = heat_rates.astype(float)

        # switched from 5 to 15 or 2 2016-07-30 to see if that would help
        space = np.linspace(0, 1, num=15)[np.newaxis, :]
        power_points = (1-space) * heat_rates[["min_load"]].values + space * heat_rates[["max_load"]].values
        mmbtu = (
            heat_rates[["mmbtu"]].values 
            + power_points * heat_rates[["mmbtu_per_kwh"]].values
            + power_points**2 * heat_rates[["mmbtu_per_kwh2"]].values
        )
        # get incremental heat rates for each segment
        inc_heat_rate = (mmbtu[:, 1:] - mmbtu[:, :-1]) / (power_points[:, 1:] - power_points[:, :-1])
        hr_table = [[
            'generation_technology', 
            'power_start_mw', 'power_end_mw', 'incremental_heat_rate_mbtu_per_mwhr',
            'fuel_use_rate_mmbtu_per_h'
        ]]
        for i, gt in enumerate(heat_rates.index):
            hr_table.append([gt, power_points[i, 0], '.', '.', mmbtu[i, 0]])
            if power_points[i, 0] == power_points[i, -1]:
                # special treatment if full load = min load (otherwise we'd get several rows of nans)
                hr_table.append([
                    gt, power_points[i, 0], power_points[i, -1], mmbtu[i, 0]/power_points[i, 0], '.'
                ])
            else:
                for start, end, hr in zip(power_points[i, :-1], power_points[i, 1:], inc_heat_rate[i]):
                    hr_table.append([gt, start, end, hr, '.'])
        # write heat rates to file
        with open(os.path.join(scenario_dir, "gen_inc_heat_rates.tab"), "w") as f:
            f.write("\n".join(["\t".join(map(str, row)) for row in hr_table]))
            # f.writelines("\t".join(map(str, row)) + "\n" for row in hr_table)




def project_info():
    """ create project_info.tab and proj_existing_builds.tab, showing details on each project """

    # every unit in the RPS Study is treated as a separate project
    proj_info = get_rps_project_info([
        ['Plant', 'PROJECT'],
        ['Plant ID', 'proj_gen_tech'],
        ['Load Zone', 'proj_load_zone'],
        ['Project Reporting Type', 'proj_reporting_type'],
        ['Max Capacity', 'proj_existing_cap'],
        ['Unit Type', 'unit_type'],
        ['Active in Scenario 1', 'active1'],
        ['Active in Scenario 2-18', 'active2'],
    ])

    proj_info.set_index('PROJECT', inplace=True, drop=False)
    # convert active flags to boolean
    proj_info['active1'] = (proj_info['active1'] == 'Y')
    proj_info['active2'] = (proj_info['active2'] == 'Y')

    if oahu_only:
        proj_info = proj_info[proj_info['proj_load_zone']=='Oahu']

    # add in renewable projects not shown in the unit list
    renewable_proj_info = data_frame_from_xlsx("RPS Study Resources.xlsx", "renewable_project_info")
    renewable_proj_info = renewable_proj_info.T.set_index(0).T    # set column headers
    # renewable_proj_energy_source = renewable_proj_info["Energy Source"] # save for use later
    r_proj_cols = map(list, zip(
        ['Project Name', 'PROJECT'],
        ['Generation Technology', 'proj_gen_tech'],
        ['Load Zone', 'proj_load_zone'],
        ['Energy Source', 'proj_energy_source'],
        ['Project Reporting Type', 'proj_reporting_type'],
    ))
    # select and rename columns
    renewable_proj_info = renewable_proj_info[r_proj_cols[0]]
    renewable_proj_info.columns = r_proj_cols[1]
    renewable_proj_info.set_index('PROJECT', inplace=True)
    # set active flags
    renewable_proj_info['active1'] = True
    renewable_proj_info['active2'] = True

    if oahu_only:
        renewable_proj_info = renewable_proj_info[renewable_proj_info['proj_load_zone']=='Oahu']

    # combine both tables
    proj_info = pd.concat([proj_info, renewable_proj_info]).copy() # copy to avoid errors later

    # assign project build info where available (we have to put in something for all of these)
    proj_info['build_year'] = 2010    # dummy build year, close enough to active date
    proj_info["proj_connect_cost_per_mw"] = 0.0
    proj_info['proj_overnight_cost'] = 0.0
    proj_info['proj_fixed_om'] = 0.0
    proj_costs = data_frame_from_xlsx("RPS Study Resources.xlsx", "new_generator_capital_costs")
    proj_costs = proj_costs.T.set_index(0).T.set_index("Plant Name")  # set headers and index
    match_rows = proj_info.index.intersection(proj_costs.index)
    proj_info.loc[match_rows, 'proj_overnight_cost'] = proj_costs.loc[match_rows, 'Capital Cost ($/MW)']
    proj_info.loc[match_rows, 'proj_connect_cost_per_mw'] = proj_costs.loc[match_rows, 'Connect Cost ($/MW)']

    # create table of build years
    # first, get capacity of wind and solar built in each scenario
    re_cap = data_frame_from_xlsx("RPS Study Resources.xlsx", "scenario_capacities")
    re_cap.set_index(0, inplace=True)    # use first column (scenario number) as index
    re_cap.index.name = "Scenario"
    # switch orientation and set first col as index
    re_cap = re_cap.T.set_index(re_cap.index[0])
    re_cap = re_cap.loc[renewable_proj_info.index]    # choose only the relevant projects

    # convert spaces in proj_reporting_type to underscores for use in .tab file
    proj_info["proj_reporting_type"].replace(' ', '_', inplace=True, regex=True) 
    
    # store data in tab files
    for scenario, scenario_dir in scenario_dirs:
        # identify active projects, and assign the right renewable capacity for this scenario
        proj_active = proj_info[proj_info['active1' if scenario == 1 else 'active2']].copy()
        proj_active.loc[re_cap.index, 'proj_existing_cap'] = re_cap[scenario]
        proj_active = proj_active.loc[proj_active['proj_existing_cap'] > 0]

        # project capacity
        proj_active.to_csv(
            os.path.join(scenario_dir, "proj_existing_builds.tab"), 
            sep='\t',
            index_label=['PROJECT'],
            columns=['proj_existing_cap', 'build_year']
        )
        
        # project_info
        proj_active.to_csv(
            os.path.join(scenario_dir, "project_info.tab"), 
            sep='\t',
            index_label=['PROJECT'],
            columns=['proj_gen_tech', 'proj_load_zone', 'proj_connect_cost_per_mw']
        )

        # proj_build_costs
        proj_active.to_csv(
            os.path.join(scenario_dir, "proj_build_costs.tab"), 
            sep='\t',
            index_label=['PROJECT'],
            columns=['build_year', 'proj_overnight_cost', 'proj_fixed_om']
        )

        # proj_reporting_types
        proj_active.to_csv(
            os.path.join(scenario_dir, "proj_reporting_types.tab"), 
            sep='\t',
            index_label=['PROJECT'],
            columns=['proj_reporting_type']
        )



def proj_commit_bounds():
    proj_info = get_rps_project_info([
        ['Plant', 'PROJECT'],
        ['Plant ID', 'Plant ID'],
        ['Load Zone', 'load_zone'],
        ['Unit Type', 'unit_type'], # Baseload, Cycling, etc.
        ['Active in Scenario 1', 'active1'],
        ['Active in Scenario 2-18', 'active2']
    ])
    proj_info.set_index("PROJECT", inplace=True)

    # force commitment of baseload projects (RPS Study p. 45)
    # and Firm RE (e-mail from John Cole 9/28/16)
    proj_info["proj_min_commit_fraction"] = 0.0
    proj_info["proj_max_commit_fraction"] = 1.0
    proj_info.loc[proj_info['unit_type']=='Baseload', "proj_min_commit_fraction"] = 1.0
    proj_info.loc[proj_info['unit_type']=='Firm RE', "proj_min_commit_fraction"] = 1.0

    # turn off de-activated plants
    proj_info.loc[proj_info['unit_type']=='Off', "proj_max_commit_fraction"] = 0.0

    # get list of maintenance dates (currently missing Maui Maalaea plants because unit names in Tables 7 & 11 don't match)
    # (currently only one outage per plant per year, but a plant could be taken out
    # multiple times by adding more rows to this list)
    maint_dates = data_frame_from_xlsx("RPS Study Resources.xlsx", "planned_maintenance_dates")
    # set row and column headers
    maint_dates = maint_dates.T.set_index(0).T.set_index("Plant ID", drop=False)
    proj_lookup = proj_info[["Plant ID", "load_zone"]].reset_index().set_index("Plant ID")
    maint_dates = maint_dates.join(proj_lookup)
    if maint_dates["PROJECT"].isnull().any():
        raise ValueError(
            'These Plant IDs are in "RPS Study Resources.xlsx:planned_maintenance_dates" but not in '
            '"Hawaii RPS Study Generator Table OCR.xlsx:generator_data": {}'.format(
                maint_dates.loc[maint_dates["PROJECT"].isnull(), 'Plant ID'].values
            )
        )
    maint_dates.set_index("PROJECT", inplace=True)

    # calculate starting and ending timepoints for each maintenance outage
    dt = maint_dates['Start'].dt
    maint_dates['start_tp'] = ((dt.year * 100 + dt.month) * 100 + dt.day) * 100 + dt.hour
    dt = maint_dates['End'].dt
    maint_dates['end_tp'] = ((dt.year * 100 + dt.month) * 100 + dt.day) * 100 + dt.hour

    for scenario, scenario_dir in scenario_dirs:
        proj_active = proj_info.loc[
            (proj_info['active1' if scenario == 1 else 'active2'] == "Y"),
            ["proj_min_commit_fraction", "proj_max_commit_fraction", "load_zone"]
        ]
        # now cross with all timepoints (is there some way to do this in one step?)
        info = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [proj_active.index, pd.Index(study_timepoints)], 
                names=["PROJECT", "TIMEPOINT"]
            )
        ).sort_index()  # index must be sorted to do slicing below
        info = info.join(proj_active)
        # set proj_max_commit_fraction and proj_min_commit_fraction to 0
        # when projects are out of service
        for proj, start, end in maint_dates[["start_tp", "end_tp"]].itertuples():
            # if proj == 'Waiau_10':
            #     import pdb; pdb.set_trace()
            if proj in proj_active.index:
                # note: if the  proj isn't in the index that means it's inactive, not a data inconsistency
                # DISABLE MAINTENANCE DATES TO GET BETTER MATCH TO FIGS 8 & 9 (W5 CYCLING SHOULD BE OUT BUT ISN'T)
                # continue
                info.loc[(proj, slice(start, end)), ["proj_min_commit_fraction", "proj_max_commit_fraction"]]=0.0
        # Force Kalaeloa 2 on when Kalaeloa 1 is out of service
        # (Kalaeloa 1 will run at all other times, because it is marked as Baseload)
        cc1_out = (info.loc['Kalaeloa_CC1', 'proj_max_commit_fraction']==0)
        cc1_out = cc1_out[cc1_out] # sub-select rows
        cc2_update = pd.concat([cc1_out], keys=['Kalaeloa_CC2'], names=['PROJECT']).index 
        # see http://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex
        info.loc[cc2_update, 'proj_min_commit_fraction'] = info.loc[cc2_update, 'proj_max_commit_fraction']
        
        if oahu_only:
            info = info[info['load_zone']=='Oahu']
        # don't need load_zone column anymore
        del info['load_zone']
        
        # write out a .tab file with all records with min > 0 or max < 1
        info.loc[(info["proj_min_commit_fraction"] > 0) | (info["proj_max_commit_fraction"] < 1)].to_csv(
            os.path.join(scenario_dir, "proj_timepoint_commit_bounds.tab"), 
            sep='\t',
            index_label=['PROJECT', 'TIMEPOINT']
        )
        
if __name__ == "__main__":
    main()
