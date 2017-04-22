"""
Defines types of reserve target and components that contribute to reserves,
and enforces the reserve targets.
"""
import os
from collections import defaultdict
from pyomo.environ import *

def define_components(m):
    """
    Note: In this simple model, we assume all reserves must be spinning. In more complex
    models you could define products and portions of those products that must be spinning,
    then use that to set the spinning reserve requirement.

    Reserves don't have a deliverability requirement, so they are calculated for the whole region.
    """

    # TODO: create these sets in build_scenario_data.py and then read in
    # TODO: include Maui.

    # TODO: find out whether up regulation from peakers is committed based on 
    # day-ahead forecast or real-time conditions; i.e., do we need a day-ahead
    # queue (which tries to cover contingencies+regulation on a day-ahead basis
    # and then must cover contingencies on its own in real time) and a 
    # real-time queue (which is used for both regulation and contingencies),
    # or do we need a contingency-reserve list and a regulation+contingency list,
    # both of which are committed day-ahead (in which case, real-time regulation
    # may not be meaningful)?
    
    # # cycling, sorted by cost per kWh at full load heat rate,
    # # using scen 1 fuel (LSFO)
    # # TODO: check whether this is the right ranking and/or
    # # whether these can be committed in real time or only day-ahead
    # # (e-mail from Derek Stenclik 3/14/17 seems to confirm this
    # # is correct)
    # 'Kalaeloa_CC3',
    # 'Honolulu_8',
    # 'Honolulu_9',
    # 'Waiau_5',
    # 'Waiau_6',
    # 'Waiau_4',
    # 'Waiau_3',

    # cycling, sorted according to apparent order in Figure 9
    oahu_cycling_plants = [
        'Waiau_5',
        'Kalaeloa_CC2',
        'Kalaeloa_CC3',
        'Waiau_6',
        'Waiau_4',
        'Waiau_3',
        'Honolulu_8',
        'Honolulu_9',
    ]
    
    # These units must be committed based on day-ahead forecast,
    # and they are the only ones that can provide contingency reserves
    # (p. 61 of RPS Study says baseload units provide reserves, 
    # but e-mail from Derek Stenclik 3/14/17 says "W9, W10, Schofield, 
    # and CIPCT only provided replacement reserves and not spinning reserves.
    # So spin was always covered by baseload and cycling units, while the
    # peakers covered wind and solar forecast error." (this seems to mean 
    # regulation as I'm defining it)
    day_ahead_commit_queue_dict = {
        'Oahu': [
            # waste-to-energy plants are must-run according to e-mail
            # from John Cole 9/28/16, so we put them at the top
            'H-Power', 
            'Honua',
            # baseload, sorted in order shown on p. 46 of RPS Study
            # (roughly by variable cost)
            'AES',
            'Kalaeloa_CC1',
            'Kahe_5',
            'Kahe_3',
            'Kahe_4',
            'Kahe_2',
            'Kahe_6',
            'Kahe_1',
            'Waiau_7',
            'Waiau_8',
        ] + oahu_cycling_plants + [
        ]
    }

    # note: we put bigger plants first, per Figure 30 of RPS Study
    # this also matches emphasis on CIPCT in Derek Stenclik's e-mail 3/14/17
    # and gives the right balance between peaking/biodiesel and peaking/baseload
    # to match Table 8
    # note: e-mail from Derek Stenclik 3/14/17 implies CIP is providing
    # peaking power on 6/23/20 and says Schofield may be last to run
    # because it uses biodiesel. But RPS Study generator table says 
    # Schofield, Airport and CIP all run on biodiesel, and CIP is most
    # expensive. 
    # On the other hand, slide 33 of
    # http://www.hnei.hawaii.edu/sites/dev.hnei.hawaii.edu/files/news/Full%20Slide%20Deck.pdf 
    # shows all of the peakers with high variable costs (probably based on biodiesel)
    # and therefore ranks them as CIP_CT, Airport_DSG, Waiau_9, Waiau_10, Schofield
    
    real_time_commit_queue_dict = {
        'Oahu': [
            # renewable
            'Oahu_Distributed_PV',
            'Oahu_Central_PV',
            'Oahu_Wind',
            'Oahu_Gen-Tie_Wind',
            # peaking
            'CIP_CT',
            'Airport_DSG',
            'Waiau_9',
            'Waiau_10',
            'Schofield'
        ]
    }

    m.UTILITY_SCALE_RENEWABLE_PROJECTS = Set(
        m.LOAD_ZONES,
        initialize=lambda m, z: (p for p in m.LZ_PROJECTS[z] if 'Central_PV' in p or 'Wind' in p)
    )
    
    m.DAY_AHEAD_COMMIT_QUEUE = Set(
        m.LOAD_ZONES,
        ordered=True,
        initialize=lambda m, z:
            [p for p in day_ahead_commit_queue_dict[z] if p in m.PROJECTS]
    )
    m.REAL_TIME_COMMIT_QUEUE = Set(
        m.LOAD_ZONES,
        ordered=True,
        initialize=lambda m, z:
            [p for p in real_time_commit_queue_dict[z] if p in m.PROJECTS]
    )
    def Show_Missing_Projects_rule(m):
        # for z, projects in day_ahead_commit_queue_dict.items():
        #     for p in projects:
        #         if p not in m.DAY_AHEAD_COMMIT_QUEUE[z]:
        #             print "WARNING: {} not added to m.DAY_AHEAD_COMMIT_QUEUE".format((z, p))
        # for z, projects in real_time_commit_queue_dict.items():
        #     for p in projects:
        #         if p not in m.REAL_TIME_COMMIT_QUEUE[z]:
        #             print "WARNING: {} not added to m.REAL_TIME_COMMIT_QUEUE".format((z, p))
        for p in m.PROJECTS:
            z = m.proj_load_zone[p]
            if p not in m.DAY_AHEAD_COMMIT_QUEUE[z] and p not in m.REAL_TIME_COMMIT_QUEUE[z]:
                print "WARNING: {} not added to m.DAY_AHEAD_COMMIT_QUEUE or m.REAL_TIME_COMMIT_QUEUE".format((z, p))
    m.Show_Missing_Projects = BuildAction(rule=Show_Missing_Projects_rule)

    
    # e-mail from Derek Stenclek says "The peaking plants were assumed to provide
    # supplemental replacement reserves only." (i.e., energy to free up other plants)
    # NOTE: we assume only HECO plants can provide up reserves, 
    # matching the rule about down-reserves (presumably when AES/Kalaeloa are backed
    # down it's due to excess energy rather than reserve shortage anyway)
    m.UP_RESERVES_STACK = Set(
        m.LOAD_ZONES,
        initialize=lambda m, z:
            [p for p in m.DAY_AHEAD_COMMIT_QUEUE[z] if p.startswith('Kahe') or p.startswith('Waiau')]
            # cycling units
            + [p
                for p in oahu_cycling_plants
                if p in m.PROJECTS and p not in m.DAY_AHEAD_COMMIT_QUEUE[z]
                    and (p.startswith('Kahe') or p.startswith('Waiau'))
            ]
    )

    # up- and down-reserve providers (used in real-time dispatch)
    # p. 61 of RPS Study says "All of the baseload HECO units and utility-scale
    # wind and solar generating were modeled to provide down reserves"
    # We assume renewables will automatically be dispatched high enough to cover
    # their share, so we just focus on the share assigned to thermal plants.
    m.DOWN_RESERVES_STACK = Set(
        m.LOAD_ZONES, 
        initialize=lambda m, z:
            [p for p in m.DAY_AHEAD_COMMIT_QUEUE[z] if p.startswith('Kahe') or p.startswith('Waiau')]
            # + [p for p in m.LZ_PROJECTS[z] if p in {'Oahu_Wind', 'Maui_Wind'}]
            # + [p for p in m.LZ_PROJECTS[z] if m.proj_gen_tech[p] in ['Wind', 'CentralPV']]
    )

    # tabulate pre-built capacity and max possible output for use in 
    # reserve allocation
    # (since this is a production cost model, this is all there will be)
    m.proj_prebuilt_capacity = Param(m.PROJECTS, m.PERIODS, rule=lambda m, proj, period:
        sum(
            m.proj_existing_cap[proj, bld_yr] 
                for bld_yr in m.PROJECT_PERIOD_ONLINE_BUILD_YRS[proj, period]
        )
    )
    m.proj_potential_output = Param(m.PROJ_DISPATCH_POINTS, rule=lambda m, p, tp:
        m.proj_prebuilt_capacity[p, m.tp_period[tp]]
        * m.proj_availability[p]
        * (
            m.proj_max_capacity_factor[p, tp] 
            if (p, tp) in m.proj_max_capacity_factor 
            else 1.0
        )
    )
    
    
    # Calculate spinning reserve requirements.
    # This model is just a production cost model for a fixed portfolio, 
    # so we can specify the reserve requirements exogenously.
    # real-time contingency requirement (on top of loads)
    m.contingency_reserve_requirement_mw = Param(m.LOAD_ZONES, m.TIMEPOINTS)
    # real-time regulation requirement (on top of loads)
    m.regulating_reserve_requirement_mw = Param(m.LOAD_ZONES, m.TIMEPOINTS)
    # energy+regulation targets for each hour based on day-ahead forecasts
    m.renewable_forecast_mw = Param(m.LOAD_ZONES, m.TIMEPOINTS)
    
    # require 10% down reserves from thermal plants at all times, 
    # minus a prorated share assigned to utility-scale wind & solar 
    # note: 0.1 * load * (1 -(w+s)/load) = 0.1 * (load - (w+s))
    m.down_reserve_requirement_mw = Param(
        m.LOAD_ZONES, m.TIMEPOINTS, 
        initialize=lambda m, z, tp:
            # 0.10 * (
            #     m.lz_demand_mw[z, tp]
            #     - sum(m.proj_potential_output[p, tp] for p in m.UTILITY_SCALE_RENEWABLE_PROJECTS[z])
            # )
            0.10 * m.lz_demand_mw[z, tp] 
            # note: t / (r + t) = 1 / (1 + r/t)
            / (
                1 
                + sum(m.proj_potential_output[p, tp] for p in m.UTILITY_SCALE_RENEWABLE_PROJECTS[z])
                / sum(m.proj_potential_output[p, tp] for p in m.UP_RESERVES_STACK[z])
            )
    )

    # Calculate contingency reserve requirements
    m.ContingencyReserveUpRequirement = Var(m.LOAD_ZONES, m.TIMEPOINTS, within=NonNegativeReals)
    # Apply a simple n-1 contingency reserve requirement; 
    # we treat each project as a separate contingency
    # Note: we provide reserves for the full committed amount of the project so that
    # if any of the capacity is being used for regulating reserves, that will be backed
    # up by contingency reserves.
    # m.ContingencyReserveUpRequirement_Calculate = Constraint(
    #     m.PROJ_DISPATCH_POINTS,
    #     rule=lambda m, p, tp:
    #         (m.ContingencyReserveUpRequirement[m.proj_load_zone[p], tp] >= m.CommitProject[p, tp])
    #         if m.proj_gen_tech[p] in m.GEN_TECH_WITH_UNIT_SIZES
    #         else Constraint.Skip
    # )
    m.ContingencyReserveUpRequirement_Calculate = Constraint(
        m.LOAD_ZONES, m.TIMEPOINTS,
        rule=lambda m, z, tp:
            m.ContingencyReserveUpRequirement[z, tp] == m.contingency_reserve_requirement_mw[z, tp]
    )
       
    # Calculate total spinning reserve requirement
    m.SpinningReserveUpRequirement = Expression(m.LOAD_ZONES, m.TIMEPOINTS, rule=lambda m, z, tp:
        m.ContingencyReserveUpRequirement[z, tp]
        # not used for real-time in Figure 8, but maybe in Figure 9?: 
        # + 1.0 * m.regulating_reserve_requirement_mw[z, tp]
    )

    m.SpinningReserveDownRequirement = Expression(
        m.LOAD_ZONES, m.TIMEPOINTS, 
        rule=lambda m, z, tp: m.down_reserve_requirement_mw[z, tp]
    )

    # Available reserves
    m.SpinningReservesUpAvailable = Expression(m.LOAD_ZONES, m.TIMEPOINTS, rule=lambda m, z, tp:
        sum(m.DispatchSlackUp[p, tp] for p in m.UP_RESERVES_STACK[z] if (p, tp) in m.PROJ_DISPATCH_POINTS)
    )
    m.SpinningReservesDownAvailable = Expression(m.LOAD_ZONES, m.TIMEPOINTS, rule=lambda m, z, tp:
        sum(m.DispatchSlackDown[p, tp] for p in m.DOWN_RESERVES_STACK[z] if (p, tp) in m.PROJ_DISPATCH_POINTS)
    )

    # Meet the reserve requirements
    m.Satisfy_Spinning_Reserve_Up_Requirement = Constraint(m.LOAD_ZONES, m.TIMEPOINTS, 
        rule=lambda m, z, tp:
            m.SpinningReservesUpAvailable[z, tp] >= m.SpinningReserveUpRequirement[z, tp]
    )
    m.Satisfy_Spinning_Reserve_Down_Requirement = Constraint(m.LOAD_ZONES, m.TIMEPOINTS, 
        rule=lambda m, z, tp:
            m.SpinningReservesDownAvailable[z, tp] >= m.SpinningReserveDownRequirement[z, tp]
    )
    
    # tabulate pre-built capacity for use in heuristic unit commitment
    # (since this is a production cost model, this is all the capacity)
    m.prebuilt_capacity = Param(m.PROJECTS, m.PERIODS, rule=lambda m, proj, period:
        sum(
            m.proj_existing_cap[proj, bld_yr] 
                for bld_yr in m.PROJECT_PERIOD_ONLINE_BUILD_YRS[proj, period]
        )
    )
    # heuristic unit commitment
    m.commit_capacity = Param(
        m.PROJ_DISPATCH_POINTS,
        initialize=commit_capacity_rule
    )
    m.Enforce_Commitment = Constraint(
        m.PROJ_DISPATCH_POINTS,
        rule=lambda m, p, tp: 
            m.CommitProject[p, tp] 
            ==  
            m.ProjCapacityTP[p, tp] * m.proj_availability[p]
            * m.commit_capacity[p, tp]
    )
    
    # make sure the model is always feasible
    # cost per MWh for unserved load (high)
    m.ge_unserved_load_penalty_per_mwh = Param(initialize=10000)
    # amount of unserved load during each timepoint
    m.GEUnservedLoad = Var(m.LOAD_ZONES, m.TIMEPOINTS, within=NonNegativeReals)
    # total cost for unserved load
    m.GE_Unserved_Load_Penalty = Expression(m.TIMEPOINTS, rule=lambda m, tp:
        sum(m.GEUnservedLoad[lz, tp] * m.ge_unserved_load_penalty_per_mwh for lz in m.LOAD_ZONES)
    )
    # add the unserved load to the model's energy balance
    m.LZ_Energy_Components_Produce.append('GEUnservedLoad')
    # add the unserved load penalty to the model's objective function
    m.cost_components_tp.append('GE_Unserved_Load_Penalty')
    
    # NOTE: the shutdown constraints below are not used, because they conflict with
    # the baseload status set in build_scenario_data.py. You should set the plant type
    # to "Off" in "source_data/Hawaii RPS Study Generator Table OCR.xlsx" instead.
    
    # # shutdown Kahe_6
    # m.KAHE_6_TIMEPOINTS = Set(initialize=lambda m: m.PROJ_ACTIVE_TIMEPOINTS['Kahe_6'])
    # m.Shutdown_Kahe_6 = Constraint(m.KAHE_6_TIMEPOINTS, rule=lambda m, tp:
    #     m.CommitProject['Kahe_6', tp] == 0
    # )

    # # shutdown Kahe_1 and Kahe_2
    # m.SHUTDOWN_TIMEPOINTS = Set(dimen=2, initialize=lambda m: [
    #     (p, tp) for p in ['Kahe_1', 'Kahe_2'] for tp in m.PROJ_ACTIVE_TIMEPOINTS[p]
    # ])
    # m.Shutdown_Projects = Constraint(m.SHUTDOWN_TIMEPOINTS, rule=lambda m, p, tp:
    #     m.CommitProject[p, tp] == 0
    # )
    
    # Force cycling plants to be online 0700-2000 and offline at other times
    # (based on inspection of Fig. 8)
    # project reporting types are defined in save_custom_results.py
    # Note: this assumes timepoints are evenly spaced, and timeseries begin at midnight
    # m.CYCLING_PLANTS_TIMEPOINTS = Set(dimen=2, initialize=lambda m: [
    #     (pr, tp) for pr in m.REPORTING_TYPE_PROJECTS['Cycling']
    #         for tp in m.PROJ_ACTIVE_TIMEPOINTS[pr]
    # ])
    # m.Cycle_Plants = Constraint(m.CYCLING_PLANTS_TIMEPOINTS, rule=lambda m, pr, tp:
    #     m.CommitSlackUp[pr, tp] == 0
    #         if (7 <= ((m.TS_TPS[m.tp_ts[tp]].ord(tp)-1) * m.tp_duration_hrs[tp]) % 24 <= 20)
    #         else m.CommitProject[pr, tp] == 0
    # )
    # def show_it(m):
    #     print "CYCLING_PLANTS_TIMEPOINTS:"
    #     print list(m.CYCLING_PLANTS_TIMEPOINTS)
    # m.ShowCyclingPlants = BuildAction(rule=show_it)

def commit_capacity_rule(m, project, timepoint):
    if hasattr(m, 'commit_capacity_dict'):
        commit = m.commit_capacity_dict
    else:
        commit = m.commit_capacity_dict = defaultdict(float)
        cap = defaultdict(float)   # committed capacity (energy+reserves)
        reserve_cap = defaultdict(float)  # reserves available
        max_contingency = defaultdict(float)
                     
        def prev_tp(tp, *steps):
            return m.TS_TPS[m.tp_ts[tp]].prevw(tp, *steps)
        def next_tp(tp, *steps):
            return m.TS_TPS[m.tp_ts[tp]].nextw(tp, *steps)
        
        def commit_proj(p, tp, level):
            reserve_eligible = (p in m.UP_RESERVES_STACK[m.proj_load_zone[p]])
            z = m.proj_load_zone[p]
            inc = level - commit[p, tp]
            commit[p, tp] += inc
            proj_capacity = (
                m.prebuilt_capacity[p, m.tp_period[tp]]
                * m.proj_availability[p]
            )
            proj_cap_factor = (
                m.proj_max_capacity_factor[p, tp] 
                if (p, tp) in m.proj_max_capacity_factor 
                else 1.0
            )
            # note: level and inc are fractions of the full project size
            cap[z, tp] += inc * proj_capacity * proj_cap_factor
            if reserve_eligible:
                reserve_cap[z, tp] += (
                    inc * proj_capacity 
                    * (proj_cap_factor - m.g_min_load_fraction[m.proj_gen_tech[p]])
                )
            if m.proj_gen_tech[p] in m.GEN_TECH_WITH_UNIT_SIZES:
                # note: we assume the whole project is a contingency 
                # (not individual units). This is consistent with the main n-1
                # constraint.
                max_contingency[z, tp] = max(
                    max_contingency[z, tp],
                    proj_capacity * proj_cap_factor * level
                )

        def fix_commit_schedule():
            # extend up time to meet minimum requirement
            for z in m.LOAD_ZONES:
                for p in m.LZ_PROJECTS[z]:
                    for tp in m.PROJ_ACTIVE_TIMEPOINTS[p]:
                        if commit[p, tp] > commit[p, prev_tp(tp)]:
                            # project just started up
                            # how many timepoints should the project stay up?
                            n_tp = int(round(
                                m.g_min_uptime[m.proj_gen_tech[p]]
                                / m.ts_duration_of_tp[m.tp_ts[tp]]
                            ))
                            # make sure it is up at least that long
                            for i in range(1, n_tp):
                                ntp = next_tp(tp, i)
                                if commit[p, ntp] < commit[p, tp]:
                                    if m.proj_max_commit_fraction[p, ntp] >= commit[p, tp]:
                                        print "min up: extending commitment for {} forward from {} to {}" \
                                            .format(p, tp, ntp)
                                        commit_proj(p, ntp, commit[p, tp])
                                    else:
                                        # mandatory outage at time ntp;
                                        # extend time earlier instead to reach min up time
                                        for j in range(1, n_tp-i+1):
                                            print "min up: extending commitment for {} backward from {} to {}" \
                                                .format(p, tp, prev_tp(tp, j))
                                            commit_proj(p, prev_tp(tp, j), commit[p, tp])
                                        break
                                    
            # don't decommit if downtime falls short of minimum requirement
            # note: this uses a bottom-up fill; if there are multiple
            # units, we don't account for the possibility of turning one
            # off and a different one on.
            for z in m.LOAD_ZONES:
                for p in m.LZ_PROJECTS[z]:
                    for tp in m.PROJ_ACTIVE_TIMEPOINTS[p]:
                        prev_commit = commit[p, prev_tp(tp)]
                        if commit[p, tp] < prev_commit:
                            # project just shut down
                            # how many timepoints should it stay down?
                            n_tp = int(round(
                                m.g_min_downtime[m.proj_gen_tech[p]]
                                / m.ts_duration_of_tp[m.tp_ts[tp]]
                            ))
                            # find highest commit level in this window
                            max_commit = commit[p, tp]
                            max_commit_i = 0
                            for i in range(1, n_tp):
                                if commit[p, next_tp(tp, i)] > max_commit:
                                    max_commit = commit[p, next_tp(tp, i)]
                                    max_commit_i = i
                            if max_commit_i > 0:
                                # turned some capacity back on too soon; 
                                # we need to fill in the gap
                                new_commit = min(prev_commit, max_commit)
                                for i in range(max_commit_i):
                                    print "min down: extending commitment for {} from {} to {}" \
                                        .format(p, prev_tp(tp), next_tp(tp, i))
                                    commit_proj(p, next_tp(tp, i), new_commit)

        # Set all units to minimum commitment.
        # This accounts for must-run plants defined in
        # build_scenario_data.py (baseload and firm RE)
        for p, tp in m.PROJ_DISPATCH_POINTS:
            commit_proj(p, tp, m.proj_min_commit_fraction[p, tp])
        # increment day-ahead commitment until it exceeds the target
        for tp in m.TIMEPOINTS:
            for z in m.LOAD_ZONES:
                for p in m.DAY_AHEAD_COMMIT_QUEUE[z]:
                    if cap[z, tp] >= (
                        m.lz_demand_mw[z, tp] 
                        - m.renewable_forecast_mw[z, tp]
                        + m.regulating_reserve_requirement_mw[z, tp]
                        + m.contingency_reserve_requirement_mw[z, tp]
                        # down reserves don't actually need to be in the total-energy
                        # supply, but this makes the commitment match GE MAPS better
                        + m.down_reserve_requirement_mw[z, tp]
                    ) and reserve_cap[z, tp] >= (
                        # note: down reserves only come from utility-scale baseload plants
                        # but this looks at both baseload and cycling plants;
                        # however, the baseload plants get committed first, so if there are
                        # enough reserves in the whole pool to provide both up and down
                        # reserves, then there should be enough in the baseload pool to
                        # provide all the down reserves. (anything more complex turns into
                        # a system of inequalities that has to be tested)
                        m.regulating_reserve_requirement_mw[z, tp] 
                        + m.contingency_reserve_requirement_mw[z, tp]
                        + m.down_reserve_requirement_mw[z, tp]
                    ):
                        # finished allocating reserves
                        break
                    if (p, tp) in m.PROJ_DISPATCH_POINTS:
                        commit_proj(p, tp, m.proj_max_commit_fraction[p, tp])
        fix_commit_schedule()

        # Commit Kalaeloa_CC3 if and only if Kalaeloa_CC1 and Kalaeloa_CC2 are committed
        # (ugh). This seems to be necessary to get it committed up until CC2 is
        # decommitted in late evening on 6/25/20 and 6/28/20 as shown in Figure 8.
        for tp in m.TIMEPOINTS:
            commit_proj(
                'Kalaeloa_CC3', tp, min(commit['Kalaeloa_CC1', tp], commit['Kalaeloa_CC1', tp])
            )

        # commit real-time projects to meet real-time energy+reserve requirements
        # note: m.REAL_TIME_COMMIT_QUEUE includes renewable projects, so this 
        # automatically includes their contribution to energy+reserves
        # Also note: this does not consider minimum up- and down-times, which is
        # fine in this case since those don't bind for any of the real-time generators.
        for tp in m.TIMEPOINTS:
            for z in m.LOAD_ZONES:
                for p in m.REAL_TIME_COMMIT_QUEUE[z]:
                    # note: we don't consider reserves as a separate category 
                    # here because none of these plants expand the reserve 
                    # margin
                    if (p, tp) not in m.PROJ_DISPATCH_POINTS:
                        do_commit = False
                    elif m.proj_gen_tech[p] not in m.GEN_TECH_WITH_UNIT_SIZES:
                        # always commit non-lumpy plants (i.e., renewables)
                        do_commit = True
                    elif cap[z, tp] < (
                        m.lz_demand_mw[z, tp] 
                        + m.regulating_reserve_requirement_mw[z, tp]
                        + m.contingency_reserve_requirement_mw[z, tp]
                        # down reserves don't actually need to be in the total-energy
                        # supply, but this makes the commitment match GE MAPS better
                        + m.down_reserve_requirement_mw[z, tp]
                    ): 
                        # not enough energy yet
                        do_commit = True
                    elif reserve_cap[z, tp] < (
                        m.regulating_reserve_requirement_mw[z, tp]
                        + m.contingency_reserve_requirement_mw[z, tp]
                        + m.down_reserve_requirement_mw[z, tp]
                    ) and p in m.UP_RESERVES_STACK:
                        # not enough reserves yet, and this could provide some
                        do_commit = True
                    else:
                        do_commit = False
                    if do_commit:
                        commit_proj(p, tp, m.proj_max_commit_fraction[p, tp])
        fix_commit_schedule()

    return commit.pop((project, timepoint), 0.0) # default=don't commit unlisted projects

def load_inputs(m, switch_data, inputs_dir):
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, 'reserve_requirements.tab'),
        auto_select=True,
        param=(
            m.regulating_reserve_requirement_mw, 
            m.contingency_reserve_requirement_mw, 
            m.renewable_forecast_mw
        )
    )


