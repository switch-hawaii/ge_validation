import os
from pyomo.environ import *

def define_components(m):
    # by inspection of figure 8 & 9 in the RPS Study, it appears that Kalaeloa has 3 modes:
    # commit unit 1, run between 65 and 90 MW
    # commit units 1 & 2, run each between 65 and 90 MW
    # run both 1 & 2 at 90 MW, and run 3 at 28 MW

    ### now in hawaii.kalaeloa ###
    # # run kalaeloa at full power or not
    # m.RunKalaeloaFull = Var(m.TIMEPOINTS, within=Binary)
    #
    # more_than_kalaeloa_capacity = 220   # used for big-m constraints on individual units
    #
    # m.Run_Kalaeloa_Full_Enforce = Constraint(
    #     ["Kalaeloa_CC1", "Kalaeloa_CC2"], m.TIMEPOINTS,
    #     rule=lambda m, proj, tp:
    #         m.DispatchProj[proj, tp] + (1-m.RunKalaeloaFull[tp]) * more_than_kalaeloa_capacity
    #         >=
    #         m.ProjCapacityTP[proj, tp] * m.proj_availability[proj]
    #         # m.DispatchSlackUp[proj, tp]   # doesn't work, because the plant can be decommitted
    #         # <=
    #         # (1-m.RunKalaeloaFull[tp]) * more_than_kalaeloa_capacity
    # )
    # m.Run_Kalaeloa_CC3_Only_When_Full = Constraint(m.TIMEPOINTS, rule=lambda m, tp:
    #     m.DispatchProj["Kalaeloa_CC3", tp]
    #     <=
    #     m.RunKalaeloaFull[tp] * more_than_kalaeloa_capacity
    #     # note: min load is set to 100%, so the unit will be forced fully on if it is committed;
    #     # the constraint above keeps it from being committed unless the plant is fully on
    # )
    ### previous parts now in hawaii.kalaeloa ###

    # # force Kalaeloa_CC2 offline at midnight every night (to match Fig. 8 & 9)
    # m.MIDNIGHTS = Set(initialize=m.TIMEPOINTS, filter=lambda m, tp:
    #     int(round((m.TS_TPS[m.tp_ts[tp]].ord(tp)-1) * m.tp_duration_hrs[tp])) % 24 == 0
    # )
    # m.NOONS = Set(initialize=m.TIMEPOINTS, filter=lambda m, tp:
    #     int(round((m.TS_TPS[m.tp_ts[tp]].ord(tp)-1) * m.tp_duration_hrs[tp])) % 24 == 12
    # )
    # m.Cycle_Kalealoa_CC2_Off = Constraint(m.MIDNIGHTS, rule=lambda m, tp:
    #     m.CommitProject['Kalaeloa_CC2', tp] == 0
    # )
    # m.Cycle_Kalealoa_CC2_On = Constraint(m.NOONS, rule=lambda m, tp:
    #     m.CommitSlackUp['Kalaeloa_CC2', tp] == 0
    # )


    # force H-POWER and HONUA on at full power, per e-mail from John Cole 2016-09-28
    # (now implemented via min_load_fraction in build_scenario_data.py)
    # m.Run_Waste_at_Full_Power = Constraint(['H-Power', 'Honua'], m.TIMEPOINTS,
    #     rule=lambda m, p, tp:
    #         (m.DispatchProj[p, tp] == m.DispatchUpperLimit[p, tp])
    #             if (p, tp) in m.PROJ_DISPATCH_POINTS
    #             else Constraint.Skip
    # )