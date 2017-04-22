"""Save results of current model in customized files.

Add this module to a standard SWITCH model, and the post_solve() callback will
be called automatically to store results.
"""

# TODO: eventually make this code more generic, e.g., create a general reporting module
# with functions like 
# add_report(report_name, indexes, reporting_times=["post_iterate", "post_solve"], restart_times=["scenario batch", "scenario", "iteration"])
# add_columns(report_name, headers, value_rule)
# Then re-create the file with the right headers at each restart time 
# (with a name reflecting the degree of generality for restart and reporting)
# Then add the rows of data at each reporting time.
# The reporting module could also define some or all of the reports below as standard reports.
# There could also be a drop_report() function or command line argument to prevent generating
# some of the standard reports or even reports defined by some of the loaded modules.
# (But generally we may want custom modules just to report their own results? Or maybe all
# standard modules add columns to the standard reports?)
# These definitions should be stored in model.config, so maybe the reporting functions should be
# added as methods (possibly by util rather than a separate reporting module).

import os
from collections import defaultdict
from pyomo.environ import *
import switch_mod.hawaii.util as util

def define_components(m):
    m.proj_reporting_type = Param(m.PROJECTS)

    def PROJECT_REPORTING_TYPES_init(m):
        types = defaultdict(list)
        for p in m.PROJECTS:
            types[m.proj_load_zone[p], m.proj_reporting_type[p]].append(p)
        m.ZONE_REPORTING_TYPES_dict = types
        return sorted(types.keys())
    m.ZONE_REPORTING_TYPES = Set(initialize=PROJECT_REPORTING_TYPES_init, ordered=True, dimen=2)

    # list of projects that match each load zone and reporting type
    m.ZONE_REPORTING_TYPE_PROJECTS = Set(
        m.ZONE_REPORTING_TYPES, 
        initialize=lambda m, z, t: m.ZONE_REPORTING_TYPES_dict[z, t]
    )
    # list of projects and timepoints that match each load zone and reporting type
    m.ZONE_REPORTING_TYPE_DISPATCH_POINTS = Set(
        m.ZONE_REPORTING_TYPES,
        dimen=2,
        initialize=lambda m, lz, rtype: [
            (p, tp) 
                for p in m.ZONE_REPORTING_TYPE_PROJECTS[lz, rtype] 
                    for tp in m.PROJ_ACTIVE_TIMEPOINTS[p]
        ]
    )

    # list of PPA projects (costs are accounted for separately)
    m.PPA_PROJECTS = Set(
        initialize=m.PROJECTS, 
        filter=lambda m, proj: any(proj.startswith(p) for p in ['AES', 'Kalaeloa'])
    )

def post_solve(m, outputs_dir):
    tag = "_" + m.options.scenario_name if m.options.scenario_name else ""
    
    # write out results

    util.write_table(m, m.TIMEPOINTS,
        output_file=os.path.join(outputs_dir, "supply{t}.tsv".format(t=tag)),
        headings=["timepoint_label"] + [lz+'_'+rtype for lz, rtype in m.ZONE_REPORTING_TYPES],
        values=lambda m, tp: 
            [m.tp_timestamp[tp]] 
            + 
            [
                sum(m.DispatchProj[p, tp] for p in m.ZONE_REPORTING_TYPE_PROJECTS[lz, rtype])
                    for lz, rtype in m.ZONE_REPORTING_TYPES
            ] 
    )
    
    # summary data for each reporting type:
    # available GWh, produced GWh, curtailed GWh (difference), installed MW, 
    # TODO: total variable cost, total capital recovery
    summary = {}
    for lz, rtype in m.ZONE_REPORTING_TYPES:
        row = []
        summary[lz, rtype] = row
        row.append((
            'GWh available',
            sum(
                (m.DispatchUpperLimit[p, tp] + m.CommitSlackUp[p, tp])  * m.tp_weight[tp] / 1000.0
                    for p, tp in m.ZONE_REPORTING_TYPE_DISPATCH_POINTS[lz, rtype]
            )
        ))
        row.append((
            'GWh produced',
            sum(
                m.DispatchProj[p, tp] * m.tp_weight[tp] / 1000.0
                    for p, tp in m.ZONE_REPORTING_TYPE_DISPATCH_POINTS[lz, rtype]
            )
        ))
        row.append((
            'MW installed',     # note: this assumes no retirement
            sum(
                m.ProjCapacity[p, per]
                    for p in m.ZONE_REPORTING_TYPE_PROJECTS[lz, rtype]
                        for per in m.PERIODS
            )
        ))

    # get headings from the last row created
    headings = ['scenario', 'load_zone', 'reporting_type'] + [c[0] for c in row]
        
    # write results
    util.write_table(m, m.ZONE_REPORTING_TYPES,
        output_file=os.path.join(outputs_dir, "totals_by_reporting_type{t}.tsv".format(t=tag)),
        headings=headings,
        values=lambda m, lz, rtype: 
            [m.options.scenario_name, lz, rtype] + [c[1] for c in summary[lz, rtype]]
    )   
    
    # total costs for this scenario
    util.write_table(m, [m.options.scenario_name],
        output_file=os.path.join(outputs_dir, "costs{t}.tsv".format(t=tag)),
        headings=['scenario', 'production cost', 'capital recovery', 'ppa'],
        values=lambda m, scen: [
            scen, 
            'xx',
            'xx',
            'xx (total cost for AES and Kalaeloa)'
        ] 
    )   
    
    

def load_inputs(m, switch_data, inputs_dir):
    switch_data.load_aug(
        filename=os.path.join(inputs_dir, 'proj_reporting_types.tab'),
        auto_select=True,
        param=(m.proj_reporting_type))
