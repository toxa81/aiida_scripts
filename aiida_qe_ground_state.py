from aiida.orm.data.parameter import ParameterData
#from aiida.orm import load_node, Code
from aiida.orm.data.array.kpoints import KpointsData
#from aiida.orm.group import Group
#from math import *
from aiida.common.example_helpers import test_and_get_code

import calculation_helpers

def submit_ground_state(**kwargs):
    # get code
    #code = Code.get(label='pw.sirius.x', computername='piz_daint', useremail='antonk@cscs.ch')
    code = test_and_get_code('pw.sirius.x', expected_code_type='quantumespresso.pw')
    #code.set_prepend_text(prep_text)

    # calculation should always belong to some group, otherwise things get messy after some time
    calc_grp, created = Group.get_or_create(name=kwargs['group'])

    # create parameters
    params = calculation_helpers.create_calculation_parameters(code,
                                                               kwargs.get('partition', 'cpu'),
                                                               kwargs.get('num_ranks_per_node', 36),
                                                               kwargs.get('num_ranks_kp', 1),
                                                               kwargs.get('num_ranks_diag', 1))
    # load structure
    structure = load_node(kwargs['structure_pk'])
    
    # generate k-points
    params['kpoints'] = KpointsData()
    params['kpoints'].set_kpoints_mesh(kwargs.get('kmesh', [1, 1, 1]), offset=(0.0, 0.0, 0.0))
    params['atomic_files'] = kwargs['atomic_files']
    params['calculation_wallclock_seconds'] = kwargs.get('time_limit', 3600)
    
    # create calculation
    calc = calculation_helpers.create_calculation(structure, params)

    calc.store_all()

    calc_grp.add_nodes([calc])
    
    print "created calculation; with uuid='{}' and PK={}".format(calc.uuid, calc.pk)

    calc.submit()
    ##calc.submit_test()

if __name__ == "__main__":
    submit_ground_state(structure_pk=2,
                        atomic_files='SSSP_acc_PBE',
                        partition='cpu',
                        num_ranks_per_node=36,
                        num_ranks_kp=4,
                        num_ranks_diag=9,
                        kmesh=[4, 4, 4],
                        group='test_ground_state')

