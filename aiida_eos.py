from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.common.example_helpers import test_and_get_code

import calculation_helpers
import click
import numpy as np

def scaled_structure(structure, scale):

    new_structure = StructureData(cell=np.array(structure.cell)*scale)

    for site in structure.sites:
        new_structure.append_atom(position=np.array(site.position)*scale, \
                                  symbols=structure.get_kind(site.kind_name).symbol,\
                                  name=site.kind_name)
    new_structure.label = 'created inside EOS run'
    new_structure.description = "auxiliary structure for EOS "\
                                "created from the original structure with PK=%i, "\
                                "lattice constant scaling: %f"%(structure.pk, scale)

    return new_structure

def submit_eos(**kwargs):
    # get code
    #code = Code.get(label='pw.sirius.x', computername='piz_daint', useremail='antonk@cscs.ch')
    code = test_and_get_code('pw.sirius.x', expected_code_type='quantumespresso.pw')
    #code.set_prepend_text(prep_text)

    # calculation should always belong to some group, otherwise things get messy after some time
    eos_grp, created = Group.get_or_create(name=kwargs['group'])
    
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
    params['structure'] = structure
    params['num_points'] = 7
    params['group'] = kwargs['group']
    params['kpoints'].store()
    params['calculation_parameters'].store()
    params['calculation_settings'].store()

    eos_dict = {}
    eos_dict['label'] = 'eos_' + structure.get_formula() + '_' + code.label
    eos_dict['description'] = "Equaiton of states for structure with PK=%i"%structure.pk
    eos_dict['calc_pk'] = []
    eos_dict['num_points'] = params['num_points']
    eos_dict['structure_pk'] = structure.pk
    eos_dict['code_pk'] = code.pk

    # volume scales from 0.94 to 1.06, alat scales as pow(1/3)
    scales = np.linspace(0.979586108715562, 1.019612822422216, num=params['num_points']).tolist()

    for scale in scales:

        structure_new = scaled_structure(structure, scale)
        structure_new.store()
        
        calc_label = 'gs_' + structure.get_formula() + '_' + code.label
        calc_desc = params['job_tag']
    
        # create calculation
        calc = calculation_helpers.create_calculation(structure_new, params, calc_label, calc_desc)
        calc.store()
        print "created calculation with uuid='{}' and PK={}".format(calc.uuid, calc.pk)
        eos_grp.add_nodes([calc])
        calc.submit()
        eos_dict['calc_pk'].append(calc.pk)
    
    eos_node = ParameterData(dict=eos_dict)
    eos_node.store()
    eos_grp.add_nodes([eos_node])
    print "created EOS ndnode with uuid='{}' and PK={}".format(eos_node.uuid, eos_node.pk)
 

@click.command()
@click.option('--structure_pk', type=int, help='PK of the structure', required=True)
@click.option('--atomic_files', type=str, help='label for the atomic files dataset', required=True)
@click.option('--group', type=str, help='label of the EOS workflow group', required=True)
@click.option('--partition', type=str, help='run on "cpu" or "gpu" partition', default='cpu')
@click.option('--ranks_per_node', type=int, help='number of ranks to put on a single node', default=36)
@click.option('--ranks_kp', type=int, help='number of ranks for k-point parallelization', default=1)
@click.option('--ranks_diag', type=int, help='number of ranks for band parallelization', default=36)
@click.option('--kmesh', type=(int, int, int), help='k-point grid', default=(4, 4, 4))
def run(structure_pk, atomic_files, group, partition, ranks_per_node, ranks_kp, ranks_diag, kmesh):
    submit_eos(structure_pk=structure_pk,
               atomic_files=atomic_files,
               partition=partition,
               num_ranks_per_node=ranks_per_node,
               num_ranks_kp=ranks_kp,
               num_ranks_diag=ranks_diag,
               kmesh=kmesh,
               group=group)

if __name__ == "__main__":
    run()

