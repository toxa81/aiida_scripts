from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.common.example_helpers import test_and_get_code
from aiida.orm import Group, load_node
import numpy as np

def create_calculation_parameters(code, partition, num_ranks_per_node, num_ranks_kp, num_ranks_diag):
    """
    Create a dictionary with parameters for a job execution.
    """

    if partition not in ['cpu', 'gpu']:
        raise RuntimeError('wrong partition name')
    
    num_cores = {'cpu' : 36, 'gpu' : 12}

    if num_cores[partition] % num_ranks_per_node != 0:
        raise RuntimeError('wrong number of ranks per node')
    
    num_threads = num_cores[partition] / num_ranks_per_node

    # total number of ranks
    num_ranks = num_ranks_kp * num_ranks_diag
    
    # get number of nodes
    num_nodes = max(1, num_ranks / num_ranks_per_node)
    
    print("partition: %s"%partition)
    print("number of nodes : %i"%num_nodes)

    # create dictionary to store parameters
    params = {}
    
    params['job_tag'] = "%iN:%iR:%iT @ %s"%(num_nodes, num_ranks_per_node, num_threads, partition)

    environment_variables = {'OMP_NUM_THREADS': str(num_threads),\
                             'MKL_NUM_THREADS': str(num_threads),\
                             'KMP_AFFINITY': 'granularity=fine,compact,1'}
    if partition == 'gpu' and num_ranks_per_node > 1:
        environment_variables['CRAY_CUDA_MPS'] = '1'

    params['environment_variables'] = environment_variables

    #calc.set_custom_scheduler_commands('#SBATCH -A, --account=mr21')
    if partition == 'cpu':
        params['custom_scheduler_commands'] = u"#SBATCH -C mc"
    if partition == 'gpu':
        params['custom_scheduler_commands'] = u"#SBATCH -C gpu"

    # create settings
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        if partition == 'cpu':
            settings = ParameterData(dict={
                'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag)]})
        if partition == 'gpu':
            settings = ParameterData(dict={
                #'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag), '-sirius', '-sirius_cfg', '/users/antonk/codes/config.json']})
                'CMDLINE' : ['-npool', str(num_ranks_kp), '-ndiag', str(num_ranks_diag), '-sirius']})
            
        parameters = ParameterData(dict={
            'CONTROL': {
                'calculation'  : 'scf',
                'restart_mode' : 'from_scratch',
                'disk_io'      : 'none'
                },
            'SYSTEM': {
                'ecutwfc': 80.,
                'ecutrho': 1200.,
                #'nbnd': 40,
                'occupations': 'smearing',
                'smearing': 'gauss',
                'degauss': 0.1
                },
            'ELECTRONS': {
                'conv_thr': 1.e-9,
                'electron_maxstep': 100,
                'mixing_beta': 0.7
                }})

    if code.get_input_plugin_name() == 'exciting.exciting':
        parameters = ParameterData(dict={'groundstate' : {'xctype' : 'GGA_PBE',
                                         'swidth' : '0.001',
                                         'beta0'  : '0.4',
                                         'gmaxvr' : '20.0',
                                         'rgkmax' : '7.0',
                                         'lmaxmat' : '10',
                                         'lmaxapw' : '10',
                                         'lmaxvr'  : '10',
                                         'fracinr' : '1d-12',
                                         'maxscl'  : '40',
                                         'nempty'  : '10'}})
        settings = ParameterData(dict={})


    params['calculation_settings'] = settings
    params['calculation_parameters'] = parameters
    params['mpirun_extra_params'] = ['-n', str(num_ranks), '-c', str(num_threads), '--hint=nomultithread','--unbuffered']
    params['calculation_resources'] = {'num_machines': num_nodes, 'num_mpiprocs_per_machine': num_ranks_per_node}
    params['code'] = code

    return params

def create_calculation(structure, params, calc_label, calc_desc):
    """
    Create calculation object from structure and a dictionary of parameters.
    Calculation has to be stored in DB by the caller.
    """
    code = params['code']
    
    calc = code.new_calc()
    calc.set_max_wallclock_seconds(params.get('calculation_wallclock_seconds', 3600)) # in second
    calc.set_resources(params['calculation_resources'])
    
    calc.use_structure(structure)
    calc.use_parameters(params['calculation_parameters'])
    calc.use_kpoints(params['kpoints'])
    calc.use_settings(params['calculation_settings'])
    if code.get_input_plugin_name() == 'quantumespresso.pw':
        calc.use_pseudos_from_family(params['atomic_files'])
    calc.set_environment_variables(params['environment_variables'])
    calc.set_mpirun_extra_params(params['mpirun_extra_params'])
    calc.set_custom_scheduler_commands(params['custom_scheduler_commands'])
    calc.label = calc_label
    calc.description = calc_desc

    return calc

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
    params = create_calculation_parameters(code,
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
    eos_dict['description'] = "Equation of states for structure with PK=%i"%structure.pk
    eos_dict['calc_pk'] = []
    eos_dict['num_points'] = params['num_points']
    eos_dict['structure_pk'] = structure.pk
    eos_dict['code_pk'] = code.pk
    eos_dict['job_tag'] = params['job_tag']

    # volume scales from 0.94 to 1.06, alat scales as pow(1/3)
    scales = np.linspace(0.979586108715562, 1.019612822422216, num=params['num_points']).tolist()
    #scales = np.linspace(0.99, 1.05, num=params['num_points']).tolist()

    for scale in scales:

        structure_new = scaled_structure(structure, scale)
        structure_new.store()
        
        calc_label = 'gs_' + structure.get_formula() + '_' + code.label
        calc_desc = params['job_tag']
    
        # create calculation
        calc = create_calculation(structure_new, params, calc_label, calc_desc)
        calc.store()
        print "created calculation with uuid='{}' and PK={}".format(calc.uuid, calc.pk)
        eos_grp.add_nodes([calc])
        calc.submit()
        eos_dict['calc_pk'].append(calc.pk)
    
    eos_node = ParameterData(dict=eos_dict)
    eos_node.store()
    eos_grp.add_nodes([eos_node])
    print "created EOS node with uuid='{}' and PK={}".format(eos_node.uuid, eos_node.pk)
