import mod_eos
import numpy as np
import matplotlib.pyplot as plt
import click
import copy

def get_E_of_V(eos_pk):
    n = load_node(eos_pk)

    x = []
    y = []
    for e in n.get_outputs():
        if isinstance(e, JobCalculation):
            #print("calculation PK: %i"%e.pk)
            V = e.res.volume
            E = e.res.energy
            #Eha = e.res.energy_hartree
            #Eoe = e.res.energy_one_electron
            #Esm = e.res.energy_smearing
            #Eew = e.res.energy_ewald
            #Exc = e.res.energy_xc

            x.append(V)
            y.append(E)
            #print("volume: %f, energy: %f"%(V, E))

    x, y = zip(*sorted(zip(x, y)))

    return {"eos_pk" : eos_pk, "V" : x, "E" : y}

def get_V0_V1_fit(E_of_V):

    fit_params = mod_eos.fit_birch_murnaghan_params(E_of_V["V"], E_of_V["E"])
    if fit_params != None:
        fit_params[0] = 0
    
    V0 = E_of_V["V"][0]
    V1 = E_of_V["V"][-1]

    return (V0, V1, fit_params)

def eos_delta(E_of_V_a, E_of_V_b):

    (V0_a, V1_a, fit_a) = get_V0_V1_fit(E_of_V_a)
    (V0_b, V1_b, fit_b) = get_V0_V1_fit(E_of_V_b)

    if np.abs(V0_a - V0_b) > 1e-4 or np.abs(V1_a - V1_b) > 1e-4:
        print("Wrong boundaries")
        sys.exit(0)
    
    V0 = max(V0_a, V0_b)
    V1 = min(V1_a, V1_b)

    from scipy.integrate import quad
    I = quad(mod_eos.deltaE2, V0, V1, args=(fit_a, fit_b))[0]
    delta = np.sqrt(I / (V1 - V0)) * 1000
    #print("Delta: %f meV/atom"%delta)
    return delta

def add_to_plot(p, E_of_V, zero, eos_id):
    x = copy.deepcopy(E_of_V["V"])
    y = copy.deepcopy(E_of_V["E"])
    
    fit_params = mod_eos.fit_birch_murnaghan_params(x, y)
    if zero:
        y0 = fit_params[0] if fit_params != None else min(y)
        y = [e - y0 for e in y]
        if fit_params != None:
            fit_params[0] = 0
    
    V0 = x[0]
    V1 = x[-1]
    
    #print("V0=%f, V1=%f, Vinput=%f %f"%(V0, V1, V1 / 1.06, V0 / 0.94))
    
    new_x = np.arange(V0, V1, 0.01)
     
    c="C%i"%eos_id

    eos_pk = E_of_V["eos_pk"]
    n = load_node(eos_pk)
    code = n.get_inputs_dict()['code']
    partition = n.get_inputs_dict()['partition']
    
    label = "%i: "%(eos_id+1) + " " + code.label + " (EoS PK=%i, %s partition)"%(eos_pk, str(partition))
    p.plot(x, y, "o", linewidth = 2.0, label=label, color=c)

    if fit_params == None:
        p.plot(x, y, linewidth = 2.0, color=c)
    else:
        p.plot(new_x, mod_eos.birch_murnaghan(new_x, *fit_params), color=c)


@click.command()
@click.argument('eos_pk', type=int, required=True, nargs=-1)
def plot_eos(eos_pk):
   
    # collect the structure PKs into a list
    spk = [load_node(i).get_inputs_dict()['structure'].pk for i in eos_pk]
    # check if all structure PKs are identical
    if spk.count(spk[0]) != len(spk):
        raise RuntimeError('structures are different')
    
    struct = load_node(eos_pk[0]).get_inputs_dict()['structure']
    num_atoms = len(struct.sites)
    print("struct: %s, number of atoms: %i"%(struct.get_formula(), num_atoms))

    E_of_V = [get_E_of_V(i) for i in eos_pk]
    
    plt.clf()

    f, axarr = plt.subplots(2, sharex=True)
    
    for i in (0, 1):
        box = axarr[i].get_position()
        axarr[i].set_position([box.x0 + 0.04, box.y0, box.width * 0.7, box.height])
        axarr[i].grid(which = "major", axis = "both")

    for i in range(len(E_of_V)):
        add_to_plot(axarr[0], E_of_V[i], False, i)
        add_to_plot(axarr[1], E_of_V[i], True, i)

    #plt.legend(loc=2, bbox_to_anchor=(1, 1), prop={'size':8})
    axarr[0].legend(loc=2, prop={'size':8})
    #plt.grid(which = "major", axis = "both")

    #axarr[0].xlabel('Unit cell volume [A^3]')
    #plt.ylabel('Total energy [eV]')
    #plt.title("EOS for %s"%struct.get_formula())
    #axarr[0].set_title("EOS for %s (structure PK: %i), absolute"%(struct.get_formula(), struct.pk))
    #axarr[1].set_title("EOS for %s (structure PK: %i), relative"%(struct.get_formula(), struct.pk))
    #axarr[0].set_title("absolute")
    #axarr[1].set_title("relative")
    #plt.ylim(ymin=-0.0)
    f.text(0.45, 0.95, "%s EoS"%struct.get_formula(), ha='center', va='center', size=15)
    f.text(0.5, 0.04, 'Unit cell volume [A^3]', ha='center', va='center')
    f.text(0.04, 0.5, 'Total energy [eV]', ha='center', va='center',  rotation=90)

    k = 0
    for i in range(len(E_of_V)):
        for j in range(i, len(E_of_V)):
            if i != j:
                delta = eos_delta(E_of_V[i], E_of_V[j]) / num_atoms
                f.text(0.75, 0.8 - 0.07 * k , r"$ \Delta_{%i%i} = %8.2f \; \frac{meV}{atom} $"%(i+1, j+1, delta))
                #print("Delta between %i and %i, %f meV / atom"%(i+1, j+1, eos_delta(E_of_V[i], E_of_V[j]) / num_atoms))
                k = k + 1

  
    plt.savefig("eos.%s.png"%struct.get_formula(), format="png", dpi=300)
    plt.savefig("eos.%s.pdf"%struct.get_formula(), format="pdf")

if __name__ == "__main__":
    plot_eos()

