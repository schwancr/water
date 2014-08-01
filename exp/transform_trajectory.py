import numpy as np
import mdtraj as md
from mdtraj import io
import bubble
import water
import argparse
from time import time
import multiprocessing as mp
import os

parser = argparse.ArgumentParser()
parser.add_argument('-t', dest='traj', help='trajectory to calculate features for')
parser.add_argument('--top', dest='top', default=None, help='topology file if needed.')
parser.add_argument('-k', dest='n_features', type=int, default=2, help='number of features to calculate. Note we start with the 2-body term since the 1-body term is unity for everything')
parser.add_argument('-o', dest='output', help='output filename to save to (mdtraj.io.saveh)')
parser.add_argument('-e', dest='eps', type=float, help='epsilon for picking neighbor atoms')
parser.add_argument('-p', dest='procs', type=int, default=1, help='number of processes to use')

args = parser.parse_args()

# first I need to compute the number of atoms

if os.path.exists(args.output):
    print "file exists"
    exit()

if args.top is None:
    print "not implemented. Give me a pdb file."
    exit()

struct = md.load(args.top)

type_dict = {}
atom_types = []
# keys are of the form HOH-H or HOH-O
# i.e. <res name>-<atom name>
for atom in struct.top.atoms:
    key = '%s-%s' % (atom.residue.name.upper(), atom.element.symbol)
    # note ^^^ this is bad. I want to do this so that the water H's are 
    # the same type. But I should really do it a different way.

    if not key in type_dict:
        if len(type_dict.keys()) != 0:
            type_dict[key] = np.max(type_dict.values()) + 1
        else:
            type_dict[key] = 0

    atom_types.append(type_dict[key])

atom_types = np.array(atom_types)
num_types = len(type_dict.keys())

epsilon = args.eps

def get_features(frame):

    try:
        a = time()
        atom_distances, atom_pairs = water.get_atom_distances(frame)
        atom_distances = md.geometry.contact.squareform(atom_distances, atom_pairs)

        features, combos = bubble.get_features(atom_distances[0], atom_types, n_terms=args.n_features, bubble_radius=args.eps)
        b = time()
        print "finished frame in %.2f seconds" % (b-a,)

        return features, combos

    except Exception as e:
        print str(e)
        raise Exception(str(e))

traj = md.load(args.traj, top=struct)
n_frames = len(traj)

pool = mp.Pool(args.procs)
result = pool.map_async(get_features, traj)

# this try/except allows you to kill the script with ctrl-c
try:
    result.wait(1E100)
except KeyboardInterrupt:
    exit()

pool.close()
pool.join()
result_list = result.get()

type_dict = np.array(list(type_dict.iteritems()))

ary_features = np.vstack([res[0] for res in result_list])
combos = result_list[0][1] # the combos should be the same for each result

ary_combos = np.zeros((len(combos), args.n_features + 1)) - 1

for i, c in enumerate(combos):
    ary_combos[i, :len(c)] = c

io.saveh(args.output, features=ary_features, combos=ary_combos,
            type_dict=type_dict, atom_types=atom_types)

print "Saved output to %s" % args.output
