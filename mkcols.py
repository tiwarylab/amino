import subprocess

plumed_file = "input.plu"
traj_file = "md1.xtc"
bashCommand = "plumed driver --plumed " + plumed_file + " --mf_xtc " + traj_file

process = subprocess.Popen(bashCommand.split())
