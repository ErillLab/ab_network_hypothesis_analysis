"""
Load data from AB_data_Boolean.csv
"""
from utils import transpose
import re

def extract_treatment_variable(txt):
    ans = re.findall(r":(.*?)i?:",txt) # grab everything between colons, ignoring terminal i if present
    return ans[0]

def extract_response_variable(txt):
    # could do this more simply, but should fail noisily if string not prepended with "DV:"
    return re.findall(r"DV:(.*)",txt)[0]
    
with open("AB_data_Boolean.csv") as f:
    lines = [line.strip().split(",") for line in f.readlines()]

header = lines[0]
data_lines = [map(float,line) for line in lines[1:]]
num_variables = 28
num_experiments = 30

experiment_lines = [(data_lines[i],data_lines[30+i]) for i in range(num_experiments)]
for i,(before,after) in enumerate(experiment_lines):
    agreed = before[:num_variables] == after[:num_variables]
    print i,agreed
    if not agreed:
        print before[:num_variables]
        print after[:num_variables]

raw_exp_dicts = [{h:data for h,data in zip(header,transpose(experiment))} for experiment in experiment_lines]

experimental_data = []
for red in raw_exp_dicts:
    # select treatment vars where at least one inhibitor or stimulus has been applied
    treatment_vars = {extract_treatment_variable(k):int("Stimuli" in k)
                      for (k,v) in red.items()
                      if k.startswith("TR:") and any(v) # consider only treatment vars where stim or inhib applied
                      and not "HIPPO" in k} # ignore HIPPO cell line thing
    response_vars = {extract_response_variable(k):v for k,v in red.items() if k.startswith("DV")}
    experimental_data.append((treatment_vars,response_vars))

