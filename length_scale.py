#!/usr/bin/env python

import pickle

with open('./result/11_hypEVs/flow_network_e0.pkl', 'rb') as f:
    graph0 = pickle.load(f)

with open('./result/11_hypEVs/flow_network_e1.pkl', 'rb') as f:
    graph1 = pickle.load(f)

with open('./result/11_hypEVs/flow_network_e2.pkl', 'rb') as f:
    graph2 = pickle.load(f)

# diameter *= 10, length *= 10 -->
#   flow_rate *= 10^3
#   rbc_velocity *= 10
print(
    f'diameter\tlength\tflow_rate\trbc_velocity\n'
    f'{graph0.es[1000]["diameter"]:.2f}\t{graph0.es[1000]["length"]:.2f}'
    f'\t{graph0.es[1000]["flow_rate"]:.2f}\t{graph0.es[1000]["rbc_velocity"]:.2f}'
)
# diameter        length  flow_rate       rbc_velocity
# 10.258323069753654      4.484468206017623       362925.580204021        4391.116441838841
print(
    f'diameter\tlength\tflow_rate\trbc_velocity\n'
        f'{graph1.es[1000]["diameter"]:.2f}\t{graph1.es[1000]["length"]:.2f}'
        f'\t{graph1.es[1000]["flow_rate"]:.2f}\t{graph1.es[1000]["rbc_velocity"]:.2f}'
)
# diameter        length  flow_rate       rbc_velocity
# 102.58323069753655      44.84468206017623       362925580.34797555      43911.16443580578
print(
    f'diameter\tlength\tflow_rate\trbc_velocity\n'
    f'{graph2.es[1000]["diameter"]:.2f}\t{graph2.es[1000]["length"]:.2f}'
    f'\t{graph2.es[1000]["flow_rate"]:.2f}\t{graph2.es[1000]["rbc_velocity"]:.2f}'
)
# diameter        length  flow_rate       rbc_velocity
# 1025.8323069753653      448.4468206017623       362925580318.2634       439111.6443221087
