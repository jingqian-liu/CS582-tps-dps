""" This file contains code for calculating TPS metrics Jensen-Shannon divergence, comparing MSM state probabilities and with transition negative log liklihood and fraction of valid paths from MSM compared to MSM paths from ground truth MD simulations. The first step to construct an MSM model for given a simulation"""

"""Ground truth centres"""
CLUSTER_CENTERS = {
    "chignolin": np.array(
        [[0.69400153, -0.34598462], [-0.48732213, 0.00642035], [1.87483537, 0.06285344]]
    ),
    "trp_cage": np.array(
        [[-2.15921372, 0.0062795], [0.47752285, -0.38050238], [0.40182245, 2.0690773]]
    ),
    "bba": np.array(
        [
            [-0.5756589, -0.60663654],
            [1.7861676, -0.87717611],
            [0.91295128, 1.07518898],
            [-0.49210152, 0.40313689],
        ]
    ),
    "villin": np.array(
        [
            [1.08971813, -0.98522752],
            [-2.49001353, -2.31375028],
            [-0.12929561, 0.53703407],
        ]
    ),
}



