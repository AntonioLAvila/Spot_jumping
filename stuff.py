
##### Write everything explicitly myself
def manipulator_equations(vars):
    '''
    Function that given the current configuration, velocity,
    acceleration, and contact force at the stance foot, evaluates
    the manipulator equations.

    vars: concatenated q, q', q'', and friction vectors

    Returns: vector with dimensions equal to the number of configuration
    variables. If the output of this function is equal to zero
    then the given arguments verify the manipulator equations
    '''
    assert vars.size == 3*nq + 3*nf
    q, qd, qdd, f_fl, f_fr, f_bl, f_br = np.split(vars, [nq, 2*nq, 3*nq, 3*nq+nf, 3*nq+2*nf])

    # set plant accordingly
    m_plant, context = spot_ad, plant_context_ad if isinstance(vars[0], AutoDiffXd) else plant, plant_context
    m_plant = plant # TODO remove this line

    # set state
    m_plant.SetPositions(context, q)
    m_plant.SetVelocities(context, qd)

    # matrices for manipulator equations
    M = m_plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = m_plant.CalcBiasTerm(context)
    tauG = m_plant.CalcGravityGeneralizedForces(context)

    # Jacobian of feet for contact modeling
    J_fl = None # calc_foot_jacobian(plant, context, )
    J_fr = None # calc_foot_jacobian(plant, context, )
    J_bl = None # calc_foot_jacobian(plant, context, )
    J_br = None # calc_foot_jacobian(plant, context, )

    return M@qdd + Cv - tauG - J_fl.T@f_fl - J_fr.T@f_fr - J_bl.T@f_bl - J_br.T@f_br

def calc_foot_jacobian(plant_, context, foot_frame, wrt_frame, jacobian_var):
    ground_frame = plant.GetBodyByName("ground").body_frame()
    J = plant_.CalcJacobianSpatialVelocity(
        context,
        JacobianWrtVariable(jacobian_var),
        foot_frame,
        wrt_frame,
        ground_frame,
        ground_frame
    )
    return J