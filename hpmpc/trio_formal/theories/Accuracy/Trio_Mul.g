# Trio Multiplication 

# Locally Compute
# P_0 = lambda_1, lambda_2
# P_1 : V_1 = m_a2 * lambda_b1 + m_bd2 * lambda_a1 + r_1
# P_2 : V_2 = m_a1 * m_b1 + M_0

# Communicate
# M_1 = V_1 - lambda_c1
# M_1 = V_2 - lambda_c2

# Locally Compute
# P_1 : m_c2 = M_2 - V_1 
# P_2 : m_c1 = V_2 - M_1

@rnd_na = fixed<-20, na>;
@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

#inputs
a = rnd_na(a_in);
b = rnd_na(b_in);

# shares of a and b_in
m_a1 = a + lambda_a1;
m_a2 = a + lambda_a2;
m_b1 = b + lambda_b1;
m_b2 = b + lambda_b2;

# preprocessing message
M_0 = lambda_a2 * lambda_b2 - (lambda_a1 - lambda_a2) * (lambda_b1 - lambda_b2) + r_1;

# local computation
V_1 = m_a2 * lambda_b1 + m_b2 * lambda_a1 + r_1;
V_2 = m_a1 * m_b1 + M_0;

# communicatoin
M_1 = V_1 - lambda_c1;
M_2 = V_2 + lambda_c2;

# output shares of c
m_c1 = V_2 - M_1;
m_c2 = M_2 - V_1;

c_1 = m_c1 - lambda_c1;
c_2 = m_c2 - lambda_c2;

# exact product
p = a * b;
z_dn = rnd_dn(c_1);
z_up = rnd_up(c_1);

{
    # hypothesis
    a_in in [-1b11, 1b11] /\ b_in in [-1b11, 1b11] /\
    lambda_a1 in [-1b43, 1b43] /\ lambda_a2 in [-1b43, 1b43] /\ 
    lambda_b1 in [-1b43, 1b43] /\ lambda_b2 in [-1b43, 1b43] /\
    lambda_c1 in [-1b43, 1b43] /\ lambda_c2 in [-1b43, 1b43] /\
    r_1 in [-1b43, 1b43] ->
    # goals
    # compare with exact
    c_1 - p in [0,0] /\
    c_2 - p in [0,0] /\

    # overflow of input
    p in [-1b22, 1b22] /\
    # truncated ulp
    (z_dn - p) * 1b20 in ? /\
    (z_up - p) * 1b20 in ? /\
    z_dn in ? /\
    z_up in ?
}

# Hints for Gappa, identites:
# identity between V and p
V_2 - V_1 - p -> 0;

# identity between c_ and p
c_1 - p -> 0;
c_2 - p -> 0;
