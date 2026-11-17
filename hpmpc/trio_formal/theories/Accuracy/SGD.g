# Gradient descent, one update 
# l = 64, f = 20, lr = 10e-4

# *_r : rounded value into fixed point
# eta : public lr

@rnd_na = fixed<-20, na>;
@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

eta = 1e-4;
eta_r = rnd_na(eta);

p = eta_r * g;
u_dn = rnd_dn(p);
u_up = rnd_up(p);
w_dn = w - u_dn;
w_up = w - u_up;
w_x = w - eta * g;

# w - eta * g
{
    # hypo
    g in [-1, 1] /\ # assumptions of input ranges
    w in [-1b10, 1b10] ->
    # goals
    p in [-1b22, 1b22] /\ # check overflow of product
    w_dn in [-1b43, 1b43] /\
    w_up in [-1b43, 1b43] /\
    eta_r - eta in ? /\ # error of rounding the learning rate
    w_dn - w_x in ? /\ # error between truncated fixed and reals
    w_up - w_x in ? 
}

# Results:
#  eta_r - eta in [1282625173875117b-73 {1.35803e-07, 2^(-22.812)}, 641312586937559b-72 {1.35803e-07, 2^(-22.812)}]
#  w_dn - w_x in [-641312586937559b-72 {-1.35803e-07, -2^(-22.812)}, 5144912214308055b-72 {1.08948e-06, 2^(-19.8079)}]
#  w_up - w_x in [-5144912214308055b-72 {-1.08948e-06, -2^(-19.8079)}, 641312586937559b-72 {1.35803e-07, 2^(-22.812)}]