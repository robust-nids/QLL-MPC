# Truncation of integer and stochastic rounding
# l = 64, f = 20

# a : exact real 
# z_dn : rounded down in fixed point
# x_up: rounded up in fixed point
# z : stochastic rounding return (up or down)

@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

z_dn = rnd_dn(a);
z_up = rnd_up(a);

{
    # hypo
    a in [-1b22, 1b22] # range within 2^(l-2-2f)
    /\ z - a in [-1b-20, 1b-20]
    ->
    # goals, error of rounding up, down, output
    (z_dn - a) * 1b20 in ? /\ 
    (z_up - a) * 1b20 in ? /\
    z in ? /\
    # (z_up - z_dn) * 1b20 in ?
}