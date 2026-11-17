# multiply fixed with real
# z = sr(rnf(c) * x)

# x : fixed point number
# c : real
# c_r : fixed point representation of c
# p : exact product 
# z_dn : rounded down fixed point product
# z_up : rounded up fixed point product

@rnd_na = fixed<-20, na>;
@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

c = 1e-3;
c_r = rnd_na(c);

p = c_r * x;
z_dn = rnd_dn(p);
z_up = rnd_up(p);

{
    # hypothesis
    x in [-1b20, 1b20]
    ->
    # goals
    p in [-1b22, 1b22] /\
    c_r - c in ? /\
    (z_dn - p) * 1b20 in ? /\ # truncation ulp
    (c_r - c)/ c in ? /\ # relative error
    z_up - c * x in ?
}