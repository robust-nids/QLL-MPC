# multiply x and y
# l = 64, f = 20
# x,y : fixed point numbers
# p : exact product
# z : truncated product

@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

p = x * y;
z_dn = rnd_dn(p);
z_up = rnd_up(p);

{
    # hypo
    x in [-1b11, 1b11] /\
    y in [-1b11, 1b11] ->
    # goal
    p in [-1b22, 1b22] /\ # 2^(l - 2 - 2f) i.e. @ (64 - 2 - 2*20)
    (z_dn - x * y) * 1b20 in ? /\
    (z_up - x * y) * 1b20 in ? /\
    z_dn in ? /\
    z_up in ?
}