# Polynomial using Horner's Rule
# Horner's rule: a + ax + ax^2 + ax^3 = a + x(a + x(a + x(a)))

# a : real valued coefficents
# c : fixed-point roudnding of a

@rnd_na = fixed<-20, na>;

a0 = 0.999999801887; c0 = rnd_na(a0);
a1 = 0.999999978076; c1 = rnd_na(a1);
a2 = 0.500006337442; c2 = rnd_na(a2);
a3 = 0.166667368038; c3 = rnd_na(a3);
a4 = 0.0416350384549; c4 = rnd_na(a4);
a5 = 0.00832983182561; c5 = rnd_na(a5);
a6 = 0.00143925671568; c6 = rnd_na(a6);
a7 = 0.0002039935802; c7 = rnd_na(a7);

# exact 
s7 = a7 * x;
s6 = (s7 + a6) * x;
s5 = (s6 + a5) * x;
s4 = (s5 + a4) * x;
s3 = (s4 + a3) * x;
s2 = (s3 + a2) * x;
s1 = (s2 + a1) * x;
y_exact = s1 + a0;

# compute Horner's Rule w/Truncation
q7 = c7 * x; u7 = q7 + e7;
q6 = (u7 + c6) * x; u6 = q6 + e6;
q5 = (u6 + c5) * x; u5 = q5 + e5;
q4 = (u5 + c4) * x; u4 = q4 + e4;
q3 = (u4 + c3) * x; u3 = q3 + e3;
q2 = (u3 + c2) * x; u2 = q2 + e2;
q1 = (u2 + c1) * x; u1 = q1 + e1;
y = u1 + c0;

# no final truncation
t6 = (q7 + c6) * x;
t5 = (t6 + c5) * x;
t4 = (t5 + c4) * x;
t3 = (t4 + c3) * x;
t2 = (t3 + c2) * x;
t1 = (t2 + c1) * x;
yt = t1 + c0;

{
    # hypothesis
    x in [-1, 1] /\
    e1 in [-1b-20, 1b-20] /\ e2 in [-1b-20, 1b-20] /\ e3 in [-1b-20, 1b-20] /\ e4 in [-1b-20, 1b-20] /\
    e5 in [-1b-20, 1b-20] /\ e6 in [-1b-20, 1b-20] /\ e7 in [-1b-20, 1b-20] ->

    # goals
    y in ? /\ # range of result
    (y - yt) * 1b20 in ? /\ # error from truncations
    (yt - y_exact) * 1b20 in ? /\ # error from rounding coefficents
    (y - y_exact) * 1b20 in ? # total error from real poly
}

# Gappa hints of how truncation is carried out
# e = u - q, rounded - result, i.e truncation error
u6 - t6 -> e7 * x + e6;
u5 - t5 -> (u6 - t6) * x + e5;
u4 - t4 -> (u5 - t5) * x + e4;
u3 - t3 -> (u4 - t4) * x + e3;
u2 - t2 -> (u3 - t3) * x + e2;
u1 - t1 -> (u2 - t2) * x + e1;