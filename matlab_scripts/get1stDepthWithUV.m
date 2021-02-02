syms f d width height u1 v1 u2 v2 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3

k=[f 0 -width/2; 0 -f -height/2; 0 0 -1];
kinv = inv(K);
T= [r11 r12 r13 t1; r21 r22 r23 t2; r31 r32 r33 t3; 0 0 0 1];

p_proj1 = [u1*d;v1*d;d];
p_proj2 = [u2*f;v2*f;f];

a=[kinv*p_proj2;1];
b=[simplify(kinv*p_proj1);1];

mlt=T*b;

disp("depth")
expr1 = mlt(3) == a(3);
slv = solve(expr1,d);
pretty(simplify(slv,'Steps',1000));

disp("u2")
expr2 = mlt(1) == a(1);
slv = solve(expr2,u2);
pretty(simplify(slv,'Steps',1000));

disp("v2")
expr3 = mlt(2) == a(2);
slv = solve(expr3,v2);
pretty(simplify(slv,'Steps',1000));



