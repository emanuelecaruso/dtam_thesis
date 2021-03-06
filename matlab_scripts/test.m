clear
clc

syms f d_1 d_2 d2_1 d2_2 width height u1 v1 u2 v2 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3

k=[f 0 -width/2; 0 -f -height/2; 0 0 -1];
kinv = inv(k);
T= [r11 r12 r13 t1; r21 r22 r23 t2; r31 r32 r33 t3; 0 0 0 1];

p_proj1_1 = [u1*d_1;v1*d_1;d_1];
p_proj1_2 = [u1*d_2;v1*d_2;d_2];
p_proj2_1 = [u2*d2_1;v2*d2_1;d2_1];
p_proj2_2 = [u2*d2_2;v2*d2_2;d2_2];

a_1=[kinv*p_proj2_1;1];
a_2=[kinv*p_proj2_2;1];
b_1=[simplify(kinv*p_proj1_1);1];
b_2=[simplify(kinv*p_proj1_2);1];

mlt_1=T*b_1;
mlt_2=T*b_2;

disp("depth_1")
expr1 = mlt_1(3) == a_1(3);
d2_1expr = solve(expr1,d2_1);
pretty(simplify(d2_1expr,'Steps',1000));

disp("depth_2")
expr1 = mlt_2(3) == a_2(3);
d2_2expr = solve(expr1,d2_2);
pretty(simplify(d2_2expr,'Steps',1000));

disp("u2_1")
expr2_1 = mlt_1(1) == a_1(1);
slv_1 = solve(subs(expr2_1,d2_1,d2_1expr),u2);
pretty(simplify(slv_1,'Steps',1000));

disp("u2_2")
expr2_2 = mlt_2(1) == a_2(1);
slv_2 = solve(subs(expr2_2,d2_2,d2_2expr),u2);
pretty(simplify(slv_2,'Steps',1000));

disp("delta")
delta = slv_1-slv_2;
pretty(simplify(delta,'Steps',1000));
