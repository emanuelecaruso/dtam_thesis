%% INITIALIZATION

clc
clear

%image
image=[0 0 0; 0 1 0; 0 0 0];

d=image(:);
a=image(:);

q=zeros(2*size(d,1),1);


%derivative fwd

%SOBEL
A_sobel=[	0  -2	0	0	-1	0	0	0	0;
            2	0	-2	1	0	-1	0	0	0;
            0	2	0	0	1	0	0	0	0;
            0	-1	0	0	-2	0	0	-1	0;
            1	0	-1	2	0	-2	1	0	1;
            0	1	0	0	2	0	0	-1	0;
            0	0	0	0	-1	0	0	-2	0;	
            0	0	0	1	0	-1	2	0	-2;
            0	0	0	0	1	0	0	2	0;
            0	0	0	-2	-1	0	0	0	0;
            0	0	0	-1	-2	-1	0	0	0;
            0	0	0	0	-1	-2	0	0	0;
            2	1	0	0	0	0	-2	-1	0;
            1	2	1	0	0	0	-1	-2	-1;
            0	1	2	0	0	0	0	-1	-2;
            0	0	0	2	1	0	0	0	0;
            0	0	0	1	2	1	0	0	0;
            0	0	0	0	1	2	0	0	0];

At_sobel=[	0  -2	0	0	-1	0	0	0	0   0	0	0	-2	-1	0	0	0	0;
            2	0	-2	1	0	-1	0	0	0   0	0	0	-1	-2	-1	0	0	0;
            0	2	0	0	1	0	0	0	0   0	0	0	0	-1	-2	0	0	0;
            0	-1	0	0	-2	0	0	-1	0   2	1	0	0	0	0	-2	-1	0;
            1	0	-1	2	0	-2	1	0	1   1	2	1	0	0	0	-1	-2	-1;
            0	1	0	0	2	0	0	-1	0   0	1	2	0	0	0	0	-1	-2;
            0	0	0	0	-1	0	0	-2	0   0	0	0	2	1	0	0	0	0;
            0	0	0	1	0	-1	2	0	-2  0	0	0	1	2	1	0	0	0;
            0	0	0	0	1	0	0	2	0   0	0	0	0	1	2	0	0	0];
    
    
    
    
    
vec=At_sobel*A_sobel*d
a=1/norm(vec)
    
    
    
%derivative bwd

% A=A_sobel;
% A=A_grad;

% sigma_q = 0.1;
% sigma_d = 1;
% eps=0.001;
% theta=0.5;
% theta_end=0.0001;
% beta=0.0001;
% lambda=1;
% At = transpose(A);


%% ALGORITHM

n=0;
while theta>theta_end
    
    %q next
    q_next=(q+sigma_q*A*d)/(1+sigma_q*eps);
    norm_q_next=norm(q_next);
    if norm_q_next<1
        norm_q_next=1;
    end
    q_next=q_next/norm_q_next

    %d next
    d_next=(d+sigma_d*(-At*q_next+(1/theta)*a))/(1+(sigma_d/theta))
    
    %a next
    a0=(1/(2*theta))*(d_next(5)-0)^2+lambda*1;
    a1=(1/(2*theta))*(d_next(5)-1)^2+lambda*0.6;
    a_next=[0; 0; 0; 0; 0; 0; 0; 0; 0];
    if a1<a0
        a_next=[0; 0; 0; 0; 1; 0; 0; 0; 0];
    end
        
    
    if n==5
        break
    end
    
    %update
    theta=theta*(1-beta*n);
    q=q_next;
    d=d_next;
    a=a_next;
    n=n+1;

end

