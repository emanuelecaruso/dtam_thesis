%% INITIALIZATION

clc
clear

%image
image=imread("./data.png");
I = im2double(image);

% imshow(I);


d=I;
a=I;
q=zeros(size(d,1),2*size(d,2));

sigma_q = 0.000347;
sigma_d = 100;
eps=0.0001;
theta=10000;
theta_end=5;
beta=0.005;
% beta=0.0001;
lambda=0;

%% ALGORITHM

n=0;
tic
% while theta>theta_end
    
    [Gx, Gy] = imgradientxy(d);
    grad= [Gx Gy];
    
    %q next
    q_next=(q+sigma_q*grad)/(1+sigma_q*eps);
    norm_q_next=norm(q_next);
    if norm_q_next<1
        norm_q_next=1;
    end
    q_next=q_next/norm_q_next;
%     norm(q_next)
    
    Qx= q_next(:, 1:(size(q_next,2)/2));
    Qy= q_next(:, (size(q_next,2)/2)+1:size(q_next,2));
    [Dx, ~] = imgradientxy(Qx);
    [~, Dy] = imgradientxy(Qy);
    div=Dx+Dy;
    
    %d next
%     d_next=(d+sigma_d*(-At*q_next+(1/theta)*a))/(1+(sigma_d/theta))
    d_next=(d+sigma_d*(div)+(1/theta)*d)/(1+(sigma_d/theta));
%     d_next=(d+sigma_d*(div))/(1+(sigma_d/theta));
    norm(d_next);
%     if n==5
%         break
%     end

    %update
    theta=theta*(1-beta*n);
    q=q_next;
    d=d_next;
    n=n+1;
% %     
    imshow(d);
    drawnow;
% end
timeElapsed = toc
n
