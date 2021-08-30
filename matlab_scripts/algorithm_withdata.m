%% INITIALIZATION

clc
clear

%image
image=imread("./data2.png");
I = im2double(image);


d=I;
a=I;
q=zeros(size(d,1),2*size(d,2));

% sigma_q = 0.000347;

sigma_q0 = 0.1;
sigma_d0 = 0.1;
sigma_q = sigma_q0;
sigma_d = sigma_d0;
eps=0.0001;
theta=0.2;
theta_end=0.001;
beta=0.005;
% beta=0.0001;
lambda=0;

%% ALGORITHM

n=0;
tic
grad=[];
r=0.9;
    
disp("norm d 0: ")
disp(norm(d))

mode='sobel';
while theta>theta_end
    
    
    [Gx, Gy] = imgradientxy(d,mode);
    grad= [Gx Gy];
    
    
    %q next
    q_next=(q+sigma_q*grad)/(1+sigma_q*eps);
%     max_q_next=norm(q_next);
    max_q_next=max(abs(q_next(:)));
    if max_q_next<1
        max_q_next=1;
    end
    q_next=q_next/max_q_next;
    
    Qx= q_next(:, 1:(size(q_next,2)/2));
    Qy= q_next(:, (size(q_next,2)/2)+1:size(q_next,2));
    [Dx, ~] = imgradientxy(Qx,mode);
    [~, Dy] = imgradientxy(Qy,mode);
    div=Dx+Dy;
    
    %d next
    d_next=(d+sigma_d*((div)+(1/theta)*d))/(1+(sigma_d/theta));
    
%     if n==5
%         break
%     end

    %update
    theta=theta*(1-beta*n);
    q=q_next;
    d=d_next;
    n=n+1;
    
    sigma_q=sigma_q0/r^n;
    sigma_d=sigma_d0*r^n;
    
%     sigma_q=sigma_q/theta;
%     sigma_d=sigma_d*theta;
    
%     disp("norm d: ")
%     disp(norm(d))
%     disp("norm q: ")
%     disp(norm(q))
    disp("sobel d norm: ")
    disp(norm(grad))
%     disp("sigma q: ")
%     disp(sigma_q)
%     disp("sigma d: ")
%     disp(sigma_d)
%     disp("theta: ")
%     disp(theta)

    imshow(d);
%     imshow(grad);
    drawnow;
%     while (true)
%         w = waitforbuttonpress;
%         if w
%             break;
%         end
%     end
end
timeElapsed = toc
n
