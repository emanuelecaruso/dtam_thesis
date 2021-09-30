%% INITIALIZATION

clc
clear

%image
image=imread("./data2.png");
I = im2double(image);

filter_x_=  [0 0 0;
            1 -2 1;
            0 0 0];
        
filter_y_= [0 1 0;
            0 -2 0;
            0 1 0];

filter_x=  [1 2 1;
            0 0 0;
            -1 -2 -1];
        
filter_y= [1 0 -1;
           2 0 -2;
           1 0 -1];
        
        
d=I;
a=I;
q=zeros(2*size(d,1),2*size(d,2));

% sigma_q = 0.000347;

sigma_q0 = 0.1;
sigma_d0 = 0.1;
sigma_q = sigma_q0;
sigma_d = sigma_d0;
eps=0.0001;
theta=1;
theta_end=0.00001;
beta=0.001;
% beta=0.0001;
lambda=0;
a_=1;
b_=0;
syms a b real

%% ALGORITHM

n=0;
tic
grad=[];
r=0.99;
    
disp("norm d 0: ")
disp(norm(d))

mode='sobel';
% mode='intermediate';
while theta>theta_end
    
    Gx = imfilter(d, filter_x);
    Gy = imfilter(d, filter_y);
    Gx_ = imfilter(d, filter_x_);
    Gy_ = imfilter(d, filter_y_);
    
%     [Gx, Gy] = imgradientxy(d,mode);
%     [Gx_, Gy_] = imgradientxy(d,"intermediate");
    grad= [Gx Gy; Gx_ Gy_];
%     grad= [Gx+Gx_ Gy+Gy_];
    
    
    %q next
    q_next=(q+sigma_q*grad)/(1+sigma_q*eps);
%     q_next=(sigma_q*grad)/(1+sigma_q*eps);
%     max_q_next=norm(q_next);
    max_q_next=max(abs(q_next(:)));
    if max_q_next<1
        max_q_next=1;
    end
    q_next=q_next/max_q_next;
    
    Qx= q_next(1:(size(q_next,1)/2), 1:(size(q_next,2)/2));
    Qy= q_next(1:(size(q_next,1)/2), (size(q_next,2)/2)+1:size(q_next,2));
    Qx_= q_next( (size(q_next,1)/2)+1:size(q_next,1), 1:(size(q_next,2)/2));
    Qy_= q_next( (size(q_next,1)/2)+1:size(q_next,1), (size(q_next,2)/2)+1:size(q_next,2));
    
    Dx = imfilter(Qx, filter_x);
    Dy = imfilter(Qy, filter_y);
    Dx_ = imfilter(Qx_, filter_x_);
    Dy_ = imfilter(Qy_, filter_y_);
    
%     [Dx, ~] = imgradientxy(Qx,mode);
%     [~, Dy] = imgradientxy(Qy,mode);
%     [Dx_, ~] = imgradientxy(Qx,"intermediate");
%     [~, Dy_] = imgradientxy(Qy,"intermediate");

    div=(a_*Dx+a_*Dy-b_*Dx_-b_*Dy_);
%     a=0.0238;
%     b=0.25;
%     a=1;
%     b=5;
%     div=(a*Dx-b*Dx_+a*Dy-b*Dy_);
    
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
    while (true)
        w = waitforbuttonpress;
        if w
            break;
        end
    end
end
timeElapsed = toc
n
