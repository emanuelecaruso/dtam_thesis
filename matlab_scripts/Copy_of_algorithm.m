%% INITIALIZATION

clc
clear

%image
image=im2double(imread("./data2.png"));
% image=[0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0];
% image=[1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 0 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1; 1 1 1 1 1 1 1];
% image=[1 1 1 1 1; 1 1 0 1 1; 1 1 1 1 1];
% image=[1 0 0; 0 1 0; 0 0 1];

% sobel
filter_x=  [-1 0 1;
            -2 0 2;
            -1 0 1];
        
filter_y= [-1 -2 -1;
            0 0 0;
            1 2 1];

% other
% filter_x_=  [-1 0 1;
%             -2 0 2;
%             -1 0 1];
%         
% filter_y_= [-1 -2 -1;
%             0 0 0;
%             1 2 1];

% filter_x_=  [0 0 0;
%             -1 0 1;
%             0 0 0];
%         
% filter_y_= [0 -1 0;
%             0 0 0;
%             0 1 0];
        
% filter_x_=  [0 0 0;
%             -1 1 0;
%             0 0 0];
%         
% filter_y_= [0 -1 0;
%             0 1 0;
%             0 0 0];
        
filter_x_=  [0 0 0;
            1 -2 1;
            0 0 0];
        
filter_y_= [0 1 0;
            0 -2 0;
            0 1 0];
        
% filter_x_= [-1 1 0;
%             -2 2 0;
%             -1 1 0];
%         
% filter_y_= [-1 -2 -1;
%             1 2 1;
%             0 0 0];

% filter_x_= [5 8 10 8 5;
%             4 10 20 10 4;
%             0 0 0 0 0;
%             -4 -10 -20 -10 -4;
%             5 8 10 8 5];
%         
% filter_y_= [-5 -4 0 4 5;
%             -8 -10 0 10 8;
%             -10 -20 0 20 10;
%             -8 -10 0 10 8;
%             -5 -4 0 4 5];

n=0
sign=1
while(true)

    Gx = imfilter(image, filter_x);
    Gy = imfilter(image, filter_y);

    norm_grad=norm(Gx)+norm(Gy)
    Gx_ = imfilter(image, filter_x_);
    Gy_ = imfilter(image, filter_y_);
%     norm_grad_=norm(Gx_)+norm(Gy_)

    grad= [Gx Gy];

    Dx = imfilter(Gx, filter_x);
    Dy = imfilter(Gy, filter_y);
    Dx_ = imfilter(Gx_, filter_x_);
    Dy_ = imfilter(Gy_, filter_y_);

%     div=(Dx+Dy)
%     div=sign*(Dx_+Dy_);
    div=(Dx-Dx_+Dy-Dy_);
    
    a=1;
    b=5;
%     div=(a*Dx-b*Dx_+a*Dy-b*Dy_);

%     a=1/100;
    c=0.02;
    fin=image+c*div;
    imshow(fin);
%     imshow(grad);
    while (true)
        w = waitforbuttonpress;
        if w
            break;
        end
    end
    image=fin;
    n=n+1
    if (n==25)
        filter_x_=  [0 0 0;
            1 -2 1;
            0 0 0];
                
        filter_y_= [0 1 0;
                    0 -2 0;
                    0 1 0];
        sign=-1
    end
end
    


