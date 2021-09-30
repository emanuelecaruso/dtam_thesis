%% INITIALIZATION

clc
clear

%image
image=im2double(imread("./data2.png"));
% image=[0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 1 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0; 0 0 0 0 0 0 0];
% image=[1 1 1 1 1; 1 1 0 1 1; 1 1 1 1 1];

% sobel
filter=  [1 2 1;
        2 4 2;
        1 2 1];
filter=(1/16)*filter;

while(true)

    fin = imfilter(image, filter);

    imshow(fin);
    while (true)
        w = waitforbuttonpress;
        if w
            break;
        end
    end
    image=fin;
end
    


